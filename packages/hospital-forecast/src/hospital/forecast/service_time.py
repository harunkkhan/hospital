"""Service-time and LOS estimation (doc 06 §6).

Two complementary estimators, because they answer different questions:

* :class:`ServiceTimeTable` — a per-``(Activity, EsiAcuity, complaint)`` lognormal
  fit by method of moments. It is keyed on exactly the axes
  ``sim.physics.service_times`` samples on, and parameterized as the ``(mean_s,
  cv)`` pair ``core.rng.sample_lognormal`` takes, so a fitted table can be handed
  straight back to the sampler. Fit and draw cannot drift apart because they name
  the same thing.
* :class:`ServiceTimeRegressor` — a per-patient GBT over the full feature row,
  predicting **log-duration**. Durations are lognormal-ish, so log space keeps the
  loss symmetric and stops a single four-hour stay from dominating the fit.

What they feed: the expected durations behind ``solver.placement``'s ``w[p,b]``,
the ``solver.objective`` acuity-weighted-time term, turnaround value, and
discharge estimates. In M1 those are static scenario means; the
``PredictionAdapter`` swaps in these predictions with no change at the call site.

Durations are read from paired ``*Started``/``*Completed`` events. An unpaired
start (the run ended mid-visit) is dropped rather than closed at the horizon —
closing it at the cut would report a duration that never happened.

**Known bias, not fixed here: informative right-censoring.** Dropping the
incomplete observation does not remove the censoring, it only hides it. Whether an
episode completes before the horizon depends on *how long it is*, so the surviving
sample is length-biased: of two otherwise identical late arrivals, the ten-minute
stay is retained and the four-hour stay is discarded. Both
:func:`activity_durations` and :func:`patient_los` are therefore biased **downward**,
and the bias grows with the fraction of the horizon that lies near the end.

Correcting it properly needs survival analysis — a Kaplan-Meier or
accelerated-failure-time fit that uses the censored observations as the inequalities
they are — which is out of scope for v1 and would change this module's output type.
What v1 does instead is make the exposure visible: :func:`censoring_report` states
how many episodes were dropped and what share of the sample that is, so a caller can
judge whether the bias is negligible for their horizon rather than assume it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

from hospital.core import (
    Activity,
    Duration,
    EsiAcuity,
    EventLog,
    FrozenModel,
    Patient,
    PatientId,
    StaffId,
    seconds,
)
from hospital.core.events import (
    DischargeCompleted,
    DisruptionInjected,
    DocumentationCompleted,
    DocumentationStarted,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientArrived,
    ProviderVisitCompleted,
    ProviderVisitStarted,
    TestOrdered,
    TriageCompleted,
    TriageStarted,
)
from hospital.forecast._estimators import GbtRegressor, GbtSettings
from hospital.forecast.features import (
    PATIENT_FEATURE_NAMES,
    ComplaintEncoder,
    FeatureFrame,
    PatientFeatures,
    to_matrix,
)

if TYPE_CHECKING:
    from hospital.core import RandomStreams

# Below this many observations a key is not estimated on its own — it borrows the
# per-Activity fallback. A three-sample lognormal fit is noise wearing a number.
DEFAULT_MIN_SAMPLES: Final[int] = 30

# The paired start/complete event types each Activity is measured from.
_PAIRS: Final[tuple[tuple[Activity, type, type], ...]] = (
    (Activity.TRIAGE, TriageStarted, TriageCompleted),
    (Activity.PROVIDER_VISIT, ProviderVisitStarted, ProviderVisitCompleted),
    (Activity.NURSE_VISIT, NurseVisitStarted, NurseVisitCompleted),
    (Activity.DOCUMENTATION, DocumentationStarted, DocumentationCompleted),
)

TargetKind = Literal["los", "activity_duration"]

# One run's event log paired with the roster that log's patient ids refer to. Ids are
# unique only within a run, so the two must travel together.
Run = tuple[EventLog, Mapping[PatientId, Patient]]


class ServiceTimeKey(FrozenModel):
    """The fit's key — the SAME axes ``sim.physics.service_times`` samples on."""

    activity: Activity
    esi: EsiAcuity
    complaint: str


class LognormalParams(FrozenModel):
    """``(mean_s, cv)`` — exactly the arguments ``sample_lognormal`` takes."""

    mean_s: float
    cv: float
    n: int


class ServiceTimeTable(FrozenModel):
    """Per-key lognormal parameters, with a per-Activity fallback for thin keys."""

    params: Mapping[ServiceTimeKey, LognormalParams]
    fallbacks: Mapping[Activity, LognormalParams]

    def _lookup(self, key: ServiceTimeKey) -> LognormalParams | None:
        return self.params.get(key) or self.fallbacks.get(key.activity)

    def expected(self, key: ServiceTimeKey) -> Duration:
        """Mean duration for ``key``. Raises rather than invent a zero.

        A missing key with no fallback means the corpus never observed this
        activity at all; returning ``Duration(0)`` would quietly tell the solver
        that the work is free.
        """
        found = self._lookup(key)
        if found is None:
            raise KeyError(f"no fitted service time for {key.activity}/{key.esi}/{key.complaint}")
        return seconds(found.mean_s)

    def cv(self, key: ServiceTimeKey) -> float:
        found = self._lookup(key)
        if found is None:
            raise KeyError(f"no fitted service time for {key.activity}/{key.esi}/{key.complaint}")
        return found.cv

    def has(self, key: ServiceTimeKey) -> bool:
        return self._lookup(key) is not None


def _moments(samples: Sequence[float]) -> tuple[float, float]:
    """Method of moments on the natural scale: ``(mean, cv)``.

    Fitting on the natural scale rather than log space is deliberate — it is the
    parameterization ``sample_lognormal`` consumes, so a round trip through
    fit-then-sample reproduces the observed mean exactly. A log-space MLE would
    recover ``(mu, sigma)`` and need converting, which is one more place for the
    two halves to disagree.
    """
    n = len(samples)
    mean = math.fsum(samples) / n
    if n < 2 or mean <= 0.0:
        return mean, 0.0
    variance = math.fsum((s - mean) ** 2 for s in samples) / (n - 1)
    return mean, math.sqrt(variance) / mean


def _patient_context(log: EventLog) -> tuple[dict[PatientId, EsiAcuity], dict[PatientId, int]]:
    """Triaged acuity and arrival instant per patient, from the log alone."""
    esi: dict[PatientId, EsiAcuity] = {}
    arrived: dict[PatientId, int] = {}
    for env in log.ordered():
        event = env.event
        if isinstance(event, TriageCompleted):
            esi[event.patient] = event.esi
        elif isinstance(event, PatientArrived):
            arrived[event.patient] = event.occurred_at.root
    return esi, arrived


def activity_durations(runs: Sequence[Run]) -> dict[ServiceTimeKey, list[float]]:
    """Observed durations in seconds, keyed by ``(activity, esi, complaint)``.

    Takes ``(log, roster)`` **pairs**, and each log is read against its own roster.
    Two reasons, both load-bearing:

    * Separate runs are separate timelines that each start at the week's origin, so
      splicing the logs together would interleave unrelated instants.
    * Patient ids are only unique *within* a run — ``data.workload`` mints
      ``p_000_00`` afresh every week. Pooling the rosters into one mapping would let
      a later week's registration silently overwrite an earlier one, and week 1's
      durations would be filed under week 2's complaint and acuity.

    Only the resulting durations are pooled. Starts are matched to completions **in
    order per patient**, so a patient with two provider visits contributes two
    independent durations rather than one span covering both.
    """
    out: dict[ServiceTimeKey, list[float]] = {}
    for log, roster in runs:
        _accumulate_durations(log, roster, out)
    return out


def _accumulate_durations(
    log: EventLog,
    roster: Mapping[PatientId, Patient],
    out: dict[ServiceTimeKey, list[float]],
) -> None:
    """Walk one log, attributing each closed service episode to its true activity.

    ``sim`` deliberately emits the **nurse-visit** event pair for a bedside lab draw
    too — the schema has no ``Lab*`` pair, and a draw genuinely is nurse direct care
    (``physics/executor.py``). But the sampler draws the two from *different*
    distributions (480s vs 300s base means), so filing both under ``NURSE_VISIT``
    pooled them into a fitted mean that matched neither, and no ``LAB`` estimate
    existed at all. That silently broke this module's central guarantee: that a
    fitted key names the same thing the sampler keys on.

    They are separable from the log. The flow orders a lab, draws it, then results it
    (``flow/patient.py``), so the first nurse-visit episode opened after a
    ``TestOrdered(LAB)`` and before its ``TestResulted`` IS the draw. One pending
    marker per ordered lab, consumed in order.
    """
    esi_of, _ = _patient_context(log)
    # (instant, staff) of every staff-absence injection. The engine emits a
    # `*Completed` at the interruption instant AND requeues the unfinished task, so a
    # 20-minute visit cut short after two contributes BOTH a 2-minute observation and
    # the later full one -- biasing the fit down and inflating `n`. The disruption
    # event names the instant and the staff member, so the truncated episode can be
    # identified exactly rather than guessed at.
    truncations: set[tuple[int, str]] = {
        (env.event.occurred_at.root, env.event.detail)
        for env in log.ordered()
        if isinstance(env.event, DisruptionInjected) and env.event.disruption == "staff_absence"
    }
    truncated_instants = {instant for instant, _ in truncations}

    def is_truncated(instant: int, staff: StaffId | None) -> bool:
        if staff is not None:
            return (instant, staff.root) in truncations
        # Triage completions carry `esi`, not `staff`, so they can only be matched on
        # the instant. Looser, and stated rather than hidden.
        return instant in truncated_instants

    open_starts: dict[tuple[Activity, PatientId], list[int]] = {}
    # Per patient: how many ordered-but-not-yet-drawn labs are outstanding. A
    # nurse-visit start while one is outstanding is that draw.
    pending_draws: dict[PatientId, int] = {}
    # Per patient, per open nurse-visit episode: whether it was tagged as a lab draw,
    # so the *completed* event attributes the duration the same way its start did.
    draw_flags: dict[PatientId, list[bool]] = {}

    def record(
        activity: Activity,
        patient_id: PatientId,
        began: int,
        ended: int,
        staff: StaffId | None = None,
    ) -> None:
        if is_truncated(ended, staff):
            return  # an interrupted episode, not a completed service
        patient = roster.get(patient_id)
        if patient is None:
            return
        key = ServiceTimeKey(
            activity=activity,
            esi=esi_of.get(patient_id, patient.esi),
            complaint=patient.complaint,
        )
        out.setdefault(key, []).append((ended - began) / 1_000_000)

    for env in log.ordered():
        event = env.event
        if isinstance(event, TestOrdered) and event.activity is Activity.LAB:
            pending_draws[event.patient] = pending_draws.get(event.patient, 0) + 1
            continue
        if isinstance(event, NurseVisitStarted):
            is_draw = pending_draws.get(event.patient, 0) > 0
            if is_draw:
                pending_draws[event.patient] -= 1
            open_starts.setdefault((Activity.NURSE_VISIT, event.patient), []).append(
                event.occurred_at.root
            )
            draw_flags.setdefault(event.patient, []).append(is_draw)
            continue
        if isinstance(event, NurseVisitCompleted):
            pending = open_starts.get((Activity.NURSE_VISIT, event.patient))
            flags = draw_flags.get(event.patient)
            if not pending or not flags:
                continue
            record(
                Activity.LAB if flags.pop(0) else Activity.NURSE_VISIT,
                event.patient,
                pending.pop(0),
                event.occurred_at.root,
                getattr(event, "staff", None),
            )
            continue
        for activity, start_type, end_type in _PAIRS:
            if activity is Activity.NURSE_VISIT:
                continue  # handled above, with lab-draw attribution
            if isinstance(event, start_type):
                open_starts.setdefault((activity, event.patient), []).append(event.occurred_at.root)
            elif isinstance(event, end_type):
                pending = open_starts.get((activity, event.patient))
                if not pending:
                    continue
                record(
                    activity,
                    event.patient,
                    pending.pop(0),
                    event.occurred_at.root,
                    getattr(event, "staff", None),
                )


def fit_service_time_table(
    runs: Sequence[Run],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> ServiceTimeTable:
    """Method-of-moments lognormal fits per key, with per-Activity fallbacks.

    ``runs`` are ``(log, roster)`` pairs — see :func:`activity_durations` for why the
    roster travels with its own log rather than being pooled.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    observed = activity_durations(runs)

    by_activity: dict[Activity, list[float]] = {}
    for key, samples in observed.items():
        by_activity.setdefault(key.activity, []).extend(samples)

    params: dict[ServiceTimeKey, LognormalParams] = {}
    for key, samples in observed.items():
        if len(samples) < min_samples:
            continue
        mean, cv = _moments(samples)
        params[key] = LognormalParams(mean_s=mean, cv=cv, n=len(samples))

    fallbacks: dict[Activity, LognormalParams] = {}
    for activity, samples in by_activity.items():
        mean, cv = _moments(samples)
        fallbacks[activity] = LognormalParams(mean_s=mean, cv=cv, n=len(samples))

    return ServiceTimeTable(params=params, fallbacks=fallbacks)


class CensoringReport(FrozenModel):
    """How much of the sample right-censoring removed (see the module docstring).

    ``dropped_share`` is the number to look at: a few tenths of a percent means the
    downward bias is negligible, while a tenth of the sample means the fitted means
    are meaningfully short and a survival fit is needed before trusting them.
    """

    completed: int
    censored: int

    @property
    def dropped_share(self) -> float:
        total = self.completed + self.censored
        return self.censored / total if total else 0.0


def censoring_report(runs: Sequence[Run]) -> CensoringReport:
    """Count completed vs still-open stays, so the censoring bias can be bounded.

    Reported rather than corrected: dropping the open stays is length-biased (a long
    stay is likelier to be cut off), and quantifying the exposure is what lets a
    caller decide whether that matters at their horizon.
    """
    completed = 0
    censored = 0
    for log, _roster in runs:
        arrived: set[PatientId] = set()
        exited: set[PatientId] = set()
        for env in log.ordered():
            if isinstance(env.event, PatientArrived):
                arrived.add(env.event.patient)
            elif isinstance(env.event, DischargeCompleted):
                exited.add(env.event.patient)
        completed += len(exited)
        censored += len(arrived - exited)
    return CensoringReport(completed=completed, censored=censored)


def patient_los(log: EventLog) -> dict[PatientId, float]:
    """Completed length of stay in seconds, per patient.

    Only patients whose discharge is in the log appear. A patient still in the
    department has no LOS yet — imputing one from the cut would train the model
    to predict the cut.
    """
    _, arrived = _patient_context(log)
    out: dict[PatientId, float] = {}
    for env in log.ordered():
        event = env.event
        if isinstance(event, DischargeCompleted) and event.patient in arrived:
            out[event.patient] = (event.occurred_at.root - arrived[event.patient]) / 1_000_000
    return out


class ServiceTimeRegressor:
    """A GBT over log-duration, with optional quantile heads for p90 KPI needs."""

    def __init__(
        self,
        point: GbtRegressor,
        quantile_heads: Mapping[float, GbtRegressor],
        *,
        feature_names: tuple[str, ...],
        target: TargetKind,
        encoder: ComplaintEncoder,
        smearing: float = 1.0,
    ) -> None:
        self.feature_names = feature_names
        self.target: TargetKind = target
        self.quantiles = tuple(sorted(quantile_heads))
        self._point = point
        self._heads = dict(quantile_heads)
        self._encoder = encoder
        # Duan smearing factor: mean(exp(residual)) on the training fold. 1.0 means
        # "uncorrected", which is only right for a noiseless fit.
        self._smearing = smearing

    def _row(self, features: PatientFeatures) -> list[list[float]]:
        frame = to_matrix(
            [features],
            feature_names=self.feature_names,
            row_ids=[features.patient.root],
            complaints=self._encoder,
        )
        return [list(frame.matrix[0])]

    def predict_median_los(self, features: PatientFeatures) -> Duration:
        """The conditional **median** stay.

        Exponentiating a squared-error fit on ``log(duration)`` recovers
        ``exp(E[log LOS])``, which is the median of a lognormal — not its mean. Named
        for what it is, because the difference is large and one-directional.
        """
        (predicted,) = self._point.predict(self._row(features))
        return seconds(math.exp(predicted))

    def predict_expected_los(self, features: PatientFeatures) -> Duration:
        """The conditional **mean** stay — what an occupancy cost actually needs.

        ``exp(E[log LOS])`` understates ``E[LOS]`` by ``exp(sigma^2/2)``: at a
        residual log-SD of 1 a one-hour median is a 1.65-hour mean, so feeding the
        median into a bay-occupancy term systematically under-books every long stay.

        Corrected by **Duan's smearing estimator** — the empirical mean of
        ``exp(residual)`` on the training fold — rather than the parametric
        ``exp(sigma^2/2)``. Smearing needs no lognormality assumption, and the
        residuals here are not guaranteed to be Gaussian.
        """
        return seconds(math.exp(self._log_point(features)) * self._smearing)

    @property
    def smearing(self) -> float:
        """Duan's factor, ``mean(exp(residual))`` on the training fold.

        Exposed because it is the size of the median-to-mean correction, and a
        consumer weighing whether the distinction matters for its data should be able
        to read it rather than infer it.
        """
        return self._smearing

    def _log_point(self, features: PatientFeatures) -> float:
        (predicted,) = self._point.predict(self._row(features))
        return predicted

    def point_log_error(self, frame: FeatureFrame) -> float:
        """MAE on the log scale the point head was fit on — the honest scoring axis.

        Exposed here rather than making callers reach for the wrapped estimator:
        the frame's column order has to match ``feature_names``, and that check
        belongs with the model that owns the contract.
        """
        if frame.labels is None:
            raise ValueError("scoring needs labels")
        if frame.feature_names != self.feature_names:
            raise ValueError("frame columns do not match the model's feature_names")
        predicted = self._point.predict(frame.matrix)
        return math.fsum(abs(p - y) for p, y in zip(predicted, frame.labels, strict=True)) / len(
            frame.labels
        )

    def predict_quantile(self, features: PatientFeatures, q: float) -> Duration:
        head = self._heads.get(q)
        if head is None:
            raise KeyError(f"no fitted head for quantile {q}; have {self.quantiles}")
        (predicted,) = head.predict(self._row(features))
        return seconds(math.exp(predicted))


def los_training_frame(
    rows: Sequence[PatientFeatures],
    los_seconds: Mapping[PatientId, float],
    encoder: ComplaintEncoder,
    *,
    feature_names: tuple[str, ...] = PATIENT_FEATURE_NAMES,
) -> FeatureFrame:
    """Join feature rows to their realized LOS, keeping only completed stays."""
    paired = [(row, los_seconds[row.patient]) for row in rows if row.patient in los_seconds]
    return to_matrix(
        [row for row, _ in paired],
        feature_names=feature_names,
        row_ids=[row.patient.root for row, _ in paired],
        labels=[math.log(max(value, 1.0)) for _, value in paired],
        complaints=encoder,
    )


def fit_service_time_regressor(
    frame: FeatureFrame,
    encoder: ComplaintEncoder,
    *,
    streams: RandomStreams,
    target: TargetKind = "los",
    quantiles: Sequence[float] = (0.9,),
    settings: GbtSettings | None = None,
) -> ServiceTimeRegressor:
    """Fit the point head plus any quantile heads on an already-labelled frame."""
    if frame.labels is None:
        raise ValueError("a training frame must carry labels")
    if not frame.matrix:
        raise ValueError("a training frame must have rows")

    def state(name: str) -> int:
        return int(streams.substream("forecast", name).integers(0, 2**31 - 1))

    point = GbtRegressor(random_state=state(f"{target}_point"), settings=settings).fit(
        frame.matrix, frame.labels
    )
    # Duan's smearing estimator, on the fold the point head was fit on: the empirical
    # mean of exp(residual). Distribution-free, unlike exp(sigma^2/2).
    fitted = point.predict(frame.matrix)
    residuals = [y - p for y, p in zip(frame.labels, fitted, strict=True)]
    smearing = math.fsum(math.exp(r) for r in residuals) / len(residuals) if residuals else 1.0
    heads = {
        q: GbtRegressor(quantile=q, random_state=state(f"{target}_q{q}"), settings=settings).fit(
            frame.matrix, frame.labels
        )
        for q in quantiles
    }
    return ServiceTimeRegressor(
        point,
        heads,
        feature_names=frame.feature_names,
        target=target,
        encoder=encoder,
        smearing=smearing,
    )


def table_baseline_log_error(
    rows: Sequence[PatientFeatures],
    los_seconds: Mapping[PatientId, float],
    predicted_seconds: float,
) -> float:
    """The "predict one global mean" baseline the regressor has to beat."""
    paired = [los_seconds[row.patient] for row in rows if row.patient in los_seconds]
    target = math.log(max(predicted_seconds, 1.0))
    return math.fsum(abs(target - math.log(max(v, 1.0))) for v in paired) / len(paired)


def static_service_table(
    means_by_activity: Mapping[Activity, float], *, cv: float = 0.4
) -> ServiceTimeTable:
    """A table from scenario constants — the A/B baseline arm's path (doc 06 §9)."""
    return ServiceTimeTable(
        params={},
        fallbacks={
            activity: LognormalParams(mean_s=mean, cv=cv, n=0)
            for activity, mean in means_by_activity.items()
        },
    )


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "CensoringReport",
    "LognormalParams",
    "ServiceTimeKey",
    "ServiceTimeRegressor",
    "ServiceTimeTable",
    "activity_durations",
    "censoring_report",
    "fit_service_time_regressor",
    "fit_service_time_table",
    "los_training_frame",
    "patient_los",
    "static_service_table",
    "table_baseline_log_error",
]
