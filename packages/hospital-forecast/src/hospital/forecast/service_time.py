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
censoring it to the cut would bias every mean downward.
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
    seconds,
)
from hospital.core.events import (
    DischargeCompleted,
    DocumentationCompleted,
    DocumentationStarted,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientArrived,
    ProviderVisitCompleted,
    ProviderVisitStarted,
    TriageCompleted,
    TriageStarted,
)
from hospital.forecast._estimators import GbtRegressor
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
    esi_of, _ = _patient_context(log)
    open_starts: dict[tuple[Activity, PatientId], list[int]] = {}

    for env in log.ordered():
        event = env.event
        for activity, start_type, end_type in _PAIRS:
            if isinstance(event, start_type):
                open_starts.setdefault((activity, event.patient), []).append(event.occurred_at.root)
            elif isinstance(event, end_type):
                pending = open_starts.get((activity, event.patient))
                if not pending:
                    continue
                began = pending.pop(0)
                patient = roster.get(event.patient)
                if patient is None:
                    continue
                key = ServiceTimeKey(
                    activity=activity,
                    esi=esi_of.get(event.patient, patient.esi),
                    complaint=patient.complaint,
                )
                out.setdefault(key, []).append((event.occurred_at.root - began) / 1_000_000)


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
) -> ServiceTimeRegressor:
    """Fit the point head plus any quantile heads on an already-labelled frame."""
    if frame.labels is None:
        raise ValueError("a training frame must carry labels")
    if not frame.matrix:
        raise ValueError("a training frame must have rows")

    def state(name: str) -> int:
        return int(streams.substream("forecast", name).integers(0, 2**31 - 1))

    point = GbtRegressor(random_state=state(f"{target}_point")).fit(frame.matrix, frame.labels)
    # Duan's smearing estimator, on the fold the point head was fit on: the empirical
    # mean of exp(residual). Distribution-free, unlike exp(sigma^2/2).
    fitted = point.predict(frame.matrix)
    residuals = [y - p for y, p in zip(frame.labels, fitted, strict=True)]
    smearing = math.fsum(math.exp(r) for r in residuals) / len(residuals) if residuals else 1.0
    heads = {
        q: GbtRegressor(quantile=q, random_state=state(f"{target}_q{q}")).fit(
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
    "LognormalParams",
    "ServiceTimeKey",
    "ServiceTimeRegressor",
    "ServiceTimeTable",
    "activity_durations",
    "fit_service_time_regressor",
    "fit_service_time_table",
    "los_training_frame",
    "patient_los",
    "static_service_table",
    "table_baseline_log_error",
]
