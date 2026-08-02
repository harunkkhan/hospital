"""Leakage-safe feature extraction from the ``EventLog``, vitals, and the roster (doc 06 §4).

**The invariant of this module is the cutoff.** Every extractor takes an
``as_of`` (or a window end) and may use only facts observable at that instant.
It is enforced *structurally*, not by discipline: each extractor filters the
envelope stream exactly once, up front, and everything downstream sees only that
prefix. A row built for a placement decision at time *t* therefore cannot know
the patient's eventual length of stay, later visits, or disposition — which is
the difference between a model that works and a model that only appears to.

Two feeds, one feature definition. Offline training reads
``data.vitals.VitalsStream``; the live monitor is handed ``VitalsSampled``
envelopes and rebuilds the *same* :class:`VitalsWindowFeatures` through
:func:`online_vitals_features`. If those two ever diverged, a model validated
offline would be scoring different inputs in production.

**Deviation from doc 06 §4's signature, on purpose.** ``patient_features`` also
takes the arrival roster. ``complaint`` and ``isolation_required`` are listed as
features there but appear nowhere in ``core.events`` — they are registration
facts, known at arrival and never written to the log. Passing the roster is not
leakage (these are arrival-time facts, and the cutoff still governs every
event-derived field); inventing them from later events would have been.

Encoding lives beside each row type as ``numeric_features()`` so the mapping
from typed row to model input has exactly one definition. The one *fitted*
encoding — ``complaint`` — is a :class:`ComplaintEncoder` fit on the training
split alone and shipped inside the artifact (doc 06 §13-9).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from hospital.core import (
    ArrivalMode,
    Duration,
    EsiAcuity,
    EventLog,
    FrozenModel,
    OperatingWeek,
    Patient,
    PatientId,
    SimTime,
    TimeWindow,
    hours,
)
from hospital.core.events import (
    BayAssigned,
    BayCleaningCompleted,
    DischargeCompleted,
    EventEnvelope,
    PatientArrived,
    StaffIdle,
    StaffMoved,
    TriageCompleted,
    VitalsSampled,
)
from hospital.data.vitals import VitalsSample, VitalsStream

if TYPE_CHECKING:
    from collections.abc import Callable

_MICROS_PER_HOUR: Final[int] = 3_600 * 1_000_000
_HOURS_PER_DAY: Final[int] = 24
_DAYS_PER_WEEK: Final[int] = 7

# Distinct staff seen inside this lookback count as "on shift" (see the note on
# `staff_on_shift` in `patient_features`).
_STAFF_ACTIVITY_LOOKBACK: Final[Duration] = hours(1)

# The bucket every complaint unseen during training maps to (doc 06 §13-9).
UNSEEN_COMPLAINT: Final[float] = -1.0


# --------------------------------------------------------------------- rows
class PatientFeatures(FrozenModel):
    """One row per patient, as of a cutoff. Nothing after ``as_of`` informs it."""

    patient: PatientId
    as_of: SimTime
    esi: EsiAcuity
    arrival_mode: ArrivalMode
    complaint: str
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    is_weekend: bool
    provider_visits: int
    nurse_visits: int
    imaging_count: int
    labs: int
    procedures: int
    isolation_required: bool
    # Congestion AT arrival — every value reduced from events at or before `as_of`.
    wip: int
    bays_occupied: int
    queue_len_by_esi: tuple[int, int, int, int, int]
    staff_on_shift: int
    arrivals_last_hour: int

    def numeric_features(self) -> Mapping[str, float]:
        """The model-facing encoding. ``complaint`` is fitted, so it is added later."""
        queues = {f"queue_len_esi_{i + 1}": float(n) for i, n in enumerate(self.queue_len_by_esi)}
        return {
            "esi": float(int(self.esi)),
            "is_ambulance": float(self.arrival_mode is ArrivalMode.AMBULANCE),
            "hour_sin": self.hour_sin,
            "hour_cos": self.hour_cos,
            "dow_sin": self.dow_sin,
            "dow_cos": self.dow_cos,
            "is_weekend": float(self.is_weekend),
            "provider_visits": float(self.provider_visits),
            "nurse_visits": float(self.nurse_visits),
            "imaging_count": float(self.imaging_count),
            "labs": float(self.labs),
            "procedures": float(self.procedures),
            "isolation_required": float(self.isolation_required),
            "wip": float(self.wip),
            "bays_occupied": float(self.bays_occupied),
            **queues,
            "staff_on_shift": float(self.staff_on_shift),
            "arrivals_last_hour": float(self.arrivals_last_hour),
        }


class WindowFeatures(FrozenModel):
    """One row per time bin over the week — the arrivals/surge feed."""

    window: TimeWindow
    hour_of_day: int
    dow: int
    is_weekend: bool
    lag_1h: int
    lag_2h: int
    lag_24h: int
    lag_1w: int
    roll_mean_3h: float
    roll_std_3h: float
    count: int

    def numeric_features(self) -> Mapping[str, float]:
        return {
            "hour_sin": _sin_of(self.hour_of_day, _HOURS_PER_DAY),
            "hour_cos": _cos_of(self.hour_of_day, _HOURS_PER_DAY),
            "dow_sin": _sin_of(self.dow, _DAYS_PER_WEEK),
            "dow_cos": _cos_of(self.dow, _DAYS_PER_WEEK),
            "is_weekend": float(self.is_weekend),
            "lag_1h": float(self.lag_1h),
            "lag_2h": float(self.lag_2h),
            "lag_24h": float(self.lag_24h),
            "lag_1w": float(self.lag_1w),
            "roll_mean_3h": self.roll_mean_3h,
            "roll_std_3h": self.roll_std_3h,
        }


class VitalsWindowFeatures(FrozenModel):
    """One row per rolling vitals window — the deterioration feed."""

    patient: PatientId
    window_end: Duration
    hr_mean: float
    hr_min: float
    hr_max: float
    hr_slope: float
    spo2_min: float
    spo2_mean: float
    sbp_min: float
    temp_mean: float
    resp_max: float
    resp_slope: float
    news2_total: int
    news2_sub: tuple[int, ...]
    time_since_arrival: Duration
    esi: EsiAcuity

    def numeric_features(self) -> Mapping[str, float]:
        subs = {f"news2_sub_{i}": float(v) for i, v in enumerate(self.news2_sub)}
        return {
            "hr_mean": self.hr_mean,
            "hr_min": self.hr_min,
            "hr_max": self.hr_max,
            "hr_slope": self.hr_slope,
            "spo2_min": self.spo2_min,
            "spo2_mean": self.spo2_mean,
            "sbp_min": self.sbp_min,
            "temp_mean": self.temp_mean,
            "resp_max": self.resp_max,
            "resp_slope": self.resp_slope,
            "news2_total": float(self.news2_total),
            **subs,
            "time_since_arrival_h": self.time_since_arrival.root / _MICROS_PER_HOUR,
            "esi": float(int(self.esi)),
        }


class FeatureFrame(FrozenModel):
    """The dense bridge to sklearn/lightgbm.

    ``row_ids`` is identity (a patient or window key), never a feature — it is
    carried so a prediction can be traced back to its subject, and deliberately
    excluded from ``matrix`` so no model can learn from an id.
    """

    feature_names: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    row_ids: tuple[str, ...]
    labels: tuple[float, ...] | None = None

    def __len__(self) -> int:
        return len(self.matrix)


class ComplaintEncoder(FrozenModel):
    """Ordinal encoding for ``complaint``, fit on the training split only.

    Fitting on the full dataset would leak the holdout's category distribution
    into training. Categories are assigned in sorted order so the encoding is a
    pure function of the training vocabulary, and anything unseen collapses to
    :data:`UNSEEN_COMPLAINT` rather than silently taking some other complaint's
    code — a wrong-but-plausible code is worse than an explicit unknown.
    """

    codes: Mapping[str, float]

    @classmethod
    def fit(cls, complaints: Iterable[str]) -> ComplaintEncoder:
        return cls(codes={name: float(i) for i, name in enumerate(sorted(set(complaints)))})

    def encode(self, complaint: str) -> float:
        return self.codes.get(complaint, UNSEEN_COMPLAINT)


# ------------------------------------------------------------------ helpers
def _sin_of(value: int, period: int) -> float:
    return math.sin(2.0 * math.pi * value / period)


def _cos_of(value: int, period: int) -> float:
    return math.cos(2.0 * math.pi * value / period)


def _hour_of_week(t: SimTime, week: OperatingWeek) -> int:
    """Hours since the week's start, wrapped — the seasonal index (doc 06 §5)."""
    elapsed_h = (t.root - week.start.root) // _MICROS_PER_HOUR
    return int(elapsed_h % (_HOURS_PER_DAY * _DAYS_PER_WEEK))


def prefix(log: EventLog, as_of: SimTime | None) -> tuple[EventEnvelope, ...]:
    """Canonically-ordered envelopes at or before ``as_of`` — the ONE cutoff filter.

    Every extractor routes through this. Centralizing it is what makes the
    no-leakage claim structural: there is a single place a future event could
    slip in, and it is three lines long.
    """
    ordered = log.ordered()
    if as_of is None:
        return ordered
    return tuple(env for env in ordered if env.event.occurred_at.root <= as_of.root)


def _slope_per_hour(values: Sequence[float], elapsed_us: Sequence[int]) -> float:
    """Least-squares slope in units per hour. Fewer than two points has no slope."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = [t / _MICROS_PER_HOUR for t in elapsed_us]
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(values) / n
    denominator = math.fsum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return 0.0
    numerator = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=True))
    return numerator / denominator


# ------------------------------------------------------- patient extraction
class _Congestion(FrozenModel):
    wip: int
    bays_occupied: int
    queue_len_by_esi: tuple[int, int, int, int, int]
    staff_on_shift: int
    arrivals_last_hour: int


def _congestion_at_arrivals(envelopes: Sequence[EventEnvelope]) -> dict[PatientId, _Congestion]:
    """Snapshot floor congestion at each patient's arrival, in one O(events) pass.

    The ``EventLog`` is the only source (nuance 1.4): no parallel state is kept
    anywhere else, so these counters cannot drift from what actually happened.
    Waiting-queue depth is by *triaged* acuity, so a patient who has arrived but
    not yet been triaged is in ``wip`` without being in any ESI queue — which is
    exactly the state the floor is in.
    """
    snapshots: dict[PatientId, _Congestion] = {}
    in_system: set[PatientId] = set()
    occupied: set[str] = set()
    waiting_esi: dict[PatientId, EsiAcuity] = {}
    placed: set[PatientId] = set()
    arrival_times: list[int] = []
    staff_seen: list[tuple[int, str]] = []

    for env in envelopes:
        event = env.event
        now = event.occurred_at.root
        if isinstance(event, StaffMoved | StaffIdle):
            staff_seen.append((now, event.staff.root))
            continue
        if isinstance(event, TriageCompleted):
            if event.patient not in placed:
                waiting_esi[event.patient] = event.esi
            continue
        if isinstance(event, BayAssigned):
            occupied.add(event.bay.root)
            placed.add(event.patient)
            waiting_esi.pop(event.patient, None)
            continue
        if isinstance(event, BayCleaningCompleted):
            occupied.discard(event.bay.root)
            continue
        if isinstance(event, DischargeCompleted):
            in_system.discard(event.patient)
            waiting_esi.pop(event.patient, None)
            continue
        if not isinstance(event, PatientArrived):
            continue

        # A PatientArrived: snapshot the floor as the arriving patient finds it,
        # BEFORE counting them into it.
        cutoff = now - _STAFF_ACTIVITY_LOOKBACK.root
        queues = [0, 0, 0, 0, 0]
        for esi in waiting_esi.values():
            queues[int(esi) - 1] += 1
        snapshots[event.patient] = _Congestion(
            wip=len(in_system),
            bays_occupied=len(occupied),
            queue_len_by_esi=(queues[0], queues[1], queues[2], queues[3], queues[4]),
            staff_on_shift=len({sid for at, sid in staff_seen if at >= cutoff}),
            arrivals_last_hour=sum(1 for t in arrival_times if t >= cutoff),
        )
        in_system.add(event.patient)
        arrival_times.append(now)

    return snapshots


def patient_features(
    log: EventLog,
    roster: Mapping[PatientId, Patient],
    week: OperatingWeek,
    *,
    as_of: SimTime | None = None,
) -> tuple[PatientFeatures, ...]:
    """One row per patient who had arrived by ``as_of``.

    ``roster`` supplies the arrival-time registration facts the log does not
    carry (``complaint``, ``isolation_required``, the workup order) — see the
    module docstring. Workup counts are what the chart *calls for* at
    registration, not what has been delivered so far; they are an input to the
    stay, not a summary of it.
    """
    envelopes = prefix(log, as_of)
    congestion = _congestion_at_arrivals(envelopes)
    triaged: dict[PatientId, EsiAcuity] = {
        env.event.patient: env.event.esi
        for env in envelopes
        if isinstance(env.event, TriageCompleted)
    }

    rows: list[PatientFeatures] = []
    for env in envelopes:
        event = env.event
        if not isinstance(event, PatientArrived):
            continue
        patient = roster.get(event.patient)
        if patient is None:
            continue
        arrival = event.occurred_at
        how = _hour_of_week(arrival, week)
        hour_of_day, dow = how % _HOURS_PER_DAY, how // _HOURS_PER_DAY
        crowd = congestion[event.patient]
        rows.append(
            PatientFeatures(
                patient=event.patient,
                as_of=as_of if as_of is not None else arrival,
                # Acuity is known once triage completes; before that the arrival
                # descriptor's own acuity is the best available estimate.
                esi=triaged.get(event.patient, patient.esi),
                arrival_mode=event.mode,
                complaint=patient.complaint,
                hour_sin=_sin_of(hour_of_day, _HOURS_PER_DAY),
                hour_cos=_cos_of(hour_of_day, _HOURS_PER_DAY),
                dow_sin=_sin_of(dow, _DAYS_PER_WEEK),
                dow_cos=_cos_of(dow, _DAYS_PER_WEEK),
                is_weekend=dow >= 5,
                provider_visits=patient.workup.provider_visits,
                nurse_visits=patient.workup.nurse_visits,
                imaging_count=len(patient.workup.imaging),
                labs=patient.workup.labs,
                procedures=patient.workup.procedures,
                isolation_required=patient.isolation_required,
                wip=crowd.wip,
                bays_occupied=crowd.bays_occupied,
                queue_len_by_esi=crowd.queue_len_by_esi,
                staff_on_shift=crowd.staff_on_shift,
                arrivals_last_hour=crowd.arrivals_last_hour,
            )
        )
    return tuple(rows)


# -------------------------------------------------------- window extraction
def window_features(
    log: EventLog, week: OperatingWeek, *, bin_width: Duration | None = None
) -> tuple[WindowFeatures, ...]:
    """Binned arrival counts with lag/rolling context — the surge model's feed.

    Lags reach backwards only; a bin's own ``count`` is the label and never a
    feature, so a row cannot predict itself.
    """
    width = bin_width if bin_width is not None else hours(1)
    if width.root <= 0:
        raise ValueError("bin_width must be positive")
    span = week.end.root - week.start.root
    n_bins = span // width.root
    counts = [0] * n_bins
    for env in prefix(log, None):
        event = env.event
        if not isinstance(event, PatientArrived):
            continue
        offset = event.occurred_at.root - week.start.root
        if 0 <= offset < span:
            counts[offset // width.root] += 1

    per_hour = max(1, _MICROS_PER_HOUR // width.root)
    rows: list[WindowFeatures] = []
    for index in range(n_bins):
        start = SimTime(week.start.root + index * width.root)
        how = _hour_of_week(start, week)
        recent = counts[max(0, index - 3 * per_hour) : index]
        rows.append(
            WindowFeatures(
                window=TimeWindow(start=start, end=SimTime(start.root + width.root)),
                hour_of_day=how % _HOURS_PER_DAY,
                dow=how // _HOURS_PER_DAY,
                is_weekend=(how // _HOURS_PER_DAY) >= 5,
                lag_1h=_lag(counts, index, per_hour),
                lag_2h=_lag(counts, index, 2 * per_hour),
                lag_24h=_lag(counts, index, 24 * per_hour),
                lag_1w=_lag(counts, index, 168 * per_hour),
                roll_mean_3h=math.fsum(recent) / len(recent) if recent else 0.0,
                roll_std_3h=_std(recent),
                count=counts[index],
            )
        )
    return tuple(rows)


def _lag(counts: Sequence[int], index: int, back: int) -> int:
    source = index - back
    return counts[source] if source >= 0 else 0


def _std(values: Sequence[int]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / n)


# -------------------------------------------------------- vitals extraction
def _vitals_row(
    patient: PatientId,
    esi: EsiAcuity,
    window: Sequence[VitalsSample],
    news2: Callable[[VitalsSample], tuple[int, tuple[int, ...]]],
) -> VitalsWindowFeatures:
    """Reduce one window of samples to a row. NEWS2 is scored on the LATEST sample."""
    latest = window[-1]
    elapsed = [s.elapsed.root for s in window]
    total, sub = news2(latest)
    return VitalsWindowFeatures(
        patient=patient,
        window_end=latest.elapsed,
        hr_mean=math.fsum(s.hr for s in window) / len(window),
        hr_min=float(min(s.hr for s in window)),
        hr_max=float(max(s.hr for s in window)),
        hr_slope=_slope_per_hour([float(s.hr) for s in window], elapsed),
        spo2_min=float(min(s.spo2 for s in window)),
        spo2_mean=math.fsum(s.spo2 for s in window) / len(window),
        sbp_min=float(min(s.sbp for s in window)),
        temp_mean=math.fsum(s.temp_c_x10 for s in window) / len(window) / 10.0,
        resp_max=float(max(s.rr for s in window)),
        resp_slope=_slope_per_hour([float(s.rr) for s in window], elapsed),
        news2_total=total,
        news2_sub=sub,
        time_since_arrival=latest.elapsed,
        esi=esi,
    )


def vitals_window_features(
    stream: VitalsStream,
    esi: EsiAcuity,
    news2: Callable[[VitalsSample], tuple[int, tuple[int, ...]]],
    *,
    window: Duration,
    stride: Duration,
) -> tuple[VitalsWindowFeatures, ...]:
    """Rolling windows over a full stream — the offline (training) feed.

    A window is emitted only once ``window`` worth of history exists, so an
    early-stay row is never padded with imaginary readings. ``news2`` is injected
    rather than imported to keep this module free of the scorer that depends on
    it, and to guarantee the offline and online feeds score identically.
    """
    if window.root <= 0 or stride.root <= 0:
        raise ValueError("window and stride must be positive")
    samples = stream.samples
    rows: list[VitalsWindowFeatures] = []
    if not samples:
        return ()
    end = samples[0].elapsed.root + window.root
    last = samples[-1].elapsed.root
    while end <= last:
        low = end - window.root
        current = [s for s in samples if low <= s.elapsed.root <= end]
        if current:
            rows.append(_vitals_row(stream.patient, esi, current, news2))
        end += stride.root
    return tuple(rows)


def online_vitals_features(
    patient: PatientId,
    esi: EsiAcuity,
    recent: Sequence[VitalsSample],
    news2: Callable[[VitalsSample], tuple[int, tuple[int, ...]]],
    *,
    window: Duration,
) -> VitalsWindowFeatures | None:
    """The live feed: the same row from the last ``window`` of readings, or ``None``.

    Returns ``None`` until a full window exists — the monitor must not be asked
    to judge a patient on a single reading, which measurement noise alone can
    push into an alarming range.
    """
    if not recent:
        return None
    latest = recent[-1].elapsed.root
    low = latest - window.root
    if recent[0].elapsed.root > low:
        return None
    current = [s for s in recent if s.elapsed.root >= low]
    if not current:
        return None
    return _vitals_row(patient, esi, current, news2)


def vitals_from_events(
    log: EventLog, patient: PatientId, *, as_of: SimTime | None = None
) -> tuple[int, ...]:
    """The NEWS2 totals this patient's ``VitalsSampled`` events carry, up to ``as_of``.

    The live path reads the score the engine stamped on each event rather than
    re-deriving it, so the console, the log, and the monitor cannot disagree
    about what NEWS2 was at a given instant.
    """
    return tuple(
        env.event.news2
        for env in prefix(log, as_of)
        if isinstance(env.event, VitalsSampled) and env.event.patient == patient
    )


# ------------------------------------------------------------- the bridge
def to_matrix(
    rows: Sequence[PatientFeatures | WindowFeatures | VitalsWindowFeatures],
    *,
    feature_names: tuple[str, ...],
    row_ids: Sequence[str],
    labels: Sequence[float] | None = None,
    complaints: ComplaintEncoder | None = None,
) -> FeatureFrame:
    """Pack typed rows into a dense numeric frame in a fixed column order.

    ``feature_names`` is the contract: the same tuple is stored on the artifact
    and replayed at inference, so a column can never silently change position
    between fit and predict. A requested name the rows cannot supply is an error
    rather than a zero — a quietly-zeroed feature trains a model that is wrong in
    a way no metric reveals.
    """
    matrix: list[tuple[float, ...]] = []
    for row in rows:
        values = dict(row.numeric_features())
        if complaints is not None and isinstance(row, PatientFeatures):
            values["complaint"] = complaints.encode(row.complaint)
        missing = [name for name in feature_names if name not in values]
        if missing:
            raise KeyError(f"rows cannot supply feature(s): {sorted(missing)}")
        matrix.append(tuple(values[name] for name in feature_names))
    return FeatureFrame(
        feature_names=feature_names,
        matrix=tuple(matrix),
        row_ids=tuple(row_ids),
        labels=tuple(labels) if labels is not None else None,
    )


PATIENT_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "esi",
    "is_ambulance",
    "complaint",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "provider_visits",
    "nurse_visits",
    "imaging_count",
    "labs",
    "procedures",
    "isolation_required",
    "wip",
    "bays_occupied",
    "queue_len_esi_1",
    "queue_len_esi_2",
    "queue_len_esi_3",
    "queue_len_esi_4",
    "queue_len_esi_5",
    "staff_on_shift",
    "arrivals_last_hour",
)

WINDOW_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "lag_1h",
    "lag_2h",
    "lag_24h",
    "lag_1w",
    "roll_mean_3h",
    "roll_std_3h",
)

VITALS_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_slope",
    "spo2_min",
    "spo2_mean",
    "sbp_min",
    "temp_mean",
    "resp_max",
    "resp_slope",
    "news2_total",
    "time_since_arrival_h",
    "esi",
)


__all__ = [
    "PATIENT_FEATURE_NAMES",
    "UNSEEN_COMPLAINT",
    "VITALS_FEATURE_NAMES",
    "WINDOW_FEATURE_NAMES",
    "ComplaintEncoder",
    "FeatureFrame",
    "PatientFeatures",
    "VitalsWindowFeatures",
    "WindowFeatures",
    "online_vitals_features",
    "patient_features",
    "prefix",
    "to_matrix",
    "vitals_from_events",
    "vitals_window_features",
    "window_features",
]
