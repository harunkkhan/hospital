"""NEWS2 scoring, a calibrated risk classifier, and the online monitor (doc 06 §7).

Three layers, deliberately separable:

1. :func:`news2_score` — the published National Early Warning Score 2 rubric, as a
   pure table-driven function. **Not** a model: it is deterministic, auditable
   against the reference chart, and serves both as a feature and as the fallback a
   clinician can reason about when the ML head is unavailable.
2. :class:`DeteriorationModel` — a calibrated gradient-boosted classifier over a
   rolling vitals window, predicting whether the patient will **cross into**
   deterioration within a horizon. It predicts the near future, not the present:
   scoring "is this patient sick now" would just re-derive NEWS2.
3. :class:`RollingDeteriorationMonitor` — the live object injected into ``sim``
   through ``core.seam.RiskMonitor``. It buffers readings per patient and declines
   to judge until a full window exists.

The threshold is **chosen, not assumed**, under two constraints that pull against
each other: hit a target sensitivity, but stay inside a false-alarm ceiling. A
missed deterioration is far costlier than a false page — yet an alarm firing on
most well patients is one staff learn to silence, which costs every future
deterioration too. When both cannot hold, the model reports which bound was
binding instead of quietly missing the recall it was asked for.

v1 uses SpO2 **Scale 1** only (doc 06 §13-4); there is no hypercapnic Scale 2 and
no consciousness/ACVPU sub-score, because ``data.vitals`` emits neither. Both
absences are scored as 0 and stated here rather than silently assumed normal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

from hospital.core import (
    Duration,
    EsiAcuity,
    FrozenModel,
    PatientId,
    RiskAssessment,
    SimTime,
    VitalsReading,
)
from hospital.core.events import VitalsSampled
from hospital.forecast._estimators import GbtClassifier
from hospital.forecast.features import (
    FeatureFrame,
    VitalsWindowFeatures,
    online_vitals_features,
    to_matrix,
)

if TYPE_CHECKING:
    from hospital.core import RandomStreams
    from hospital.data.vitals import VitalsSample

Band = Literal["low", "medium", "high"]

# The NEWS2 chart, as (inclusive-low, inclusive-high, score) bands per parameter.
# Written as data so it can be read against the published table line by line.
_RESP: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 8, 3),
    (9, 11, 1),
    (12, 20, 0),
    (21, 24, 2),
    (25, math.inf, 3),
)
_SPO2_SCALE1: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 91, 3),
    (92, 93, 2),
    (94, 95, 1),
    (96, math.inf, 0),
)
_SBP: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 90, 3),
    (91, 100, 2),
    (101, 110, 1),
    (111, 219, 0),
    (220, math.inf, 3),
)
_PULSE: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 40, 3),
    (41, 50, 1),
    (51, 90, 0),
    (91, 110, 1),
    (111, 130, 2),
    (131, math.inf, 3),
)
_TEMP_C: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 35.0, 3),
    (35.1, 36.0, 1),
    (36.1, 38.0, 0),
    (38.1, 39.0, 1),
    (39.1, math.inf, 2),
)

# A single parameter scoring 3 triggers escalation on its own, even at a low
# total — one catastrophically deranged vital is not offset by five normal ones.
_SINGLE_PARAM_RED: Final[int] = 3
_HIGH_TOTAL: Final[int] = 7
_MEDIUM_TOTAL: Final[int] = 5

# The order `news2_sub` is reported in; also the order the features use.
NEWS2_PARAMETERS: Final[tuple[str, ...]] = (
    "resp",
    "spo2",
    "oxygen",
    "sbp",
    "pulse",
    "temp",
    "consciousness",
)


class News2Result(FrozenModel):
    """A scored NEWS2 observation: per-parameter sub-scores, total, and band."""

    sub: Mapping[str, int]
    total: int
    band: Band

    def ordered_sub(self) -> tuple[int, ...]:
        return tuple(self.sub[name] for name in NEWS2_PARAMETERS)


def _band_score(bands: Sequence[tuple[float, float, int]], value: float) -> int:
    for low, high, score in bands:
        if low <= value <= high:
            return score
    # Unreachable while the tables span the real line; a loud failure beats a
    # silent 0 that would read as "this vital is normal".
    raise ValueError(f"value {value} fell outside every NEWS2 band")


def news2_score(reading: VitalsReading, *, on_oxygen: bool = False) -> News2Result:
    """Score one observation against the published NEWS2 rubric (SpO2 Scale 1).

    Pure and table-driven, so it can be checked against the reference chart rather
    than against itself. ``consciousness`` is always 0: ``data.vitals`` emits no
    ACVPU level, and assuming "alert" is the honest reading of an absent
    observation — but it is a real gap, not a modelling choice.
    """
    sub = {
        "resp": _band_score(_RESP, reading.rr),
        "spo2": _band_score(_SPO2_SCALE1, reading.spo2),
        "oxygen": 2 if on_oxygen else 0,
        "sbp": _band_score(_SBP, reading.sbp),
        "pulse": _band_score(_PULSE, reading.hr),
        "temp": _band_score(_TEMP_C, reading.temp_c),
        "consciousness": 0,
    }
    total = sum(sub.values())
    if total >= _HIGH_TOTAL or max(sub.values()) >= _SINGLE_PARAM_RED:
        band: Band = "high"
    elif total >= _MEDIUM_TOTAL:
        band = "medium"
    else:
        band = "low"
    return News2Result(sub=sub, total=total, band=band)


def news2_for_features(sample: VitalsSample) -> tuple[int, tuple[int, ...]]:
    """The ``(total, sub-scores)`` adapter ``features`` injects — one scorer, two feeds."""
    scored = news2_score(sample)
    return scored.total, scored.ordered_sub()


class ThresholdChoice(FrozenModel):
    """The chosen operating point and what it actually achieves.

    ``meets_target`` and ``binding`` exist so a caller can tell "we hit the recall
    we asked for" apart from "the alarm ceiling stopped us"; a bare threshold
    would look identical in both cases.
    """

    threshold: float
    sensitivity: float
    false_alarm_rate: float
    meets_target: bool
    binding: Literal["sensitivity", "false_alarms"]


class DeteriorationModel:
    """A calibrated classifier plus the threshold chosen for it.

    A plain object, not a ``FrozenModel``: it wraps a fitted estimator, which is
    an opaque payload rather than a validated value (doc 06 §2).

    ``threshold`` belongs to the model, not to its caller. It was selected on
    validation to hit a sensitivity target, so a consumer comparing ``probability``
    against a number of its own would silently discard that choice — which is why
    :class:`~hospital.core.RiskAssessment` carries a decided ``escalate``.
    """

    def __init__(
        self,
        classifier: GbtClassifier,
        *,
        threshold: float,
        horizon: Duration,
        feature_names: tuple[str, ...],
        sensitivity: float,
        false_alarm_rate: float,
        meets_target: bool = True,
    ) -> None:
        self.classifier = classifier
        self.threshold = threshold
        self.horizon = horizon
        self.feature_names = feature_names
        self.sensitivity = sensitivity
        self.false_alarm_rate = false_alarm_rate
        self.meets_target = meets_target

    def score(self, features: VitalsWindowFeatures) -> float:
        frame = to_matrix(
            [features],
            feature_names=self.feature_names,
            row_ids=[features.patient.root],
        )
        (probability,) = self.classifier.predict_proba(frame.matrix)
        return probability

    def assess(self, features: VitalsWindowFeatures, at: SimTime) -> RiskAssessment:
        probability = self.score(features)
        return RiskAssessment(
            patient=features.patient,
            at=at,
            probability=probability,
            news2=features.news2_total,
            escalate=probability >= self.threshold,
        )


def _sensitivity_and_false_alarms(
    probabilities: Sequence[float], labels: Sequence[int], threshold: float
) -> tuple[float, float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    hits = sum(1 for p, y in zip(probabilities, labels, strict=True) if y == 1 and p >= threshold)
    false_alarms = sum(
        1 for p, y in zip(probabilities, labels, strict=True) if y == 0 and p >= threshold
    )
    sensitivity = hits / positives if positives else 0.0
    false_alarm_rate = false_alarms / negatives if negatives else 0.0
    return sensitivity, false_alarm_rate


def choose_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    target_sensitivity: float,
    max_false_alarm_rate: float = 0.25,
) -> ThresholdChoice:
    """The strictest threshold meeting the sensitivity target within the alarm ceiling.

    Two constraints, and they pull against each other (doc 06 §7): a missed
    deterioration is far costlier than a false page, but an alarm that fires on
    most well patients is one staff learn to silence, which costs every future
    deterioration too.

    So the search is ordered: among thresholds whose false-alarm rate is within
    ``max_false_alarm_rate``, take the most sensitive. If that clears
    ``target_sensitivity``, both constraints hold. If it does not, the ceiling is
    the binding constraint and the result says so via ``meets_target`` rather than
    quietly returning a threshold that misses the recall it was asked for.

    Sweeping the observed probabilities rather than a fixed grid makes the choice
    exact for this validation set instead of quantized to arbitrary steps.
    """
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must lie in (0, 1]")
    if not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("max_false_alarm_rate must lie in [0, 1]")

    candidates = sorted({*probabilities, 1.0 + 1e-9}, reverse=True)
    best: ThresholdChoice | None = None
    for threshold in candidates:
        sensitivity, false_alarms = _sensitivity_and_false_alarms(probabilities, labels, threshold)
        if false_alarms > max_false_alarm_rate:
            continue
        if best is None or sensitivity > best.sensitivity:
            best = ThresholdChoice(
                threshold=threshold,
                sensitivity=sensitivity,
                false_alarm_rate=false_alarms,
                meets_target=sensitivity >= target_sensitivity,
                binding="sensitivity" if sensitivity >= target_sensitivity else "false_alarms",
            )
        if sensitivity >= target_sensitivity:
            break
    if best is None:
        # Even the strictest setting breaches the ceiling: alarm on nothing and
        # report a sensitivity of zero rather than pretend either bound was met.
        top = candidates[0]
        sensitivity, false_alarms = _sensitivity_and_false_alarms(probabilities, labels, top)
        best = ThresholdChoice(
            threshold=top,
            sensitivity=sensitivity,
            false_alarm_rate=false_alarms,
            meets_target=False,
            binding="false_alarms",
        )
    return best


def auroc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC. Undefined without both classes, which is reported as NaN."""
    positives = [p for p, y in zip(probabilities, labels, strict=True) if y == 1]
    negatives = [p for p, y in zip(probabilities, labels, strict=True) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of the probabilities — the calibration check."""
    if not labels:
        return float("nan")
    return math.fsum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(labels)


def fit_deterioration_model(
    train: FeatureFrame,
    validation: FeatureFrame,
    *,
    streams: RandomStreams,
    horizon: Duration,
    target_sensitivity: float = 0.90,
    max_false_alarm_rate: float = 0.25,
) -> DeteriorationModel:
    """Fit, calibrate, and threshold — each on a fold that has not seen the others.

    The classifier trains on ``train``; the isotonic calibrator and the threshold
    are both fit on ``validation``. Sharing a fold between the classifier and its
    calibrator would map the model's own over-confidence onto itself and report a
    reliability it does not have.
    """
    if train.labels is None or validation.labels is None:
        raise ValueError("both frames must carry labels")
    if not train.matrix or not validation.matrix:
        raise ValueError("both frames must have rows")
    if train.feature_names != validation.feature_names:
        raise ValueError("train and validation frames must share a column contract")

    train_labels = [int(v) for v in train.labels]
    validation_labels = [int(v) for v in validation.labels]
    if len(set(train_labels)) < 2:
        raise ValueError("the training frame has only one class; nothing to learn")

    state = int(streams.substream("forecast", "deterioration").integers(0, 2**31 - 1))
    classifier = GbtClassifier(random_state=state).fit(train.matrix, train_labels)
    classifier.calibrate(validation.matrix, validation_labels)

    probabilities = classifier.predict_proba(validation.matrix)
    choice = choose_threshold(
        probabilities,
        validation_labels,
        target_sensitivity=target_sensitivity,
        max_false_alarm_rate=max_false_alarm_rate,
    )
    return DeteriorationModel(
        classifier,
        threshold=choice.threshold,
        horizon=horizon,
        feature_names=train.feature_names,
        sensitivity=choice.sensitivity,
        false_alarm_rate=choice.false_alarm_rate,
        meets_target=choice.meets_target,
    )


class RollingDeteriorationMonitor:
    """The live monitor injected into ``sim`` (implements ``core.seam.RiskMonitor``).

    Buffers each patient's readings and scores the trailing ``window``. Returns
    ``None`` until a full window exists — measurement noise alone can push a single
    reading into an alarming range, and paging on one spike is how an alarm becomes
    something staff learn to ignore.

    Stateful by necessity, and per-patient: one monitor serves a whole floor, so a
    busy neighbour's readings must never enter someone else's window.
    """

    def __init__(
        self,
        model: DeteriorationModel,
        *,
        window: Duration,
        acuity: Mapping[PatientId, EsiAcuity] | None = None,
    ) -> None:
        self.model = model
        self.window = window
        self._acuity: dict[PatientId, EsiAcuity] = dict(acuity or {})
        self._buffers: dict[PatientId, list[VitalsSample]] = {}
        self._arrived: dict[PatientId, int] = {}

    def set_acuity(self, patient: PatientId, esi: EsiAcuity) -> None:
        """Record a triaged acuity — a model feature the vitals stream does not carry."""
        self._acuity[patient] = esi

    def set_arrival(self, patient: PatientId, at: SimTime) -> None:
        """Anchor ``time_since_arrival``. Defaults to the first reading seen."""
        self._arrived[patient] = at.root

    def forget(self, patient: PatientId) -> None:
        """Drop a departed patient's buffer so a week-long run does not grow forever."""
        self._buffers.pop(patient, None)
        self._acuity.pop(patient, None)
        self._arrived.pop(patient, None)

    def features_for(self, patient: PatientId) -> VitalsWindowFeatures | None:
        buffered = self._buffers.get(patient)
        if not buffered:
            return None
        return online_vitals_features(
            patient,
            self._acuity.get(patient, EsiAcuity.ESI3),
            buffered,
            news2_for_features,
            window=self.window,
        )

    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        """Take one reading; assess once the patient's window is full.

        Elapsed time is derived from ``event.occurred_at`` against the patient's
        recorded arrival — the monitor never trusts a caller-supplied offset,
        because the same reading replayed at a different instant must land in a
        different window.
        """
        from hospital.data.vitals import VitalsSample

        patient = event.patient
        anchor = self._arrived.setdefault(patient, event.occurred_at.root)
        elapsed = Duration(max(0, event.occurred_at.root - anchor))
        # Take only the reading's own fields: `VitalsSample` extends `VitalsReading`,
        # so a sample passed in as the reading would already carry an `elapsed` --
        # and its offset is the generator's, not this run's clock.
        vitals = reading.model_dump(include=set(VitalsReading.model_fields))
        sample = VitalsSample(elapsed=elapsed, **vitals)

        buffered = [*self._buffers.get(patient, []), sample]
        # Keep only the trailing window; an unbounded buffer would grow with the
        # length of the stay, and a week-long run has thousands of ticks.
        oldest = elapsed.root - self.window.root
        self._buffers[patient] = [s for s in buffered if s.elapsed.root >= oldest]

        features = self.features_for(patient)
        if features is None:
            return None
        return self.model.assess(features, event.occurred_at)


__all__ = [
    "NEWS2_PARAMETERS",
    "Band",
    "DeteriorationModel",
    "News2Result",
    "RollingDeteriorationMonitor",
    "ThresholdChoice",
    "auroc",
    "brier_score",
    "choose_threshold",
    "fit_deterioration_model",
    "news2_for_features",
    "news2_score",
]
