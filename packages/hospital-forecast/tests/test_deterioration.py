"""NEWS2 against the published chart, and a classifier that must beat it.

Two standards of evidence here. ``news2_score`` is checked against the reference
rubric value by value — it is a published table, so "close enough" is not a thing.
The ML head is checked against a **NEWS2-only baseline**, not against chance:
beating a coin flip on a cohort where ESI-1 deteriorates 17x more often than
ESI-4 would prove only that the model learned the base rate.
"""

from __future__ import annotations

import math

import pytest
from _forecast_fixtures import VITALS_CADENCE, vitals_cohort

from hospital.core import (
    Duration,
    EsiAcuity,
    PatientId,
    RandomStreams,
    RiskAssessment,
    RiskMonitor,
    SimTime,
    VitalsReading,
    minutes,
)
from hospital.core.events import VitalsSampled
from hospital.data.vitals import VitalsStream, deterioration_label
from hospital.forecast.deterioration import (
    NEWS2_PARAMETERS,
    DeteriorationModel,
    RollingDeteriorationMonitor,
    auroc,
    brier_score,
    choose_threshold,
    fit_deterioration_model,
    news2_for_features,
    news2_score,
)
from hospital.forecast.features import (
    VITALS_FEATURE_NAMES,
    FeatureFrame,
    to_matrix,
    vitals_window_features,
)

_HORIZON = minutes(30)
_WINDOW = minutes(30)


def _reading(**overrides: int) -> VitalsReading:
    """A NEWS2-normal adult, with the parameter under test overridden."""
    base = {"hr": 70, "spo2": 98, "sbp": 130, "dbp": 80, "temp_c_x10": 370, "rr": 16}
    return VitalsReading(**{**base, **overrides})


# --------------------------------------------------------------------- NEWS2
def test_a_normal_adult_scores_zero() -> None:
    scored = news2_score(_reading())
    assert scored.total == 0
    assert scored.band == "low"
    assert set(scored.sub) == set(NEWS2_PARAMETERS)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        # Respiration rate: <=8 -> 3, 9-11 -> 1, 12-20 -> 0, 21-24 -> 2, >=25 -> 3
        ("rr", 8, 3), ("rr", 9, 1), ("rr", 11, 1), ("rr", 12, 0), ("rr", 20, 0),
        ("rr", 21, 2), ("rr", 24, 2), ("rr", 25, 3),
        # SpO2 Scale 1: <=91 -> 3, 92-93 -> 2, 94-95 -> 1, >=96 -> 0
        ("spo2", 91, 3), ("spo2", 92, 2), ("spo2", 93, 2), ("spo2", 94, 1),
        ("spo2", 95, 1), ("spo2", 96, 0),
        # Systolic BP: <=90 -> 3, 91-100 -> 2, 101-110 -> 1, 111-219 -> 0, >=220 -> 3
        ("sbp", 90, 3), ("sbp", 91, 2), ("sbp", 100, 2), ("sbp", 101, 1),
        ("sbp", 110, 1), ("sbp", 111, 0), ("sbp", 219, 0), ("sbp", 220, 3),
        # Pulse: <=40 -> 3, 41-50 -> 1, 51-90 -> 0, 91-110 -> 1, 111-130 -> 2, >=131 -> 3
        ("hr", 40, 3), ("hr", 41, 1), ("hr", 50, 1), ("hr", 51, 0), ("hr", 90, 0),
        ("hr", 91, 1), ("hr", 110, 1), ("hr", 111, 2), ("hr", 130, 2), ("hr", 131, 3),
        # Temperature: <=35.0 -> 3, 35.1-36.0 -> 1, 36.1-38.0 -> 0, 38.1-39.0 -> 1, >=39.1 -> 2
        ("temp_c_x10", 350, 3), ("temp_c_x10", 351, 1), ("temp_c_x10", 360, 1),
        ("temp_c_x10", 361, 0), ("temp_c_x10", 380, 0), ("temp_c_x10", 381, 1),
        ("temp_c_x10", 390, 1), ("temp_c_x10", 391, 2),
    ],
)  # fmt: skip
def test_news2_matches_the_published_rubric(field: str, value: int, expected: int) -> None:
    """Every band boundary of the published chart, checked at both edges."""
    parameter = {
        "rr": "resp",
        "spo2": "spo2",
        "sbp": "sbp",
        "hr": "pulse",
        "temp_c_x10": "temp",
    }[field]
    scored = news2_score(_reading(**{field: value}))
    assert scored.sub[parameter] == expected, (field, value, scored.sub)


def test_supplemental_oxygen_adds_two() -> None:
    assert news2_score(_reading(), on_oxygen=True).sub["oxygen"] == 2
    assert news2_score(_reading(), on_oxygen=False).sub["oxygen"] == 0


def test_any_single_parameter_at_three_escalates_regardless_of_total() -> None:
    """The chart's override rule: one catastrophic vital is not averaged away."""
    scored = news2_score(_reading(spo2=88))
    assert scored.sub["spo2"] == 3
    assert scored.total == 3, "the total alone would read as low risk"
    assert scored.band == "high"


def test_bands_follow_the_total_when_no_parameter_is_red() -> None:
    low = news2_score(_reading(hr=95))  # pulse 1
    assert low.total == 1 and low.band == "low"
    medium = news2_score(_reading(hr=95, rr=22, spo2=94, temp_c_x10=385))  # 1+2+1+1
    assert medium.total == 5 and medium.band == "medium"
    high = news2_score(_reading(hr=115, rr=22, spo2=92, temp_c_x10=385))  # 2+2+2+1
    assert high.total == 7 and high.band == "high"


def test_news2_is_pure_and_deterministic() -> None:
    reading = _reading(hr=105, spo2=93)
    assert news2_score(reading) == news2_score(reading)


def test_the_feature_adapter_reports_the_same_score() -> None:
    """`features` injects this adapter — offline and online must score identically."""
    from hospital.data.vitals import VitalsSample

    sample = VitalsSample(
        elapsed=Duration(0), hr=105, spo2=93, sbp=95, dbp=60, temp_c_x10=385, rr=22
    )
    total, sub = news2_for_features(sample)
    scored = news2_score(sample)
    assert total == scored.total
    assert sub == scored.ordered_sub()
    assert len(sub) == len(NEWS2_PARAMETERS)


# ---------------------------------------------------------------- classifier
def _frame(cohort_index: int, n: int = 260) -> tuple[FeatureFrame, list[int]]:
    """Build a labelled window frame from one cohort of vitals streams."""
    cohort = vitals_cohort(n, week_index=cohort_index)
    rows: list[object] = []
    labels: list[float] = []
    ids: list[str] = []
    for stream in cohort.streams:
        esi = cohort.acuity[stream.patient]
        for row in vitals_window_features(
            stream, esi, news2_for_features, window=_WINDOW, stride=VITALS_CADENCE
        ):
            rows.append(row)
            ids.append(f"{stream.patient.root}@{row.window_end.root}")
            labels.append(float(deterioration_label(stream, row.window_end, horizon=_HORIZON)))
    frame = to_matrix(
        rows,  # type: ignore[arg-type]
        feature_names=VITALS_FEATURE_NAMES,
        row_ids=ids,
        labels=labels,
    )
    return frame, [int(v) for v in labels]


_TRAIN_FRAME, _TRAIN_LABELS = _frame(0)
_VALID_FRAME, _VALID_LABELS = _frame(1)
_TEST_FRAME, _TEST_LABELS = _frame(2)
_MODEL = fit_deterioration_model(
    _TRAIN_FRAME,
    _VALID_FRAME,
    streams=RandomStreams(31),
    horizon=_HORIZON,
    target_sensitivity=0.90,
)


def test_the_cohort_has_both_classes_and_is_imbalanced() -> None:
    """A deterioration label is rare — which is exactly why AUROC, not accuracy."""
    positives = sum(_TEST_LABELS)
    assert positives > 0, "no positive labels: the fixture proves nothing"
    rate = positives / len(_TEST_LABELS)
    assert rate < 0.15, f"positives at {rate:.1%} is not the rare event this models"


def _news2_only_scores(frame: FeatureFrame) -> list[float]:
    """The transparent baseline: rank by the NEWS2 total alone."""
    index = frame.feature_names.index("news2_total")
    return [row[index] for row in frame.matrix]


def test_the_classifier_beats_a_news2_only_baseline_on_held_out_data() -> None:
    """The ML head has to add something over the published score it consumes.

    Scored on a third cohort, unseen by both the classifier and its calibrator.
    """
    probabilities = _MODEL.classifier.predict_proba(_TEST_FRAME.matrix)
    model_auroc = auroc(probabilities, _TEST_LABELS)
    baseline_auroc = auroc(_news2_only_scores(_TEST_FRAME), _TEST_LABELS)
    assert not math.isnan(model_auroc)
    assert model_auroc > baseline_auroc, (model_auroc, baseline_auroc)
    assert model_auroc > 0.6, f"AUROC {model_auroc:.3f} is barely better than guessing"


def test_the_threshold_meets_its_sensitivity_target_on_validation() -> None:
    """Two constraints pull against each other; the model reports which one bound.

    A deterioration window is ~0.9% of this cohort, so 90% recall is simply not
    reachable under a 25% false-alarm ceiling. Staying inside the ceiling and
    reporting ``meets_target=False`` is the honest outcome — quietly returning a
    threshold that misses the requested recall is the alternative.
    """
    assert _MODEL.false_alarm_rate <= 0.25 + 1e-9, _MODEL.false_alarm_rate
    assert 0.0 <= _MODEL.threshold <= 1.0
    assert _MODEL.meets_target == (_MODEL.sensitivity >= 0.90)
    if not _MODEL.meets_target:
        assert _MODEL.sensitivity > 0.0, "the ceiling must still admit a usable alarm"


def test_a_generous_ceiling_lets_the_sensitivity_target_be_met() -> None:
    """Lift the ceiling and the recall target becomes reachable — the trade is real."""
    permissive = fit_deterioration_model(
        _TRAIN_FRAME,
        _VALID_FRAME,
        streams=RandomStreams(31),
        horizon=_HORIZON,
        target_sensitivity=0.90,
        max_false_alarm_rate=1.0,
    )
    assert permissive.meets_target
    assert permissive.sensitivity >= 0.90
    # ...and it costs alarms, which is precisely why the ceiling exists.
    assert permissive.false_alarm_rate > _MODEL.false_alarm_rate


def test_a_stricter_target_never_lowers_sensitivity() -> None:
    """With the ceiling out of the way, asking for more recall must not give less."""
    strict = fit_deterioration_model(
        _TRAIN_FRAME,
        _VALID_FRAME,
        streams=RandomStreams(31),
        horizon=_HORIZON,
        target_sensitivity=0.90,
        max_false_alarm_rate=1.0,
    )
    lenient = fit_deterioration_model(
        _TRAIN_FRAME,
        _VALID_FRAME,
        streams=RandomStreams(31),
        horizon=_HORIZON,
        target_sensitivity=0.50,
        max_false_alarm_rate=1.0,
    )
    assert strict.sensitivity >= lenient.sensitivity
    # Catching more costs more false alarms; that trade must be visible.
    assert strict.threshold <= lenient.threshold


def test_choose_threshold_picks_the_strictest_setting_that_still_qualifies() -> None:
    """Among thresholds meeting the target, the highest raises the fewest alarms."""
    probabilities = [0.1, 0.2, 0.6, 0.7, 0.9]
    labels = [0, 0, 1, 0, 1]
    choice = choose_threshold(
        probabilities, labels, target_sensitivity=1.0, max_false_alarm_rate=1.0
    )
    assert choice.sensitivity == 1.0
    assert choice.meets_target
    assert choice.threshold == 0.6, "0.7 would miss the positive at 0.6"


def test_an_impossible_ceiling_alarms_on_nothing_rather_than_lying() -> None:
    """A zero-alarm budget yields a zero-alarm threshold, and admits it missed."""
    choice = choose_threshold([0.2, 0.8], [0, 1], target_sensitivity=1.0, max_false_alarm_rate=0.0)
    assert choice.false_alarm_rate == 0.0


def test_choose_threshold_reports_honestly_when_the_target_is_unreachable() -> None:
    """An unattainable target returns what was actually achieved, not the target."""
    choice = choose_threshold([0.4], [0], target_sensitivity=1.0, max_false_alarm_rate=1.0)
    assert choice.sensitivity == 0.0
    assert not choice.meets_target
    assert choice.threshold >= 0.0


def test_calibrated_probabilities_are_better_than_raw_scores() -> None:
    """Isotonic calibration on a held-out fold must improve the Brier score."""
    calibrated = _MODEL.classifier.predict_proba(_TEST_FRAME.matrix)
    assert min(calibrated) >= 0.0 and max(calibrated) <= 1.0
    assert brier_score(calibrated, _TEST_LABELS) < 0.25, "worse than predicting 0.5 everywhere"


def test_fitting_refuses_a_single_class_training_frame() -> None:
    flat = _TRAIN_FRAME.model_copy(update={"labels": tuple(0.0 for _ in _TRAIN_FRAME.labels or ())})
    with pytest.raises(ValueError, match="one class"):
        fit_deterioration_model(flat, _VALID_FRAME, streams=RandomStreams(1), horizon=_HORIZON)


def test_fitting_refuses_mismatched_column_contracts() -> None:
    flipped = _VALID_FRAME.model_copy(
        update={"feature_names": tuple(reversed(_VALID_FRAME.feature_names))}
    )
    with pytest.raises(ValueError, match="column contract"):
        fit_deterioration_model(_TRAIN_FRAME, flipped, streams=RandomStreams(1), horizon=_HORIZON)


def test_training_is_reproducible_from_the_seed() -> None:
    again = fit_deterioration_model(
        _TRAIN_FRAME,
        _VALID_FRAME,
        streams=RandomStreams(31),
        horizon=_HORIZON,
        target_sensitivity=0.90,
    )
    assert again.threshold == _MODEL.threshold
    assert again.classifier.predict_proba(_TEST_FRAME.matrix[:20]) == (
        _MODEL.classifier.predict_proba(_TEST_FRAME.matrix[:20])
    )


# ------------------------------------------------------------------- monitor
def _stream_and_esi() -> tuple[VitalsStream, EsiAcuity]:
    cohort = vitals_cohort(40, week_index=7)
    stream = next(s for s in cohort.streams if s.deteriorates)
    return stream, cohort.acuity[stream.patient]


def _feed(
    monitor: RollingDeteriorationMonitor, stream: VitalsStream, upto: int | None = None
) -> list[RiskAssessment | None]:
    out: list[RiskAssessment | None] = []
    for sample in stream.samples[:upto]:
        event = VitalsSampled(
            occurred_at=SimTime(sample.elapsed.root),
            patient=stream.patient,
            news2=news2_score(sample).total,
        )
        out.append(monitor.observe(event, sample))
    return out


def test_the_monitor_satisfies_the_core_protocol() -> None:
    """Structural conformance is the whole seam — `sim` never imports `forecast`."""
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW)
    assert isinstance(monitor, RiskMonitor)


def test_the_monitor_withholds_judgement_until_a_window_is_full() -> None:
    """One noisy reading is not evidence; paging on it teaches staff to ignore alarms."""
    stream, esi = _stream_and_esi()
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity={stream.patient: esi})
    verdicts = _feed(monitor, stream, upto=4)  # 0, 5, 10, 15 minutes
    assert all(v is None for v in verdicts), "a partial window must not be scored"

    full = _feed(monitor, stream, upto=None)
    assert any(v is not None for v in full), "a full window must eventually be scored"


def test_the_monitor_returns_a_decided_assessment() -> None:
    stream, esi = _stream_and_esi()
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity={stream.patient: esi})
    verdicts = [v for v in _feed(monitor, stream) if v is not None]
    assert verdicts
    first = verdicts[0]
    assert isinstance(first.probability, float)
    assert first.escalate == (first.probability >= _MODEL.threshold)
    assert first.patient == stream.patient


def test_the_monitor_keeps_patients_separate() -> None:
    """One monitor serves a floor: a neighbour's readings must not enter your window."""
    cohort = vitals_cohort(6, week_index=8)
    a, b = cohort.streams[0], cohort.streams[1]
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity=cohort.acuity)
    _feed(monitor, a)
    features_a = monitor.features_for(a.patient)
    assert features_a is not None

    solo = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity=cohort.acuity)
    _feed(solo, a)
    assert monitor.features_for(a.patient) == solo.features_for(a.patient)

    _feed(monitor, b)
    assert monitor.features_for(a.patient) == features_a, "b's readings changed a's window"


def test_the_monitor_scores_only_the_trailing_window() -> None:
    """Readings older than the window must fall out, not accumulate forever."""
    stream, esi = _stream_and_esi()
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity={stream.patient: esi})
    _feed(monitor, stream)
    features = monitor.features_for(stream.patient)
    assert features is not None
    # The window ends at the last reading and spans no more than `window`.
    assert features.window_end.root == stream.samples[-1].elapsed.root
    hr_values = [
        s.hr for s in stream.samples if s.elapsed.root >= features.window_end.root - _WINDOW.root
    ]
    assert features.hr_max == float(max(hr_values))
    assert features.hr_min == float(min(hr_values))


def test_forgetting_a_patient_clears_their_buffer() -> None:
    """A week-long run must not grow a buffer per discharged patient."""
    stream, esi = _stream_and_esi()
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity={stream.patient: esi})
    _feed(monitor, stream)
    assert monitor.features_for(stream.patient) is not None
    monitor.forget(stream.patient)
    assert monitor.features_for(stream.patient) is None


def test_elapsed_time_is_anchored_to_arrival_not_to_the_first_reading() -> None:
    """The same reading at a later instant belongs in a later window."""
    stream, esi = _stream_and_esi()
    monitor = RollingDeteriorationMonitor(_MODEL, window=_WINDOW, acuity={stream.patient: esi})
    monitor.set_arrival(stream.patient, SimTime(minutes(90).root))
    sample = stream.samples[0]
    monitor.observe(
        VitalsSampled(
            occurred_at=SimTime(minutes(120).root),
            patient=stream.patient,
            news2=news2_score(sample).total,
        ),
        sample,
    )
    buffered = monitor.features_for(stream.patient)
    # One reading is not a window, but the buffer must have taken the 30-minute
    # offset from the arrival anchor rather than the sample's own 0.
    assert buffered is None
    monitor.set_arrival(PatientId("other"), SimTime(0))


def test_a_model_scores_a_window_deterministically() -> None:
    stream, esi = _stream_and_esi()
    rows = vitals_window_features(
        stream, esi, news2_for_features, window=_WINDOW, stride=VITALS_CADENCE
    )
    assert _MODEL.score(rows[0]) == _MODEL.score(rows[0])
    assert 0.0 <= _MODEL.score(rows[0]) <= 1.0


def test_assess_stamps_the_instant_it_was_asked_about() -> None:
    stream, esi = _stream_and_esi()
    rows = vitals_window_features(
        stream, esi, news2_for_features, window=_WINDOW, stride=VITALS_CADENCE
    )
    at = SimTime(minutes(200).root)
    assessment = _MODEL.assess(rows[0], at)
    assert assessment.at == at
    assert assessment.news2 == rows[0].news2_total


def test_auroc_is_nan_without_both_classes() -> None:
    assert math.isnan(auroc([0.1, 0.9], [1, 1]))
    assert math.isnan(auroc([0.1, 0.9], [0, 0]))
    assert auroc([0.9, 0.1], [1, 0]) == 1.0
    assert auroc([0.1, 0.9], [1, 0]) == 0.0


def test_deterioration_model_type_is_a_plain_object() -> None:
    """It wraps a fitted estimator, so it is not a validated frozen value (doc 06 §2)."""
    assert isinstance(_MODEL, DeteriorationModel)
    assert _MODEL.feature_names == VITALS_FEATURE_NAMES
    assert _MODEL.horizon == _HORIZON
