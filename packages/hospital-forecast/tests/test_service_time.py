"""Service time: recover known lognormal parameters, and beat the mean baseline.

The fixture samples every visit from ``TRUE_SERVICE[(activity, esi, complaint)]``
via the same ``core.rng.sample_lognormal`` the sim uses, so ``fit_service_time_table``
is scored against the parameters that actually produced the durations — not
against its own output.
"""

from __future__ import annotations

import math
import statistics

import pytest
from _forecast_fixtures import TRUE_SERVICE, SynthWeek, synth_week, synth_weeks

from hospital.core import (
    Activity,
    EsiAcuity,
    EventLog,
    Patient,
    PatientId,
    RandomStreams,
    SimTime,
    WorkupNeeds,
    hours,
)
from hospital.core.events import PatientArrived, ProviderVisitCompleted, TriageCompleted
from hospital.forecast.features import (
    PATIENT_FEATURE_NAMES,
    ComplaintEncoder,
    FeatureFrame,
    concat_frames,
    patient_features,
)
from hospital.forecast.service_time import (
    LognormalParams,
    ServiceTimeKey,
    ServiceTimeTable,
    activity_durations,
    fit_service_time_regressor,
    fit_service_time_table,
    los_training_frame,
    patient_los,
    static_service_table,
    table_baseline_log_error,
)

# Several weeks: a per-key lognormal needs samples, and the point of the
# `min_samples` guard is that thin keys should NOT be fit on their own.
#
# The weeks stay SEPARATE logs. Each synthesized week runs on its own 0..168h
# timeline, so splicing them would interleave unrelated instants; the fit pools
# the resulting durations, not the events.
_WEEKS = synth_weeks(4)
_WEEK = _WEEKS[0].week
_LOGS: list[EventLog] = [w.log for w in _WEEKS]
# Each log paired with its OWN roster -- ids are unique only within a run, so pooling
# the rosters would misattribute one week's durations to another week's complaint.
_RUNS = [(w.log, w.roster) for w in _WEEKS]
_ROSTER: dict[PatientId, Patient] = {}
for _w in _WEEKS:
    _ROSTER.update(_w.roster)


def _table() -> ServiceTimeTable:
    return fit_service_time_table(_RUNS, min_samples=30)


def _frame_for(week: SynthWeek, encoder: ComplaintEncoder) -> FeatureFrame:
    """One week's labelled LOS frame, extracted on that week's own timeline."""
    rows = patient_features(week.log, week.roster, week.week)
    return los_training_frame(rows, patient_los(week.log), encoder)


def test_fit_recovers_the_generating_lognormal_parameters() -> None:
    """Both moments, per key — a mean-only check would pass a wrong-shape fit."""
    table = _table()
    checked = 0
    for (activity, esi, complaint), (true_mean, true_cv) in TRUE_SERVICE.items():
        key = ServiceTimeKey(activity=activity, esi=esi, complaint=complaint)
        fitted = table.params.get(key)
        if fitted is None:
            continue  # a thin key legitimately falls back; see the fallback test
        checked += 1
        assert fitted.mean_s == pytest.approx(true_mean, rel=0.12), (key, fitted.mean_s, true_mean)
        assert fitted.cv == pytest.approx(true_cv, rel=0.30), (key, fitted.cv, true_cv)
    assert checked >= 6, f"only {checked} keys were directly fit; the corpus is too thin"


def test_expected_returns_the_fitted_mean_as_a_duration() -> None:
    table = _table()
    key = ServiceTimeKey(
        activity=Activity.PROVIDER_VISIT, esi=EsiAcuity.ESI3, complaint="chest_pain"
    )
    true_mean, _ = TRUE_SERVICE[(Activity.PROVIDER_VISIT, EsiAcuity.ESI3, "chest_pain")]
    assert table.expected(key).root / 1_000_000 == pytest.approx(true_mean, rel=0.12)


def test_keys_are_exactly_activity_by_esi_by_complaint() -> None:
    """The axes must not drift from ``sim.physics.service_times``' sampling key."""
    table = _table()
    assert table.params, "the corpus must fit at least one key directly"
    for key in table.params:
        assert isinstance(key.activity, Activity)
        assert isinstance(key.esi, EsiAcuity)
        assert isinstance(key.complaint, str)
    observed_activities = {key.activity for key in table.params}
    assert Activity.PROVIDER_VISIT in observed_activities
    assert Activity.NURSE_VISIT in observed_activities


def test_a_thin_key_borrows_the_activity_fallback_never_a_zero() -> None:
    """Under `min_samples` the key is not fit — but it must still answer.

    A zero would tell the solver the work is free, which is worse than a coarse
    estimate borrowed from the activity as a whole.
    """
    strict = fit_service_time_table(_RUNS, min_samples=10_000)
    assert strict.params == {}, "nothing should clear an impossible threshold"
    key = ServiceTimeKey(
        activity=Activity.PROVIDER_VISIT, esi=EsiAcuity.ESI2, complaint="chest_pain"
    )
    assert strict.has(key)
    assert strict.expected(key).root > 0


def test_an_unobserved_activity_raises_rather_than_returning_zero() -> None:
    table = _table()
    key = ServiceTimeKey(activity=Activity.IMAGING, esi=EsiAcuity.ESI3, complaint="chest_pain")
    if not table.has(key):
        with pytest.raises(KeyError, match="no fitted service time"):
            table.expected(key)


def test_repeated_visits_are_separate_observations() -> None:
    """Two provider visits are two durations, not one span covering both."""
    durations = activity_durations(_RUNS)
    provider = [
        n
        for key, samples in durations.items()
        if key.activity is Activity.PROVIDER_VISIT
        for n in [len(samples)]
    ]
    total_provider = sum(provider)
    # The fixture gives every patient 1 or 2 provider visits, so the count of
    # observed durations must exceed the patient count.
    assert total_provider > len(_ROSTER), (total_provider, len(_ROSTER))


def test_an_unpaired_start_is_dropped_not_censored() -> None:
    """A visit still running at the horizon must not enter the fit as a short one."""
    from hospital.core import StaffId
    from hospital.core.events import ProviderVisitStarted

    truncated = EventLog()
    for env in _WEEKS[0].log.ordered():
        truncated.append(env.event, caused_by=env.caused_by)
    # An open visit with no completion anywhere.
    victim = next(iter(_WEEKS[0].roster))
    truncated.append(
        ProviderVisitStarted(
            occurred_at=SimTime(hours(200).root), patient=victim, staff=StaffId("staff_x")
        )
    )
    before = activity_durations([(_WEEKS[0].log, _WEEKS[0].roster)])
    after = activity_durations([(truncated, _WEEKS[0].roster)])
    assert before == after, "an unmatched start must contribute nothing"


def test_fit_is_deterministic() -> None:
    assert _table() == _table()


def test_static_service_table_answers_from_scenario_constants() -> None:
    """The A/B baseline arm needs the same type without any fitting."""
    table = static_service_table({Activity.PROVIDER_VISIT: 600.0}, cv=0.35)
    key = ServiceTimeKey(activity=Activity.PROVIDER_VISIT, esi=EsiAcuity.ESI1, complaint="anything")
    assert table.expected(key).root / 1_000_000 == pytest.approx(600.0)
    assert table.cv(key) == pytest.approx(0.35)


# --------------------------------------------------------------- regressor
_TRAIN_WEEKS = _WEEKS[:3]
_TEST_WEEK = _WEEKS[3]
_ENCODER = ComplaintEncoder.fit(p.complaint for w in _TRAIN_WEEKS for p in w.roster.values())
_TRAIN_FRAME = concat_frames([_frame_for(w, _ENCODER) for w in _TRAIN_WEEKS])
_TEST_FRAME = _frame_for(_TEST_WEEK, _ENCODER)
_MODEL = fit_service_time_regressor(
    _TRAIN_FRAME, _ENCODER, streams=RandomStreams(5), quantiles=(0.9,)
)


def test_regressor_beats_the_global_mean_baseline_on_a_held_out_week() -> None:
    """The model must earn its keep against "predict the average stay, always".

    Scored on a week it never saw, on the log scale it was fit on. Beating the
    baseline on the training weeks would prove only that a GBT can memorize.
    """
    assert len(_TRAIN_FRAME) > 200, "the training frame is too small to be meaningful"

    model_error = _MODEL.point_log_error(_TEST_FRAME)
    # The strongest constant predictor under MAE-on-log is the training median,
    # not the arithmetic mean -- beating a weaker baseline would prove nothing.
    train_los = [value for w in _TRAIN_WEEKS for value in patient_los(w.log).values()]
    baseline_error = table_baseline_log_error(
        patient_features(_TEST_WEEK.log, _TEST_WEEK.roster, _TEST_WEEK.week),
        patient_los(_TEST_WEEK.log),
        statistics.median(train_los),
    )
    assert model_error < baseline_error, (model_error, baseline_error)


def test_predicted_los_rises_with_workup_and_acuity() -> None:
    """Monotone sanity on crafted rows — a model that ignores workup is not useful."""
    rows = patient_features(_TRAIN_WEEKS[0].log, _TRAIN_WEEKS[0].roster, _WEEK)
    light = rows[0].model_copy(
        update={"provider_visits": 1, "nurse_visits": 1, "labs": 0, "imaging_count": 0}
    )
    heavy = rows[0].model_copy(
        update={"provider_visits": 2, "nurse_visits": 2, "labs": 2, "imaging_count": 1}
    )
    assert _MODEL.predict_expected_los(heavy).root > _MODEL.predict_expected_los(light).root


def test_quantile_head_sits_above_the_point_estimate() -> None:
    """A p90 stay must be longer than a typical one, or the head is mislabelled."""
    rows = patient_features(_TRAIN_WEEKS[0].log, _TRAIN_WEEKS[0].roster, _WEEK)
    above = sum(
        _MODEL.predict_quantile(row, 0.9).root >= _MODEL.predict_median_los(row).root
        for row in rows[:60]
    )
    assert above >= 57, f"only {above}/60 p90 predictions exceeded the point estimate"


def test_asking_for_an_unfitted_quantile_is_an_error() -> None:
    rows = patient_features(_TRAIN_WEEKS[0].log, _TRAIN_WEEKS[0].roster, _WEEK)
    with pytest.raises(KeyError, match="no fitted head"):
        _MODEL.predict_quantile(rows[0], 0.5)


def test_frames_from_different_column_orders_cannot_be_stacked() -> None:
    """A silent stack of mismatched columns trains on scrambled inputs."""
    other = _frame_for(_TEST_WEEK, _ENCODER)
    flipped = other.model_copy(update={"feature_names": tuple(reversed(other.feature_names))})
    with pytest.raises(ValueError, match="feature_names"):
        concat_frames([_TRAIN_FRAME, flipped])


def test_only_completed_stays_become_labels() -> None:
    """A patient still in the department has no LOS — imputing one trains a lie."""
    week = synth_week(days=2, week_index=42)
    los = patient_los(week.log)
    assert los, "the fixture must complete some stays"
    assert set(los) <= set(week.roster)
    for value in los.values():
        assert value > 0.0


def test_los_frame_labels_are_log_scaled() -> None:
    week = synth_week(days=3, week_index=43)
    encoder = ComplaintEncoder.fit(p.complaint for p in week.roster.values())
    rows = patient_features(week.log, week.roster, week.week)
    los = patient_los(week.log)
    frame = los_training_frame(rows, los, encoder)
    assert frame.labels is not None
    assert frame.feature_names == PATIENT_FEATURE_NAMES
    by_id = {row_id: label for row_id, label in zip(frame.row_ids, frame.labels, strict=True)}
    for patient, seconds_value in los.items():
        if patient.root in by_id:
            assert by_id[patient.root] == pytest.approx(math.log(seconds_value), rel=1e-9)


def test_fitting_without_labels_is_refused() -> None:
    week = synth_week(days=2, week_index=44)
    encoder = ComplaintEncoder.fit(p.complaint for p in week.roster.values())
    rows = patient_features(week.log, week.roster, week.week)
    from hospital.forecast.features import to_matrix

    unlabelled = to_matrix(
        rows,
        feature_names=PATIENT_FEATURE_NAMES,
        row_ids=[r.patient.root for r in rows],
        complaints=encoder,
    )
    with pytest.raises(ValueError, match="labels"):
        fit_service_time_regressor(unlabelled, encoder, streams=RandomStreams(1))


def test_lognormal_params_round_trip_through_the_sampler() -> None:
    """The fitted pair is exactly what `sample_lognormal` takes — check it draws.

    This is the anti-drift guarantee: a table fit here can be handed straight back
    to the sim's sampler with no conversion step to get wrong.
    """
    from hospital.core import sample_lognormal

    params = LognormalParams(mean_s=700.0, cv=0.4, n=100)
    streams = RandomStreams(3)
    draws = [
        sample_lognormal(streams.substream("t", i), params.mean_s, params.cv).root / 1_000_000
        for i in range(4000)
    ]
    assert statistics.mean(draws) == pytest.approx(params.mean_s, rel=0.05)
    assert statistics.pstdev(draws) / statistics.mean(draws) == pytest.approx(params.cv, rel=0.12)


def test_repeated_patient_ids_across_runs_are_not_conflated() -> None:
    """Patient ids are unique only WITHIN a run — `data.workload` reuses them weekly.

    Pooling rosters into one mapping lets a later week's registration overwrite an
    earlier one, so week 1's durations get filed under week 2's complaint and acuity.
    Here the same id is a 600s chest-pain visit in run A and a 1200s abdominal visit
    in run B; each must land under its own key.
    """
    from hospital.core import StaffId
    from hospital.core.events import ProviderVisitStarted

    shared = PatientId("p_000_00")
    staff = StaffId("doc")

    def run(complaint: str, seconds_long: int) -> tuple[EventLog, dict[PatientId, Patient]]:
        patient = Patient(
            id=shared,
            arrival_time=SimTime(0),
            arrival_mode=_WEEKS[0].roster[next(iter(_WEEKS[0].roster))].arrival_mode,
            esi=EsiAcuity.ESI3,
            complaint=complaint,
            isolation_required=False,
            workup=WorkupNeeds(provider_visits=1, nurse_visits=0, imaging=(), labs=0, procedures=0),
        )
        log = EventLog()
        log.append(
            PatientArrived(occurred_at=SimTime(0), patient=shared, mode=patient.arrival_mode)
        )
        log.append(TriageCompleted(occurred_at=SimTime(0), patient=shared, esi=EsiAcuity.ESI3))
        log.append(ProviderVisitStarted(occurred_at=SimTime(0), patient=shared, staff=staff))
        log.append(
            ProviderVisitCompleted(
                occurred_at=SimTime(seconds_long * 1_000_000), patient=shared, staff=staff
            )
        )
        return log, {shared: patient}

    durations = activity_durations([run("chest_pain", 600), run("abdominal", 1200)])
    chest = ServiceTimeKey(
        activity=Activity.PROVIDER_VISIT, esi=EsiAcuity.ESI3, complaint="chest_pain"
    )
    abdo = ServiceTimeKey(
        activity=Activity.PROVIDER_VISIT, esi=EsiAcuity.ESI3, complaint="abdominal"
    )
    assert durations[chest] == [600.0], "run A's duration must keep run A's complaint"
    assert durations[abdo] == [1200.0], "run B's duration must keep run B's complaint"


def test_the_expected_stay_exceeds_the_median_stay() -> None:
    """`exp(E[log LOS])` is a median; an occupancy cost needs a mean.

    The gap is one-directional and large — at a residual log-SD of 1 a one-hour
    median is a 1.65-hour mean — so feeding the median into a bay-occupancy term
    under-books every long stay. Duan smearing corrects it without assuming
    lognormality.
    """
    rows = patient_features(_TRAIN_WEEKS[0].log, _TRAIN_WEEKS[0].roster, _WEEK)[:40]
    assert rows
    for row in rows:
        median = _MODEL.predict_median_los(row).root
        expected = _MODEL.predict_expected_los(row).root
        assert expected >= median, "the smeared mean can never fall below the median"
    # And on a fit with real residual spread it is strictly larger somewhere.
    assert any(
        _MODEL.predict_expected_los(row).root > _MODEL.predict_median_los(row).root for row in rows
    ), "a noiseless fit would make the correction a no-op; this one is not noiseless"


def test_the_smearing_factor_is_at_least_one() -> None:
    """mean(exp(residual)) >= exp(mean(residual)) by Jensen, and the mean residual ~0.

    On this fixture the factor is only ~1.004 — LOS is nearly determined by the
    features, so residual spread is small. The correction is still the right one; its
    magnitude just scales with how noisy the fit is, and on real data it is larger.
    """
    assert _MODEL.smearing >= 1.0 - 1e-9
