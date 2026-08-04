"""Feature extraction: the no-leakage cutoff, determinism, and honest encodings.

The headline test is :func:`test_features_cannot_see_past_the_cutoff`. Stated as
"a row equals what it would be if the future had not happened yet", it catches
leakage through *any* field, including ones added later — which a hand-written
per-field assertion never could.
"""

from __future__ import annotations

import math

from _forecast_fixtures import synth_week
from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import (
    ArrivalMode,
    Duration,
    EsiAcuity,
    EventLog,
    OperatingWeek,
    Patient,
    PatientId,
    SimTime,
    StaffId,
    WorkupNeeds,
    hours,
    minutes,
)
from hospital.core.events import (
    BayAssigned,
    BayCleaningCompleted,
    DischargeCompleted,
    PatientArrived,
    StaffIdle,
    TriageCompleted,
)
from hospital.core.ids import BayId, NodeId
from hospital.data.vitals import VitalsSample, VitalsStream
from hospital.forecast.features import (
    PATIENT_FEATURE_NAMES,
    UNSEEN_COMPLAINT,
    ComplaintEncoder,
    online_vitals_features,
    patient_features,
    to_matrix,
    vitals_window_features,
    window_features,
)

_WEEK = synth_week(days=7)


def _news2_stub(sample: VitalsSample) -> tuple[int, tuple[int, ...]]:
    """A stand-in scorer: features must not depend on the real rubric to be correct."""
    total = (sample.hr > 110) + (sample.spo2 < 92) + (sample.rr > 24)
    return int(total), (int(sample.hr > 110), int(sample.spo2 < 92), int(sample.rr > 24))


def _truncate(log: EventLog, as_of: SimTime) -> EventLog:
    """A log in which nothing after ``as_of`` ever happened.

    Filters inline rather than calling ``features.prefix``: a helper built from
    the function under test would be disabled by the very bug it is meant to
    catch, and the equivalence would hold vacuously.
    """
    out = EventLog()
    for env in log.ordered():
        if env.event.occurred_at.root <= as_of.root:
            out.append(env.event, caused_by=env.caused_by)
    return out


@settings(max_examples=15, deadline=None)
@given(cut_hours=st.integers(min_value=1, max_value=167))
def test_features_cannot_see_past_the_cutoff(cut_hours: int) -> None:
    """A row must be identical whether or not the future is present in the log.

    This is the module's whole invariant, and stating it as an equivalence is
    what makes it total: any field that consulted a later event — a future visit,
    the eventual disposition, tomorrow's congestion — would make the two sides
    differ, including fields that do not exist yet.
    """
    as_of = SimTime(hours(cut_hours).root)
    from_full = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week, as_of=as_of)
    from_truncated = patient_features(
        _truncate(_WEEK.log, as_of), _WEEK.roster, _WEEK.week, as_of=as_of
    )
    assert from_full == from_truncated


def test_a_later_cutoff_only_ever_adds_rows() -> None:
    """Rows are append-only in the cutoff: an earlier row never changes shape later.

    Congestion is snapshotted at arrival, so a patient's row is frozen the moment
    they walk in. If a later cutoff rewrote an earlier row, the training set
    would depend on when it happened to be built.
    """
    early = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week, as_of=SimTime(hours(48).root))
    late = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week, as_of=SimTime(hours(96).root))
    assert len(late) > len(early)
    by_id = {row.patient: row for row in late}
    for row in early:
        # `as_of` is the only field that legitimately moves with the cutoff.
        assert row.model_copy(update={"as_of": by_id[row.patient].as_of}) == by_id[row.patient]


def test_extraction_is_deterministic() -> None:
    a = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week)
    b = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week)
    assert a == b


def test_cyclical_encodings_wrap_around_midnight() -> None:
    """23:00 and 00:00 must be neighbours — the reason for sin/cos over a raw hour."""
    rows = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week)
    by_hour: dict[int, tuple[float, float]] = {}
    for row in rows:
        hour = round(math.atan2(row.hour_sin, row.hour_cos) / (2 * math.pi) * 24) % 24
        by_hour[hour] = (row.hour_sin, row.hour_cos)
    assert {0, 23} <= set(by_hour), "fixture must cover both sides of midnight"

    def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)

    midnight, late_evening, midday = by_hour[0], by_hour[23], by_hour[12]
    assert distance(midnight, late_evening) < distance(midnight, midday)


def test_congestion_counters_match_a_hand_built_log() -> None:
    """The counters are reduced from the log alone (nuance 1.4) — checked by hand."""
    log = EventLog()
    week = OperatingWeek(start=SimTime(0), end=SimTime(hours(24).root))
    roster: dict[PatientId, Patient] = {}

    def register(name: str, at_min: int, esi: EsiAcuity) -> PatientId:
        pid = PatientId(name)
        roster[pid] = Patient(
            id=pid,
            arrival_time=SimTime(minutes(at_min).root),
            arrival_mode=ArrivalMode.WALK_IN,
            esi=esi,
            complaint="chest_pain",
            isolation_required=False,
            workup=WorkupNeeds(provider_visits=1, nurse_visits=0, imaging=(), labs=0, procedures=0),
        )
        return pid

    a = register("a", 0, EsiAcuity.ESI2)
    b = register("b", 10, EsiAcuity.ESI3)
    c = register("c", 20, EsiAcuity.ESI4)

    log.append(StaffIdle(occurred_at=SimTime(0), staff=StaffId("s1"), at=NodeId("n")))
    log.append(PatientArrived(occurred_at=SimTime(0), patient=a, mode=ArrivalMode.WALK_IN))
    log.append(TriageCompleted(occurred_at=SimTime(minutes(5).root), patient=a, esi=EsiAcuity.ESI2))
    # b arrives with `a` triaged-and-waiting: wip 1, one ESI-2 in queue, no bays used.
    log.append(
        PatientArrived(occurred_at=SimTime(minutes(10).root), patient=b, mode=ArrivalMode.WALK_IN)
    )
    log.append(
        BayAssigned(
            occurred_at=SimTime(minutes(12).root), patient=a, bay=BayId("bay_0"), by="baseline"
        )
    )
    log.append(
        TriageCompleted(occurred_at=SimTime(minutes(15).root), patient=b, esi=EsiAcuity.ESI3)
    )
    # c arrives: `a` is in a bay, `b` is queued at ESI-3.
    log.append(
        PatientArrived(occurred_at=SimTime(minutes(20).root), patient=c, mode=ArrivalMode.WALK_IN)
    )

    rows = {row.patient: row for row in patient_features(log, roster, week)}
    assert rows[a].wip == 0
    assert rows[a].queue_len_by_esi == (0, 0, 0, 0, 0)
    assert rows[a].staff_on_shift == 1

    assert rows[b].wip == 1
    assert rows[b].queue_len_by_esi == (0, 1, 0, 0, 0), "a is triaged and waiting at ESI-2"
    assert rows[b].bays_occupied == 0

    assert rows[c].wip == 2
    assert rows[c].bays_occupied == 1, "a holds bay_0"
    assert rows[c].queue_len_by_esi == (0, 0, 1, 0, 0), "only b is still waiting"
    assert rows[c].arrivals_last_hour == 2


def test_a_discharged_patient_leaves_wip_and_a_cleaned_bay_frees() -> None:
    log = EventLog()
    week = OperatingWeek(start=SimTime(0), end=SimTime(hours(24).root))
    roster: dict[PatientId, Patient] = {}
    for name, at_min in (("a", 0), ("b", 30)):
        pid = PatientId(name)
        roster[pid] = Patient(
            id=pid,
            arrival_time=SimTime(minutes(at_min).root),
            arrival_mode=ArrivalMode.WALK_IN,
            esi=EsiAcuity.ESI3,
            complaint="chest_pain",
            isolation_required=False,
            workup=WorkupNeeds(provider_visits=1, nurse_visits=0, imaging=(), labs=0, procedures=0),
        )
    a, b = PatientId("a"), PatientId("b")
    log.append(PatientArrived(occurred_at=SimTime(0), patient=a, mode=ArrivalMode.WALK_IN))
    log.append(
        BayAssigned(occurred_at=SimTime(minutes(5).root), patient=a, bay=BayId("x"), by="baseline")
    )
    log.append(DischargeCompleted(occurred_at=SimTime(minutes(20).root), patient=a))
    log.append(
        BayCleaningCompleted(
            occurred_at=SimTime(minutes(25).root), bay=BayId("x"), staff=StaffId("s")
        )
    )
    log.append(
        PatientArrived(occurred_at=SimTime(minutes(30).root), patient=b, mode=ArrivalMode.WALK_IN)
    )

    rows = {row.patient: row for row in patient_features(log, roster, week)}
    assert rows[b].wip == 0, "a was discharged before b arrived"
    assert rows[b].bays_occupied == 0, "the bay was released by cleaning completion"


def test_window_features_lag_backwards_only() -> None:
    """A bin's own count is the label; lags may only reach into the past."""
    rows = window_features(_WEEK.log, _WEEK.week)
    assert len(rows) == 168
    assert rows[0].lag_1h == 0 and rows[0].lag_24h == 0, "no history before the week starts"
    counts = [row.count for row in rows]
    for index, row in enumerate(rows[1:], start=1):
        assert row.lag_1h == counts[index - 1]
        if index >= 24:
            assert row.lag_24h == counts[index - 24]
    # `count` is never offered as an input feature.
    assert "count" not in rows[0].numeric_features()


def test_window_counts_sum_to_the_weeks_arrivals() -> None:
    rows = window_features(_WEEK.log, _WEEK.week)
    arrivals = sum(1 for env in _WEEK.log.ordered() if isinstance(env.event, PatientArrived))
    assert sum(row.count for row in rows) == arrivals


def _stream(n: int, *, cadence_min: int = 5) -> VitalsStream:
    return VitalsStream(
        patient=PatientId("p"),
        samples=tuple(
            VitalsSample(
                elapsed=Duration(minutes(i * cadence_min).root),
                hr=90 + i,
                spo2=97,
                sbp=120,
                dbp=75,
                temp_c_x10=370,
                rr=16,
            )
            for i in range(n)
        ),
        deteriorates=False,
    )


def test_vitals_windows_start_only_once_a_full_window_exists() -> None:
    """No padded early rows: a partial window is not a window."""
    stream = _stream(13)  # 0..60 min at 5-min cadence
    rows = vitals_window_features(
        stream, EsiAcuity.ESI3, _news2_stub, window=minutes(30), stride=minutes(15)
    )
    assert rows, "a 60-minute stream must yield at least one 30-minute window"
    assert min(row.window_end.root for row in rows) >= minutes(30).root


def test_vitals_window_summaries_use_only_the_window() -> None:
    """A window's stats come from its own samples — later readings are invisible."""
    stream = _stream(25)
    rows = vitals_window_features(
        stream, EsiAcuity.ESI3, _news2_stub, window=minutes(30), stride=minutes(30)
    )
    first = rows[0]
    inside = [s for s in stream.samples if s.elapsed.root <= first.window_end.root]
    window_start = first.window_end.root - minutes(30).root
    inside = [s for s in inside if s.elapsed.root >= window_start]
    assert first.hr_max == float(max(s.hr for s in inside))
    # HR rises monotonically in this fixture, so a leak from later samples would
    # push hr_max above the window's own maximum.
    assert first.hr_max < float(max(s.hr for s in stream.samples))
    assert first.hr_slope > 0.0


def test_online_and_offline_vitals_rows_agree() -> None:
    """One feature definition, two feeds (doc 06 §4) — they must not drift apart."""
    stream = _stream(13)
    offline = vitals_window_features(
        stream, EsiAcuity.ESI3, _news2_stub, window=minutes(30), stride=minutes(5)
    )
    last_offline = offline[-1]
    online = online_vitals_features(
        stream.patient, EsiAcuity.ESI3, stream.samples, _news2_stub, window=minutes(30)
    )
    assert online == last_offline


def test_online_features_withhold_judgement_until_the_window_fills() -> None:
    partial = _stream(3).samples  # only 10 minutes of history
    assert (
        online_vitals_features(
            PatientId("p"), EsiAcuity.ESI3, partial, _news2_stub, window=minutes(30)
        )
        is None
    )
    assert (
        online_vitals_features(PatientId("p"), EsiAcuity.ESI3, (), _news2_stub, window=minutes(30))
        is None
    )


def test_complaint_encoder_is_fit_on_training_categories_only() -> None:
    """An unseen complaint gets an explicit unknown bucket, not another's code."""
    encoder = ComplaintEncoder.fit(["chest_pain", "abdominal", "chest_pain"])
    assert encoder.encode("chest_pain") != encoder.encode("abdominal")
    assert encoder.encode("never_seen") == UNSEEN_COMPLAINT
    assert encoder.encode("never_seen") not in {
        encoder.encode("chest_pain"),
        encoder.encode("abdominal"),
    }


def test_to_matrix_pins_the_column_order_and_refuses_unknown_features() -> None:
    """`feature_names` is the fit/predict contract — a missing column is an error."""
    rows = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week)[:20]
    encoder = ComplaintEncoder.fit(p.complaint for p in _WEEK.roster.values())
    ids = [row.patient.root for row in rows]
    frame = to_matrix(rows, feature_names=PATIENT_FEATURE_NAMES, row_ids=ids, complaints=encoder)
    assert frame.feature_names == PATIENT_FEATURE_NAMES
    assert len(frame) == 20
    assert all(len(r) == len(PATIENT_FEATURE_NAMES) for r in frame.matrix)
    assert frame.row_ids == tuple(ids)

    # Reordering the request reorders the columns -- position is by name, always.
    flipped = tuple(reversed(PATIENT_FEATURE_NAMES))
    other = to_matrix(rows, feature_names=flipped, row_ids=ids, complaints=encoder)
    assert other.matrix[0] == tuple(reversed(frame.matrix[0]))

    try:
        to_matrix(rows, feature_names=("no_such_feature",), row_ids=ids, complaints=encoder)
    except KeyError as exc:
        assert "no_such_feature" in str(exc)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("a missing feature must raise, never silently zero-fill")


def test_row_ids_are_identity_and_never_columns() -> None:
    """A model must not be able to learn from a patient id."""
    rows = patient_features(_WEEK.log, _WEEK.roster, _WEEK.week)[:5]
    encoder = ComplaintEncoder.fit(p.complaint for p in _WEEK.roster.values())
    frame = to_matrix(
        rows,
        feature_names=PATIENT_FEATURE_NAMES,
        row_ids=[row.patient.root for row in rows],
        complaints=encoder,
    )
    assert "patient" not in frame.feature_names
    assert not any(name.endswith("_id") for name in frame.feature_names)


def test_the_default_cutoff_is_each_rows_own_arrival() -> None:
    """`as_of=None` means "as of each patient's arrival" — not "the whole log".

    The property test above always passes an explicit `as_of`, under which every row
    shares one cutoff and the prefix filter alone is correct. `as_of=None` — the path
    LOS training uses — has a *per-row* cutoff, and that is where a later triage
    result leaked in: a row stamped `as_of=00:00` reported the ESI revealed at 00:10.
    """
    pid = PatientId("late_triage")
    patient = Patient(
        id=pid,
        arrival_time=SimTime(0),
        arrival_mode=ArrivalMode.WALK_IN,
        esi=EsiAcuity.ESI5,
        complaint="chest_pain",
        isolation_required=False,
        workup=WorkupNeeds(provider_visits=1, nurse_visits=0, imaging=(), labs=0, procedures=0),
    )
    log = EventLog()
    log.append(PatientArrived(occurred_at=SimTime(0), patient=pid, mode=ArrivalMode.WALK_IN))
    log.append(
        TriageCompleted(occurred_at=SimTime(minutes(10).root), patient=pid, esi=EsiAcuity.ESI1)
    )
    week = OperatingWeek(start=SimTime(0), end=SimTime(hours(24).root))
    roster = {pid: patient}

    # At arrival nothing has revealed ESI-1, so the roster estimate stands.
    (default_row,) = patient_features(log, roster, week)
    assert default_row.as_of == SimTime(0)
    assert default_row.esi is EsiAcuity.ESI5, "a future triage must not reach an arrival-time row"

    # Once triage has completed by the cutoff, the triaged value is correct to use.
    (later_row,) = patient_features(log, roster, week, as_of=SimTime(minutes(10).root))
    assert later_row.esi is EsiAcuity.ESI1


@settings(max_examples=12, deadline=None)
@given(cut_hours=st.integers(min_value=1, max_value=167))
def test_the_default_path_agrees_with_a_per_arrival_explicit_cutoff(cut_hours: int) -> None:
    """`as_of=None` must equal asking for each patient's own arrival instant.

    The equivalence the explicit-cutoff property test could not see: it fixes one
    cutoff for all rows, so it cannot detect a row reading past its *own*.
    """
    del cut_hours
    default_rows = {
        row.patient: row for row in patient_features(_WEEK.log, _WEEK.roster, _WEEK.week)
    }
    assert default_rows
    for patient, row in list(default_rows.items())[:25]:
        (explicit,) = [
            r
            for r in patient_features(_WEEK.log, _WEEK.roster, _WEEK.week, as_of=row.as_of)
            if r.patient == patient
        ]
        assert row == explicit, f"{patient} differs between the default and explicit cutoff"
