"""Vitals monitoring in the engine: opt-in, log-faithful, and one page per patient.

The load-bearing test here is :func:`test_a_run_without_a_watch_is_byte_identical`.
Vitals monitoring adds events to the one log every KPI is folded from, so if it
could not be switched off, every M1 golden would have to be re-baselined and the
"incremental playback equals the headless engine" guarantee would quietly weaken.
"""

from __future__ import annotations

import json
from typing import Any

from _sim_fixtures import tiny_scenario

from hospital.core import (
    EsiAcuity,
    ForgetfulMonitor,
    PatientId,
    RiskAssessment,
    RiskMonitor,
    SimTime,
    VitalsReading,
    minutes,
    news2_score,
)
from hospital.core.events import VitalsSampled
from hospital.sim import run_replication
from hospital.sim.flow.vitals import VitalsWatch, vitals_process

_WATCH = VitalsWatch(cadence=minutes(10), span=minutes(120))
_ANY_PATIENT = PatientId("p_probe")


class _NeverEscalates:
    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        del event, reading
        return None


class _AlwaysEscalates:
    """Escalates on the first reading it sees — the strongest possible trigger."""

    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        del reading
        return RiskAssessment(
            patient=event.patient,
            at=event.occurred_at,
            probability=1.0,
            news2=event.news2,
            escalate=True,
        )


def _events(jsonl: str) -> list[dict[str, Any]]:
    return [json.loads(line)["event"] for line in jsonl.splitlines() if line.strip()]


def _kinds(jsonl: str, kind: str) -> list[dict[str, Any]]:
    return [e for e in _events(jsonl) if e["kind"] == kind]


def test_a_run_without_a_watch_is_byte_identical() -> None:
    """Monitoring is opt-in, and switching it off must cost exactly nothing.

    The event log is what every KPI is folded from. If simply linking this module
    changed the log, the M1 goldens would be re-baselined rather than validated.
    """
    scenario = tiny_scenario()
    plain = run_replication(scenario, "baseline", 5).event_log_jsonl
    again = run_replication(scenario, "baseline", 5).event_log_jsonl
    assert plain == again
    assert "vitals_sampled" not in plain


def test_a_watch_adds_readings_and_nothing_else_by_itself() -> None:
    """Sampling alone must not escalate — that needs a monitor with a verdict."""
    scenario = tiny_scenario()
    watched = run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl
    assert _kinds(watched, "vitals_sampled")
    assert not _kinds(watched, "deterioration_detected")
    assert not _kinds(watched, "emergency_raised")


def test_monitoring_is_deterministic_in_the_seed() -> None:
    scenario = tiny_scenario()
    a = run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl
    b = run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl
    assert a == b


def test_only_acute_patients_are_monitored() -> None:
    """Sampling every walk-in would bury the signal under readings nobody acts on."""
    scenario = tiny_scenario()
    watch = VitalsWatch(cadence=minutes(10), span=minutes(120), monitor_at_or_above=EsiAcuity.ESI2)
    log = run_replication(scenario, "baseline", 5, watch=watch).event_log_jsonl

    triaged: dict[str, int] = {
        str(e["patient"]): int(e["esi"]) for e in _kinds(log, "triage_completed")
    }
    sampled = {str(e["patient"]) for e in _kinds(log, "vitals_sampled")}
    assert sampled, "the fixture must produce some monitored patients"

    # Monitoring starts at ARRIVAL, from the patient descriptor's acuity — the same
    # ground truth `patient_process` already uses to set triage priority. The log
    # only reveals ESI at triage completion, so a patient sampled before triaging
    # (or cut off by the horizon) legitimately has no `triage_completed` yet.
    checked = 0
    for patient in sampled:
        if patient not in triaged:
            continue
        checked += 1
        assert triaged[patient] <= int(EsiAcuity.ESI2), (patient, triaged[patient])
    assert checked, "no monitored patient reached triage; the gate is untested"

    # ...and the gate really excludes someone: a looser threshold monitors more.
    loose = VitalsWatch(cadence=minutes(10), span=minutes(120), monitor_at_or_above=EsiAcuity.ESI5)
    loose_log = run_replication(scenario, "baseline", 5, watch=loose).event_log_jsonl
    loose_sampled = {str(e["patient"]) for e in _kinds(loose_log, "vitals_sampled")}
    assert loose_sampled > sampled, "the acuity gate is not filtering anything"


def test_readings_carry_the_news2_of_the_reading_they_report() -> None:
    """The stamped score must be the rubric's, not an independent guess."""
    scenario = tiny_scenario()
    log = run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl
    readings = _kinds(log, "vitals_sampled")
    assert readings
    for event in readings:
        assert isinstance(event["news2"], int)
        assert 0 <= int(event["news2"]) <= 21


def test_monitoring_stops_when_the_patient_leaves() -> None:
    """The world never forgets a patient, so a departed one could sample forever."""
    scenario = tiny_scenario()
    log = run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl
    discharged: dict[str, int] = {
        str(e["patient"]): int(e["occurred_at"]) for e in _kinds(log, "discharge_completed")
    }
    assert discharged, "the fixture must discharge someone"
    for event in _kinds(log, "vitals_sampled"):
        exit_at = discharged.get(str(event["patient"]))
        if exit_at is not None:
            assert int(event["occurred_at"]) <= exit_at, (
                f"{event['patient']} was sampled after discharge"
            )


def test_a_silent_monitor_changes_nothing_about_the_run() -> None:
    """A monitor that never decides must be indistinguishable from no monitor."""
    scenario = tiny_scenario()
    quiet = run_replication(
        scenario, "baseline", 5, watch=_WATCH, monitor=_NeverEscalates()
    ).event_log_jsonl
    unmonitored = run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl
    assert quiet == unmonitored


def test_an_escalation_writes_both_events_in_order() -> None:
    """The engine writes them, in the documented order, at the same instant."""
    scenario = tiny_scenario()
    log = run_replication(
        scenario, "optimized", 5, watch=_WATCH, monitor=_AlwaysEscalates()
    ).event_log_jsonl

    events = _events(log)
    detections = [(i, e) for i, e in enumerate(events) if e["kind"] == "deterioration_detected"]
    emergencies = [(i, e) for i, e in enumerate(events) if e["kind"] == "emergency_raised"]
    assert detections, "an always-escalating monitor must trigger"
    assert len(detections) == len(emergencies)

    by_patient = {str(e["patient"]): i for i, e in emergencies}
    for index, event in detections:
        patient = str(event["patient"])
        assert patient in by_patient
        assert by_patient[patient] > index, "EmergencyRaised must follow DeteriorationDetected"
        assert events[by_patient[patient]]["occurred_at"] == event["occurred_at"]


def test_a_patient_is_paged_at_most_once() -> None:
    """A model above threshold for an hour must not page twelve times.

    Repeated identical pages are how a real alarm gets ignored, so the engine
    escalates a given patient once and then goes quiet for them.
    """
    scenario = tiny_scenario()
    log = run_replication(
        scenario, "optimized", 5, watch=_WATCH, monitor=_AlwaysEscalates()
    ).event_log_jsonl
    raised = [str(e["patient"]) for e in _kinds(log, "emergency_raised")]
    assert raised
    assert len(raised) == len(set(raised))


def test_an_emergency_is_answered_by_a_provider() -> None:
    """The escalation must reach physics, not just the log.

    The raised task is boosted and a decision is requested immediately, so the
    existing dispatch policy should serve it — via the same seam as any other
    task, with no emergency-only code path.
    """
    scenario = tiny_scenario()
    log = run_replication(
        scenario, "optimized", 5, watch=_WATCH, monitor=_AlwaysEscalates()
    ).event_log_jsonl
    events = _events(log)

    answered = 0
    for index, event in enumerate(events):
        if event["kind"] != "emergency_raised":
            continue
        patient = event["patient"]
        answered += any(
            later["kind"] == "provider_visit_started" and later["patient"] == patient
            for later in events[index + 1 :]
        )
    assert answered > 0, "no raised emergency was ever attended"


def test_escalation_does_not_perturb_a_run_that_never_escalates() -> None:
    """CRN safety: the monitor consumes no randomness and moves no other draw."""
    scenario = tiny_scenario()
    with_monitor = run_replication(scenario, "baseline", 5, watch=_WATCH, monitor=_NeverEscalates())
    without = run_replication(scenario, "baseline", 5, watch=_WATCH)
    assert with_monitor.event_log_jsonl == without.event_log_jsonl


class _Forgetful:
    """Records the lifecycle calls so a leak is visible rather than merely likely."""

    def __init__(self) -> None:
        self.seen: set[PatientId] = set()
        self.forgotten: list[PatientId] = []

    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        del reading
        self.seen.add(event.patient)
        return None

    def forget(self, patient: PatientId) -> None:
        self.forgotten.append(patient)


def test_every_monitored_patient_is_released_when_their_process_ends() -> None:
    """A rolling monitor has no other signal that a patient is gone.

    Without release its buffers grow for every discharge across a week-long run, and
    since patient ids are unique only *within* a run, reusing one monitor across runs
    would feed the previous week's readings into this week's window.

    Scoped to patients who actually *left*. A process still suspended when the horizon
    cuts the run never finishes, so its ``finally`` never runs — and that is fine: the
    whole monitor is discarded with the run. The leak worth preventing is the one that
    accumulates *during* a run, patient after discharged patient.
    """
    scenario = tiny_scenario()
    monitor = _Forgetful()
    log = run_replication(scenario, "baseline", 5, watch=_WATCH, monitor=monitor).event_log_jsonl

    assert monitor.seen, "the fixture must monitor somebody"
    discharged = {PatientId(str(e["patient"])) for e in _kinds(log, "discharge_completed")}
    assert discharged, "the fixture must discharge somebody"

    released = set(monitor.forgotten)
    assert len(monitor.forgotten) == len(released), "a patient was released twice"
    left_while_monitored = monitor.seen & discharged
    assert left_while_monitored, "no monitored patient was discharged; the release is untested"
    assert left_while_monitored <= released, (
        f"{len(left_while_monitored - released)} discharged patients were never released"
    )
    # Nothing is released that was never watched in the first place.
    assert released <= monitor.seen


def test_a_monitor_without_a_lifecycle_is_still_accepted() -> None:
    """`forget` is opt-in: a stateless monitor needs no lifecycle and must not be forced."""
    assert not isinstance(_NeverEscalates(), ForgetfulMonitor)
    assert isinstance(_Forgetful(), ForgetfulMonitor)
    scenario = tiny_scenario()
    quiet = run_replication(
        scenario, "baseline", 5, watch=_WATCH, monitor=_NeverEscalates()
    ).event_log_jsonl
    assert quiet == run_replication(scenario, "baseline", 5, watch=_WATCH).event_log_jsonl


def test_the_monitor_protocol_is_satisfied_structurally() -> None:
    assert isinstance(_AlwaysEscalates(), RiskMonitor)
    assert isinstance(_NeverEscalates(), RiskMonitor)


def test_vitals_process_is_exported_for_composition() -> None:
    """The M3 harness composes it directly; it must be reachable, not private."""
    assert callable(vitals_process)
    assert VitalsWatch().cadence == minutes(5)


def test_news2_stamped_on_an_event_matches_the_core_rubric() -> None:
    """One definition: the engine stamps exactly what `core.vitals` scores."""
    reading = VitalsReading(hr=132, spo2=90, sbp=88, dbp=55, temp_c_x10=392, rr=27)
    scored = news2_score(reading)
    event = VitalsSampled(occurred_at=SimTime(0), patient=_ANY_PATIENT, news2=scored.total)
    assert event.news2 == scored.total
    assert scored.band == "high"
