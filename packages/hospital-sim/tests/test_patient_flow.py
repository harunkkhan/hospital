"""The 9-step flow — milestone ordering, paired events, terminal paths, turnaround."""

from __future__ import annotations

import itertools
from collections import defaultdict

from _sim_fixtures import tiny_scenario

from hospital.core import (
    BayAssigned,
    BayCleaningCompleted,
    DischargeCompleted,
    DispositionDecided,
    DispositionKind,
    EventEnvelope,
    EventLog,
    PatientId,
    TimeWindow,
)
from hospital.data.layout import generate_floor
from hospital.data.scenario import realize_staff
from hospital.sim.experiment.replication import run_replication

_MILESTONES = (
    "patient_arrived",
    "triage_started",
    "triage_completed",
    "bay_requested",
    "bay_assigned",
    "provider_visit_started",
    "provider_visit_completed",
    "disposition_decided",
    "discharge_started",
    "discharge_completed",
)

_PAIRS = (
    ("triage_started", "triage_completed"),
    ("provider_visit_started", "provider_visit_completed"),
    ("nurse_visit_started", "nurse_visit_completed"),
    ("documentation_started", "documentation_completed"),
    ("test_ordered", "test_resulted"),
)


def _run() -> EventLog:
    rep = run_replication(tiny_scenario(), "baseline", 7)
    return EventLog.from_jsonl(rep.event_log_jsonl)


def _by_patient(log: EventLog) -> dict[str, list[EventEnvelope]]:
    out: dict[str, list[EventEnvelope]] = defaultdict(list)
    for env in log.ordered():
        patient = getattr(env.event, "patient", None)
        if isinstance(patient, PatientId):
            out[patient.root].append(env)
    return out


def _completed_patients(log: EventLog) -> list[str]:
    return [env.event.patient.root for env in log if isinstance(env.event, DischargeCompleted)]


class TestNineSteps:
    def test_every_completed_patient_walks_the_full_chain_in_order(self) -> None:
        log = _run()
        per_patient = _by_patient(log)
        completed = _completed_patients(log)
        assert completed
        for pid in completed:
            kinds = [env.event.kind for env in per_patient[pid]]
            positions = [kinds.index(m) for m in _MILESTONES if m in kinds]
            required = set(_MILESTONES) - {"discharge_started"}
            assert required <= set(kinds), f"{pid} is missing milestones"
            assert positions == sorted(positions), f"{pid} milestones out of order: {kinds}"

    def test_paired_event_discipline(self) -> None:
        # every *_started has its *_completed, or is the (at most one) open
        # service of a WIP patient at the horizon
        log = _run()
        per_patient = _by_patient(log)
        for pid, envs in per_patient.items():
            kinds = [env.event.kind for env in envs]
            for started, completed in _PAIRS:
                n_started, n_completed = kinds.count(started), kinds.count(completed)
                assert 0 <= n_started - n_completed <= 1, (pid, started)

    def test_analysis_reconstructs_the_log_without_error(self) -> None:
        # build_index raises ZeroTimeCycle on any causally-impossible pairing
        from hospital.analysis._index import build_index

        scenario = tiny_scenario()
        rep = run_replication(scenario, "baseline", 7)
        log = EventLog.from_jsonl(rep.event_log_jsonl)
        layout = generate_floor(scenario.facility)
        window = TimeWindow(start=rep.horizon.start, end=rep.horizon.end)
        roster = realize_staff(scenario.staffing, layout, window)
        index = build_index(log, layout, roster)
        assert index.patients


class TestTerminalPaths:
    def test_admits_board_then_leave_without_paperwork(self) -> None:
        log = _run()
        per_patient = _by_patient(log)
        dispositions = {
            env.event.patient.root: env.event.disposition
            for env in log
            if isinstance(env.event, DispositionDecided)
        }
        admits = [
            pid
            for pid in _completed_patients(log)
            if dispositions.get(pid) is DispositionKind.ADMIT
        ]
        assert admits  # seed 7 realizes several admits on the tiny floor
        for pid in admits:
            kinds = [env.event.kind for env in per_patient[pid]]
            assert "documentation_started" not in kinds
            decided = next(
                env.event.occurred_at
                for env in per_patient[pid]
                if isinstance(env.event, DispositionDecided)
            )
            departed = next(
                env.event.occurred_at
                for env in per_patient[pid]
                if env.event.kind == "discharge_started"
            )
            assert departed > decided  # the boarding hold elapsed in the bay

    def test_discharges_do_paperwork_first(self) -> None:
        log = _run()
        per_patient = _by_patient(log)
        dispositions = {
            env.event.patient.root: env.event.disposition
            for env in log
            if isinstance(env.event, DispositionDecided)
        }
        discharges = [
            pid
            for pid in _completed_patients(log)
            if dispositions.get(pid) in (DispositionKind.DISCHARGE, DispositionKind.TRANSFER)
        ]
        assert discharges
        for pid in discharges:
            kinds = [env.event.kind for env in per_patient[pid]]
            assert "documentation_completed" in kinds
            assert kinds.index("documentation_completed") < kinds.index("discharge_completed")


class TestBayTurnaround:
    def test_a_bay_is_never_reassigned_before_cleaning_completes(self) -> None:
        log = _run()
        timeline: dict[str, list[str]] = defaultdict(list)
        for env in log.ordered():
            if isinstance(env.event, BayAssigned):
                timeline[env.event.bay.root].append("assigned")
            elif isinstance(env.event, BayCleaningCompleted):
                timeline[env.event.bay.root].append("cleaned")
        for bay, seq in timeline.items():
            for first, second in itertools.pairwise(seq):
                if first == "assigned":
                    assert second == "cleaned", f"bay {bay} reassigned before cleaning"

    def test_wip_patients_simply_stop_at_the_horizon(self) -> None:
        log = _run()
        per_patient = _by_patient(log)
        completed = set(_completed_patients(log))
        wip = [pid for pid in per_patient if pid not in completed]
        for pid in wip:
            kinds = {env.event.kind for env in per_patient[pid]}
            assert "discharge_completed" not in kinds
