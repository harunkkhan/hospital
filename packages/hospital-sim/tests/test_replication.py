"""``run_replication`` — determinism, WIP conservation, horizon discipline."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from _sim_fixtures import tiny_scenario
from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import Duration, EventLog, PatientId, TimeWindow, hours
from hospital.data.layout import generate_floor
from hospital.data.scenario import realize_staff
from hospital.sim.experiment.replication import Replication, default_rules, run_replication


def _counts(rep: Replication) -> tuple[int, int]:
    arrivals = 0
    completions = 0
    for line in rep.event_log_jsonl.splitlines():
        kind = json.loads(line)["event"]["kind"]
        if kind == "patient_arrived":
            arrivals += 1
        elif kind == "discharge_completed":
            completions += 1
    return arrivals, completions


class TestDeterminism:
    def test_same_seed_twice_is_byte_identical(self) -> None:
        scenario = tiny_scenario()
        a = run_replication(scenario, "baseline", 11)
        b = run_replication(scenario, "baseline", 11)
        assert a.event_log_jsonl == b.event_log_jsonl
        assert a.run_id == b.run_id
        assert a.objective_hash == b.objective_hash

    def test_optimized_same_seed_twice_is_byte_identical(self) -> None:
        # the CP-SAT arm is deterministic given its input (fixed random_seed,
        # one worker, deterministic-time budget): same seed -> same bytes
        scenario = tiny_scenario(horizon_hours=6, rate_per_hour=3.0)
        a = run_replication(scenario, "optimized", 11)
        b = run_replication(scenario, "optimized", 11)
        assert a.event_log_jsonl == b.event_log_jsonl

    def test_different_seeds_differ(self) -> None:
        scenario = tiny_scenario()
        a = run_replication(scenario, "baseline", 1)
        b = run_replication(scenario, "baseline", 2)
        assert a.event_log_jsonl != b.event_log_jsonl

    def test_append_order_is_already_canonical_order(self) -> None:
        # (occurred_at, sequence) is monotone: the sim never appends into the past
        rep = run_replication(tiny_scenario(), "baseline", 7)
        log = EventLog.from_jsonl(rep.event_log_jsonl)
        assert [env.sequence for env in log.ordered()] == list(range(len(log)))


class TestHorizon:
    def test_no_event_at_or_after_the_half_open_end(self) -> None:
        rep = run_replication(tiny_scenario(), "baseline", 7)
        log = EventLog.from_jsonl(rep.event_log_jsonl)
        assert all(env.event.occurred_at < rep.horizon.end for env in log)

    def test_run_record_carries_the_scenario_horizon(self) -> None:
        scenario = tiny_scenario(horizon_hours=6)
        rep = run_replication(scenario, "baseline", 3)
        assert rep.horizon == scenario.workload.horizon
        assert rep.arm == "baseline"
        assert rep.seed == 3


class TestConservation:
    def test_arrivals_equal_completions_plus_wip(self) -> None:
        from hospital.analysis import compute_kpis

        scenario = tiny_scenario()
        rep = run_replication(scenario, "baseline", 7)
        arrivals, completions = _counts(rep)
        assert arrivals > 0

        log = EventLog.from_jsonl(rep.event_log_jsonl)
        layout = generate_floor(scenario.facility)
        window = TimeWindow(start=rep.horizon.start, end=rep.horizon.end)
        roster = realize_staff(scenario.staffing, layout, window)
        kpis = compute_kpis(log, layout, roster, window=rep.horizon, warmup=hours(1))
        assert kpis.values["completions_per_week"] == float(completions)
        assert kpis.values["wip_end_of_week"] == float(arrivals - completions)

    def test_no_patient_is_created_or_lost(self) -> None:
        rep = run_replication(tiny_scenario(), "baseline", 7)
        arrived: list[str] = []
        discharged: list[str] = []
        for line in rep.event_log_jsonl.splitlines():
            event = json.loads(line)["event"]
            if event["kind"] == "patient_arrived":
                arrived.append(event["patient"])
            elif event["kind"] == "discharge_completed":
                discharged.append(event["patient"])
        assert len(arrived) == len(set(arrived))  # each patient arrives once
        assert len(discharged) == len(set(discharged))  # completes at most once
        assert set(discharged) <= set(arrived)  # nobody completes out of thin air

    @settings(max_examples=5, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=1_000))
    def test_conservation_holds_over_random_seeds(self, seed: int) -> None:
        scenario = tiny_scenario(horizon_hours=4, rate_per_hour=3.0)
        rep = run_replication(scenario, "baseline", seed)
        arrivals, completions = _counts(rep)
        assert 0 <= completions <= arrivals


def test_rejected_plan_triggers_a_resolve_and_mutates_nothing() -> None:
    """The tick's reject-then-re-solve path: InfeasiblePlan is caught, state is
    untouched, and a fresh decision runs against the now-current world."""
    from dataclasses import dataclass, field

    from _sim_fixtures import build_physics, make_patient, tiny_rules

    from hospital.core import DecisionInput, EsiAcuity, PlanItem
    from hospital.sim.experiment.replication import (
        _make_tick,  # pyright: ignore[reportPrivateUsage]
    )
    from hospital.sim.policies.factory import make_policies
    from hospital.sim.policies.protocols import PolicySet
    from hospital.solver import GraphRoutingOracle, RoutingOracle

    h = build_physics()
    rules = tiny_rules()
    oracle = GraphRoutingOracle(h.layout.graph)
    p = make_patient("p1", esi=EsiAcuity.ESI5)  # ESI5 may not enter resus
    h.world.register_patient(p)
    h.world.request_bay(p, stage="waiting_for_bay")
    resus_bay = next(b.id for b in h.layout.bays if b.zone_type.value == "resus_trauma")

    @dataclass
    class InfeasibleOncePlacement:
        calls: list[int] = field(default_factory=list[int])

        def place(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]:
            self.calls.append(int(di.now.root))
            if len(self.calls) == 1:  # a stale/buggy first answer
                return (
                    PlanItem(stable_id="assign:p1", kind="assign_bay", patient=p.id, bay=resus_bay),
                )
            return ()

    placement = InfeasibleOncePlacement()
    baseline = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)
    policies = PolicySet(
        placement=placement,
        sequencing=baseline.sequencing,
        dispatch=baseline.dispatch,
        turnaround=baseline.turnaround,
        discharge=baseline.discharge,
        staffing=baseline.staffing,
        origin="baseline",
    )
    from hospital.sim.experiment.replication import (
        _TapEventLog,  # pyright: ignore[reportPrivateUsage]
    )

    tap = _TapEventLog()
    h.world.set_decision_hook(_make_tick(h.world, oracle, policies, rules, h.executor, tap))
    before = h.world.snapshot_bays()
    h.world.request_decision()
    h.env.run(until=1)

    assert len(placement.calls) == 2  # rejected once, re-solved same instant
    assert h.world.snapshot_bays() == before  # nothing was applied
    assert h.world.waiting_for_bay()  # the patient still waits (honest backlog)


def test_callers_objective_drives_policies_and_the_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (M1 review finding 1): run_replication hardcoded
    # DEFAULT_OBJECTIVE into make_policies while callers scored the logs with
    # their own objective — the weighted contrast reported an objective that
    # never drove a decision, and objective_hash misdescribed the run. The
    # caller's objective must reach BOTH the policies and the recorded hash.
    import hospital.sim.experiment.replication as replication_mod
    from hospital.core import CompiledRules, StaffMember
    from hospital.sim.experiment.replication import DEFAULT_OBJECTIVE
    from hospital.sim.policies.factory import Arm, make_policies
    from hospital.sim.policies.protocols import PolicySet
    from hospital.solver import ObjectiveConfig, RoutingOracle, config_hash

    captured: list[ObjectiveConfig | None] = []

    def spy(
        kind: Arm,
        *,
        oracle: RoutingOracle,
        rules: CompiledRules,
        roster: tuple[StaffMember, ...],
        objective: ObjectiveConfig | None = None,
        expected_stay: Mapping[PatientId, Duration] | None = None,
    ) -> PolicySet:
        captured.append(objective)
        return make_policies(
            kind,
            oracle=oracle,
            rules=rules,
            roster=roster,
            objective=objective,
            expected_stay=expected_stay,
        )

    monkeypatch.setattr(replication_mod, "make_policies", spy)
    custom = ObjectiveConfig(w_time=7, w_travel=2, unplaced_wait_penalty=99)
    rep = run_replication(
        tiny_scenario(horizon_hours=2, rate_per_hour=2.0), "optimized", 3, objective=custom
    )
    assert captured == [custom]  # the policies were built from the caller's weights
    assert rep.objective_hash == config_hash(custom)  # ... and the hash describes them
    assert rep.objective_hash != config_hash(DEFAULT_OBJECTIVE)


def test_solver_status_is_recorded_and_propagates_to_the_scorecard() -> None:
    # Regression (M1 review finding 4): a non-OPTIMAL solve status vanished at
    # the end of the run — Replication carried nothing and Scorecard.status
    # stayed None, so a fallback run was indistinguishable from a proven one.
    from hospital.sim.experiment.scorecard import fold_scorecard
    from hospital.solver import ObjectiveConfig, SolverStatus

    scenario = tiny_scenario(horizon_hours=2, rate_per_hour=2.0)
    optimized = run_replication(scenario, "optimized", 5)
    assert optimized.solver_status is SolverStatus.OPTIMAL  # tiny instances solve to proof
    card = fold_scorecard(optimized, ObjectiveConfig())
    assert card.status is optimized.solver_status

    baseline = run_replication(scenario, "baseline", 5)
    assert baseline.solver_status is None  # no solver ran; no claim to record
    assert fold_scorecard(baseline, ObjectiveConfig()).status is None


def test_default_rules_cover_every_acuity() -> None:
    from hospital.core import EsiAcuity, compile_rules

    kernel = compile_rules(default_rules())
    for esi in EsiAcuity:
        assert kernel.zone_types_for(esi)  # nobody is placeable-nowhere by default


def test_er_floor_reference_week_end_to_end() -> None:
    """The ONE slower end-to-end: the reference 100k-sqft ER, a full baseline week."""
    from pathlib import Path

    from hospital.data.scenario import load_scenario
    from hospital.sim.experiment.scorecard import fold_scorecard
    from hospital.solver import ObjectiveConfig

    scenario = load_scenario(Path(__file__).resolve().parents[3] / "scenarios" / "er_floor.yaml")
    rep = run_replication(scenario, "baseline", scenario.seed)
    arrivals, completions = _counts(rep)
    assert arrivals > 500  # a real week of load
    assert completions > 0

    # the log folds through the one KPI fold + the one objective without error,
    # and conservation holds at the week boundary
    card = fold_scorecard(rep, ObjectiveConfig())
    assert card.completions + card.wip == arrivals
    assert card.completions > 0.8 * arrivals  # the baseline floor actually flows
    assert card.objective_total > 0
