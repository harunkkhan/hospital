"""``run_replication`` — determinism, WIP conservation, horizon discipline."""

from __future__ import annotations

import json

from _sim_fixtures import tiny_scenario
from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import EventLog, TimeWindow, hours
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


def test_default_rules_cover_every_acuity() -> None:
    from hospital.core import EsiAcuity, compile_rules

    kernel = compile_rules(default_rules())
    for esi in EsiAcuity:
        assert kernel.zone_types_for(esi)  # nobody is placeable-nowhere by default
