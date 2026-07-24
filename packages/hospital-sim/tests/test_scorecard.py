"""Scorecard — reuses the one fold + the one objective; lexicographic ranking."""

from __future__ import annotations

from _sim_fixtures import tiny_scenario

from hospital.core import KPI_KEYS, EventLog, KpiVector, RunId, hours
from hospital.sim.experiment.replication import run_replication
from hospital.sim.experiment.scorecard import (
    Scorecard,
    fold_scorecard,
    objective_inputs,
    rank_candidates,
)
from hospital.solver import ObjectiveConfig, weighted_total


def _card(run_id: str, *, wip: int, door_s: float, walked: float, total: int) -> Scorecard:
    values = dict.fromkeys(KPI_KEYS, float("nan"))
    values.update(
        {
            "completions_per_week": 10.0,
            "wip_end_of_week": float(wip),
            "door_to_provider_s_mean": door_s,
            "staff_minutes_walked": walked,
            "staff_frac_walk": 0.1,
            "staff_frac_direct_care": 0.2,
            "staff_frac_cleaning": 0.1,
            "staff_frac_documentation": 0.1,
            "staff_frac_idle": 0.5,
        }
    )
    return Scorecard(
        run_id=RunId(run_id),
        arm="baseline",
        seed=1,
        kpis=KpiVector(values=values),
        objective_total=total,
        completions=10,
        wip=wip,
    )


class TestFold:
    def test_fold_reuses_the_one_kpi_fold_and_the_one_objective(self) -> None:
        scenario = tiny_scenario()
        rep = run_replication(scenario, "baseline", 7)
        objective = ObjectiveConfig()
        card = fold_scorecard(rep, objective, warmup=hours(1))

        # the KpiVector's closed contract validated on construction; counts are
        # read from the fold, never recomputed
        assert card.completions == int(card.kpis.values["completions_per_week"])
        assert card.wip == int(card.kpis.values["wip_end_of_week"])
        assert card.status is None  # baseline makes no solver claim

        # invariant 7 (doc 08 §3): the scorecard total equals weighted_total
        # recomputed from its parts — no hidden inline costs
        log = EventLog.from_jsonl(rep.event_log_jsonl)
        patient_time_s, staff_travel_s = objective_inputs(log, rep.horizon)
        assert card.objective_total == weighted_total(
            patient_time_s=patient_time_s, staff_travel_s=staff_travel_s, config=objective
        )
        assert card.objective_total > 0

    def test_objective_scales_with_config(self) -> None:
        scenario = tiny_scenario()
        rep = run_replication(scenario, "baseline", 7)
        base = fold_scorecard(rep, ObjectiveConfig(), warmup=hours(1))
        doubled = fold_scorecard(rep, ObjectiveConfig(scale=2), warmup=hours(1))
        assert doubled.objective_total == 2 * base.objective_total


class TestRanking:
    def test_fewer_wip_beats_a_better_scalar(self) -> None:
        efficient_but_backlogged = _card("run_a", wip=5, door_s=100.0, walked=10.0, total=1)
        cleared_the_floor = _card("run_b", wip=1, door_s=900.0, walked=999.0, total=1_000_000)
        ranked = rank_candidates((efficient_but_backlogged, cleared_the_floor))
        assert [c.run_id.root for c in ranked] == ["run_b", "run_a"]

    def test_ties_break_on_time_then_walking_then_run_id(self) -> None:
        a = _card("run_a", wip=2, door_s=100.0, walked=50.0, total=0)
        b = _card("run_b", wip=2, door_s=100.0, walked=20.0, total=0)
        c = _card("run_c", wip=2, door_s=80.0, walked=999.0, total=0)
        ranked = rank_candidates((a, b, c))
        assert [x.run_id.root for x in ranked] == ["run_c", "run_b", "run_a"]

        # full tie -> run_id keeps the order total and platform-stable
        d1 = _card("run_1", wip=2, door_s=100.0, walked=50.0, total=0)
        d2 = _card("run_0", wip=2, door_s=100.0, walked=50.0, total=0)
        assert [x.run_id.root for x in rank_candidates((d1, d2))] == ["run_0", "run_1"]
