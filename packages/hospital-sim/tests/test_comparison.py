"""Paired comparison — CRN pairing loop; stats delegated to analysis.paired_bootstrap."""

from __future__ import annotations

import math

import pytest
from _sim_fixtures import tiny_scenario

from hospital.analysis import WEIGHTED_OBJECTIVE_KEY
from hospital.core import KPI_KEYS, hours
from hospital.sim.experiment.comparison import run_paired_comparison
from hospital.solver import ObjectiveConfig


def test_null_comparison_baseline_vs_itself_shows_zero_signal() -> None:
    # arm-agnostic scaffolding: with the same arm on both sides of every pair
    # (identical CRN => byte-identical runs), every finite diff is exactly 0
    # and nothing is significant — the null experiment reads as null.
    contrasts = run_paired_comparison(
        tiny_scenario(horizon_hours=4, rate_per_hour=3.0),
        (1, 2),
        objective=ObjectiveConfig(),
        arms=("baseline", "baseline"),
        n_boot=200,
        warmup=hours(1),
    )
    # the G1 weighted-objective headline leads, then one contrast per KPI key
    assert [c.key for c in contrasts] == [WEIGHTED_OBJECTIVE_KEY, *KPI_KEYS]
    for c in contrasts:
        assert not c.significant
        if not math.isnan(c.diff):
            assert c.diff == 0.0
        if not math.isnan(c.baseline) and not math.isnan(c.optimized):
            assert c.baseline == c.optimized  # the pairing really used the same seed


def test_paired_comparison_threads_its_objective_into_the_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (M1 review finding 1): the pairing loop ran replications under
    # DEFAULT_OBJECTIVE but scored them with the caller-supplied objective —
    # the weighted contrast reported weights that never drove the optimized
    # arm. Every replication must be run under the comparison's own objective.
    import hospital.sim.experiment.comparison as comparison_mod
    from hospital.data.scenario import Scenario
    from hospital.sim.experiment.replication import Replication, run_replication
    from hospital.sim.policies.factory import Arm

    captured: list[ObjectiveConfig] = []

    def spy(scenario: Scenario, arm: Arm, seed: int, *, objective: ObjectiveConfig) -> Replication:
        captured.append(objective)
        return run_replication(scenario, arm, seed, objective=objective)

    monkeypatch.setattr(comparison_mod, "run_replication", spy)
    custom = ObjectiveConfig(w_time=5, w_travel=3)
    run_paired_comparison(
        tiny_scenario(horizon_hours=2, rate_per_hour=2.0),
        (1,),
        objective=custom,
        arms=("baseline", "baseline"),
        n_boot=50,
        warmup=hours(1),
    )
    assert captured == [custom, custom]  # both arms of the pair, same weights


def test_diff_direction_is_baseline_minus_optimized() -> None:
    contrasts = run_paired_comparison(
        tiny_scenario(horizon_hours=4, rate_per_hour=3.0),
        (5,),
        objective=ObjectiveConfig(),
        arms=("baseline", "baseline"),
        n_boot=50,
        warmup=hours(1),
    )
    by_key = {c.key: c for c in contrasts}
    completions = by_key["completions_per_week"]
    assert completions.diff == completions.baseline - completions.optimized
