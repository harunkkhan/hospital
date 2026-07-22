"""Paired comparison — CRN pairing loop; stats delegated to analysis.paired_bootstrap."""

from __future__ import annotations

import math

from _sim_fixtures import tiny_scenario

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
    assert [c.key for c in contrasts] == list(KPI_KEYS)  # one contrast per KPI key
    for c in contrasts:
        assert not c.significant
        if not math.isnan(c.diff):
            assert c.diff == 0.0
        if not math.isnan(c.baseline) and not math.isnan(c.optimized):
            assert c.baseline == c.optimized  # the pairing really used the same seed


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
