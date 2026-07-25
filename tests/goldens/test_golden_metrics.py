"""Golden metrics (doc 08 §4): the committed M1 proof, pinned to the REAL numbers.

``metrics.json`` is the M1 comparison on ``scenarios/er_floor_stressed.yaml``
(reference seed 100, 10 paired replications, seeds 100-109, both arms under
CRN). It is too expensive to recompute per test run (~20 stressed weeks), so
this test pins the committed values: any regeneration that moves a number
fails here until the pins are deliberately updated and reviewed as a semantic
change (`uv run python tests/goldens/regenerate.py --only metrics`).

The pins assert the HONEST M1 story, trade included:

* G1 headline — ``weighted_objective_total`` (the solver's acuity-weighted
  objective) drops significantly: the optimized arm wins the milestone metric.
* Decisive significant wins: ESI-1/ESI-2 mean LOS, staff minutes walked.
* The explicit, quantified trade: unweighted mean door-to-provider is
  significantly WORSE (negative diff), and so is ESI-3 mean LOS. These pins
  are deliberate — the golden asserts the real numbers, never a fake win.
* Throughput is demand-limited at this operating point: completions flat (ns).
"""

from __future__ import annotations

import json
import math
from typing import Any

from _golden_helpers import GOLDENS_DIR, METRICS_REPS, METRICS_SEED

# Exact values from the committed golden (full float precision). diff is
# baseline - optimized: positive favors OPTIMIZED for lower-is-better metrics.
_EXPECTED_CONTRASTS: dict[str, dict[str, float | bool]] = {
    "weighted_objective_total": {
        "diff_mean": 3618535.4,
        "ci_lo": 1723472.4775,
        "ci_hi": 5429655.154999999,
        "significant": True,
    },
    "door_to_provider_s_mean": {
        "diff_mean": -173.58247093519554,
        "ci_lo": -239.49739476282102,
        "ci_hi": -111.33609384007795,
        "significant": True,
    },
    "staff_minutes_walked": {
        "diff_mean": 108.90191358833336,
        "ci_lo": 63.048642065861216,
        "ci_hi": 148.44848623702782,
        "significant": True,
    },
    "completions_per_week": {
        "diff_mean": 0.0,
        "ci_lo": -1.9,
        "ci_hi": 1.8,
        "significant": False,
    },
    "los_s_mean_by_esi_1": {
        "diff_mean": 847.9509552326311,
        "ci_lo": 418.44940983419474,
        "ci_hi": 1402.2544568269213,
        "significant": True,
    },
    "los_s_mean_by_esi_2": {
        "diff_mean": 827.3230083339377,
        "ci_lo": 521.2332294124935,
        "ci_hi": 1203.8423401525047,
        "significant": True,
    },
    "los_s_mean_by_esi_3": {
        "diff_mean": -388.76352301879615,
        "ci_lo": -563.0262690860152,
        "ci_hi": -227.32916379146008,
        "significant": True,
    },
    "los_s_mean_by_esi_4": {
        "diff_mean": -135.97904623941903,
        "ci_lo": -499.8210907101257,
        "ci_hi": 409.8452498482151,
        "significant": False,
    },
    "los_s_mean_by_esi_5": {
        "diff_mean": -739.6140432848233,
        "ci_lo": -2057.031157237418,
        "ci_hi": 558.9315919419952,
        "significant": False,
    },
}


def _load() -> dict[str, Any]:
    return json.loads((GOLDENS_DIR / "metrics.json").read_text())


def test_golden_metrics_provenance() -> None:
    data = _load()
    assert data["schema_version"] == "1"
    assert data["scenario"] == "er_floor_stressed"
    assert data["seed"] == METRICS_SEED
    assert data["arms"]["baseline"]["n_reps"] == METRICS_REPS
    assert data["arms"]["optimized"]["n_reps"] == METRICS_REPS
    assert data["contrasts"]["weighted_objective_total"]["n_pairs"] == METRICS_REPS


def test_golden_metrics_pinned_contrasts() -> None:
    data = _load()
    for key, expected in _EXPECTED_CONTRASTS.items():
        actual = data["contrasts"][key]
        for field, value in expected.items():
            assert actual[field] == value, (
                f"{key}.{field} drifted: committed golden has {actual[field]!r}, "
                f"pinned {value!r}. Review as a semantic change, then update BOTH "
                "the golden (tests/goldens/regenerate.py --only metrics) and these pins"
            )


def test_golden_metrics_g1_headline_is_a_significant_weighted_win() -> None:
    """The milestone claim itself: the acuity-weighted objective drops, CI > 0."""
    c = _load()["contrasts"]["weighted_objective_total"]
    assert c["significant"] is True
    assert c["alpha_adjusted"] == 0.05  # the one pre-registered primary contrast
    assert c["diff_mean"] > 0.0
    assert c["ci_lo"] > 0.0 and c["ci_hi"] > 0.0
    assert c["baseline_mean"] > c["optimized_mean"]


def test_golden_metrics_pins_the_real_trade_not_a_fake_win() -> None:
    """The negative rows are load-bearing: door-to-provider mean and ESI-3 LOS
    are significantly WORSE for the optimized arm. If either ever flips to a
    win, that is a real (welcome) semantic change — repin deliberately."""
    contrasts = _load()["contrasts"]
    d2p = contrasts["door_to_provider_s_mean"]
    assert d2p["diff_mean"] < 0.0 and d2p["significant"] is True
    esi3 = contrasts["los_s_mean_by_esi_3"]
    assert esi3["diff_mean"] < 0.0 and esi3["significant"] is True
    # demand-limited throughput: completions cannot distinguish the arms
    assert contrasts["completions_per_week"]["significant"] is False


def test_golden_metrics_headline_echoes_contrasts() -> None:
    data = _load()
    headline = data["headline"]
    contrasts = data["contrasts"]
    # The weighted headline is a direct echo of the contrast's diff_mean.
    assert (
        headline["weighted_objective_total_saved"]
        == contrasts["weighted_objective_total"]["diff_mean"]
    )
    # These two are recomputed from the rep-averaged arm summaries, so they
    # agree with the per-rep contrast means only up to float associativity.
    assert math.isclose(
        headline["staff_minutes_walked_saved"],
        contrasts["staff_minutes_walked"]["diff_mean"],
        rel_tol=1e-9,
    )
    assert math.isclose(
        headline["extra_completions_per_week"],
        -contrasts["completions_per_week"]["diff_mean"],
        abs_tol=1e-9,
    )
