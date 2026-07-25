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
        "diff_mean": 3649887.6,
        "ci_lo": 1772938.9525000001,
        "ci_hi": 5444257.3549999995,
        "significant": True,
    },
    "door_to_provider_s_mean": {
        "diff_mean": -175.3940930754121,
        "ci_lo": -241.1904833604466,
        "ci_hi": -113.57734980786567,
        "significant": True,
    },
    "staff_minutes_walked": {
        "diff_mean": 109.84885813000001,
        "ci_lo": 64.51001675354179,
        "ci_hi": 149.0043047587777,
        "significant": True,
    },
    "completions_per_week": {
        "diff_mean": -0.1,
        "ci_lo": -1.9,
        "ci_hi": 1.7,
        "significant": False,
    },
    "los_s_mean_by_esi_1": {
        "diff_mean": 848.553831845413,
        "ci_lo": 419.1310189790879,
        "ci_hi": 1402.8858580370331,
        "significant": True,
    },
    "los_s_mean_by_esi_2": {
        "diff_mean": 826.3164328646739,
        "ci_lo": 520.000249158824,
        "ci_hi": 1203.8241054814002,
        "significant": True,
    },
    "los_s_mean_by_esi_3": {
        "diff_mean": -386.81699092937487,
        "ci_lo": -562.568034102977,
        "ci_hi": -225.43050539956266,
        "significant": True,
    },
    "los_s_mean_by_esi_4": {
        "diff_mean": -128.6644707584579,
        "ci_lo": -499.78006304558124,
        "ci_hi": 414.4674551888207,
        "significant": False,
    },
    "los_s_mean_by_esi_5": {
        "diff_mean": -732.6016454713905,
        "ci_lo": -2056.811245931604,
        "ci_hi": 569.4197596815125,
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
