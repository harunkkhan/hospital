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
* ``deadline_breach_hours_total`` (added M4b+) states that same trade in the sharpest
  available terms: the optimized arm breaches acuity care deadlines by **38 more
  patient-hours a week**, significantly. It is the acuity-relative, extensive form of
  the door-to-provider regression — an ESI-1 is due immediately and an ESI-5 in two
  hours, so this weights the delay by who was waiting, which an unweighted mean cannot.
  It is also now *priced*, so the trade shows up in money and not only in seconds.
* Throughput is demand-limited at this operating point: completions flat (ns).

**One of M1's three stated acceptance criteria is not met, and that is recorded here rather
than engineered around.** Section 12 accepts M1 "when a reference run produces metrics.json
in which OPTIMIZED beats BASELINE on acuity-weighted time, staff-minutes walked, *and weekly
completions* with significant confidence intervals". This golden delivers the first two and
not the third: ``completions_per_week`` is flat and non-significant.

The reason is the operating point, not the policy. This floor runs at ``bay_utilization``
~0.21 — 76 bays against six arrivals an hour — so throughput is limited by demand and no
placement or dispatch decision can add a completion. ``er_floor_stressed`` stresses
*staffing*, not beds, which is why it does not change that.

Measured, so the claim is not hand-waving: on a bay-constrained variant of the same week (22
bays, ``bay_utilization`` ~0.75) the criterion *does* come alive — the optimized arm completes
about 15 more patients a week than the baseline, significantly, with end-of-week WIP falling
by the same 15. But at six paired reps the *other* two criteria go non-significant there: on a
congested floor the travel saving and the objective gain are swamped by queueing variance.

So no committed scenario satisfies all three at once, and moving this golden to a congested
floor would trade two passes for one. It has not been moved. Choosing an operating point until
an acceptance test goes green is selecting the evidence, which is the same failure as a fake
win — and this file's whole purpose is to refuse that. The gap is a property of the criterion
(the three measures pull toward different floors), not a defect the golden should hide.

Re-baselined three times. Twice for KPI-contract growth (27 -> 30 at M4b, then 31 when
``deadline_breach_hours_total`` landed), where **only the interval bounds moved** —

``paired_bootstrap`` corrects across the whole KPI family, so each extra key tightens the
per-key alpha and widens every exploratory CI a little. Every ``diff_mean`` stayed
byte-identical — the runs, seeds, and CRN draws did not change, and measuring more things
cannot move an estimate.

The third is different in kind: a **behaviour** change, when the documentation load gate
started firing. §7's "schedule documentation into low-load windows" had a mechanism from M1
whose only production input was the neutral ``FloorLoad()``, and 0.0 utilization can never
cross a 0.8 threshold — so it had never once fired. Measuring the load from the staff states
the projection already carries puts it at peak for about half of all decision ticks, and 29
of 32 estimates moved as a result.

**It cost the milestone about 6.5% of its headline win**, and that is the honest reading
rather than a regression to explain away: ``weighted_objective_total`` fell from +3.65M to
+3.41M. Deferring paperwork at peak frees staff for patient-facing work — mean
door-to-provider's regression shrinks from -175 s to -168 s — but patients then wait longer
for the documentation that ends their stay, holding bays and adding acuity-weighted time. The
gate is what the spec asks for; this is what it costs here, at the pre-existing 0.8 threshold,
which was deliberately not tuned to make the number look better.

**No significance verdict flipped in any of the three re-baselines.**

That the milestone survived both is by construction rather than luck:
``weighted_objective_total`` is a pre-registered primary endpoint tested at the full alpha
*outside* the multiplicity family (see ``analysis.compare``), which is exactly what lets the
exploratory family grow without putting the G1 claim at risk.
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
        "diff_mean": 3410959.7,
        "ci_lo": 1415096.39,
        "ci_hi": 5531594.2,
        "significant": True,
    },
    "door_to_provider_s_mean": {
        "diff_mean": -167.7901546954885,
        "ci_lo": -228.12349965492854,
        "ci_hi": -113.71550718758597,
        "significant": True,
    },
    "staff_minutes_walked": {
        "diff_mean": 106.0984132616666,
        "ci_lo": 61.995834087619635,
        "ci_hi": 145.2012721491132,
        "significant": True,
    },
    "completions_per_week": {
        "diff_mean": 0.2,
        "ci_lo": -1.9,
        "ci_hi": 2.1,
        "significant": False,
    },
    "deadline_breach_hours_total": {
        "diff_mean": -38.708130808638884,
        "ci_lo": -61.37200528885143,
        "ci_hi": -24.564418329223994,
        "significant": True,
    },
    "los_s_mean_by_esi_1": {
        "diff_mean": 823.0502519186772,
        "ci_lo": 373.0594256157192,
        "ci_hi": 1396.4187685802337,
        "significant": True,
    },
    "los_s_mean_by_esi_2": {
        "diff_mean": 819.4891513252902,
        "ci_lo": 520.447448894836,
        "ci_hi": 1189.8733430009047,
        "significant": True,
    },
    "los_s_mean_by_esi_3": {
        "diff_mean": -332.519133427469,
        "ci_lo": -520.8627246204493,
        "ci_hi": -180.82952251582554,
        "significant": True,
    },
    "los_s_mean_by_esi_4": {
        "diff_mean": -200.92783711722913,
        "ci_lo": -542.3783773932706,
        "ci_hi": 82.13850001500697,
        "significant": False,
    },
    "los_s_mean_by_esi_5": {
        "diff_mean": -397.8850298881292,
        "ci_lo": -1446.4039967650242,
        "ci_hi": 603.5502607972602,
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
