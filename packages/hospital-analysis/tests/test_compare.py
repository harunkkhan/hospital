"""``compare.paired_bootstrap`` — CI + significance flag, NaN tolerance, reproducibility."""

from __future__ import annotations

import math

from _analysis_fixtures import full_kpi_values

from hospital.analysis.compare import paired_bootstrap, paired_scalar_contrast
from hospital.core import KPI_KEYS, KpiVector

_N_BOOT = 500  # small for test speed; reproducibility/coverage don't need 10_000


def _reps(n: int, **overrides_per_key: list[float]) -> list[KpiVector]:
    """Build ``n`` KpiVectors, each key defaulting to 0.0 unless overridden per-index."""
    out: list[KpiVector] = []
    for i in range(n):
        overrides = {key: values[i] for key, values in overrides_per_key.items()}
        out.append(KpiVector(values=full_kpi_values(**overrides)))
    return out


def test_known_difference_is_significant_with_correct_sign() -> None:
    n = 8
    baseline = _reps(n, completions_per_week=[800.0 + i for i in range(n)])
    optimized = _reps(n, completions_per_week=[850.0 + i for i in range(n)])
    result = paired_bootstrap(baseline, optimized, n_boot=_N_BOOT, seed=1)
    c = result.contrasts["completions_per_week"]
    assert c.n_pairs == n
    assert math.isclose(c.diff_mean, -50.0)
    assert c.significant
    assert c.ci_lo < 0.0  # whole CI is negative (excludes 0) — baseline < optimized
    assert c.ci_hi < 0.0


def test_identical_arms_not_significant() -> None:
    n = 6
    reps = _reps(n, completions_per_week=[100.0 + i for i in range(n)])
    result = paired_bootstrap(reps, reps, n_boot=_N_BOOT, seed=2)
    c = result.contrasts["completions_per_week"]
    assert math.isclose(c.diff_mean, 0.0, abs_tol=1e-9)
    assert not c.significant
    assert c.ci_lo <= 0.0 <= c.ci_hi


def test_reproducible_ci_given_same_seed() -> None:
    n = 6
    baseline = _reps(n, wip_end_of_week=[10.0 + i for i in range(n)])
    optimized = _reps(n, wip_end_of_week=[8.0 + i for i in range(n)])
    r1 = paired_bootstrap(baseline, optimized, n_boot=_N_BOOT, seed=42)
    r2 = paired_bootstrap(baseline, optimized, n_boot=_N_BOOT, seed=42)
    c1 = r1.contrasts["wip_end_of_week"]
    c2 = r2.contrasts["wip_end_of_week"]
    assert c1.ci_lo == c2.ci_lo
    assert c1.ci_hi == c2.ci_hi


def test_nan_in_one_arm_drops_that_pair_only() -> None:
    n = 6
    key = "los_s_mean_by_esi_1"
    baseline_values = [100.0] * n
    baseline_values[0] = float("nan")
    baseline = _reps(n, **{key: baseline_values})
    optimized = _reps(n, **{key: [90.0] * n})
    result = paired_bootstrap(baseline, optimized, n_boot=_N_BOOT, seed=3)
    c = result.contrasts[key]
    assert c.n_pairs == n - 1
    assert not math.isnan(c.diff_mean)
    # every other key is unaffected (still n_pairs == n)
    assert result.contrasts["completions_per_week"].n_pairs == n


def test_all_kpi_keys_present_and_n_pairs_less_than_two_is_nan_ci() -> None:
    baseline = _reps(1, completions_per_week=[100.0])
    optimized = _reps(1, completions_per_week=[90.0])
    result = paired_bootstrap(baseline, optimized, n_boot=_N_BOOT, seed=4)
    assert set(result.contrasts.keys()) == set(KPI_KEYS)
    c = result.contrasts["completions_per_week"]
    assert c.n_pairs == 1
    assert math.isnan(c.ci_lo)
    assert math.isnan(c.ci_hi)
    assert not c.significant


def test_bonferroni_alpha_adjusted() -> None:
    baseline = _reps(4, completions_per_week=[100.0, 101.0, 102.0, 103.0])
    optimized = _reps(4, completions_per_week=[90.0, 91.0, 92.0, 93.0])
    result = paired_bootstrap(baseline, optimized, n_boot=_N_BOOT, family_alpha=0.05, seed=5)
    c = result.contrasts["completions_per_week"]
    assert math.isclose(c.alpha_adjusted, 0.05 / len(KPI_KEYS))
    assert result.n_comparisons == len(KPI_KEYS)


def test_scalar_contrast_known_difference_significant_at_full_alpha() -> None:
    # A single pre-registered endpoint runs at m=1: alpha_adjusted is the full
    # alpha, never a Bonferroni share of the KPI family.
    baseline = [1000.0 + i for i in range(8)]
    optimized = [900.0 + i for i in range(8)]
    c = paired_scalar_contrast(baseline, optimized, key="obj", n_boot=_N_BOOT, seed=6)
    assert c.key == "obj"
    assert c.alpha_adjusted == 0.05
    assert c.n_pairs == 8
    assert math.isclose(c.diff_mean, 100.0)
    assert c.significant
    assert c.ci_lo > 0.0 and c.ci_hi > 0.0


def test_scalar_contrast_identical_arms_not_significant_and_reproducible() -> None:
    xs = [50.0, 51.0, 49.0, 52.0, 48.0]
    c1 = paired_scalar_contrast(xs, xs, key="obj", n_boot=_N_BOOT, seed=7)
    c2 = paired_scalar_contrast(xs, xs, key="obj", n_boot=_N_BOOT, seed=7)
    assert not c1.significant
    assert math.isclose(c1.diff_mean, 0.0, abs_tol=1e-9)
    assert (c1.ci_lo, c1.ci_hi) == (c2.ci_lo, c2.ci_hi)


def test_scalar_contrast_nan_pair_dropped_and_single_pair_has_nan_ci() -> None:
    baseline = [float("nan"), 10.0, 11.0]
    optimized = [5.0, 8.0, 9.0]
    c = paired_scalar_contrast(baseline, optimized, key="obj", n_boot=_N_BOOT, seed=8)
    assert c.n_pairs == 2
    assert math.isclose(c.diff_mean, 2.0)

    single = paired_scalar_contrast([10.0], [8.0], key="obj", n_boot=_N_BOOT, seed=9)
    assert single.n_pairs == 1
    assert math.isnan(single.ci_lo) and math.isnan(single.ci_hi)
    assert not single.significant
