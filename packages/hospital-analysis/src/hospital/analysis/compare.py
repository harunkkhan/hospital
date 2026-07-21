"""``paired_bootstrap`` — the ONE bootstrap-comparison routine in the repo.

CRN paired diffs (``baseline - optimized``) resampled over REPLICATIONS, never
patients (doc 05 §4.5 / nuance 5.7 — patient-level resampling would treat
correlated within-run observations as independent and understate variance).
One shared index vector per bootstrap iteration is applied to all
``len(KPI_KEYS)`` keys, preserving cross-KPI correlation. CI bounds use the
same type-7 percentile as ``fold``/``waits`` (``_stats.percentile``), with a
Bonferroni family-wise correction across all keys. ``sim.experiment.comparison``
and ``api.compare`` call this rather than re-deriving statistics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from hospital.analysis._stats import percentile
from hospital.core import KPI_KEYS, FrozenModel, KpiVector, RandomStreams

__all__ = ["ComparisonResult", "Contrast", "paired_bootstrap"]


class Contrast(FrozenModel):
    key: str
    baseline_mean: float
    optimized_mean: float
    diff_mean: float
    ci_lo: float
    ci_hi: float
    significant: bool
    alpha_adjusted: float
    n_pairs: int


class ComparisonResult(FrozenModel):
    contrasts: Mapping[str, Contrast]
    n_reps: int
    n_boot: int
    family_alpha: float
    n_comparisons: int


def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return math.fsum(vals) / len(vals) if vals else float("nan")


def paired_bootstrap(
    baseline_reps: Sequence[KpiVector],
    optimized_reps: Sequence[KpiVector],
    *,
    n_boot: int = 10_000,
    family_alpha: float = 0.05,
    seed: int = 0,
) -> ComparisonResult:
    n_reps = len(baseline_reps)
    if len(optimized_reps) != n_reps:
        raise ValueError("baseline_reps and optimized_reps must have the same length")
    m = len(KPI_KEYS)
    alpha_adjusted = family_alpha / m
    q_lo = alpha_adjusted / 2.0
    q_hi = 1.0 - alpha_adjusted / 2.0

    rng = RandomStreams(seed).substream("bootstrap")

    # Per-key arrays, NaN where either arm is NaN for that rep (pairwise-complete
    # dropping — a rep missing a key is dropped from THAT key's diff vector only).
    baseline_arr: dict[str, np.ndarray] = {}
    optimized_arr: dict[str, np.ndarray] = {}
    diff_arr: dict[str, np.ndarray] = {}
    for key in KPI_KEYS:
        b = np.array([rep.values[key] for rep in baseline_reps], dtype=float)
        o = np.array([rep.values[key] for rep in optimized_reps], dtype=float)
        mask = ~(np.isnan(b) | np.isnan(o))
        baseline_arr[key] = np.where(mask, b, np.nan)
        optimized_arr[key] = np.where(mask, o, np.nan)
        diff_arr[key] = np.where(mask, b - o, np.nan)

    # ONE shared index vector per bootstrap iteration, applied to every key —
    # keeps the resample a coherent reweighting of the same set of replications
    # across all KPIs (preserves joint/cross-KPI structure).
    boot_samples: dict[str, list[float]] = {key: [] for key in KPI_KEYS}
    if n_reps > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, n_reps, size=n_reps)
            for key in KPI_KEYS:
                draws = diff_arr[key][idx]
                valid = draws[~np.isnan(draws)]
                theta = float(np.mean(valid)) if valid.size > 0 else float("nan")
                boot_samples[key].append(theta)

    contrasts: dict[str, Contrast] = {}
    for key in KPI_KEYS:
        d = diff_arr[key]
        n_pairs = int((~np.isnan(d)).sum())
        baseline_mean = _nanmean(baseline_arr[key].tolist())
        optimized_mean = _nanmean(optimized_arr[key].tolist())
        diff_mean = _nanmean(d.tolist())
        if n_pairs < 2:
            ci_lo, ci_hi = float("nan"), float("nan")
        else:
            ci_lo = percentile(boot_samples[key], q_lo)
            ci_hi = percentile(boot_samples[key], q_hi)
        significant = (
            not math.isnan(ci_lo) and not math.isnan(ci_hi) and not (ci_lo <= 0.0 <= ci_hi)
        )
        contrasts[key] = Contrast(
            key=key,
            baseline_mean=baseline_mean,
            optimized_mean=optimized_mean,
            diff_mean=diff_mean,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            significant=significant,
            alpha_adjusted=alpha_adjusted,
            n_pairs=n_pairs,
        )

    return ComparisonResult(
        contrasts=contrasts,
        n_reps=n_reps,
        n_boot=n_boot,
        family_alpha=family_alpha,
        n_comparisons=m,
    )
