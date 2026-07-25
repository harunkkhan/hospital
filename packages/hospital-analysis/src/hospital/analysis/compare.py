"""``paired_bootstrap``/``paired_scalar_contrast`` — the ONE bootstrap core.

CRN paired diffs (``baseline - optimized``) resampled over REPLICATIONS, never
patients (doc 05 §4.5 / nuance 5.7 — patient-level resampling would treat
correlated within-run observations as independent and understate variance).
One shared index vector per bootstrap iteration is applied to all
``len(KPI_KEYS)`` keys, preserving cross-KPI correlation. CI bounds use the
same type-7 percentile as ``fold``/``waits`` (``_stats.percentile``), with a
Bonferroni family-wise correction across all keys. ``sim.experiment.comparison``
and ``api.compare`` call this rather than re-deriving statistics.

Two entry points share one resampling implementation:

* :func:`paired_bootstrap` — the 27-key exploratory KPI family, Bonferroni-
  corrected (``alpha / len(KPI_KEYS)`` per key).
* :func:`paired_scalar_contrast` — ONE pre-registered primary endpoint (the
  G1 acuity-weighted objective), tested at full ``alpha`` (``m = 1``). It is
  deliberately kept OUT of the KPI family: folding it in would widen every
  exploratory CI (``m = 28``), and a single pre-specified primary contrast
  needs no multiplicity correction. Each :class:`Contrast` self-describes via
  ``alpha_adjusted``, so mixed families remain readable downstream.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from hospital.analysis._stats import percentile
from hospital.core import KPI_KEYS, FrozenModel, KpiVector, RandomStreams

__all__ = [
    "WEIGHTED_OBJECTIVE_KEY",
    "ComparisonResult",
    "Contrast",
    "paired_bootstrap",
    "paired_scalar_contrast",
]

# The G1 headline contrast: per-replication ``solver.objective.weighted_total``
# scorecard totals, compared baseline - optimized. This is a REPORT-layer key —
# deliberately NOT a member of ``KPI_KEYS`` (the ``KpiVector`` contract stays
# closed; the objective is a solver-priced scalar over physical inputs, not an
# output of the one KPI fold). ``sim.experiment.comparison`` produces it and
# ``analysis.report``/the CLI surface it under this name.
WEIGHTED_OBJECTIVE_KEY = "weighted_objective_total"


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


def _paired_arrays(
    baseline: Sequence[float], optimized: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise-complete arrays: NaN where EITHER arm is NaN for that rep."""
    b = np.array(baseline, dtype=float)
    o = np.array(optimized, dtype=float)
    mask = ~(np.isnan(b) | np.isnan(o))
    return np.where(mask, b, np.nan), np.where(mask, o, np.nan), np.where(mask, b - o, np.nan)


def _boot_mean(diffs: np.ndarray, idx: np.ndarray) -> float:
    draws = diffs[idx]
    valid = draws[~np.isnan(draws)]
    return float(np.mean(valid)) if valid.size > 0 else float("nan")


def _contrast(
    key: str,
    baseline_arr: np.ndarray,
    optimized_arr: np.ndarray,
    diff_arr: np.ndarray,
    boot_samples: Sequence[float],
    *,
    alpha_adjusted: float,
) -> Contrast:
    """Assemble one key's contrast from its arrays + bootstrap distribution."""
    n_pairs = int((~np.isnan(diff_arr)).sum())
    if n_pairs < 2:
        ci_lo, ci_hi = float("nan"), float("nan")
    else:
        ci_lo = percentile(boot_samples, alpha_adjusted / 2.0)
        ci_hi = percentile(boot_samples, 1.0 - alpha_adjusted / 2.0)
    significant = not math.isnan(ci_lo) and not math.isnan(ci_hi) and not (ci_lo <= 0.0 <= ci_hi)
    return Contrast(
        key=key,
        baseline_mean=_nanmean(baseline_arr.tolist()),
        optimized_mean=_nanmean(optimized_arr.tolist()),
        diff_mean=_nanmean(diff_arr.tolist()),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        significant=significant,
        alpha_adjusted=alpha_adjusted,
        n_pairs=n_pairs,
    )


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

    rng = RandomStreams(seed).substream("bootstrap")

    # Per-key arrays, NaN where either arm is NaN for that rep (pairwise-complete
    # dropping — a rep missing a key is dropped from THAT key's diff vector only).
    baseline_arr: dict[str, np.ndarray] = {}
    optimized_arr: dict[str, np.ndarray] = {}
    diff_arr: dict[str, np.ndarray] = {}
    for key in KPI_KEYS:
        b = [rep.values[key] for rep in baseline_reps]
        o = [rep.values[key] for rep in optimized_reps]
        baseline_arr[key], optimized_arr[key], diff_arr[key] = _paired_arrays(b, o)

    # ONE shared index vector per bootstrap iteration, applied to every key —
    # keeps the resample a coherent reweighting of the same set of replications
    # across all KPIs (preserves joint/cross-KPI structure).
    boot_samples: dict[str, list[float]] = {key: [] for key in KPI_KEYS}
    if n_reps > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, n_reps, size=n_reps)
            for key in KPI_KEYS:
                boot_samples[key].append(_boot_mean(diff_arr[key], idx))

    contrasts = {
        key: _contrast(
            key,
            baseline_arr[key],
            optimized_arr[key],
            diff_arr[key],
            boot_samples[key],
            alpha_adjusted=alpha_adjusted,
        )
        for key in KPI_KEYS
    }

    return ComparisonResult(
        contrasts=contrasts,
        n_reps=n_reps,
        n_boot=n_boot,
        family_alpha=family_alpha,
        n_comparisons=m,
    )


def paired_scalar_contrast(
    baseline: Sequence[float],
    optimized: Sequence[float],
    *,
    key: str,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Contrast:
    """One pre-registered scalar endpoint, same CRN-paired bootstrap, ``m = 1``.

    Same estimator, resampler, percentile, and significance rule as
    :func:`paired_bootstrap`; the only statistical difference is the
    multiplicity family — a single pre-specified primary contrast is tested at
    the full ``alpha`` rather than a Bonferroni share of it (rationale in the
    module docstring).
    """
    if len(optimized) != len(baseline):
        raise ValueError("baseline and optimized must have the same length")
    n_reps = len(baseline)
    baseline_arr, optimized_arr, diff_arr = _paired_arrays(baseline, optimized)

    rng = RandomStreams(seed).substream("bootstrap")
    boot_samples: list[float] = []
    if n_reps > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, n_reps, size=n_reps)
            boot_samples.append(_boot_mean(diff_arr, idx))

    return _contrast(
        key, baseline_arr, optimized_arr, diff_arr, boot_samples, alpha_adjusted=alpha
    )
