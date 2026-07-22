"""``run_paired_comparison`` — paired-arm orchestration (doc 04 §3.15 / nuance 4.11).

This module owns the *pairing loop* only; the statistics (percentile bootstrap
CIs, Bonferroni correction, significance flags) are
``analysis.compare.paired_bootstrap`` — the ONE bootstrap routine in the repo.

The pairing invariant: both arms of a pair run under the SAME seed, so CRN
gives them the identical realized week (arrivals, acuities, service draws,
dispositions, disruptions) and the per-seed diff ``baseline - optimized``
isolates *decisions*, never weather. For a "lower is better" KPI a positive
diff means the optimized arm helped.

The loop is arm-agnostic (``arms`` defaults to the headline pair, and the
optimized ``PolicySet`` arrives next phase) — pass ``("baseline", "baseline")``
for a null-comparison smoke run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hospital.analysis import paired_bootstrap
from hospital.core import KPI_KEYS, FrozenModel, KpiVector
from hospital.sim.experiment.replication import run_replication
from hospital.sim.experiment.scorecard import fold_scorecard
from hospital.solver import ObjectiveConfig

if TYPE_CHECKING:
    from hospital.core import Duration
    from hospital.data.scenario import Scenario
    from hospital.sim.policies.factory import Arm


class KpiContrast(FrozenModel):
    """One KPI's paired contrast: per-seed mean diff with its bootstrap CI."""

    key: str
    baseline: float
    optimized: float
    diff: float  # baseline - optimized (positive = optimized helped, for lower-is-better)
    ci_lo: float
    ci_hi: float
    significant: bool


def run_paired_comparison(
    scenario: Scenario,
    seeds: tuple[int, ...],
    *,
    objective: ObjectiveConfig,
    arms: tuple[Arm, Arm] = ("baseline", "optimized"),
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
    warmup: Duration | None = None,
) -> tuple[KpiContrast, ...]:
    """Per seed: run both arms under identical CRN, fold, then delegate the stats."""
    baseline_vectors: list[KpiVector] = []
    optimized_vectors: list[KpiVector] = []
    for seed in seeds:
        baseline_rep = run_replication(scenario, arms[0], seed)
        optimized_rep = run_replication(scenario, arms[1], seed)  # SAME seed: the pairing
        baseline_vectors.append(fold_scorecard(baseline_rep, objective, warmup=warmup).kpis)
        optimized_vectors.append(fold_scorecard(optimized_rep, objective, warmup=warmup).kpis)

    result = paired_bootstrap(
        baseline_vectors, optimized_vectors, n_boot=n_boot, seed=bootstrap_seed
    )
    contrasts: list[KpiContrast] = []
    for key in KPI_KEYS:
        contrast = result.contrasts[key]
        contrasts.append(
            KpiContrast(
                key=key,
                baseline=contrast.baseline_mean,
                optimized=contrast.optimized_mean,
                diff=contrast.diff_mean,
                ci_lo=contrast.ci_lo,
                ci_hi=contrast.ci_hi,
                significant=contrast.significant,
            )
        )
    return tuple(contrasts)


__all__ = ["KpiContrast", "run_paired_comparison"]
