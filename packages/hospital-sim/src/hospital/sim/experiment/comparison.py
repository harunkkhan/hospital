"""``run_paired_comparison`` — paired-arm orchestration (doc 04 §3.15 / nuance 4.11).

This module owns the *pairing loop* only; the statistics (percentile bootstrap
CIs, Bonferroni correction, significance flags) live in ``analysis.compare`` —
the ONE bootstrap core in the repo.

The pairing invariant: both arms of a pair run under the SAME seed, so CRN
gives them the identical realized week (arrivals, acuities, service draws,
dispositions, disruptions) and the per-seed diff ``baseline - optimized``
isolates *decisions*, never weather. For a "lower is better" KPI a positive
diff means the optimized arm helped.

Alongside the 27-key KPI family, the comparison carries the **G1 headline**:
the per-seed ``Scorecard.objective_total`` (``solver.objective.weighted_total``
— acuity-weighted patient-seconds + staff travel) contrasted under
``analysis.compare.paired_scalar_contrast`` and reported as
``WEIGHTED_OBJECTIVE_KEY``. It rides in ``ComparisonResult.contrasts`` next to
the KPI keys but is NOT a ``KPI_KEYS`` member — the ``KpiVector`` contract
stays closed; the objective is the solver's priced scalar, not a fold output —
and its ``alpha_adjusted`` self-describes the m=1 primary-endpoint family
(``ComparisonResult.n_comparisons`` keeps counting the Bonferroni KPI family
only).

The loop is arm-agnostic (``arms`` defaults to the headline pair, and the
optimized ``PolicySet`` arrives next phase) — pass ``("baseline", "baseline")``
for a null-comparison smoke run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hospital.analysis import (
    WEIGHTED_OBJECTIVE_KEY,
    ComparisonResult,
    paired_bootstrap,
    paired_scalar_contrast,
)
from hospital.core import KPI_KEYS, FrozenModel
from hospital.sim.experiment.replication import Replication, run_replication
from hospital.sim.experiment.scorecard import fold_scorecard
from hospital.solver import ObjectiveConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def compare_replications(
    baseline: Sequence[Replication],
    optimized: Sequence[Replication],
    *,
    objective: ObjectiveConfig,
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
    warmup: Duration | None = None,
) -> ComparisonResult:
    """Fold two already-run arms (paired by position) and delegate the stats.

    The CRN pairing invariant is *checked*, never assumed: pair ``i`` of both
    sequences must carry the same seed, or the diff would compare different
    realized weeks and the bootstrap would measure weather, not decisions.
    Callers that already hold the ``Replication``s (the CLI folds them into
    per-arm summaries too) reuse this instead of re-running the week.
    """
    if len(baseline) != len(optimized):
        raise ValueError("paired comparison needs the same number of reps per arm")
    for b, o in zip(baseline, optimized, strict=True):
        if b.seed != o.seed:
            raise ValueError(f"pair must share a seed (CRN): baseline={b.seed} optimized={o.seed}")
    baseline_cards = [fold_scorecard(rep, objective, warmup=warmup) for rep in baseline]
    optimized_cards = [fold_scorecard(rep, objective, warmup=warmup) for rep in optimized]
    result = paired_bootstrap(
        [card.kpis for card in baseline_cards],
        [card.kpis for card in optimized_cards],
        n_boot=n_boot,
        seed=bootstrap_seed,
    )
    weighted = paired_scalar_contrast(
        [float(card.objective_total) for card in baseline_cards],
        [float(card.objective_total) for card in optimized_cards],
        key=WEIGHTED_OBJECTIVE_KEY,
        n_boot=n_boot,
        seed=bootstrap_seed,
    )
    # The G1 headline leads the mapping; n_comparisons still counts the
    # Bonferroni KPI family only (the scalar self-describes alpha_adjusted).
    return result.model_copy(
        update={"contrasts": {WEIGHTED_OBJECTIVE_KEY: weighted, **result.contrasts}}
    )


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
    """Per seed: run both arms under identical CRN, fold, then delegate the stats.

    The caller's ``objective`` both DRIVES the optimized arm's decisions
    (threaded into ``run_replication`` -> ``make_policies``) and scores the
    logs (``fold_scorecard``) — one weight set, decisions and headline agree.
    """
    baseline_reps = [
        run_replication(scenario, arms[0], seed, objective=objective) for seed in seeds
    ]
    optimized_reps = [
        run_replication(scenario, arms[1], seed, objective=objective)  # SAME seeds (CRN)
        for seed in seeds
    ]
    result = compare_replications(
        baseline_reps,
        optimized_reps,
        objective=objective,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
        warmup=warmup,
    )
    contrasts: list[KpiContrast] = []
    for key in (WEIGHTED_OBJECTIVE_KEY, *KPI_KEYS):  # G1 headline first
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


__all__ = ["KpiContrast", "compare_replications", "run_paired_comparison"]
