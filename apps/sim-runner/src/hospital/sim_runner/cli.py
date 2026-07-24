"""``hospital-sim`` — the headless M1 CLI (doc 00 §2, doc 08 §7 step 7).

``hospital-sim run --scenario scenarios/er_floor.yaml [--seed N] [--reps N]
[--arm baseline|optimized|both] [--out metrics.json]`` composes the M1 stack:
``run_replication`` per ``(seed, arm)`` cell under CRN, the one KPI fold
(``analysis.fold`` via ``analysis.report.fold_arm``), the pairing + the one
bootstrap (``sim.experiment.comparison.compare_replications``, which folds via
``sim.fold_scorecard`` and delegates to ``analysis.compare.paired_bootstrap``),
and the canonical ``metrics.json`` writer (``analysis.report``). The CLI itself
computes no statistics and no KPIs — it is orchestration + presentation only.

Conventions:

* ``--seed`` defaults to the scenario's own reference seed; replication ``i``
  runs at ``seed + i`` and both arms of a pair share the same seed (CRN).
* The warmup mirrors ``sim.experiment.scorecard``'s default — 24h for a real
  week, a quarter of the horizon for short scenarios — and the SAME warmup is
  passed to both the per-rep fold (contrast vectors) and the arm summaries, so
  the table and the report cannot disagree.
* ``--arm baseline|optimized`` runs a single arm and writes a ``metrics.json``
  with that arm's summary only (``contrasts``/``headline`` empty — a contrast
  needs both arms); ``--arm both`` (the default) is the full paired comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from hospital.analysis import ArmSummary, fold_arm, write_metrics
from hospital.analysis.compare import ComparisonResult
from hospital.analysis.report import Metrics, build_metrics
from hospital.core import KPI_KEYS, Duration, EventLog, TimeWindow, hours
from hospital.data.layout import generate_floor
from hospital.data.scenario import Scenario, load_scenario, realize_staff
from hospital.sim import Replication, run_replication
from hospital.sim.experiment.comparison import compare_replications
from hospital.solver import ObjectiveConfig

Arm = Literal["baseline", "optimized"]

# The three KPIs the M1 milestone is judged on (doc 08 §4).
HEADLINE_KEYS: tuple[str, ...] = (
    "door_to_provider_s_mean",
    "staff_minutes_walked",
    "completions_per_week",
)


def _default_warmup(scenario: Scenario) -> Duration:
    """24h for a real week; a quarter of the horizon for short test scenarios.

    Mirrors ``sim.experiment.scorecard``'s default so per-rep contrast vectors
    and the arm summaries censor identically.
    """
    horizon = scenario.workload.horizon
    span = horizon.end.root - horizon.start.root
    return Duration(min(hours(24).root, span // 4))


def _run_arm(
    scenario: Scenario, arm: Arm, seeds: tuple[int, ...], *, echo: bool
) -> list[Replication]:
    reps: list[Replication] = []
    for seed in seeds:
        if echo:
            print(f"  running {arm} seed={seed} ...", flush=True)
        reps.append(run_replication(scenario, arm, seed))
    return reps


def _summarize(scenario: Scenario, reps: list[Replication], warmup: Duration) -> ArmSummary:
    layout = generate_floor(scenario.facility)
    horizon = scenario.workload.horizon
    roster = realize_staff(
        scenario.staffing,
        layout,
        TimeWindow(start=horizon.start, end=horizon.end),
    )
    logs = [EventLog.from_jsonl(rep.event_log_jsonl) for rep in reps]
    return fold_arm(logs, layout, roster, window=horizon, warmup=warmup)


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "-"
    return f"{value:,.1f}"


def _print_table(comparison: ComparisonResult) -> None:
    header = (
        f"{'KPI':34s} {'baseline':>12s} {'optimized':>12s} {'diff':>10s} "
        f"{'95% CI (Bonferroni)':>24s} {'sig':>4s}"
    )
    print(header)
    print("-" * len(header))
    ordered = [*HEADLINE_KEYS, *[k for k in KPI_KEYS if k not in HEADLINE_KEYS]]
    for key in ordered:
        c = comparison.contrasts[key]
        marker = " *" if key in HEADLINE_KEYS else ""
        ci = f"[{_fmt(c.ci_lo)}, {_fmt(c.ci_hi)}]"
        sig = "yes" if c.significant else "no"
        print(
            f"{key + marker:34s} {_fmt(c.baseline_mean):>12s} {_fmt(c.optimized_mean):>12s} "
            f"{_fmt(c.diff_mean):>10s} {ci:>24s} {sig:>4s}"
        )
    print("-" * len(header))
    print(
        "diff = baseline - optimized (positive favors OPTIMIZED for lower-is-better KPIs); "
        "* = M1 headline KPI"
    )


def _print_single_arm(arm: str, summary: ArmSummary) -> None:
    header = f"{'KPI':34s} {arm:>14s}"
    print(header)
    print("-" * len(header))
    for key in KPI_KEYS:
        print(f"{key:34s} {_fmt(summary.kpis.values[key]):>14s}")


def run_command(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    seed = args.seed if args.seed is not None else scenario.seed
    seeds = tuple(seed + i for i in range(args.reps))
    warmup = _default_warmup(scenario)
    objective = ObjectiveConfig()
    out = Path(args.out)

    print(f"scenario={scenario.name} seeds={list(seeds)} arm={args.arm}")

    if args.arm == "both":
        baseline_reps = _run_arm(scenario, "baseline", seeds, echo=True)
        optimized_reps = _run_arm(scenario, "optimized", seeds, echo=True)
        baseline_summary = _summarize(scenario, baseline_reps, warmup)
        optimized_summary = _summarize(scenario, optimized_reps, warmup)
        comparison = compare_replications(
            baseline_reps, optimized_reps, objective=objective, warmup=warmup
        )
        metrics = build_metrics(
            scenario.name,
            seed,
            baseline_summary,
            optimized_summary,
            comparison,
            horizon_s=(scenario.workload.horizon.end.root - scenario.workload.horizon.start.root)
            / 1_000_000,
            warmup_s=warmup.root / 1_000_000,
        )
        write_metrics(metrics, out)
        print(f"\nwrote {out}\n")
        _print_table(comparison)
        return 0

    arm = cast("Arm", args.arm)
    reps = _run_arm(scenario, arm, seeds, echo=True)
    summary = _summarize(scenario, reps, warmup)
    metrics = Metrics(
        schema_version="1",
        scenario=scenario.name,
        seed=seed,
        horizon_s=(scenario.workload.horizon.end.root - scenario.workload.horizon.start.root)
        / 1_000_000,
        warmup_s=warmup.root / 1_000_000,
        arms={arm: summary},
        contrasts={},
        headline={},
    )
    write_metrics(metrics, out)
    print(f"\nwrote {out}\n")
    _print_single_arm(arm, summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hospital-sim",
        description="Headless M1 runner: BASELINE vs OPTIMIZED under CRN -> metrics.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a scenario and write metrics.json")
    run.add_argument("--scenario", required=True, help="path to a scenario YAML")
    run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed (default: the scenario's own seed); rep i runs at seed+i",
    )
    run.add_argument("--reps", type=int, default=5, help="paired replications per arm (default 5)")
    run.add_argument(
        "--arm",
        choices=("baseline", "optimized", "both"),
        default="both",
        help="which arm(s) to run (default both -> paired comparison)",
    )
    run.add_argument("--out", default="metrics.json", help="output path (default metrics.json)")
    run.set_defaults(func=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
