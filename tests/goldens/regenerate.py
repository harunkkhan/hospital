"""Regenerate the committed goldens (doc 08 §4).

Usage::

    uv run python tests/goldens/regenerate.py [--only {trace,metrics}]

A golden diff is a SEMANTIC change: regenerate, inspect the diff, update the
pinned expectations in ``test_golden_metrics.py`` if the metrics moved, and
say WHY in the commit — never auto-accept as churn.

``metrics.json`` runs the real ``hospital-sim run`` CLI path (10 paired
replications of the stressed week, both arms) and takes several minutes.
"""

from __future__ import annotations

import argparse

from _golden_helpers import (
    GOLDENS_DIR,
    METRICS_REPS,
    METRICS_SEED,
    STRESSED_SCENARIO,
    slice_trace_canonical,
)

from hospital.sim_runner import cli


def regenerate_trace() -> None:
    path = GOLDENS_DIR / "er_slice_trace.json"
    path.write_text(slice_trace_canonical())
    print(f"wrote {path}")


def regenerate_metrics() -> None:
    out = GOLDENS_DIR / "metrics.json"
    rc = cli.main(
        [
            "run",
            "--scenario",
            str(STRESSED_SCENARIO),
            "--seed",
            str(METRICS_SEED),
            "--reps",
            str(METRICS_REPS),
            "--out",
            str(out),
        ]
    )
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("trace", "metrics"),
        default=None,
        help="regenerate a single golden (default: both)",
    )
    args = parser.parse_args()
    if args.only in (None, "trace"):
        regenerate_trace()
    if args.only in (None, "metrics"):
        regenerate_metrics()


if __name__ == "__main__":
    main()
