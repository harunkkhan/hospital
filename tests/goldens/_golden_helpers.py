"""Shared builders for the committed goldens (imported by name; no conftest, D10).

Two goldens live here (doc 08 §4):

* ``er_slice_trace.json`` — a SMALL fixed-seed ER run (the committed stressed
  floor cut to a 6-hour slice), canonicalized. Cheap enough that the golden
  test *recomputes* it and compares byte-for-byte.
* ``metrics.json`` — the M1 headline comparison on
  ``scenarios/er_floor_stressed.yaml`` at the reference seed
  (:data:`METRICS_SEED`, :data:`METRICS_REPS` paired replications), produced by
  the exact ``hospital-sim run`` CLI path. Too expensive to recompute per test
  run; its golden test pins the committed values instead.

Both are regenerated ONLY via ``uv run python tests/goldens/regenerate.py`` and
reviewed as a semantic change — never auto-accepted as churn.
"""

from __future__ import annotations

import json
from pathlib import Path

from hospital.data.scenario import Scenario, apply_overlay, load_scenario
from hospital.sim import Replication, run_replication

GOLDENS_DIR = Path(__file__).resolve().parent
REPO_ROOT = GOLDENS_DIR.parents[1]
STRESSED_SCENARIO = REPO_ROOT / "scenarios" / "er_floor_stressed.yaml"

# metrics.json provenance: the reference seed and paired-rep count of the
# committed M1 comparison (seeds 100..109, both arms under CRN).
METRICS_SEED = 100
METRICS_REPS = 10

_SLICE_HORIZON_US = 6 * 3600 * 1_000_000  # [0, 6h): small enough to recompute in-test


def slice_scenario() -> Scenario:
    """The committed stressed floor, cut to a six-hour slice (same demand model)."""
    base = load_scenario(STRESSED_SCENARIO)
    return apply_overlay(
        base,
        {"name": "er_slice", "workload": {"horizon": {"start": 0, "end": _SLICE_HORIZON_US}}},
    )


def slice_replication() -> Replication:
    """One OPTIMIZED-arm run of the slice at the scenario's own seed.

    The optimized arm pins the deepest stack (solver placement/dispatch +
    policies + physics); baseline-arm drift shows up through the paired
    comparison and the determinism property tests instead.
    """
    scenario = slice_scenario()
    return run_replication(scenario, "optimized", scenario.seed)


def canonical_trace(rep: Replication) -> str:
    """Sorted-keys, stable-format JSON for a run's event stream (doc 08 §4).

    ``objective_hash`` rides along so a change to the experiment weight set
    (``DEFAULT_OBJECTIVE``) fails the golden even if the realized events happen
    to coincide.
    """
    events = [json.loads(line) for line in rep.event_log_jsonl.splitlines()]
    payload = {
        "arm": rep.arm,
        "events": events,
        "objective_hash": rep.objective_hash,
        "run_id": rep.run_id.root,
        "scenario": rep.scenario.name,
        "seed": rep.seed,
    }
    return json.dumps(payload, sort_keys=True, indent=1) + "\n"


def slice_trace_canonical() -> str:
    return canonical_trace(slice_replication())
