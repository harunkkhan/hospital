"""hospital.sim — the SimPy digital twin: physics, policies, and the experiment root.

Re-exports ONLY the public surface (doc 00 §3 / doc 04 §2): the mutable-state
owner, the composition root and its run record, the fold/compare entry points,
and the policy factory. Everything else (executor, seam adapter, flow
processes, injectors) is import-by-module internal.
"""

from __future__ import annotations

from hospital.sim.experiment.comparison import run_paired_comparison
from hospital.sim.experiment.replication import Replication, run_replication
from hospital.sim.experiment.scorecard import Scorecard, fold_scorecard
from hospital.sim.physics.world import World
from hospital.sim.policies.factory import make_policies
from hospital.sim.policies.protocols import PolicySet

__all__ = [
    "PolicySet",
    "Replication",
    "Scorecard",
    "World",
    "fold_scorecard",
    "make_policies",
    "run_paired_comparison",
    "run_replication",
]
