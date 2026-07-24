"""``make_policies(kind)`` — the single spot an arm is chosen (doc 04 §3.9).

Everything else in ``sim`` is arm-agnostic: the composition root asks for a
``PolicySet`` by kind and never looks inside. The factory is justified by two
real implementations (anti-dup rule 8): ``baseline`` wires the myopic levers;
``optimized`` wires the thin solver adapters (``policies.optimized``).

Signature note (deviation from the doc's ``(kind, *, oracle, objective)``,
recorded in the build report): the baseline levers also need ``rules`` (the
compatible-bay filter is the compiled-rule kernel, not a re-derivation) and
``roster`` (staff roles/skills for "qualified" — ``StaffState`` carries
neither). Both are static run-scoped facts, threaded once here.
"""

from __future__ import annotations

from typing import Literal

from hospital.core import CompiledRules, StaffMember
from hospital.sim.policies.baseline import (
    FifoDischarge,
    FifoTurnaround,
    FifoWithinAcuity,
    FirstAvailablePlacement,
    InputStaffing,
    NearestIdleDispatch,
)
from hospital.sim.policies.optimized import make_optimized_policies
from hospital.sim.policies.protocols import PolicySet
from hospital.solver import ObjectiveConfig, RoutingOracle

Arm = Literal["baseline", "optimized"]


def make_policies(
    kind: Arm,
    *,
    oracle: RoutingOracle,
    rules: CompiledRules,
    roster: tuple[StaffMember, ...],
    objective: ObjectiveConfig | None = None,
) -> PolicySet:
    """Build the ``PolicySet`` for an arm. ``objective`` is required for ``optimized``."""
    if kind == "baseline":
        return PolicySet(
            placement=FirstAvailablePlacement(rules=rules),
            sequencing=FifoWithinAcuity(),
            dispatch=NearestIdleDispatch(rules=rules, roster=roster),
            turnaround=FifoTurnaround(),
            discharge=FifoDischarge(),
            staffing=InputStaffing(),
            origin="baseline",
        )
    if objective is None:
        raise ValueError("the optimized arm requires an ObjectiveConfig")
    return make_optimized_policies(oracle=oracle, objective=objective, rules=rules, roster=roster)


__all__ = ["Arm", "make_policies"]
