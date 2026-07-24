"""The six lever protocols + ``PolicySet`` (doc 04 §3.6).

Policies **decide, never mutate or validate**: each lever returns only
``PlanItem``s of its own kind, computed from the immutable ``DecisionInput``
(plus the read-only ``RoutingOracle`` for distance *queries*). Mutation is the
seam adapter's job; judgment is ``core.validation.validate``'s. A policy that
reaches into ``World`` has broken the whole seam.

``PolicySet.decide`` concatenates the six levers into one complete
``core.seam.Plan`` plus a ``WakeDirective``. An empty item set is a
``mode="keep"`` response (the seam contract forbids a ``keep`` carrying a
plan). ``origin`` is the plan authorship stamped onto ``BayAssigned.by`` —
``"baseline"`` for the myopic arm, ``"solver"`` for the optimized arm,
``"operator"`` reserved for the M2 override path.

Policy randomness, if any lever ever needs a tie-break, must come from
``substream("policy", ...)`` — never the ``world`` domain, or changing the
policy would change the weather (doc 04 §4.5). The baseline levers are fully
deterministic and draw nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from hospital.core import DecisionInput, DecisionResponse, Plan, PlanItem, WakeDirective
from hospital.solver import RoutingOracle

PlanOrigin = Literal["baseline", "solver", "operator"]


class PlacementPolicy(Protocol):
    def place(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]: ...


class SequencingPolicy(Protocol):
    def sequence(self, di: DecisionInput) -> tuple[PlanItem, ...]: ...


class DispatchPolicy(Protocol):
    def dispatch(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]: ...


class TurnaroundPolicy(Protocol):
    def turnaround(self, di: DecisionInput) -> tuple[PlanItem, ...]: ...


class DischargePolicy(Protocol):
    def discharge(self, di: DecisionInput) -> tuple[PlanItem, ...]: ...


class StaffingPolicy(Protocol):
    def staffing(self, di: DecisionInput) -> tuple[PlanItem, ...]: ...


@dataclass(frozen=True)
class PolicySet:
    """One arm: the six levers composed into a single plan per decision tick."""

    placement: PlacementPolicy
    sequencing: SequencingPolicy
    dispatch: DispatchPolicy
    turnaround: TurnaroundPolicy
    discharge: DischargePolicy
    staffing: StaffingPolicy
    origin: PlanOrigin = "baseline"

    def decide(self, di: DecisionInput, oracle: RoutingOracle) -> DecisionResponse:
        """Compose the levers into one complete ``Plan`` (or ``keep`` if empty)."""
        items = (
            *self.placement.place(di, oracle),
            *self.sequencing.sequence(di),
            *self.dispatch.dispatch(di, oracle),
            *self.turnaround.turnaround(di),
            *self.discharge.discharge(di),
            *self.staffing.staffing(di),
        )
        if not items:
            return DecisionResponse(mode="keep", plan=None, wake=WakeDirective(kind="keep"))
        return DecisionResponse(
            mode="replace", plan=Plan(items=items), wake=WakeDirective(kind="keep")
        )


__all__ = [
    "DischargePolicy",
    "DispatchPolicy",
    "PlacementPolicy",
    "PlanOrigin",
    "PolicySet",
    "SequencingPolicy",
    "StaffingPolicy",
    "TurnaroundPolicy",
]
