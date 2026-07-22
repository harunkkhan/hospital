"""The pure-surface contracts (doc 03 §3.1) — no OR-Tools on this surface.

``protocol.py`` imports only ``core`` types + ``typing.Protocol`` so
``sim.policies`` can type-annotate against ``Solver`` / ``RoutingOracle``
without pulling the solver stack.

``SolverStatus`` is a *claim*, not a label: ``OPTIMAL`` means CP-SAT proved the
gap closed within the model as posed; ``FEASIBLE`` means a valid incumbent was
found before the time cap fired (best-found, not proven optimal); ``HEURISTIC``
means a constructive plan with no optimality claim. The downgrade is
one-directional — a cap-truncated solve is never upgraded to ``OPTIMAL``.

Note (deviation from doc 03 §3.1): ``solve`` takes ``rules: CompiledRules`` as a
keyword argument. The compat kernel and the self-validation both need the rule
set, and ``DecisionInput`` carries none — so the backend must receive it. This
is the one place the doc's signature is extended, for correctness.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol

from hospital.core import (
    CompiledRules,
    DecisionInput,
    Duration,
    FrozenModel,
    NodeId,
    Plan,
    RoutePath,
)

if TYPE_CHECKING:
    from hospital.solver.objective import ObjectiveConfig
    from hospital.solver.oracle import RouteMask


class SolverStatus(StrEnum):
    """A solve's optimality *claim* — telemetry, never a gate on applying the plan."""

    OPTIMAL = "optimal"  # proven optimal within the model as posed
    FEASIBLE = "feasible"  # valid, best-found, not proven optimal (hit time cap)
    HEURISTIC = "heuristic"  # constructive; no optimality claim


class SolveResult(FrozenModel):
    """A backend's output: a validated plan plus its status and provenance stubs."""

    plan: Plan
    status: SolverStatus
    objective_value: int | None  # integer-scaled surrogate value; None for pure-heuristic
    solve_wall_us: int  # wall time spent solving (µs), for the real-time budget
    backend: str  # registry name, e.g. "placement_cpsat"


class RoutingOracle(Protocol):
    """The spatial-cost surface every lever routes through (doc 03 §3.3)."""

    def distance(self, src: NodeId, dst: NodeId, *, mask: RouteMask = ...) -> Duration: ...

    def path(self, src: NodeId, dst: NodeId, *, mask: RouteMask = ...) -> RoutePath: ...


class Solver(Protocol):
    """The placement-backend contract (the registry-backed family, doc 03 §3.1)."""

    name: ClassVar[str]
    version: ClassVar[str]

    def solve(
        self,
        di: DecisionInput,
        oracle: RoutingOracle,
        *,
        config: ObjectiveConfig,
        rules: CompiledRules,
        time_cap: Duration | None = None,
        warm_start: Plan | None = None,
    ) -> SolveResult: ...


__all__ = ["RoutingOracle", "SolveResult", "Solver", "SolverStatus"]
