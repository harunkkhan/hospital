"""Typed exceptions for ``hospital.core``.

Every failure the contract can produce has a distinct type so callers (and
tests) can assert *which* thing went wrong rather than matching on message
strings. All inherit from :class:`HospitalCoreError`.

This module imports nothing from the rest of ``hospital.core`` at runtime (only
a ``TYPE_CHECKING`` reference to :class:`~hospital.core.validation.Violation`),
so it sits at the very bottom of the dependency order and can never introduce a
cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hospital.core.validation import Violation


class HospitalCoreError(Exception):
    """Base class for every typed error raised out of ``hospital.core``."""


class SeamViolation(HospitalCoreError):  # noqa: N818
    """The decision/physics seam contract was broken.

    Raised when a decision-layer producer tries to do something the seam
    forbids (e.g. reading hidden state, or a malformed ``DecisionResponse``).
    """


class InfeasiblePlan(HospitalCoreError):  # noqa: N818
    """A plan failed validation; it carries the concrete violations.

    The engine *rejects, never repairs*: on an infeasible plan (whether from the
    solver or an operator override) it raises this with the full violation tuple
    and mutates no state. The violations are surfaced verbatim to the operator.
    """

    def __init__(self, violations: tuple[Violation, ...]) -> None:
        self.violations: tuple[Violation, ...] = tuple(violations)
        detail = ", ".join(f"{v.kind}:{v.entity}" for v in self.violations) or "<none>"
        super().__init__(f"plan is infeasible ({len(self.violations)} violation(s)): {detail}")


class UnknownEntity(HospitalCoreError):  # noqa: N818
    """A plan or query referenced an id absent from the context."""


class ZeroTimeCycle(HospitalCoreError):  # noqa: N818
    """A same-instant causal loop was detected in the executor.

    Guards against zero-duration services chaining into an infinite loop at a
    single ``SimTime`` (see the zero-distance-edge edge case in the layout).
    """


class KpiContractError(HospitalCoreError):
    """A :class:`~hospital.core.kpi.KpiVector` violated the closed KPI contract.

    Raised for an unknown/extra key, a missing key, or ``staff_frac_*`` that do
    not sum to 1.0 within tolerance.
    """


class LayoutError(HospitalCoreError):
    """The route graph / floor layout is malformed for the requested operation.

    Raised by :func:`~hospital.core.graph.RouteGraph.dijkstra` when ``src``/``dst``
    is unknown or closed, or when no path exists; and by ``data.generate_floor``
    when the generated floor is disconnected.
    """


__all__ = [
    "HospitalCoreError",
    "InfeasiblePlan",
    "KpiContractError",
    "LayoutError",
    "SeamViolation",
    "UnknownEntity",
    "ZeroTimeCycle",
]
