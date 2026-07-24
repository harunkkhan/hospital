"""The OPTIMIZED arm — thin adapters over ``hospital.solver`` backends (doc 04 §3.8).

**Documented stub — NEXT PHASE.** This module is the factory's registration
point for ``kind="optimized"``; the adapters themselves land with the
optimized-policies phase (doc 08 §7 step 7). The shape they must take:

* each ``Solver*`` policy is a *marshaller*: it passes the ``DecisionInput`` +
  ``RoutingOracle`` to a backend obtained via ``solver.registry.get_backend``,
  ``solver.stamping.stamp``s the ``SolveResult`` with the ``ObjectiveConfig``
  (provenance choke point: backend version + ``config_hash`` ride the wire),
  and returns the stamped plan's items;
* NO optimization math, pathfinding, RNG, or validation lives here (anti-dup
  rule 3) — the solver self-checks with ``validate()`` before returning and the
  seam adapter re-checks on apply (one implementation, two enforcement points);
* a capped-wall-time solve may return ``HEURISTIC``/``FEASIBLE`` — the status
  is recorded onto the scorecard, never silently treated as ``OPTIMAL``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hospital.core import CompiledRules
    from hospital.sim.policies.protocols import PolicySet
    from hospital.solver import ObjectiveConfig, RoutingOracle


def make_optimized_policies(
    *,
    oracle: RoutingOracle,
    objective: ObjectiveConfig,
    rules: CompiledRules,
) -> PolicySet:
    """Wire the solver-backed ``PolicySet`` — not yet implemented (next phase)."""
    raise NotImplementedError(
        "the optimized arm arrives with the solver-adapter phase (doc 08 §7 step 7); "
        "only kind='baseline' is available in this build"
    )


__all__ = ["make_optimized_policies"]
