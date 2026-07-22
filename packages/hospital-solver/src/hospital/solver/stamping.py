"""The provenance choke point (doc 03 §4.10). Pure attribution — no mutation, no re-validation.

``stamp`` is the single doorway a plan passes through on its way out of the
solver. It attaches ``backend``/``backend_version``, ``objective_config_hash``,
``rules_hash``, ``solver_version``, and ``stamped_at`` so ``sim``/``api``/golden
tooling can assert *which backend + which weights + which rule set* produced any
plan, and detect config/rules drift.

It does **not** mutate the plan (immutable anyway) and does **not** re-validate
(the backend already did, rule 5); ``status`` and ``objective_value`` are copied
straight through, so a ``FEASIBLE`` claim is never upgraded on stamp.
"""

from __future__ import annotations

from importlib import metadata

from hospital.core import FrozenModel, Plan, SimTime
from hospital.solver.objective import ObjectiveConfig, config_hash
from hospital.solver.protocol import SolveResult, SolverStatus


def _solver_version() -> str:
    try:
        return metadata.version("hospital-solver")
    except metadata.PackageNotFoundError:  # pragma: no cover - editable install always resolves
        return "0.0.0"


SOLVER_VERSION: str = _solver_version()


class StampedPlan(FrozenModel):
    """An immutable plan with its full solver/objective/rules provenance."""

    plan: Plan
    status: SolverStatus
    objective_value: int | None
    backend: str
    backend_version: str
    objective_config_hash: str
    rules_hash: str | None
    solver_version: str
    stamped_at: SimTime | None


def _backend_version(name: str) -> str:
    """Resolve a registry backend's version by name (empty for non-registry producers)."""
    from hospital.core import UnknownEntity
    from hospital.solver.registry import get_backend

    try:
        return get_backend(name).version
    except UnknownEntity:
        return ""


def stamp(
    result: SolveResult,
    config: ObjectiveConfig,
    *,
    rules_hash: str | None = None,
    now: SimTime | None = None,
    backend_version: str | None = None,
) -> StampedPlan:
    """Wrap a ``SolveResult`` in a fully-attributed :class:`StampedPlan`."""
    resolved_version = (
        backend_version if backend_version is not None else _backend_version(result.backend)
    )
    return StampedPlan(
        plan=result.plan,
        status=result.status,
        objective_value=result.objective_value,
        backend=result.backend,
        backend_version=resolved_version,
        objective_config_hash=config_hash(config),
        rules_hash=rules_hash,
        solver_version=SOLVER_VERSION,
        stamped_at=now,
    )


__all__ = ["SOLVER_VERSION", "StampedPlan", "stamp"]
