"""The lazy, ``importlib``-based backend registry (doc 03 §3.2).

Two *real* placement backends exist (CP-SAT + heuristic), so a registry earns
its place (doc 00 §5 rule 8). Lazy import means ``import hospital.solver`` costs
nothing in OR-Tools until a CP-SAT backend is actually requested, and the
heuristic backend stays usable where OR-Tools is absent.

The registry caches *constructed* instances, so a backend must be **stateless
across solves** — everything a solve needs arrives as ``solve`` arguments.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

from hospital.core import UnknownEntity

if TYPE_CHECKING:
    from hospital.solver.protocol import Solver

# name -> "module:Class", imported lazily on first request.
_BACKENDS: Final[dict[str, str]] = {
    "placement_cpsat": "hospital.solver.placement:CpSatPlacement",
    "placement_greedy": "hospital.solver.heuristic:HeuristicPlacement",
}

# Cache of constructed, reused backend instances.
_INSTANCES: dict[str, Solver] = {}


def available_backends() -> tuple[str, ...]:
    """The registered backend names, sorted."""
    return tuple(sorted(_BACKENDS))


def get_backend(name: str) -> Solver:
    """Import + construct + cache the backend ``name``; ``UnknownEntity`` on miss."""
    cached = _INSTANCES.get(name)
    if cached is not None:
        return cached
    target = _BACKENDS.get(name)
    if target is None:
        raise UnknownEntity(f"unknown solver backend: {name!r} (known: {sorted(_BACKENDS)})")
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    instance: Solver = backend_cls()
    _INSTANCES[name] = instance
    return instance


__all__ = ["available_backends", "get_backend"]
