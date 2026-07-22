"""hospital.solver — the pure optimization core.

Re-exports **only** the public surface (doc 03 §3). Everything else is
import-by-module internal. Importing this package is OR-Tools-free: the two
placement backends live behind the lazy :func:`get_backend` registry, and every
lever imports ``ortools`` inside the function that needs it — so
``import hospital.solver`` never pulls the solver stack.
"""

from __future__ import annotations

from hospital.solver.discharge import prioritize_discharge
from hospital.solver.dispatch import assign_staff, route_visits
from hospital.solver.objective import ObjectiveConfig, config_hash, weighted_total
from hospital.solver.oracle import GraphRoutingOracle, RouteMask
from hospital.solver.protocol import RoutingOracle, Solver, SolveResult, SolverStatus
from hospital.solver.registry import available_backends, get_backend
from hospital.solver.scheduling import load_roster
from hospital.solver.sequencing import sequence
from hospital.solver.stamping import StampedPlan, stamp
from hospital.solver.turnaround import prioritize_cleaning

__all__ = [
    "GraphRoutingOracle",
    "ObjectiveConfig",
    "RouteMask",
    "RoutingOracle",
    "SolveResult",
    "Solver",
    "SolverStatus",
    "StampedPlan",
    "assign_staff",
    "available_backends",
    "config_hash",
    "get_backend",
    "load_roster",
    "prioritize_cleaning",
    "prioritize_discharge",
    "route_visits",
    "sequence",
    "stamp",
    "weighted_total",
]
