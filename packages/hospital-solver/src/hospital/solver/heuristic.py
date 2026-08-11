"""``HeuristicPlacement`` -- a fast greedy placement backend (doc 03 §4.9).

Registered beside CP-SAT (rule 8's "two real backends"), it is also CP-SAT's
warm-start seed and the OR-Tools-free fallback: it has **no ``ortools`` import**,
so importing/using it never pulls OR-Tools.

It reads the *same* ``placement_weights`` table and ``candidates`` compat set as
CP-SAT (no second definition, rules 1/6), so it is a different search over the one
objective rather than a second objective: order ``P`` by
``sequencing.priority_score`` descending, then greedily take ``argmin_b w[p,b]``
among free / compatible / zone-open bays. Pricing each pair by its patient's care
phase — an ED placement or an admission — therefore comes for free.
Greedy is myopic (it can paint a later high-acuity patient into a corner), so
``objective_value = None`` and status ``HEURISTIC`` -- no optimality claim. Like
every backend it self-validates before returning and never repairs.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import ClassVar

from hospital.core import (
    BayId,
    CompiledRules,
    DecisionInput,
    Duration,
    PatientId,
    Plan,
    PlanItem,
    ZoneType,
)
from hospital.solver.objective import ObjectiveConfig
from hospital.solver.placement import (
    candidates,
    occupied_by_zone_type,
    placement_weights,
    self_validate,
    zone_remaining,
)
from hospital.solver.protocol import RoutingOracle, SolveResult, SolverStatus
from hospital.solver.sequencing import DEFAULT_STARVATION_RATE, priority_score


class HeuristicPlacement:
    """The greedy constructive placement backend (implements ``Solver``)."""

    name: ClassVar[str] = "placement_greedy"
    version: ClassVar[str] = "1.0.0"

    def solve(
        self,
        di: DecisionInput,
        oracle: RoutingOracle,
        *,
        config: ObjectiveConfig,
        rules: CompiledRules,
        time_cap: Duration | None = None,
        warm_start: Plan | None = None,
        expected_stay: Mapping[PatientId, Duration] | None = None,
    ) -> SolveResult:
        del time_cap, warm_start  # greedy is a single deterministic pass
        started_ns = time.perf_counter_ns()
        patients, bays, compat = candidates(di, rules)
        waited = {wp.patient.id: wp.waited for wp in di.waiting}

        order = sorted(
            patients,
            key=lambda p: (
                -priority_score(
                    p.esi,
                    waited.get(p.id, Duration(0)),
                    config=config,
                    starvation_rate=DEFAULT_STARVATION_RATE,
                ),
                p.id.root,
            ),
        )

        # The ONE weight table, snapshotted before any placement — so the greedy order
        # neither re-prices scarcity as it fills bays (which CP-SAT does not do either)
        # nor prices a pair differently from the exact backend it stands in for.
        weight = placement_weights(
            di, patients, bays, compat, oracle, config, expected_stay=expected_stay
        )
        remaining_by_zone = zone_remaining(di)
        remaining_by_zt: dict[ZoneType, int] = {}
        occupied_zt = occupied_by_zone_type(di)
        for bay in bays:
            cap = rules.capacity_for(bay.zone_type)
            if cap is not None and bay.zone_type not in remaining_by_zt:
                remaining_by_zt[bay.zone_type] = max(0, cap - occupied_zt.get(bay.zone_type, 0))

        taken: set[BayId] = set()
        assignments: list[tuple[PatientId, BayId]] = []
        for p in order:
            open_bays = [
                b
                for b in bays
                if (p.id, b.id) in compat
                and b.id not in taken
                and remaining_by_zone.get(b.zone, 0) > 0
                and remaining_by_zt.get(b.zone_type, 1) > 0
            ]
            if not open_bays:
                continue
            best = min(open_bays, key=lambda b: (weight[(p.id, b.id)], b.id.root))
            assignments.append((p.id, best.id))
            taken.add(best.id)
            remaining_by_zone[best.zone] -= 1
            if best.zone_type in remaining_by_zt:
                remaining_by_zt[best.zone_type] -= 1

        assignments.sort(key=lambda pb: (pb[0].root, pb[1].root))
        plan = Plan(
            items=tuple(
                PlanItem(stable_id=f"assign:{pid.root}", kind="assign_bay", patient=pid, bay=bid)
                for pid, bid in assignments
            )
        )
        self_validate(plan, di, rules)
        elapsed_us = (time.perf_counter_ns() - started_ns) // 1000
        return SolveResult(
            plan=plan,
            status=SolverStatus.HEURISTIC,
            objective_value=None,
            solve_wall_us=elapsed_us,
            backend=self.name,
        )


__all__ = ["DEFAULT_STARVATION_RATE", "HeuristicPlacement"]
