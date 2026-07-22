"""``HeuristicPlacement`` -- a fast greedy placement backend (doc 03 §4.9).

Registered beside CP-SAT (rule 8's "two real backends"), it is also CP-SAT's
warm-start seed and the OR-Tools-free fallback: it has **no ``ortools`` import**,
so importing/using it never pulls OR-Tools.

It reuses the *same* ``w[p,b]`` and ``compat`` derivation as CP-SAT (no second
definition, rules 1/6): order ``P`` by ``sequencing.priority_score`` descending,
then greedily take ``argmin_b w[p,b]`` among free / compatible / zone-open bays.
Greedy is myopic (it can paint a later high-acuity patient into a corner), so
``objective_value = None`` and status ``HEURISTIC`` -- no optimality claim. Like
every backend it self-validates before returning and never repairs.
"""

from __future__ import annotations

import time
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
from hospital.solver.objective import ObjectiveConfig, assignment_coeffs
from hospital.solver.placement import (
    candidates,
    occupied_by_zone_type,
    self_validate,
    travel_weight,
    zone_remaining,
)
from hospital.solver.protocol import RoutingOracle, SolveResult, SolverStatus
from hospital.solver.sequencing import priority_score

# The greedy seed orders by acuity urgency plus a light anti-starvation term, so
# it stays consistent with ``sequencing`` (doc 03 §4.9). A modest default rate;
# the exact value is a tuning knob and does not affect feasibility.
DEFAULT_STARVATION_RATE: int = 1


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
    ) -> SolveResult:
        del time_cap, warm_start  # greedy is a single deterministic pass
        started_ns = time.perf_counter_ns()
        patients, bays, compat = candidates(di, rules)
        layout = di.layout
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
            coeffs = assignment_coeffs(config, p.esi)
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
            best = min(
                open_bays, key=lambda b: (travel_weight(p, b, oracle, layout, coeffs), b.id.root)
            )
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
