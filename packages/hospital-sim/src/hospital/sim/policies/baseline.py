"""The BASELINE arm — myopic, greedy, no lookahead (doc 04 §3.7).

This is the honest floor the OPTIMIZED arm is measured against, so it must be
*competent, not crippled* (doc 04 nuance 4.6): it uses the same routing oracle
for its nearest-idle distance lookup (a query, not optimization), the same
compiled-rule compatible-bay filter, and the same acuity priority as the
optimized arm. The only things it lacks are lookahead and joint optimization —
so every measured gain is attributable to the solver, not to a strawman.

All levers are deterministic (fixed ``BayId`` order, distance-then-id
tie-breaks) and draw no randomness at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hospital.core import (
    Bay,
    BayId,
    BayStatus,
    CompiledRules,
    DecisionInput,
    Patient,
    PlanItem,
    StaffMember,
    StaffState,
    TaskSpec,
    WaitingPatient,
    ZoneType,
)
from hospital.solver import RoutingOracle


def _compatible(bay: Bay, patient: Patient, rules: CompiledRules, stage: str) -> bool:
    """The one compiled-rule compatibility judgment (same kernel the validator uses).

    Phase-aware for the same reason the solver's is: the baseline must be competent,
    not crippled (doc 04 nuance 4.6), and a baseline that offered a just-triaged
    patient an ICU bed would have its plans rejected by the validator forever rather
    than merely placing them worse.
    """
    if bay.zone_type not in rules.zone_types_for_stage(patient.esi, stage):
        return False
    if rules.equipment_for(patient.esi) - bay.equipment:
        return False
    return not (
        rules.isolation_enforced and patient.isolation_required and not bay.isolation_capable
    )


def _service_order(waiting: tuple[WaitingPatient, ...]) -> list[WaitingPatient]:
    """Acuity tiers first, FIFO (arrival) within a tier, id as the total-order tail."""
    return sorted(
        waiting,
        key=lambda w: (int(w.patient.esi), w.patient.arrival_time.root, w.patient.id.root),
    )


@dataclass(frozen=True)
class FirstAvailablePlacement:
    """First FREE compatible bay in fixed ``BayId`` order, per waiting patient.

    Serves both care phases off the one queue: an ED placement and an admission are
    the same greedy scan, differing only in which whitelist ``_compatible`` consults.
    That is the honest baseline for M4 — a floor that takes the first bed that fits,
    with no view of which ward the next ESI-1 will need.

    Capacity discipline: a ``CapacityRule`` can cap a zone type BELOW its
    physical bay count, and the validator counts current occupants PLUS the
    items of the plan under judgment. The policy tracks the same running
    occupancy here — proposing over-cap assignments would be rejected by the
    validator and, being deterministic, re-proposed identically every re-solve
    until the per-instant tick bound raised ``ZeroTimeCycle``.
    """

    rules: CompiledRules

    def place(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]:
        bay_by_id: dict[BayId, Bay] = {b.id: b for b in di.layout.bays}
        free = sorted(
            (bs.bay for bs in di.bays if bs.status is BayStatus.FREE), key=lambda b: b.root
        )
        # Zone-type occupancy the validator will judge against: current
        # occupants first, then every bay this plan takes.
        occupancy: dict[ZoneType, int] = {}
        for bs in di.bays:
            if bs.status is BayStatus.OCCUPIED:
                zt = bay_by_id[bs.bay].zone_type
                occupancy[zt] = occupancy.get(zt, 0) + 1

        def within_capacity(bay: Bay) -> bool:
            cap = self.rules.capacity_for(bay.zone_type)
            return cap is None or occupancy.get(bay.zone_type, 0) < cap

        items: list[PlanItem] = []
        taken: set[BayId] = set()
        for w in _service_order(di.waiting):
            chosen = next(
                (
                    b
                    for b in free
                    if b not in taken
                    and _compatible(bay_by_id[b], w.patient, self.rules, w.stage)
                    and within_capacity(bay_by_id[b])
                ),
                None,
            )
            if chosen is None:
                continue  # no compatible capacity now; the patient keeps waiting
            taken.add(chosen)
            zt = bay_by_id[chosen].zone_type
            occupancy[zt] = occupancy.get(zt, 0) + 1
            items.append(
                PlanItem(
                    stable_id=f"assign:{w.patient.id.root}",
                    kind="assign_bay",
                    patient=w.patient.id,
                    bay=chosen,
                )
            )
        return tuple(items)


@dataclass(frozen=True)
class NearestIdleDispatch:
    """Each pending task -> nearest idle qualified staff; greedy, no batching.

    "Nearest" is an ``oracle.distance`` *query* over the same one Dijkstra the
    optimized arm uses; ties break on staff id for a total order. One task per
    staff per tick (the validator's staff double-booking rule is never risked).

    Qualification is judged on ``task.required_skills | rules.skills_for(kind)``
    — the SAME union the validator (and ``solver.dispatch``) applies. Filtering
    on the spec's skills alone would propose rule-unqualified staff, and the
    deterministic reject/re-solve loop would end in ``ZeroTimeCycle``.
    """

    rules: CompiledRules
    roster: tuple[StaffMember, ...] = field(default=())

    def dispatch(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]:
        members = {m.id: m for m in self.roster}
        idle: list[StaffState] = [
            s
            for s in di.staff
            if s.current_task is None
            and (s.busy_until is None or s.busy_until <= di.now)
            and s.staff in members
        ]
        items: list[PlanItem] = []
        for task in di.pending_tasks:
            candidates = [s for s in idle if _qualified(members[s.staff], task, self.rules)]
            if not candidates:
                continue  # nobody idle and qualified; the task stays pending
            chosen = min(
                candidates,
                key=lambda s: (oracle.distance(s.at, task.at).root, s.staff.root),
            )
            idle.remove(chosen)
            items.append(
                PlanItem(
                    stable_id=f"dispatch:{task.id.root}",
                    kind="dispatch",
                    staff=chosen.staff,
                    task=task.id,
                )
            )
        return tuple(items)


def _qualified(member: StaffMember, task: TaskSpec, rules: CompiledRules) -> bool:
    if member.role != task.required_role:
        return False
    required = task.required_skills | rules.skills_for(task.kind)
    return not (required - member.skills)


@dataclass(frozen=True)
class FifoWithinAcuity:
    """Strict ESI tiers, FIFO within a tier — the ``World`` queue's native order.

    The baseline emits no ``sequence`` items: re-affirming the default order
    every tick would be pure churn. The lever exists so the optimized arm can
    override it.
    """

    def sequence(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        return ()


@dataclass(frozen=True)
class FifoTurnaround:
    """Dirty-bay FIFO — cleaning tasks are served in creation order (no items)."""

    def turnaround(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        return ()


@dataclass(frozen=True)
class FifoDischarge:
    """Discharge/documentation FIFO — served in creation order (no items)."""

    def discharge(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        return ()


@dataclass(frozen=True)
class InputStaffing:
    """Roster comes from the scenario (staffing is input-only in v1, 🟡 A7)."""

    def staffing(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        return ()


__all__ = [
    "FifoDischarge",
    "FifoTurnaround",
    "FifoWithinAcuity",
    "FirstAvailablePlacement",
    "InputStaffing",
    "NearestIdleDispatch",
]
