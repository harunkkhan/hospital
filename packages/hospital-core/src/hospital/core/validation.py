"""The one validator — the sole judge of a plan.

``validate(plan, ctx)`` returns a (possibly empty) tuple of :class:`Violation`;
an **empty tuple means feasible**. Violations are *data* — the caller decides
what to do (``sim`` raises :class:`~hospital.core.errors.InfeasiblePlan`; ``api``
returns the list to the operator). This is the only implementation; the solver
self-check, the engine's plan acceptance, and operator-override checking all call
it (nuance 1.11).

Validate against the **apply-time** context, not the solve-time snapshot: a plan
feasible when solved can be infeasible when applied (a bay filled in the interim),
and stale plans must be rejected so a re-solve is triggered.

``Violation`` is a single model discriminated by its ``kind`` field (rather than
N subclasses) — this matches doc 01 §11 and avoids colliding with the
:class:`~hospital.core.errors.UnknownEntity` *exception* of the same concept.

Note on context: the doc's 4-field context is augmented here with the static
descriptors the checks actually need — ``patients``, ``staff_members``, and
``tasks`` — because ``validate`` must be a pure function of ``(plan, ctx)`` yet
acuity/isolation/skill judgments need entity attributes the dynamic ``bays``/
``staff`` projections do not carry. All three default to empty.
"""

from __future__ import annotations

from typing import Literal

from hospital.core.entities import Bay, FloorLayout, Patient, StaffMember
from hospital.core.enums import BayStatus
from hospital.core.ids import BayId, PatientId, StaffId, TaskId
from hospital.core.models import FrozenModel
from hospital.core.rules import CompiledRules
from hospital.core.seam import BayState, Plan, PlanItem, StaffState, TaskSpec

ViolationKind = Literal[
    "unknown_entity",
    "bay_incompatible",
    "capacity_exceeded",
    "isolation_violated",
    "staff_lacks_skill",
    "double_booked",
    "precedence_violated",
]


class Violation(FrozenModel):
    """A single rule breach. ``kind`` discriminates; ``entity`` names the culprit."""

    kind: ViolationKind
    detail: str
    entity: str


class ValidationContext(FrozenModel):
    """Everything ``validate`` needs, captured at **apply time**."""

    layout: FloorLayout
    bays: tuple[BayState, ...]
    staff: tuple[StaffState, ...]
    rules: CompiledRules
    patients: tuple[Patient, ...] = ()
    staff_members: tuple[StaffMember, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()


def _v(kind: ViolationKind, detail: str, entity: str) -> Violation:
    return Violation(kind=kind, detail=detail, entity=entity)


def validate(plan: Plan, ctx: ValidationContext) -> tuple[Violation, ...]:
    """Return the violations in ``plan`` under ``ctx`` — empty tuple iff feasible."""
    violations: list[Violation] = []
    static_bay: dict[BayId, Bay] = {b.id: b for b in ctx.layout.bays}
    bay_state = {bs.bay: bs for bs in ctx.bays}
    patient_by_id = {p.id: p for p in ctx.patients}
    staff_by_id = {s.id: s for s in ctx.staff_members}
    task_by_id = {t.id: t for t in ctx.tasks}
    rules = ctx.rules

    _check_items(
        plan.items, static_bay, bay_state, patient_by_id, staff_by_id, task_by_id, rules, violations
    )
    _check_capacity(plan.items, static_bay, bay_state, rules, violations)
    _check_double_booking(plan.items, bay_state, violations)
    return tuple(violations)


def _check_items(
    items: tuple[PlanItem, ...],
    static_bay: dict[BayId, Bay],
    bay_state: dict[BayId, BayState],
    patient_by_id: dict[PatientId, Patient],
    staff_by_id: dict[StaffId, StaffMember],
    task_by_id: dict[TaskId, TaskSpec],
    rules: CompiledRules,
    out: list[Violation],
) -> None:
    for item in items:
        if item.kind == "assign_bay":
            _check_assign_bay(item, static_bay, bay_state, patient_by_id, rules, out)
        elif item.kind == "dispatch":
            _check_dispatch(item, staff_by_id, task_by_id, rules, out)
        elif item.kind == "sequence":
            _check_precedence(item, rules, out)


def _check_assign_bay(
    item: PlanItem,
    static_bay: dict[BayId, Bay],
    bay_state: dict[BayId, BayState],
    patient_by_id: dict[PatientId, Patient],
    rules: CompiledRules,
    out: list[Violation],
) -> None:
    if item.bay is None:
        out.append(_v("unknown_entity", "assign_bay item has no bay", item.stable_id))
        return
    bay = static_bay.get(item.bay)
    if bay is None:
        out.append(_v("unknown_entity", "unknown bay", item.bay.root))
        return

    state = bay_state.get(item.bay)
    if state is not None and state.status == BayStatus.CLOSED:
        out.append(_v("bay_incompatible", "bay is closed", item.bay.root))

    if item.patient is None:
        out.append(_v("unknown_entity", "assign_bay item has no patient", item.stable_id))
        return
    patient = patient_by_id.get(item.patient)
    if patient is None:
        if patient_by_id:
            out.append(_v("unknown_entity", "unknown patient", item.patient.root))
        return

    allowed = rules.zone_types_for(patient.esi)
    if allowed and bay.zone_type not in allowed:
        out.append(
            _v(
                "bay_incompatible",
                f"esi {int(patient.esi)} not allowed in {bay.zone_type.value}",
                item.bay.root,
            )
        )
    missing = rules.equipment_for(patient.esi) - bay.equipment
    if missing:
        out.append(
            _v("bay_incompatible", f"bay missing equipment {sorted(missing)}", item.bay.root)
        )
    if rules.isolation_enforced and patient.isolation_required and not bay.isolation_capable:
        out.append(
            _v("isolation_violated", "isolation patient in non-isolation bay", item.bay.root)
        )


def _check_dispatch(
    item: PlanItem,
    staff_by_id: dict[StaffId, StaffMember],
    task_by_id: dict[TaskId, TaskSpec],
    rules: CompiledRules,
    out: list[Violation],
) -> None:
    if item.staff is None:
        out.append(_v("unknown_entity", "dispatch item has no staff", item.stable_id))
        return
    staff = staff_by_id.get(item.staff)
    if staff is None:
        if staff_by_id:
            out.append(_v("unknown_entity", "unknown staff", item.staff.root))
        return
    if item.task is None:
        return
    task = task_by_id.get(item.task)
    if task is None:
        if task_by_id:
            out.append(_v("unknown_entity", "unknown task", item.task.root))
        return
    if task.required_role != staff.role:
        out.append(
            _v(
                "staff_lacks_skill",
                f"task needs role {task.required_role.value}, staff is {staff.role.value}",
                item.staff.root,
            )
        )
    required = task.required_skills | rules.skills_for(task.kind)
    missing = required - staff.skills
    if missing:
        out.append(
            _v("staff_lacks_skill", f"staff missing skills {sorted(missing)}", item.staff.root)
        )


def _check_precedence(item: PlanItem, rules: CompiledRules, out: list[Violation]) -> None:
    if item.order is None:
        return
    order = list(item.order)
    for before, after in rules.precedences:
        if (
            before.value in order
            and after.value in order
            and order.index(after.value) < order.index(before.value)
        ):
            out.append(
                _v(
                    "precedence_violated",
                    f"{after.value} sequenced before {before.value}",
                    item.stable_id,
                )
            )


def _check_capacity(
    items: tuple[PlanItem, ...],
    static_bay: dict[BayId, Bay],
    bay_state: dict[BayId, BayState],
    rules: CompiledRules,
    out: list[Violation],
) -> None:
    occupied: dict[str, set[BayId]] = {}

    def _zone_type_of(bay_id: BayId) -> str | None:
        obj = static_bay.get(bay_id)
        return obj.zone_type.value if obj is not None else None

    for state in bay_state.values():
        if state.status == BayStatus.OCCUPIED and state.occupant is not None:
            zt = _zone_type_of(state.bay)
            if zt is not None:
                occupied.setdefault(zt, set()).add(state.bay)
    for item in items:
        if item.kind == "assign_bay" and item.bay is not None:
            zt = _zone_type_of(item.bay)
            if zt is not None:
                occupied.setdefault(zt, set()).add(item.bay)

    for zt_value, bays in occupied.items():
        cap = _capacity_for_value(rules, zt_value)
        if cap is not None and len(bays) > cap:
            out.append(_v("capacity_exceeded", f"{len(bays)} > cap {cap} in {zt_value}", zt_value))


def _capacity_for_value(rules: CompiledRules, zt_value: str) -> int | None:
    for zt, cap in rules.capacities:
        if zt.value == zt_value:
            return cap
    return None


def _check_double_booking(
    items: tuple[PlanItem, ...],
    bay_state: dict[BayId, BayState],
    out: list[Violation],
) -> None:
    # Bay double-booking: distinct patients (plan + apply-time occupant) on one bay.
    bay_patients: dict[BayId, set[PatientId]] = {}
    for item in items:
        if item.kind == "assign_bay" and item.bay is not None and item.patient is not None:
            bay_patients.setdefault(item.bay, set()).add(item.patient)
    for bay_id, patients in bay_patients.items():
        distinct = set(patients)
        state = bay_state.get(bay_id)
        if state is not None and state.occupant is not None:
            distinct.add(state.occupant)
        if len(distinct) > 1:
            out.append(_v("double_booked", f"{len(distinct)} patients on one bay", bay_id.root))

    # Staff double-booking: one staff dispatched to more than one task.
    staff_counts: dict[StaffId, int] = {}
    for item in items:
        if item.kind == "dispatch" and item.staff is not None:
            staff_counts[item.staff] = staff_counts.get(item.staff, 0) + 1
    for staff_id, count in staff_counts.items():
        if count > 1:
            out.append(_v("double_booked", f"staff dispatched to {count} tasks", staff_id.root))


__all__ = ["ValidationContext", "Violation", "validate"]
