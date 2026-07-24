"""The decision/physics seam: build projection, validate-then-apply, reject (doc 04 §3.5).

Two load-bearing properties:

* **``build_decision_input`` is a pure read with NO hidden fields.** It
  assembles only observable projections (waiting patients, bay states, staff
  states, pending ``TaskSpec``s, recent events) — never a patient's true future
  LOS, a service duration, or an un-ordered workup. This is the noninterference
  invariant; a policy sees exactly this and nothing else.
* **``apply_plan`` rejects, never repairs.** It calls the ONE
  ``core.validation.validate`` against the **apply-time** context; a non-empty
  violation tuple raises :class:`~hospital.core.InfeasiblePlan` and mutates
  nothing — no partial apply, no nearest-feasible fix-up. A stale plan (state
  moved between solve and apply) is thereby rejected, and the decision tick
  catches the exception and requests a fresh solve. ``sim`` has exactly one
  caller of ``validate()`` — this function.

  ``validate()`` judges the rule contract; it cannot see facts that live only
  in physics (whether a staff member is already mid-task, whether a task is
  still pending, whether a patient is still in the bay wait queue). Those are
  exactly the preconditions the ``World`` mutators enforce by raising, and a
  raise from the Nth mutator would leave items ``1..N-1`` already applied —
  partial application, the one thing the seam forbids. So :func:`_preflight`
  re-checks every dynamic enactment precondition against apply-time physics
  BEFORE the first mutation; any failure behaves exactly like a validation
  violation (``InfeasiblePlan``, nothing mutated, re-solve triggered).

``_enact`` translates each ``PlanItem`` into a ``World`` mutator call and holds
no pathfinding, RNG, event formatting, or optimization. Items whose effect is
already in place (an ``assign_bay`` re-affirming a patient already in that bay,
a ``dispatch`` naming the staff already assigned) are no-ops rather than
cancel-and-recreate churn — the idempotence equivalent of applying only the
``stable_id`` delta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hospital.core import (
    BayAssigned,
    BayStatus,
    CompiledRules,
    DecisionInput,
    EventEnvelope,
    EventLog,
    InfeasiblePlan,
    Plan,
    PlanItem,
    SimTime,
    UnknownEntity,
    ValidationContext,
    Violation,
    validate,
)
from hospital.sim.policies.protocols import PlanOrigin

if TYPE_CHECKING:
    from hospital.core import BayId, PatientId, StaffId, TaskId

    from hospital.sim.physics.executor import TaskExecutor
    from hospital.sim.physics.world import World


def build_decision_input(
    world: World, now: SimTime, events_since: tuple[EventEnvelope, ...]
) -> DecisionInput:
    """The immutable projection a policy decides against — a pure read."""
    return DecisionInput(
        now=now,
        layout=world.layout,
        waiting=world.waiting_for_bay(),
        bays=world.snapshot_bays(),
        staff=world.snapshot_staff(),
        pending_tasks=world.pending_tasks(),
        events_since=events_since,
    )


def validation_context(world: World, rules: CompiledRules) -> ValidationContext:
    """The apply-time context for the one validator.

    Populates the static descriptors (``patients``, ``staff_members``,
    ``tasks``) alongside the dynamic projections — DECISIONS D8: the dynamic
    ``bays``/``staff`` alone cannot express acuity/isolation/skill checks, and
    an entity absent from the context is ALWAYS judged unknown.
    """
    return ValidationContext(
        layout=world.layout,
        bays=world.snapshot_bays(),
        staff=world.snapshot_staff(),
        rules=rules,
        patients=world.known_patients(),
        staff_members=world.roster(),
        tasks=world.live_task_specs(),
    )


def apply_plan(
    world: World,
    plan: Plan,
    ctx: ValidationContext,
    executor: TaskExecutor,
    event_log: EventLog,
    *,
    origin: PlanOrigin = "baseline",
) -> None:
    """Validate-then-apply — the ONLY place a ``Plan`` mutates physics.

    All-or-nothing: any violation raises ``InfeasiblePlan(violations)`` with no
    state touched, so a buggy solver (or a stale plan, or an M2 operator
    override) is a *caught* error, never silent corruption.
    """
    violations = validate(plan, ctx)
    if violations:
        raise InfeasiblePlan(violations)
    stale = _preflight(plan, world)
    if stale:
        raise InfeasiblePlan(stale)
    for item in plan.items:
        _enact(item, world, executor, event_log, origin)


def _preflight(plan: Plan, world: World) -> tuple[Violation, ...]:
    """Check every DYNAMIC enactment precondition before the first mutation.

    ``validate()`` judges the rule contract (compatibility, capacity, skills,
    double-booking); this pass re-checks, against apply-time physics, the
    preconditions the ``World`` mutators would otherwise enforce by raising
    mid-enactment: the target bay actually FREE, the granted patient actually
    still waiting, the dispatched task still pending, the dispatched staff not
    already mid-task. Items earlier in the same plan count (a bay granted by
    item 1 is not grantable by item 3; a staff dispatched by item 2 is busy for
    item 5). No-op items (idempotent re-affirmations) are exempt — they enact
    without touching any precondition.
    """
    out: list[Violation] = []
    granted_bays: dict[BayId, PatientId] = {}
    busy_staff: set[StaffId] = set()
    taken_tasks: set[TaskId] = set()
    for item in plan.items:
        if item.kind == "assign_bay":
            _preflight_assign_bay(item, world, granted_bays, out)
        elif item.kind == "dispatch":
            _preflight_dispatch(item, world, busy_staff, taken_tasks, out)
    return tuple(out)


def _preflight_assign_bay(
    item: PlanItem,
    world: World,
    granted_bays: dict[BayId, PatientId],
    out: list[Violation],
) -> None:
    if item.patient is None or item.bay is None:
        return  # malformed items were already rejected by validate()
    if (
        world.bay_status(item.bay) is BayStatus.OCCUPIED
        and world.occupant(item.bay) == item.patient
    ) or granted_bays.get(item.bay) == item.patient:
        return  # idempotent re-affirmation — enacts as a no-op
    if world.bay_status(item.bay) is not BayStatus.FREE or item.bay in granted_bays:
        out.append(
            Violation(
                kind="bay_incompatible",
                detail="bay is not free at apply time",
                entity=item.bay.root,
            )
        )
        return
    if not world.is_waiting(item.patient) or item.patient in granted_bays.values():
        out.append(
            Violation(
                kind="unknown_entity",
                detail="patient is not waiting for a bay",
                entity=item.patient.root,
            )
        )
        return
    granted_bays[item.bay] = item.patient


def _preflight_dispatch(
    item: PlanItem,
    world: World,
    busy_staff: set[StaffId],
    taken_tasks: set[TaskId],
    out: list[Violation],
) -> None:
    if item.staff is None or item.task is None:
        return  # malformed items were already rejected by validate()
    task = world.task(item.task)
    if task.completed or task.assigned_to == item.staff:
        return  # already done / already exactly this dispatch — enacts as a no-op
    if not world.is_pending(item.task) or item.task in taken_tasks:
        out.append(
            Violation(
                kind="double_booked",
                detail="task is no longer pending at apply time",
                entity=item.task.root,
            )
        )
        return
    if world.staff_task(item.staff) is not None or item.staff in busy_staff:
        out.append(
            Violation(
                kind="double_booked",
                detail="staff is already busy at apply time",
                entity=item.staff.root,
            )
        )
        return
    busy_staff.add(item.staff)
    taken_tasks.add(item.task)


def _enact(
    item: PlanItem,
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    origin: PlanOrigin,
) -> None:
    """Dispatch one validated ``PlanItem`` to its ``World`` mutator."""
    if item.kind == "assign_bay":
        _enact_assign_bay(item, world, executor, event_log, origin)
    elif item.kind == "sequence":
        world.resequence_waiting(item.order or ())
    elif item.kind == "dispatch":
        _enact_dispatch(item, world)
    elif item.kind == "clean":
        _enact_boost_cleaning(item, world)
    elif item.kind == "discharge":
        _enact_boost_documentation(item, world)
    # "staffing" is input-only in v1 (🟡 A7): the roster comes from the
    # scenario; a staffing item validates but enacts nothing.


def _enact_assign_bay(
    item: PlanItem,
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    origin: PlanOrigin,
) -> None:
    if item.patient is None or item.bay is None:
        raise UnknownEntity(f"assign_bay item {item.stable_id} lacks patient/bay")
    # Idempotence: re-affirming a patient already in their own bay is a no-op
    # (the validator accepts it for exactly this reason), never churn.
    if (
        world.bay_status(item.bay) is BayStatus.OCCUPIED
        and world.occupant(item.bay) == item.patient
    ):
        return
    world.grant_bay(item.patient, item.bay)
    event_log.append(
        BayAssigned(occurred_at=executor.now(), patient=item.patient, bay=item.bay, by=origin)
    )


def _enact_dispatch(item: PlanItem, world: World) -> None:
    if item.staff is None or item.task is None:
        raise UnknownEntity(f"dispatch item {item.stable_id} lacks staff/task")
    task = world.task(item.task)
    if task.completed or task.assigned_to == item.staff:
        return  # already done / already exactly this dispatch — no churn
    world.dispatch_task(item.task, item.staff)


def _enact_boost_cleaning(item: PlanItem, world: World) -> None:
    """A ``clean`` item prioritizes the pending cleaning task for its bay."""
    if item.bay is None:
        raise UnknownEntity(f"clean item {item.stable_id} lacks a bay")
    for spec in world.pending_tasks():
        if spec.kind == "cleaning" and world.task(spec.id).bay == item.bay:
            world.boost_task(spec.id)
            return


def _enact_boost_documentation(item: PlanItem, world: World) -> None:
    """A ``discharge`` item prioritizes the patient's documentation task."""
    if item.patient is None:
        raise UnknownEntity(f"discharge item {item.stable_id} lacks a patient")
    for spec in world.pending_tasks():
        if spec.kind == "documentation" and spec.patient == item.patient:
            world.boost_task(spec.id)
            return


__all__ = ["apply_plan", "build_decision_input", "validation_context"]
