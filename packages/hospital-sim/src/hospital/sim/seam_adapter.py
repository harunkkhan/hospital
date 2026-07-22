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
    validate,
)
from hospital.sim.policies.protocols import PlanOrigin

if TYPE_CHECKING:
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
    for item in plan.items:
        _enact(item, world, executor, event_log, origin)


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
