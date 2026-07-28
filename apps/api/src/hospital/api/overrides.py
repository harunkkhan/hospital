"""``POST /runs/{id}/override`` — the OperatorAction union, compiler, and pin registry.

The safety-critical core of the console (doc 07 §4): every operator action is
compiled into a typed edit — a ``core.seam.Plan`` for the five plan-shaped
actions, or a ``ValidationContext`` delta for the two availability actions —
and judged by the **same** ``core.validation.validate()`` the solver self-checks
with and the engine accepts with. There is no second validator, no operator
fast-path, and no repair:

* non-empty violations -> ``422 OverrideRejected`` with the ``Violation`` tuple
  verbatim; the session state is byte-identical to before the call (atomicity);
* empty violations -> the plan goes through ``sim.seam_adapter.apply_plan``,
  which RE-VALIDATES with the same function (a guard, not a duplicate rule set)
  and additionally preflights dynamic physics preconditions — a stale action
  (the bay filled since the frame the operator saw) is rejected there, never
  silently applied to different state.

Availability actions (``close_bay``/``block_edge``) are NOT new plan-item kinds
(forbidden — doc 00 §5.1): closing a bay changes *what may be assigned*, not an
assignment. They compile to a context delta, the **standing plan** (the current
occupant assignments) is re-validated against it, and only a clean verdict lets
the ``World`` mutator run — so closing a bay whose occupant is mid-workup is
rejected with the stranded ``assign_bay`` violation and the engine never evicts
a patient on the operator's behalf.

Provenance is free: an accepted plan-shaped override reaches physics with
``origin="operator"``, so ``BayAssigned.by == "operator"`` in the one log.

Pins (doc 07 §4.4): ``pin=true`` records the accepted decision in the
:class:`PinRegistry`; the session's decision tick consults it at plan-merge
time, carrying each still-applicable pinned item as a pre-fixed member of the
plan the arm's policy must keep — the merged plan still flows through the one
``validate()``, so pins add no validation path. A pin auto-releases when its
subject leaves the decidable state (patient placed/departed, task done).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from hospital.api.sessions import require_session
from hospital.core import (
    BayId,
    BayStatus,
    FrozenModel,
    InfeasiblePlan,
    LayoutError,
    NodeId,
    PatientId,
    Plan,
    PlanItem,
    SimTime,
    StaffId,
    TaskId,
    UnknownEntity,
    Violation,
    validate,
)
from hospital.core.seam import BayState
from hospital.sim.seam_adapter import apply_plan, validation_context

if TYPE_CHECKING:
    from fastapi import FastAPI

    from hospital.api.sessions import RunSession
    from hospital.sim.physics.world import World

router = APIRouter()


class ReassignAction(FrozenModel):
    """Reassign a waiting patient to a bay -> ``PlanItem(kind="assign_bay")``."""

    kind: Literal["reassign"] = "reassign"
    patient: PatientId
    bay: BayId


class BumpPriorityAction(FrozenModel):
    """Move a waiting patient to the head of the bay queue -> ``sequence`` item."""

    kind: Literal["bump_priority"] = "bump_priority"
    patient: PatientId


class RerouteAction(FrozenModel):
    """Direct a specific staff member to a pending task -> ``dispatch`` item."""

    kind: Literal["reroute"] = "reroute"
    staff: StaffId
    task: TaskId


class ExpediteCleanAction(FrozenModel):
    """Prioritize the pending cleaning task for a bay -> ``clean`` item."""

    kind: Literal["expedite_clean"] = "expedite_clean"
    bay: BayId


class ExpediteDischargeAction(FrozenModel):
    """Prioritize a patient's pending documentation -> ``discharge`` item."""

    kind: Literal["expedite_discharge"] = "expedite_discharge"
    patient: PatientId


class CloseBayAction(FrozenModel):
    """Hold/close a bay — a ``ValidationContext`` delta, not a plan item."""

    kind: Literal["close_bay"] = "close_bay"
    bay: BayId


class BlockEdgeAction(FrozenModel):
    """Block a corridor edge — a routing-mask delta, not a plan item."""

    kind: Literal["block_edge"] = "block_edge"
    edge: tuple[NodeId, NodeId]


_PlanAction = (
    ReassignAction
    | BumpPriorityAction
    | RerouteAction
    | ExpediteCleanAction
    | ExpediteDischargeAction
)
_AvailabilityAction = CloseBayAction | BlockEdgeAction

OperatorAction = Annotated[
    _PlanAction | _AvailabilityAction,
    Field(discriminator="kind"),
]


class OverrideRequest(FrozenModel):
    """``POST /runs/{id}/override`` body (doc 07 §3.4)."""

    action: OperatorAction
    pin: bool = True


class OverrideAccepted(FrozenModel):
    status: Literal["applied"] = "applied"
    plan: Plan
    applied_at: SimTime


class OverrideRejected(FrozenModel):
    status: Literal["rejected"] = "rejected"
    violations: tuple[Violation, ...]


def _unknown(detail: str, entity: str) -> tuple[Violation, ...]:
    return (Violation(kind="unknown_entity", detail=detail, entity=entity),)


def _has_pending_cleaning(world: World, bay: BayId) -> bool:
    return any(
        spec.kind == "cleaning" and world.task(spec.id).bay == bay for spec in world.pending_tasks()
    )


def _has_pending_documentation(world: World, patient: PatientId) -> bool:
    return any(
        spec.kind == "documentation" and spec.patient == patient for spec in world.pending_tasks()
    )


def compile_plan_action(action: _PlanAction, world: World) -> Plan | tuple[Violation, ...]:
    """Compile a plan-shaped action into a ``core.seam.Plan`` delta.

    Entity references the one validator can judge (bays, patients, staff,
    tasks, compatibility, occupancy) are deliberately NOT checked here — they
    flow through ``validate()`` so the rejection is the canonical
    ``Violation``. Only facts ``validate()`` cannot see (queue membership,
    pending-task existence — the same dynamic preconditions the seam adapter
    preflights) are resolved at compile time.
    """
    if isinstance(action, ReassignAction):
        return Plan(
            items=(
                PlanItem(
                    stable_id=f"op:assign_bay:{action.patient.root}",
                    kind="assign_bay",
                    patient=action.patient,
                    bay=action.bay,
                ),
            )
        )
    if isinstance(action, BumpPriorityAction):
        order = [w.patient.id.root for w in world.waiting_for_bay()]
        if action.patient.root not in order:
            return _unknown("patient is not waiting for a bay", action.patient.root)
        bumped = (action.patient.root, *(p for p in order if p != action.patient.root))
        return Plan(
            items=(
                PlanItem(
                    stable_id=f"op:sequence:{action.patient.root}",
                    kind="sequence",
                    patient=action.patient,
                    order=bumped,
                ),
            )
        )
    if isinstance(action, RerouteAction):
        return Plan(
            items=(
                PlanItem(
                    stable_id=f"op:dispatch:{action.task.root}",
                    kind="dispatch",
                    staff=action.staff,
                    task=action.task,
                ),
            )
        )
    if isinstance(action, ExpediteCleanAction):
        if not _has_pending_cleaning(world, action.bay):
            return _unknown("no pending cleaning task for bay", action.bay.root)
        return Plan(
            items=(
                PlanItem(
                    stable_id=f"op:clean:{action.bay.root}",
                    kind="clean",
                    bay=action.bay,
                ),
            )
        )
    # ExpediteDischargeAction
    if not _has_pending_documentation(world, action.patient):
        return _unknown("no pending documentation task for patient", action.patient.root)
    return Plan(
        items=(
            PlanItem(
                stable_id=f"op:discharge:{action.patient.root}",
                kind="discharge",
                patient=action.patient,
            ),
        )
    )


def standing_plan(world: World) -> Plan:
    """The assignments currently in force: one ``assign_bay`` item per occupant.

    Re-validating this plan against a context delta is how an availability edit
    is judged: a closure that would strand an occupant surfaces as a violation
    on the now-stranded item (doc 07 §4.1), and the operator must reassign
    first — the engine never evicts.
    """
    items = tuple(
        PlanItem(
            stable_id=f"standing:assign_bay:{bs.bay.root}",
            kind="assign_bay",
            patient=bs.occupant,
            bay=bs.bay,
        )
        for bs in world.snapshot_bays()
        if bs.status is BayStatus.OCCUPIED and bs.occupant is not None
    )
    return Plan(items=items)


def _edge_known(world: World, a: NodeId, b: NodeId) -> bool:
    try:
        world.edge_seconds(a, b)
    except LayoutError:
        return False
    return True


def _apply_availability(
    session: RunSession, action: _AvailabilityAction
) -> OverrideAccepted | OverrideRejected:
    """Compile a ``ValidationContext`` delta, re-validate the standing plan, enact."""
    world = session.world
    now = SimTime(int(session.env.now))
    standing = standing_plan(world)
    ctx = validation_context(world, session.rules)
    if isinstance(action, CloseBayAction):
        try:
            world.bay(action.bay)
        except UnknownEntity:
            return OverrideRejected(violations=_unknown("unknown bay", action.bay.root))
        bays = tuple(
            bs
            if bs.bay != action.bay
            else BayState(bay=bs.bay, status=BayStatus.CLOSED, occupant=bs.occupant)
            for bs in ctx.bays
        )
        ctx = ctx.model_copy(update={"bays": bays})
        violations = validate(standing, ctx)
        if violations:
            return OverrideRejected(violations=violations)
        world.close_bay(action.bay)
        return OverrideAccepted(plan=standing, applied_at=now)
    # BlockEdgeAction — the mask is consumed by the one dijkstra; validate()
    # carries no graph state, so the standing plan is judged against the
    # unchanged context (a closure blocks routing, it invalidates no assignment).
    a, b = action.edge
    if not _edge_known(world, a, b) and not _edge_known(world, b, a):
        return OverrideRejected(violations=_unknown("unknown corridor edge", f"{a.root}->{b.root}"))
    violations = validate(standing, ctx)
    if violations:
        return OverrideRejected(violations=violations)
    world.block_edge(a, b)
    return OverrideAccepted(plan=standing, applied_at=now)


def apply_override(
    session: RunSession, request: OverrideRequest
) -> OverrideAccepted | OverrideRejected:
    """Compile -> the one ``validate()`` -> apply-or-reject (caller holds the lock).

    Atomic by construction: a rejection (compile resolution, validation, or the
    seam adapter's apply-time preflight) mutates nothing; an acceptance applies
    the whole plan through the same seam the solver uses, stamped
    ``origin="operator"``.
    """
    action = request.action
    now = SimTime(int(session.env.now))
    if isinstance(action, CloseBayAction | BlockEdgeAction):
        return _apply_availability(session, action)
    compiled = compile_plan_action(action, session.world)
    if isinstance(compiled, tuple):
        return OverrideRejected(violations=compiled)
    ctx = validation_context(session.world, session.rules)
    violations = validate(compiled, ctx)
    if violations:
        return OverrideRejected(violations=violations)
    try:
        apply_plan(session.world, compiled, ctx, session.executor, session.log, origin="operator")
    except InfeasiblePlan as exc:
        return OverrideRejected(violations=exc.violations)
    if request.pin:
        session.pins.record(action, compiled)
    return OverrideAccepted(plan=compiled, applied_at=now)


class PinRegistry:
    """Entity -> pinned decision, consulted at plan-merge time (doc 07 §4.4).

    A pin is *future intent* — "keep this decision across the next re-solve" —
    and ``DecisionInput`` has no hidden fields, so a pin cannot ride a side
    channel to the policy. It is represented explicitly instead: each still-
    applicable pinned ``PlanItem`` is merged into the plan the policy returned
    (conflicting policy items yield), and the merged plan flows through the one
    ``validate()``. An empty registry is an exact pass-through, which preserves
    the no-override run's byte-identity with the headless engine.
    """

    def __init__(self) -> None:
        self._items: dict[str, PlanItem] = {}
        self._sequence: dict[str, PatientId] = {}

    def __len__(self) -> int:
        return len(self._items) + len(self._sequence)

    def record(self, action: _PlanAction | _AvailabilityAction, plan: Plan) -> None:
        """Hold an accepted decision against the solver's next re-solve.

        Availability actions are not recorded: a closure/mask is world state
        the solver cannot re-solve away, so there is nothing to hold.
        """
        if isinstance(action, CloseBayAction | BlockEdgeAction):
            return
        if isinstance(action, BumpPriorityAction):
            self._sequence[action.patient.root] = action.patient
            return
        for item in plan.items:
            self._items[item.stable_id] = item

    def _keep(self, item: PlanItem, world: World) -> bool:
        if item.kind == "assign_bay":
            if item.patient is None or item.bay is None:
                return False
            return world.is_waiting(item.patient) or world.occupant(item.bay) == item.patient
        if item.kind == "dispatch":
            return item.task is not None and world.is_pending(item.task)
        if item.kind == "clean":
            return item.bay is not None and _has_pending_cleaning(world, item.bay)
        if item.kind == "discharge":
            return item.patient is not None and _has_pending_documentation(world, item.patient)
        return False

    def _prune(self, world: World) -> None:
        """Auto-release pins whose subject left the decidable state."""
        self._items = {sid: it for sid, it in self._items.items() if self._keep(it, world)}
        self._sequence = {
            root: pid for root, pid in self._sequence.items() if world.is_waiting(pid)
        }

    def _conflicts(self, item: PlanItem) -> bool:
        for pin in self._items.values():
            if (
                pin.kind == "assign_bay"
                and item.kind == "assign_bay"
                and (item.patient == pin.patient or item.bay == pin.bay)
            ):
                return True
            if (
                pin.kind == "dispatch"
                and item.kind == "dispatch"
                and (item.task == pin.task or item.staff == pin.staff)
            ):
                return True
        return False

    def _apply_sequence_pins(self, items: list[PlanItem], world: World) -> list[PlanItem]:
        pin_order = [pid.root for pid in self._sequence.values()]
        if not pin_order:
            return items
        pinned_set = set(pin_order)
        out: list[PlanItem] = []
        transformed = False
        for item in items:
            if item.kind == "sequence" and item.order is not None:
                rest = tuple(p for p in item.order if p not in pinned_set)
                out.append(item.model_copy(update={"order": (*pin_order, *rest)}))
                transformed = True
            else:
                out.append(item)
        if not transformed:
            rest = tuple(
                w.patient.id.root
                for w in world.waiting_for_bay()
                if w.patient.id.root not in pinned_set
            )
            out.append(
                PlanItem(stable_id="op:sequence:pins", kind="sequence", order=(*pin_order, *rest))
            )
        return out

    def merge(self, plan: Plan | None, world: World) -> Plan | None:
        """Merge still-applicable pins into a policy plan (pins are pre-fixed).

        With an empty registry this is an exact pass-through — the same object
        in, the same object out — so an un-overridden run stays byte-identical
        to the headless engine's.
        """
        if not self._items and not self._sequence:
            return plan
        self._prune(world)
        if not self._items and not self._sequence:
            return plan
        base = list(plan.items) if plan is not None else []
        out = [
            item for item in base if item.stable_id not in self._items and not self._conflicts(item)
        ]
        out = self._apply_sequence_pins(out, world)
        out.extend(self._items.values())
        if not out:
            return None
        return Plan(items=tuple(out))

    def drop_conflicting(self, violations: tuple[Violation, ...]) -> None:
        """Release pins named by apply-time violations so a re-solve converges."""
        entities = {v.entity for v in violations}

        def refs(item: PlanItem) -> set[str]:
            ids = (item.patient, item.bay, item.staff, item.task)
            return {item.stable_id, *(i.root for i in ids if i is not None)}

        self._items = {
            sid: item for sid, item in self._items.items() if not (refs(item) & entities)
        }
        self._sequence = {root: pid for root, pid in self._sequence.items() if root not in entities}


@router.post(
    "/runs/{run_id}/override",
    response_model=OverrideAccepted,
    responses={422: {"model": OverrideRejected}},
)
async def post_override(
    run_id: str, body: OverrideRequest, request: Request
) -> OverrideAccepted | JSONResponse:
    """A validated operator action: applied atomically or rejected verbatim."""
    session = require_session(cast("FastAPI", request.app), run_id)
    async with session.lock:
        result = apply_override(session, body)
    if isinstance(result, OverrideRejected):
        return JSONResponse(status_code=422, content=result.model_dump(mode="json"))
    return result


__all__ = [
    "BlockEdgeAction",
    "BumpPriorityAction",
    "CloseBayAction",
    "ExpediteCleanAction",
    "ExpediteDischargeAction",
    "OperatorAction",
    "OverrideAccepted",
    "OverrideRejected",
    "OverrideRequest",
    "PinRegistry",
    "ReassignAction",
    "RerouteAction",
    "apply_override",
    "compile_plan_action",
    "router",
    "standing_plan",
]
