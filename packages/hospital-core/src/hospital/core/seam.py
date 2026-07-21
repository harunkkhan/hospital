"""The decision <-> physics seam: immutable projection in, complete plan out.

The noninterference invariant (nuance 1.10): ``DecisionInput`` has **no hidden
fields** — a policy sees only what is observable *now* (waiting patients, bay
states, staff positions, pending tasks, recent events), never a patient's true
future LOS or un-ordered workup. A human operator override produces the *same*
``Plan``/``PlanItem`` types and goes through the *same* validation — a human is
just another ``DecisionResponse`` producer.

``Plan`` is **complete, not a delta**: the policy returns its whole revisable
plan and the engine computes the delta via :meth:`Plan.diff` keyed on
``PlanItem.stable_id``, so an unchanged bay assignment is left alone rather than
cancelled and recreated (which would thrash the sim and UI).
"""

from __future__ import annotations

from typing import Literal

from hospital.core.entities import FloorLayout, Patient
from hospital.core.enums import BayStatus, StaffRole
from hospital.core.events import EventEnvelope
from hospital.core.ids import BayId, NodeId, PatientId, StaffId, TaskId
from hospital.core.models import FrozenModel
from hospital.core.time import Duration, SimTime

TaskKind = Literal[
    "provider_visit",
    "nurse_visit",
    "transport",
    "imaging",
    "lab",
    "cleaning",
    "discharge",
    "documentation",
]
PlanItemKind = Literal["assign_bay", "sequence", "dispatch", "clean", "discharge", "staffing"]


class WaitingPatient(FrozenModel):
    """A patient waiting at some stage, with how long they have waited."""

    patient: Patient
    waited: Duration
    stage: str


class BayState(FrozenModel):
    """Apply-time dynamic status of a bay (projection of ``world`` state)."""

    bay: BayId
    status: BayStatus
    occupant: PatientId | None = None


class StaffState(FrozenModel):
    """Apply-time dynamic state of a staff member."""

    staff: StaffId
    at: NodeId
    busy_until: SimTime | None = None
    current_task: TaskId | None = None


class TaskSpec(FrozenModel):
    """A unit of pending work the decision layer may route/sequence."""

    id: TaskId
    kind: TaskKind
    patient: PatientId | None
    at: NodeId
    required_role: StaffRole
    required_skills: frozenset[str] = frozenset()
    ready_at: SimTime


class DecisionInput(FrozenModel):
    """Immutable projection of current floor state — NO hidden fields."""

    now: SimTime
    layout: FloorLayout
    waiting: tuple[WaitingPatient, ...]
    bays: tuple[BayState, ...]
    staff: tuple[StaffState, ...]
    pending_tasks: tuple[TaskSpec, ...]
    events_since: tuple[EventEnvelope, ...]


class PlanItem(FrozenModel):
    """One decision. ``stable_id`` lets plans diff cleanly across re-solves.

    Payload fields are per-kind and optional; a given ``kind`` populates the
    subset it needs (e.g. ``assign_bay`` uses ``patient``+``bay``; ``dispatch``
    uses ``staff``+``task``; ``sequence`` uses ``patient``+``order``).
    """

    stable_id: str
    kind: PlanItemKind
    patient: PatientId | None = None
    bay: BayId | None = None
    staff: StaffId | None = None
    task: TaskId | None = None
    priority: int | None = None
    route: tuple[NodeId, ...] | None = None
    order: tuple[str, ...] | None = None


class PlanDiff(FrozenModel):
    """The delta between two plans, keyed on ``stable_id``.

    Computed as ``new.diff(old)``: ``added`` are items in ``new`` absent from
    ``old``; ``removed`` are items in ``old`` absent from ``new``; ``changed``
    are ``(old_item, new_item)`` pairs sharing a ``stable_id`` but differing.
    """

    added: tuple[PlanItem, ...]
    removed: tuple[PlanItem, ...]
    changed: tuple[tuple[PlanItem, PlanItem], ...]


class Plan(FrozenModel):
    """A complete, revisable set of decisions."""

    items: tuple[PlanItem, ...]

    def diff(self, other: Plan) -> PlanDiff:
        """Delta from ``other`` (previous) to ``self`` (new), keyed by ``stable_id``."""
        new_by_id = {item.stable_id: item for item in self.items}
        old_by_id = {item.stable_id: item for item in other.items}
        added = tuple(item for sid, item in new_by_id.items() if sid not in old_by_id)
        removed = tuple(item for sid, item in old_by_id.items() if sid not in new_by_id)
        changed = tuple(
            (old_by_id[sid], new_item)
            for sid, new_item in new_by_id.items()
            if sid in old_by_id and old_by_id[sid] != new_item
        )
        return PlanDiff(added=added, removed=removed, changed=changed)


class WakeDirective(FrozenModel):
    """When the policy is next consulted: keep the current wake, cancel, or reschedule."""

    kind: Literal["keep", "cancel", "schedule"]
    at: SimTime | None = None


class DecisionResponse(FrozenModel):
    """A policy's response: keep the current plan, or replace it, plus a wake."""

    mode: Literal["keep", "replace"]
    plan: Plan | None = None
    wake: WakeDirective


__all__ = [
    "BayState",
    "DecisionInput",
    "DecisionResponse",
    "Plan",
    "PlanDiff",
    "PlanItem",
    "StaffState",
    "TaskSpec",
    "WaitingPatient",
    "WakeDirective",
]
