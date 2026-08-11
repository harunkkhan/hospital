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

from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from hospital.core.entities import FloorLayout, Patient
from hospital.core.enums import BayStatus, EsiAcuity, StaffRole
from hospital.core.errors import SeamViolation
from hospital.core.events import EventEnvelope, VitalsSampled
from hospital.core.ids import BayId, NodeId, PatientId, StaffId, TaskId
from hospital.core.models import FrozenModel
from hospital.core.time import Duration, SimTime
from hospital.core.vitals import VitalsReading

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

# The two reasons a patient is in the bay queue, and the whole of the stage
# vocabulary. They live in ``core`` because three packages have to agree on them:
# ``sim`` writes the stage when it enqueues, ``solver`` reads it to decide which
# bays are even candidates, and ``core.validation`` reads it to judge the grant.
# Through M3 the ED string was written literally in ``sim.flow.patient`` and
# restated as a set in ``solver.placement``, with a comment on each warning that a
# stage outside the other's vocabulary would be silently invisible. One stage
# survived that; two would not.
WAITING_FOR_BAY: Final[str] = "waiting_for_bay"
AWAITING_ADMISSION: Final[str] = "awaiting_admission"


class WaitingPatient(FrozenModel):
    """A patient waiting at some stage, with how long they have waited.

    ``stage`` is the patient's **care phase**, and it is what makes one queue serve
    two questions: :data:`WAITING_FOR_BAY` is a patient who needs an ED bay to be
    worked up in, :data:`AWAITING_ADMISSION` one whose workup ended in an admit and
    who now needs an inpatient bed. They are not interchangeable — an ESI-2 eligible
    for a resus bay is not thereby eligible for an ICU bed — so the phase selects
    which whitelist judges the placement, in the solver and in the validator alike.
    """

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
    """A unit of pending work the decision layer may route/sequence.

    ``esi`` is the served patient's triage acuity — the urgency signal the
    dispatch lever prices (doc 03 §4.5's ``u(t)``). Every task is created
    post-triage, so acuity is observable state, never a hidden field; it is
    ``None`` only for patient-less work (cleaning).
    """

    id: TaskId
    kind: TaskKind
    patient: PatientId | None
    at: NodeId
    required_role: StaffRole
    required_skills: frozenset[str] = frozenset()
    ready_at: SimTime
    esi: EsiAcuity | None = None


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

    @model_validator(mode="after")
    def _unique_stable_ids(self) -> Plan:
        """``stable_id`` must be unique across items, else :meth:`diff` is lossy.

        The diff comprehensions key on ``stable_id``; a duplicate would silently
        collapse (last wins), dropping items from added/removed/changed.
        """
        seen: set[str] = set()
        dupes: set[str] = set()
        for item in self.items:
            if item.stable_id in seen:
                dupes.add(item.stable_id)
            seen.add(item.stable_id)
        if dupes:
            raise ValueError(f"duplicate stable_id(s) in plan: {sorted(dupes)}")
        return self

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

    @model_validator(mode="after")
    def _mode_matches_plan(self) -> DecisionResponse:
        """``mode`` and ``plan`` must agree, or the seam contract is broken.

        ``replace`` requires a concrete plan to apply; ``keep`` carries none (the
        engine retains the current plan). A mismatch is a malformed response.
        """
        if self.mode == "replace" and self.plan is None:
            raise SeamViolation("DecisionResponse mode='replace' requires a plan")
        if self.mode == "keep" and self.plan is not None:
            raise SeamViolation("DecisionResponse mode='keep' must not carry a plan")
        return self


class RiskAssessment(FrozenModel):
    """A monitor's verdict on one patient at one instant (doc 06 §7).

    ``escalate`` is the monitor's *decision*, already threshold-applied, because
    the threshold is a modelling choice (chosen on validation to hit a target
    sensitivity) and the engine must not re-derive it from ``probability`` with a
    constant of its own. ``news2`` rides along because the engine stamps it onto
    the events it writes, and because it is the transparent fallback a reviewer
    can check the model against.
    """

    patient: PatientId
    at: SimTime
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    news2: int = Field(ge=0)
    escalate: bool


@runtime_checkable
class RiskMonitor(Protocol):
    """The deterioration seam: ``sim`` calls it, ``forecast`` implements it (doc 06 §3).

    The dependency graph runs downward — ``sim`` may not import ``forecast``, and
    ``forecast`` may not import ``sim`` — so a live risk model cannot be reached
    by either side directly. It is injected instead, through this ``core``-owned
    Protocol: the engine streams each ``VitalsSampled`` it writes into
    :meth:`observe`, and acts on what comes back.

    **The monitor never writes.** ``sim`` remains the sole ``EventLog`` writer
    (nuance 1.4): on a positive decision the *engine* appends
    ``DeteriorationDetected`` then ``EmergencyRaised``. A monitor that could
    append would be a second writer, and the log would stop being a single
    replayable history.

    The raw ``reading`` travels beside the event rather than inside it.
    ``VitalsSampled`` carries only the NEWS2 aggregate because the log is a
    published artifact, but a model needs the underlying numbers — so the engine,
    which already holds the trajectory it sampled, passes both.

    :meth:`observe` returns ``None`` while the model has nothing to say — most
    often because a rolling window is not yet full — which is a normal state, not
    an error. A run with no monitor injected behaves exactly as one whose monitor
    always returns ``None``, which is what keeps the M1/M2 engine byte-identical.
    """

    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        """Take one sampled reading; return an assessment, or ``None`` if undecided."""
        ...


@runtime_checkable
class ForgetfulMonitor(Protocol):
    """A :class:`RiskMonitor` that keeps per-patient state and can be told to drop it.

    Separate from ``RiskMonitor`` on purpose. Buffering is an implementation choice — a
    stateless monitor scoring one reading at a time needs no lifecycle at all — so
    requiring ``forget`` of every monitor would make the simpler kind harder to write for
    no gain. The engine checks structurally (``isinstance``, which this Protocol supports)
    and calls it when it is there.

    Without it, a rolling monitor has no signal that a patient is gone: its buffers grow
    for every discharge across a week-long run, and because patient ids are unique only
    *within* a run, reusing one monitor across runs would feed the previous week's
    readings into this week's window.
    """

    def forget(self, patient: PatientId) -> None:
        """Drop everything retained for ``patient``. Must tolerate an unknown id."""
        ...


__all__ = [
    "AWAITING_ADMISSION",
    "WAITING_FOR_BAY",
    "BayState",
    "DecisionInput",
    "DecisionResponse",
    "ForgetfulMonitor",
    "Plan",
    "PlanDiff",
    "PlanItem",
    "RiskAssessment",
    "RiskMonitor",
    "StaffState",
    "TaskSpec",
    "WaitingPatient",
    "WakeDirective",
]
