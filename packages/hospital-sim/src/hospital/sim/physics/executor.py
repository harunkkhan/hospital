"""``TaskExecutor`` — the SimPy timing authority (doc 04 §3.2).

Three invariants live here and nowhere else:

* **No teleport.** :meth:`TaskExecutor.walk` advances ``env.now`` by exactly
  ``edge.seconds`` per hop and updates the actor's position hop-by-hop, emitting
  one ``StaffMoved``/``PatientMoved`` per edge. This is what makes
  ``staff_minutes_walked`` and every placement/routing gain *measured*, not
  assumed — a single timeout for the whole path would total the same seconds
  but emit no per-edge trace, silently breaking the movement export and the
  utilization walk fraction.
* **Integer-µs clock with tiered same-instant ordering.** Every wake is created
  through :meth:`TaskExecutor.delay`, which feeds ``int(Duration.root)`` into
  ``Environment.schedule(event, priority, delay)``. SimPy pops its heap by
  ``(time, priority, eid)``, so at one instant all ``COMPLETION`` wakes settle
  before any ``DECISION`` tick, which settles before any ``DISRUPTION`` — the
  policy reads a settled world and a plan can never pre-dodge a same-instant
  disruption (doc 04 nuance 4.2). If a future SimPy changes its scheduling
  semantics, :class:`TierTimeout` is the single adaptation point (🟡 A3).
* **Zero-time-cycle guard.** A chain of zero-duration wakes at a frozen
  ``env.now`` raises :class:`~hospital.core.ZeroTimeCycle` instead of spinning
  (layout generation avoiding 0-length edges is the other half of the defense).

``run_service`` stamps each ``*_completed`` with the sequence of its
``*_started`` (``caused_by``), so ``analysis`` reconstructs causal chains with
no side-channel state. On a :class:`simpy.Interrupt` mid-service (a staff
absence) the ``*_completed`` is still emitted at the interruption instant — a
cut-short service is truthfully closed, never left dangling — and the interrupt
propagates to the caller.
"""

from __future__ import annotations

import itertools
from enum import IntEnum
from typing import TYPE_CHECKING, Final

import simpy
from simpy.events import EventPriority

from hospital.core import (
    Activity,
    BayCleaningCompleted,
    BayCleaningStarted,
    BayId,
    DocumentationCompleted,
    DocumentationStarted,
    Duration,
    EsiAcuity,
    Event,
    EventLog,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientId,
    PatientMoved,
    ProviderVisitCompleted,
    ProviderVisitStarted,
    RoutePath,
    SimTime,
    StaffId,
    StaffMoved,
    TriageCompleted,
    TriageStarted,
    ZeroTimeCycle,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from simpy.resources.resource import PriorityRequest

    from hospital.sim.physics.world import World

# Bound on zero-delay wakes at a single instant before the executor declares a
# same-instant causal loop. Generous: legitimate same-µs bursts (simultaneous
# arrivals, coalesced grants) are orders of magnitude smaller.
_MAX_ZERO_DELAYS_PER_INSTANT: Final[int] = 10_000


class PriorityTier(IntEnum):
    """Same-instant settlement order — lower value settles first at one SimTime."""

    COMPLETION = 0  # a travel/service/cleaning finishing — frees state
    DECISION = 1  # a policy tick reads the settled world, applies one Plan
    DISRUPTION = 2  # exogenous injection perturbs the post-decision world


class TierTimeout(simpy.Event):
    """A pre-triggered event scheduled at an explicit ``(delay, priority)``.

    SimPy's own :class:`~simpy.events.Timeout` hardcodes ``NORMAL`` priority;
    this is the same construction (pre-succeeded, then scheduled) with the tier
    exposed. It is the single point that touches SimPy's scheduling internals.
    """

    def __init__(self, env: simpy.Environment, delay_us: int, tier: PriorityTier) -> None:
        if delay_us < 0:
            raise ValueError(f"negative delay: {delay_us}")
        super().__init__(env)
        # Mirrors simpy.events.Timeout.__init__: mark triggered-ok, then queue.
        self._ok = True
        self._value = None
        env.schedule(self, EventPriority(int(tier)), delay_us)


class TaskExecutor:
    """The sole advancer of ``env.now``: walking, services, resource waits."""

    def __init__(self, env: simpy.Environment, world: World, event_log: EventLog) -> None:
        self._env = env
        self._world = world
        self._log = event_log
        self._guard_at: int = -1
        self._zero_delays: int = 0

    @property
    def env(self) -> simpy.Environment:
        return self._env

    def now(self) -> SimTime:
        """The current instant as a core ``SimTime`` (the clock is integer µs)."""
        return SimTime(int(self._env.now))

    def delay(self, dt: Duration, tier: PriorityTier) -> simpy.Event:
        """A wake after ``dt`` at ``tier`` — the only way sim schedules time.

        Zero-duration wakes are legal (same-instant hand-offs) but bounded: more
        than ``_MAX_ZERO_DELAYS_PER_INSTANT`` at one frozen ``env.now`` raises
        :class:`ZeroTimeCycle` rather than spinning forever.
        """
        micros = dt.root
        if micros == 0:
            now = int(self._env.now)
            if now != self._guard_at:
                self._guard_at = now
                self._zero_delays = 0
            self._zero_delays += 1
            if self._zero_delays > _MAX_ZERO_DELAYS_PER_INSTANT:
                raise ZeroTimeCycle(
                    f"more than {_MAX_ZERO_DELAYS_PER_INSTANT} zero-delay wakes at t={now}µs"
                )
        return TierTimeout(self._env, micros, tier)

    def walk(
        self,
        actor: PatientId | StaffId,
        path: RoutePath,
        *,
        escort: StaffId | None = None,
    ) -> Generator[simpy.Event, object]:
        """Traverse ``path`` one edge at a time — NO teleporting.

        Each hop is one ``COMPLETION``-tier delay of exactly ``edge.seconds``;
        the position is updated and a ``PatientMoved``/``StaffMoved`` emitted
        only *after* the hop completes, so an interrupt mid-hop leaves the actor
        truthfully at the last completed node. An ``escort`` (staff moving a
        patient) traverses the same edges and emits its own ``StaffMoved``.
        """
        nodes = path.nodes
        if not nodes:
            return
        at = (
            self._world.patient_at(actor)
            if isinstance(actor, PatientId)
            else self._world.staff_at(actor)
        )
        if at != nodes[0]:
            raise ValueError(f"walk path starts at {nodes[0].root} but actor is at {at.root}")
        for u, v in itertools.pairwise(nodes):
            secs = self._world.edge_seconds(u, v)
            yield self.delay(secs, PriorityTier.COMPLETION)
            t = self.now()
            if isinstance(actor, PatientId):
                self._world.set_patient_position(actor, v)
                self._log.append(
                    PatientMoved(occurred_at=t, patient=actor, edge=(u, v), seconds=secs)
                )
                if escort is not None:
                    self._world.set_staff_position(escort, v)
                    self._log.append(
                        StaffMoved(occurred_at=t, staff=escort, edge=(u, v), seconds=secs)
                    )
            else:
                self._world.set_staff_position(actor, v)
                self._log.append(StaffMoved(occurred_at=t, staff=actor, edge=(u, v), seconds=secs))

    def run_service(
        self,
        activity: Activity,
        *,
        duration: Duration,
        patient: PatientId | None,
        staff: StaffId | None,
        bay: BayId | None = None,
        esi: EsiAcuity | None = None,
        caused_by: int | None = None,
    ) -> Generator[simpy.Event, object, int]:
        """Emit ``*_started`` → elapse ``duration`` → emit ``*_completed``.

        The completed event carries ``caused_by = <started sequence>``. On a
        ``simpy.Interrupt`` mid-service, the completed event is emitted at the
        interruption instant (the pair discipline holds — a cut-short service is
        closed, not dangling) and the interrupt re-raises to the caller.
        Returns the started event's sequence.
        """
        started = _started_event(activity, self.now(), patient=patient, staff=staff, bay=bay)
        seq = self._log.append(started, caused_by=caused_by)
        try:
            yield self.delay(duration, PriorityTier.COMPLETION)
        except simpy.Interrupt:
            completed = _completed_event(
                activity, self.now(), patient=patient, staff=staff, bay=bay, esi=esi
            )
            self._log.append(completed, caused_by=seq)
            raise
        completed = _completed_event(
            activity, self.now(), patient=patient, staff=staff, bay=bay, esi=esi
        )
        self._log.append(completed, caused_by=seq)
        return seq

    def acquire(
        self, resource: simpy.PriorityResource, *, priority: int
    ) -> Generator[simpy.Event, object, PriorityRequest]:
        """Queue at a contended pool; returns the granted request (caller releases).

        Lower ``priority`` values are served first — callers derive it as
        ``-esi.priority_weight()`` so the acuity sign-inversion stays in the one
        core helper (DECISIONS D8). Within a priority, SimPy's request timestamp
        gives FIFO.
        """
        req = resource.request(priority=priority)
        yield req
        return req


def _require(value: object, what: str, activity: Activity) -> None:
    if value is None:
        raise ValueError(f"run_service({activity.value}) requires {what}")


def _started_event(
    activity: Activity,
    now: SimTime,
    *,
    patient: PatientId | None,
    staff: StaffId | None,
    bay: BayId | None,
) -> Event:
    """The ``*_started`` event for ``activity`` (the paired-event vocabulary).

    ``Activity.LAB`` maps to the nurse-visit pair: the bedside draw *is* nurse
    direct care, and the event schema has no ``Lab*`` pair — the lab sojourn
    itself is measured by ``TestOrdered``/``TestResulted``.
    """
    _require(staff, "staff", activity)
    assert staff is not None
    if activity is Activity.CLEANING:
        _require(bay, "bay", activity)
        assert bay is not None
        return BayCleaningStarted(occurred_at=now, bay=bay, staff=staff)
    _require(patient, "patient", activity)
    assert patient is not None
    if activity is Activity.TRIAGE:
        return TriageStarted(occurred_at=now, patient=patient, staff=staff)
    if activity is Activity.PROVIDER_VISIT:
        return ProviderVisitStarted(occurred_at=now, patient=patient, staff=staff)
    if activity in (Activity.NURSE_VISIT, Activity.LAB):
        return NurseVisitStarted(occurred_at=now, patient=patient, staff=staff)
    if activity is Activity.DOCUMENTATION:
        return DocumentationStarted(occurred_at=now, patient=patient, staff=staff)
    raise ValueError(f"activity {activity.value} has no started/completed event pair")


def _completed_event(
    activity: Activity,
    now: SimTime,
    *,
    patient: PatientId | None,
    staff: StaffId | None,
    bay: BayId | None,
    esi: EsiAcuity | None,
) -> Event:
    """The matching ``*_completed`` event (see :func:`_started_event`)."""
    _require(staff, "staff", activity)
    assert staff is not None
    if activity is Activity.CLEANING:
        _require(bay, "bay", activity)
        assert bay is not None
        return BayCleaningCompleted(occurred_at=now, bay=bay, staff=staff)
    _require(patient, "patient", activity)
    assert patient is not None
    if activity is Activity.TRIAGE:
        _require(esi, "esi", activity)
        assert esi is not None
        return TriageCompleted(occurred_at=now, patient=patient, esi=esi)
    if activity is Activity.PROVIDER_VISIT:
        return ProviderVisitCompleted(occurred_at=now, patient=patient, staff=staff)
    if activity in (Activity.NURSE_VISIT, Activity.LAB):
        return NurseVisitCompleted(occurred_at=now, patient=patient, staff=staff)
    if activity is Activity.DOCUMENTATION:
        return DocumentationCompleted(occurred_at=now, patient=patient, staff=staff)
    raise ValueError(f"activity {activity.value} has no started/completed event pair")


__all__ = ["PriorityTier", "TaskExecutor", "TierTimeout"]
