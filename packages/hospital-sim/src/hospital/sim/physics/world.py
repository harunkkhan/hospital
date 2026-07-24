"""``World`` — the single owner of mutable state (doc 04 §3.1).

Every fact that changes during a run lives here and nowhere else: bay statuses
and occupants, staff/patient positions, the acuity-priority bay wait queue, the
live task registry, absence windows, and the routing masks. Flow processes, the
seam adapter, and disruption injectors mutate *only* through the named methods
below; nobody touches the backing dicts. The seam and the policies see only
immutable projections (:meth:`snapshot_bays`, :meth:`snapshot_staff`,
:meth:`waiting_for_bay`, :meth:`pending_tasks`).

Routing **delegates** to the one ``core.graph.RouteGraph.dijkstra`` — there is
no Dijkstra in ``sim``. ``block_edge``/``close_node`` mutate the masks and bump
``mask_version``; the per-version memo in :meth:`route` is a result cache, not
a second pathfinder (a masked ``(a, b)`` entry closes a bidirectional corridor
in both directions — core's semantics, nuance 1.7).

Closure semantics (nuance 4.1 — a closure must never crash or strand):

* **Masks are reference-counted.** Overlapping disruption windows on the same
  node/edge each count one closure; the mask relaxes only when the LAST window
  ends — an early ``open_node`` from the first window cannot reopen a node the
  second window still holds closed.
* **Egress is always legal.** :meth:`route` never treats the *source* as
  closed: an actor already standing on a closed node may leave (closure
  forbids entering/traversing, not escaping) — otherwise a closure would
  strand actors, masquerading as gridlock.
* **Ingress waits, it does not crash.** A destination that is closed — or
  unreachable because closures cut every path — is a *recoverable* condition:
  :meth:`try_route` reports it as ``None`` (instead of core's ``LayoutError``)
  and :meth:`await_route` parks the caller until a reopening relaxes the
  masks, then retries. Only a failure that would occur with NO masks (unknown
  node, disconnected floor) still raises — that is a layout bug, and waiting
  would hide it forever.

The bay four-state machine ``FREE → OCCUPIED → CLEANING → FREE`` (plus
``* → CLOSED → prior`` for disruptions) has exactly one path per transition.
:meth:`free_bay` is a *decision trigger*: freed capacity requests a decision
tick so a waiting patient can be placed — forgetting that link is a silent
starvation bug (doc 04 nuance 4.1).

:meth:`request_bay` implements **infinite patience**: the returned event is only
ever succeeded by :meth:`grant_bay` (from a validated ``assign_bay`` item);
patients never abandon, so gridlock surfaces as unbounded WIP, caught by the
``arrivals == completions + wip`` conservation invariant — never masked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

import simpy

from hospital.core import (
    Activity,
    Bay,
    BayId,
    BayStatus,
    Duration,
    EsiAcuity,
    EventLog,
    FloorLayout,
    LayoutError,
    NodeId,
    Patient,
    PatientId,
    RoutePath,
    SimTime,
    StaffId,
    StaffMember,
    StaffRole,
    TaskId,
    TaskSpec,
    UnknownEntity,
    ZeroTimeCycle,
    ZoneId,
)
from hospital.core.seam import BayState, StaffState, TaskKind, WaitingPatient
from hospital.sim.physics.executor import PriorityTier, TierTimeout

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from hospital.core import CompiledRules
    from hospital.sim.physics.resources import ResourcePool

# Bound on decision ticks at one frozen instant: a deterministic policy that is
# re-solved against unchanged state loops forever at the same SimTime; the guard
# raises rather than spinning (the executor's zero-delay guard is the other half).
_MAX_TICKS_PER_INSTANT: Final[int] = 200

# Rank assigned to non-overridden queue entries by a `sequence` plan item; an
# explicitly ordered patient always sorts ahead of unordered peers in its tier.
_UNRANKED: Final[int] = 2**31


def _callbacks(ev: simpy.Event) -> list[Callable[[simpy.Event], None]]:
    """``Event.callbacks`` with a concrete type (simpy's alias is TypeVar-unknown)."""
    callbacks = ev.callbacks  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast("list[Callable[[simpy.Event], None]]", callbacks)


@dataclass(slots=True)
class SimTask:
    """A live unit of work: the policy-visible ``TaskSpec`` plus physics-only fields.

    ``duration``/``destination``/``bay`` are *hidden from policies by design*
    (the noninterference invariant — a policy never sees true durations); the
    seam projects only ``spec``. ``done`` is the event the requesting patient
    process waits on; it is succeeded exactly once, by :meth:`World.complete_task`.
    """

    spec: TaskSpec
    activity: Activity
    duration: Duration
    done: simpy.Event
    esi: EsiAcuity | None = None
    bay: BayId | None = None
    destination: NodeId | None = None
    assigned_to: StaffId | None = None
    completed: bool = False


@dataclass(slots=True)
class _QueueEntry:
    patient: Patient
    stage: str
    enqueued_at: SimTime
    seq: int
    event: simpy.Event
    override: int = field(default=_UNRANKED)

    def sort_key(self) -> tuple[int, int, int, int]:
        return (int(self.patient.esi), self.override, self.patient.arrival_time.root, self.seq)


class World:
    """The mutable-state owner. All writes go through named mutators."""

    def __init__(
        self,
        env: simpy.Environment,
        layout: FloorLayout,
        resources: ResourcePool,
        event_log: EventLog,
    ) -> None:
        self._env = env
        self.layout = layout
        self.resources = resources
        self._event_log = event_log

        self._bay_by_id: dict[BayId, Bay] = {b.id: b for b in layout.bays}
        self._bay_status: dict[BayId, BayStatus] = {b.id: BayStatus.FREE for b in layout.bays}
        self._bay_occupant: dict[BayId, PatientId] = {}
        self._prior_status: dict[BayId, BayStatus] = {}
        # Active zone-closure windows, refcounted by ZoneId root (overlap-safe).
        self._zone_closures: dict[str, int] = {}

        self._staff: dict[StaffId, StaffMember] = {}
        self._staff_pos: dict[StaffId, NodeId] = {}
        self._staff_task: dict[StaffId, TaskId] = {}
        self._absent_until: dict[StaffId, SimTime] = {}

        self._patients: dict[PatientId, Patient] = {}
        self._patient_pos: dict[PatientId, NodeId] = {}

        self._queue: list[_QueueEntry] = []
        self._queue_seq = 0

        self._tasks: dict[TaskId, SimTask] = {}
        self._pending: list[TaskId] = []
        self._task_seq = 0

        # Masks are refcounted: overlapping closure windows on one node/edge
        # each hold a count; the mask relaxes only at zero (finding: the first
        # window's unconditional reopen must not undo a still-active second).
        self._blocked: dict[tuple[NodeId, NodeId], int] = {}
        self._closed: dict[NodeId, int] = {}
        self._mask_version = 0
        self._route_memo: dict[tuple[str, str, int], RoutePath] = {}
        self._mask_relaxed: simpy.Event = env.event()

        self._edge_secs: dict[tuple[NodeId, NodeId], Duration] = {}
        for e in layout.graph.edges:
            self._edge_secs[(e.a, e.b)] = e.seconds
            if e.bidirectional:
                self._edge_secs[(e.b, e.a)] = e.seconds

        self._decision_hook: Callable[[], None] | None = None
        self._tick_scheduled = False
        self._tick_guard_at: int = -1
        self._ticks_this_instant = 0

    # ------------------------------------------------------------------ misc
    @property
    def env(self) -> simpy.Environment:
        return self._env

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    def now(self) -> SimTime:
        return SimTime(int(self._env.now))

    # --------------------------------------------------------------- routing
    def route(self, src: NodeId, dst: NodeId) -> RoutePath:
        """The ONLY pathfinding call site in ``sim`` — delegates to core's dijkstra.

        Results are memoized per ``(src, dst, mask_version)``; every mask
        mutator bumps the version, invalidating the memo wholesale.

        Egress rule: ``src`` is never treated as closed — an actor standing on
        a closed node may always leave (nuance 4.1). Raises ``LayoutError``
        when ``dst`` is closed or every path is masked; callers that can wait
        use :meth:`await_route` instead.
        """
        key = (src.root, dst.root, self._mask_version)
        cached = self._route_memo.get(key)
        if cached is not None:
            return cached
        path = self.layout.graph.dijkstra(
            src,
            dst,
            blocked_edges=frozenset(self._blocked),
            closed_nodes=frozenset(self._closed) - {src},
        )
        self._route_memo[key] = path
        return path

    def try_route(self, src: NodeId, dst: NodeId) -> RoutePath | None:
        """:meth:`route`, with closure-induced failures softened to ``None``.

        ``None`` means "blocked by the current masks — recoverable, wait for a
        reopening". A failure that would occur even with NO masks (unknown
        node, disconnected floor) still raises: that is a genuine layout bug,
        and waiting on it would hide it forever.
        """
        if not self._blocked and not self._closed:
            return self.route(src, dst)
        try:
            return self.route(src, dst)
        except LayoutError:
            self.layout.graph.dijkstra(src, dst)  # re-raises iff genuinely broken
            return None

    def masks_relaxed(self) -> simpy.Event:
        """An event succeeded at the next mask relaxation (open/unblock)."""
        return self._mask_relaxed

    def await_route(self, src: NodeId, dst: NodeId) -> Generator[simpy.Event, object, RoutePath]:
        """Route ``src -> dst``, waiting out closures instead of crashing.

        An actor needing a closed (or closure-severed) destination parks here
        until a reopening relaxes the masks, then retries — the recoverable
        disruption semantics of nuance 4.1: a closure blocks, it never kills
        the replication.
        """
        while True:
            path = self.try_route(src, dst)
            if path is not None:
                return path
            yield self._mask_relaxed

    def _relax_masks(self) -> None:
        """Wake every actor parked on a closure — a route may have reopened."""
        ev = self._mask_relaxed
        self._mask_relaxed = self._env.event()
        if not ev.triggered:
            ev.succeed(None)

    def edge_seconds(self, u: NodeId, v: NodeId) -> Duration:
        """Traversal seconds of the directed hop ``u -> v`` (LayoutError if absent)."""
        secs = self._edge_secs.get((u, v))
        if secs is None:
            raise LayoutError(f"no edge {u.root} -> {v.root} in the floor graph")
        return secs

    def block_edge(self, a: NodeId, b: NodeId) -> None:
        """Close a corridor (refcounted — one count per overlapping window)."""
        self._blocked[(a, b)] = self._blocked.get((a, b), 0) + 1
        self._mask_version += 1

    def unblock_edge(self, a: NodeId, b: NodeId) -> None:
        """Release one closure count; the edge reopens only at zero."""
        count = self._blocked.get((a, b), 0)
        if count <= 1:
            self._blocked.pop((a, b), None)
        else:
            self._blocked[(a, b)] = count - 1
        self._mask_version += 1
        self._relax_masks()

    def close_node(self, n: NodeId) -> None:
        """Close a node (refcounted — one count per overlapping window)."""
        self._closed[n] = self._closed.get(n, 0) + 1
        self._mask_version += 1

    def open_node(self, n: NodeId) -> None:
        """Release one closure count; the node reopens only at zero."""
        count = self._closed.get(n, 0)
        if count <= 1:
            self._closed.pop(n, None)
        else:
            self._closed[n] = count - 1
        self._mask_version += 1
        self._relax_masks()

    @property
    def blocked_edges(self) -> frozenset[tuple[NodeId, NodeId]]:
        return frozenset(self._blocked)

    @property
    def closed_nodes(self) -> frozenset[NodeId]:
        return frozenset(self._closed)

    # ------------------------------------------------------------------ bays
    def bay(self, b: BayId) -> Bay:
        bay = self._bay_by_id.get(b)
        if bay is None:
            raise UnknownEntity(f"unknown bay: {b.root}")
        return bay

    def bay_status(self, b: BayId) -> BayStatus:
        status = self._bay_status.get(b)
        if status is None:
            raise UnknownEntity(f"unknown bay: {b.root}")
        return status

    def occupant(self, b: BayId) -> PatientId | None:
        self.bay(b)
        return self._bay_occupant.get(b)

    def free_compatible_bays(self, p: Patient, rules: CompiledRules) -> tuple[BayId, ...]:
        """FREE bays compatible with ``p`` (zone whitelist, equipment, isolation).

        Fixed ``BayId`` order — the deterministic order the baseline placement
        scans. The compatibility judgment reuses the compiled-rule kernel; there
        is no second acuity→zone mapping in ``sim``.
        """
        allowed_zones = rules.zone_types_for(p.esi)
        needed_equipment = rules.equipment_for(p.esi)
        out: list[BayId] = []
        for bay_id in sorted(self._bay_status, key=lambda b: b.root):
            if self._bay_status[bay_id] is not BayStatus.FREE:
                continue
            bay = self._bay_by_id[bay_id]
            if bay.zone_type not in allowed_zones:
                continue
            if needed_equipment - bay.equipment:
                continue
            if rules.isolation_enforced and p.isolation_required and not bay.isolation_capable:
                continue
            out.append(bay_id)
        return tuple(out)

    def _expect_status(self, b: BayId, expected: BayStatus, action: str) -> None:
        status = self.bay_status(b)
        if status is not expected:
            raise ValueError(f"cannot {action} bay {b.root}: status is {status.value}")

    def assign_bay(self, b: BayId, p: PatientId) -> None:
        """``FREE -> OCCUPIED`` (the caller emits ``BayAssigned``)."""
        self._expect_status(b, BayStatus.FREE, "assign")
        self._bay_status[b] = BayStatus.OCCUPIED
        self._bay_occupant[b] = p

    def vacate_bay(self, b: BayId) -> None:
        """``OCCUPIED -> CLEANING`` — the patient has physically left the bay."""
        self._expect_status(b, BayStatus.OCCUPIED, "vacate")
        self._bay_status[b] = BayStatus.CLEANING
        self._bay_occupant.pop(b, None)

    def free_bay(self, b: BayId) -> None:
        """``CLEANING -> FREE`` — capacity returns; triggers a decision tick.

        Inside an active zone-closure window the closure GOVERNS the freed
        capacity: the bay parks ``CLOSED`` (prior ``FREE``) instead, and
        returns via :meth:`end_zone_closure`'s reopen at window end — a bay
        freed mid-window must never leak back into placement.
        """
        self._expect_status(b, BayStatus.CLEANING, "free")
        if self._bay_by_id[b].zone.root in self._zone_closures:
            self._bay_status[b] = BayStatus.CLOSED
            self._prior_status[b] = BayStatus.FREE
            return
        self._bay_status[b] = BayStatus.FREE
        self.request_decision()

    def close_bay(self, b: BayId) -> None:
        """``FREE|CLEANING -> CLOSED`` (disruption). Closing an OCCUPIED bay is refused."""
        status = self.bay_status(b)
        if status is BayStatus.OCCUPIED:
            raise ValueError(f"cannot close occupied bay {b.root}")
        if status is BayStatus.CLOSED:
            return
        self._prior_status[b] = status
        self._bay_status[b] = BayStatus.CLOSED

    def reopen_bay(self, b: BayId) -> None:
        """``CLOSED -> (prior)``; a reopening into FREE triggers a decision tick."""
        self._expect_status(b, BayStatus.CLOSED, "reopen")
        prior = self._prior_status.pop(b, BayStatus.FREE)
        self._bay_status[b] = prior
        if prior is BayStatus.FREE:
            self.request_decision()

    def begin_zone_closure(self, zone: ZoneId) -> None:
        """Open a zone-closure window (refcounted — overlapping windows stack).

        FREE bays in the zone close immediately; occupied/cleaning bays are
        never yanked from under a patient, but the window governs them — a bay
        freed mid-window (:meth:`free_bay`) parks ``CLOSED`` until the LAST
        overlapping window ends.
        """
        count = self._zone_closures.get(zone.root, 0)
        self._zone_closures[zone.root] = count + 1
        if count == 0:
            for bay in self.layout.bays:
                if bay.zone == zone and self._bay_status[bay.id] is BayStatus.FREE:
                    self.close_bay(bay.id)

    def end_zone_closure(self, zone: ZoneId) -> None:
        """Close a zone-closure window; the zone reopens only at refcount zero."""
        count = self._zone_closures.get(zone.root, 0)
        if count > 1:
            self._zone_closures[zone.root] = count - 1
            return
        self._zone_closures.pop(zone.root, None)
        for bay in self.layout.bays:
            if bay.zone == zone and self._bay_status[bay.id] is BayStatus.CLOSED:
                self.reopen_bay(bay.id)

    # ----------------------------------------------------------------- staff
    def register_staff(self, member: StaffMember) -> None:
        """Add a roster member, positioned at their home station (composition root)."""
        self._staff[member.id] = member
        self._staff_pos[member.id] = member.home_station

    def roster(self) -> tuple[StaffMember, ...]:
        return tuple(self._staff.values())

    def staff_member(self, s: StaffId) -> StaffMember:
        member = self._staff.get(s)
        if member is None:
            raise UnknownEntity(f"unknown staff: {s.root}")
        return member

    def staff_at(self, s: StaffId) -> NodeId:
        pos = self._staff_pos.get(s)
        if pos is None:
            raise UnknownEntity(f"unknown staff: {s.root}")
        return pos

    def set_staff_position(self, s: StaffId, n: NodeId) -> None:
        self.staff_member(s)
        self._staff_pos[s] = n

    def set_staff_idle(self, s: StaffId) -> None:
        self.staff_member(s)
        self._staff_task.pop(s, None)

    def set_absent(self, s: StaffId, until: SimTime) -> None:
        """Mark ``s`` absent through ``until`` — overlapping windows keep the MAX.

        A later-but-shorter window must never truncate an active longer one
        (through-hour-6 then through-hour-3 still returns at 6).
        """
        self.staff_member(s)
        current = self._absent_until.get(s)
        if current is None or until.root > current.root:
            self._absent_until[s] = until

    def absent_until(self, s: StaffId) -> SimTime | None:
        return self._absent_until.get(s)

    def clear_absent(self, s: StaffId) -> None:
        self._absent_until.pop(s, None)
        self.request_decision()

    # -------------------------------------------------------------- patients
    def register_patient(self, p: Patient) -> None:
        self._patients[p.id] = p

    def patient(self, pid: PatientId) -> Patient:
        p = self._patients.get(pid)
        if p is None:
            raise UnknownEntity(f"unknown patient: {pid.root}")
        return p

    def known_patients(self) -> tuple[Patient, ...]:
        return tuple(self._patients[k] for k in sorted(self._patients, key=lambda i: i.root))

    def patient_at(self, pid: PatientId) -> NodeId:
        pos = self._patient_pos.get(pid)
        if pos is None:
            raise UnknownEntity(f"patient has no position: {pid.root}")
        return pos

    def set_patient_position(self, pid: PatientId, n: NodeId) -> None:
        self._patient_pos[pid] = n

    # --------------------------------------------- the acuity-priority bay queue
    def request_bay(self, p: Patient, stage: str) -> simpy.Event:
        """Enqueue with infinite patience; yield the returned event for the grant.

        The queue key is ``(esi, sequence-override, arrival_time, seq)`` —
        strict acuity tiers, FIFO within a tier. The event is succeeded only by
        :meth:`grant_bay` with the granted ``BayId`` as its value; there is no
        timeout branch (patients never abandon, ``PLAN §1.3``).
        """
        entry = _QueueEntry(
            patient=p,
            stage=stage,
            enqueued_at=self.now(),
            seq=self._queue_seq,
            event=self._env.event(),
        )
        self._queue_seq += 1
        self._queue.append(entry)
        return entry.event

    def waiting_for_bay(self) -> tuple[WaitingPatient, ...]:
        """Immutable projection of the queue, in service order."""
        now = self.now()
        return tuple(
            WaitingPatient(patient=e.patient, waited=now - e.enqueued_at, stage=e.stage)
            for e in sorted(self._queue, key=_QueueEntry.sort_key)
        )

    def resequence_waiting(self, order: tuple[str, ...]) -> None:
        """Apply a ``sequence`` plan item: explicit rank within each acuity tier."""
        ranks = {pid: i for i, pid in enumerate(order)}
        for entry in self._queue:
            rank = ranks.get(entry.patient.id.root)
            if rank is not None:
                entry.override = rank

    def is_waiting(self, p: PatientId) -> bool:
        """Whether ``p`` is currently in the bay wait queue (a ``grant_bay`` precondition)."""
        return any(e.patient.id == p for e in self._queue)

    def grant_bay(self, p: PatientId, b: BayId) -> None:
        """Assign ``b`` to waiting patient ``p`` and succeed their wake event."""
        entry = next((e for e in self._queue if e.patient.id == p), None)
        if entry is None:
            raise UnknownEntity(f"patient not waiting for a bay: {p.root}")
        self.assign_bay(b, p)
        self._queue.remove(entry)
        entry.event.succeed(b)

    # ----------------------------------------------------------------- tasks
    def add_task(
        self,
        *,
        kind: TaskKind,
        patient: PatientId | None,
        at: NodeId,
        required_role: StaffRole,
        activity: Activity,
        duration: Duration,
        esi: EsiAcuity | None = None,
        bay: BayId | None = None,
        destination: NodeId | None = None,
        required_skills: frozenset[str] = frozenset(),
    ) -> SimTask:
        """Register a pending unit of work; dispatch decides who serves it.

        Task ids are minted from a deterministic counter (never a UUID — RNG
        keys and event bytes must be replayable). The physics-only fields
        (duration, destination, bay) never reach the ``TaskSpec`` projection.
        """
        tid = TaskId(f"task_{self._task_seq:06d}")
        self._task_seq += 1
        spec = TaskSpec(
            id=tid,
            kind=kind,
            patient=patient,
            at=at,
            required_role=required_role,
            required_skills=required_skills,
            ready_at=self.now(),
        )
        task = SimTask(
            spec=spec,
            activity=activity,
            duration=duration,
            done=self._env.event(),
            esi=esi,
            bay=bay,
            destination=destination,
        )
        self._tasks[tid] = task
        self._pending.append(tid)
        return task

    def task(self, tid: TaskId) -> SimTask:
        task = self._tasks.get(tid)
        if task is None:
            raise UnknownEntity(f"unknown task: {tid.root}")
        return task

    def pending_tasks(self) -> tuple[TaskSpec, ...]:
        """Undispatched tasks, in FIFO order (boosts move a task to the front)."""
        return tuple(self._tasks[tid].spec for tid in self._pending)

    def is_pending(self, tid: TaskId) -> bool:
        """Whether ``tid`` is still undispatched (a ``dispatch_task`` precondition)."""
        return tid in self._pending

    def staff_task(self, s: StaffId) -> TaskId | None:
        """The task currently assigned to ``s`` — ``None`` means dispatchable."""
        self.staff_member(s)
        return self._staff_task.get(s)

    def live_task_specs(self) -> tuple[TaskSpec, ...]:
        """Every not-yet-completed task (pending or dispatched) — validation context."""
        return tuple(t.spec for t in self._tasks.values() if not t.completed)

    def boost_task(self, tid: TaskId) -> None:
        """Move a pending task to the front of the FIFO (clean/discharge items)."""
        if tid in self._pending:
            self._pending.remove(tid)
            self._pending.insert(0, tid)

    def dispatch_task(self, tid: TaskId, staff: StaffId) -> SimTask:
        """Hand a pending task to a specific staff member's mailbox (directed dispatch)."""
        task = self.task(tid)
        if tid not in self._pending:
            raise ValueError(f"task not pending: {tid.root}")
        self.staff_member(staff)
        if staff in self._staff_task:
            raise ValueError(f"staff already busy: {staff.root}")
        self._pending.remove(tid)
        task.assigned_to = staff
        self._staff_task[staff] = tid
        self.resources.mailboxes[staff].put(task)
        return task

    def requeue_task(self, task: SimTask) -> None:
        """Return a dispatched-but-unfinished task to the FRONT of the pending queue.

        Used when an absence interrupts the assigned staff: the work is still
        owed (the patient is blocked on ``task.done``), so it must be re-offered
        to the dispatch policy, not silently dropped.
        """
        if task.completed:
            return
        if task.assigned_to is not None:
            self._staff_task.pop(task.assigned_to, None)
            task.assigned_to = None
        if task.spec.id not in self._pending:
            self._pending.insert(0, task.spec.id)

    def complete_task(self, task: SimTask) -> None:
        """Mark done and wake the requesting process (succeeded exactly once)."""
        if task.completed:
            return
        task.completed = True
        if task.assigned_to is not None:
            self._staff_task.pop(task.assigned_to, None)
        task.done.succeed(None)

    # --------------------------------------------------------- decision ticks
    def set_decision_hook(self, hook: Callable[[], None]) -> None:
        """Install the one tick runner (composition root). Without a hook,
        ``request_decision`` is a no-op — physics-only tests run decision-free."""
        self._decision_hook = hook

    def request_decision(self) -> None:
        """Coalesced, event-driven re-solve trigger (doc 04 §4.1).

        Schedules at most one pending ``DECISION``-tier tick at a time; N
        simultaneous triggers cost one solve against one settled snapshot. A
        re-request *after* this instant's tick ran schedules a fresh tick at the
        same instant — bounded by ``_MAX_TICKS_PER_INSTANT`` (ZeroTimeCycle).
        """
        if self._decision_hook is None or self._tick_scheduled:
            return
        self._tick_scheduled = True
        ev = TierTimeout(self._env, 0, PriorityTier.DECISION)
        _callbacks(ev).append(self._run_tick)

    def schedule_decision_at(self, at: SimTime) -> None:
        """Honor a ``WakeDirective(kind="schedule")`` — a future decision tick."""
        delay = max(0, at.root - int(self._env.now))
        ev = TierTimeout(self._env, delay, PriorityTier.DECISION)

        def _wake(_ev: simpy.Event) -> None:
            self.request_decision()

        _callbacks(ev).append(_wake)

    def _run_tick(self, _ev: simpy.Event) -> None:
        self._tick_scheduled = False
        now = int(self._env.now)
        if now != self._tick_guard_at:
            self._tick_guard_at = now
            self._ticks_this_instant = 0
        self._ticks_this_instant += 1
        if self._ticks_this_instant > _MAX_TICKS_PER_INSTANT:
            raise ZeroTimeCycle(f"more than {_MAX_TICKS_PER_INSTANT} decision ticks at t={now}µs")
        assert self._decision_hook is not None
        self._decision_hook()

    # ------------------------------------------------------------ projections
    def snapshot_bays(self) -> tuple[BayState, ...]:
        """Immutable apply-time bay states, in the layout's canonical bay order."""
        return tuple(
            BayState(
                bay=b.id,
                status=self._bay_status[b.id],
                occupant=self._bay_occupant.get(b.id),
            )
            for b in self.layout.bays
        )

    def snapshot_staff(self) -> tuple[StaffState, ...]:
        """Immutable apply-time staff states (absence surfaces as ``busy_until``)."""
        return tuple(
            StaffState(
                staff=s,
                at=self._staff_pos[s],
                busy_until=self._absent_until.get(s),
                current_task=self._staff_task.get(s),
            )
            for s in self._staff
        )


__all__ = ["SimTask", "World"]
