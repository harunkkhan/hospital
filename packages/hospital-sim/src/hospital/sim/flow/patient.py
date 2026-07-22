"""The 9-step ER patient process (doc 04 §3.10 / §4.3, PLAN §6.2).

Arrive -> Triage -> Wait-for-bay -> Move-to-bay -> Provider eval -> Workup loop
-> Disposition -> Terminal (discharge/transfer paperwork, or admit boarding)
-> Bay turnaround. Every state change emits a ``core.events`` event at the
correct instant, and the process advances time ONLY through the executor
(travel/service, no-teleport + tier discipline) or a ``World`` wait event (the
infinite-patience bay queue) — never a bare ``env.timeout``.

Flow nuances honored here:

* **Workup order is pre-sampled and deterministic** — the loop iterates the
  frozen ``WorkupNeeds`` (imaging, then labs, then nurse visits, then remaining
  provider visits, sequentially), so both arms key the identical CRN draws.
* **Escorts** (🟡 A9): non-ambulatory patients (ESI 1-2) and *all* imaging
  transports move only with a dispatched porter (``transport`` task); the
  escort's own walking lands in ``staff_minutes_walked``.
* **Boarding** (🟡 A4): an admit holds the bay through a boarding delay, then
  leaves the modeled floor; the boarding exit emits the same
  ``DischargeStarted``/``DischargeCompleted`` pair — ``DischargeCompleted`` is
  the universal disposition-out marker ``analysis`` folds on, so admits count
  as completions, not hidden WIP.
* **Triage attribution**: the event schema requires ``TriageStarted.staff``,
  but triage rooms are a pooled resource, not dispatched staff — a synthetic
  ``triage_team`` id (never on the roster) is stamped so the pair discipline
  holds without corrupting roster utilization (judgment call, in the report).
* If the process has not terminated by the horizon it is **WIP** — the run
  simply stops; conservation ``arrivals == completions + wip`` is the tripwire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from hospital.core import (
    Activity,
    ArrivalMode,
    Bay,
    BayId,
    BayRequested,
    DischargeCompleted,
    DischargeStarted,
    DispositionDecided,
    DispositionKind,
    Duration,
    EventLog,
    LayoutError,
    NodeId,
    Patient,
    PatientArrived,
    StaffId,
    StaffRole,
    TestOrdered,
    TestResulted,
)
from hospital.sim.physics.executor import PriorityTier, TaskExecutor
from hospital.sim.physics.service_times import (
    ServiceTimes,
    sample_boarding_delay,
    sample_disposition,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import simpy

    from hospital.core import RandomStreams
    from hospital.core.seam import TaskKind
    from hospital.sim.physics.world import World

# The synthetic staff id stamped on Triage* events (never in the roster).
TRIAGE_TEAM: Final[StaffId] = StaffId("triage_team")

# Patients at or below this ESI value are non-ambulatory and move by escort.
_ESCORT_ESI_THRESHOLD: Final[int] = 2


def patient_process(
    env: simpy.Environment,
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    patient: Patient,
    *,
    service_times: ServiceTimes,
    streams: RandomStreams,
) -> Generator[simpy.Event, object]:
    """One SimPy process per patient — the full 9-step flow."""
    pid = patient.id
    layout = world.layout
    priority = -patient.esi.priority_weight()

    # 1 — arrive
    world.register_patient(patient)
    entrance = (
        layout.entrances[0] if patient.arrival_mode is ArrivalMode.WALK_IN else layout.entrances[-1]
    )
    world.set_patient_position(pid, entrance)
    event_log.append(
        PatientArrived(occurred_at=executor.now(), patient=pid, mode=patient.arrival_mode)
    )

    # 2 — triage (pooled rooms; acuity priority; ESI is ground truth, revealed here)
    triage_node = _nearest(world, entrance, world.resources.triage_nodes)
    yield from executor.walk(pid, world.route(entrance, triage_node))
    req = yield from executor.acquire(world.resources.triage, priority=priority)
    try:
        duration = service_times.sample(
            Activity.TRIAGE, patient.esi, patient.complaint, patient=pid
        )
        yield from executor.run_service(
            Activity.TRIAGE, duration=duration, patient=pid, staff=TRIAGE_TEAM, esi=patient.esi
        )
    finally:
        world.resources.triage.release(req)

    # 3 — wait for a bay (acuity-priority queue, infinite patience)
    event_log.append(BayRequested(occurred_at=executor.now(), patient=pid))
    wake = world.request_bay(patient, stage="triage->bay")
    world.request_decision()
    granted = yield wake
    bay = world.bay(cast("BayId", granted))

    # 4 — move to the bay (BayAssigned was emitted by the seam on grant)
    if int(patient.esi) <= _ESCORT_ESI_THRESHOLD:
        yield from _transport(world, patient, bay.node)
    else:
        yield from _self_walk(world, executor, patient, bay.node)

    # 5 — provider evaluation (provider visit index 0)
    yield from _bedside_service(
        world,
        patient,
        bay,
        "provider_visit",
        Activity.PROVIDER_VISIT,
        StaffRole.PHYSICIAN,
        0,
        service_times,
    )

    # 6 — workup loop (pre-sampled WorkupNeeds; sequential, deterministic order)
    for idx in range(len(patient.workup.imaging)):
        yield from _imaging(world, executor, event_log, patient, bay, idx, priority, service_times)
    for idx in range(patient.workup.labs):
        yield from _lab(world, executor, event_log, patient, bay, idx, priority, service_times)
    for idx in range(patient.workup.nurse_visits):
        yield from _bedside_service(
            world,
            patient,
            bay,
            "nurse_visit",
            Activity.NURSE_VISIT,
            StaffRole.NURSE,
            idx,
            service_times,
        )
    for idx in range(1, patient.workup.provider_visits):
        yield from _bedside_service(
            world,
            patient,
            bay,
            "provider_visit",
            Activity.PROVIDER_VISIT,
            StaffRole.PHYSICIAN,
            idx,
            service_times,
        )

    # 7 — disposition (world randomness, content-addressed on the patient)
    disposition = sample_disposition(streams, patient)
    event_log.append(
        DispositionDecided(occurred_at=executor.now(), patient=pid, disposition=disposition)
    )
    world.request_decision()

    # 8 — terminal path
    if disposition is DispositionKind.ADMIT:
        # boarding: hold the bay until the ward accepts, then leave the floor
        yield executor.delay(sample_boarding_delay(streams, patient), PriorityTier.COMPLETION)
    else:
        yield from _bedside_service(
            world,
            patient,
            bay,
            "documentation",
            Activity.DOCUMENTATION,
            StaffRole.NURSE,
            0,
            service_times,
        )
    event_log.append(DischargeStarted(occurred_at=executor.now(), patient=pid))

    # 9 — vacate; turnaround returns the capacity (housekeeping via dispatch)
    world.vacate_bay(bay.id)
    _enqueue_cleaning(world, patient, bay, service_times)
    world.request_decision()

    yield from _self_walk(world, executor, patient, layout.entrances[0])
    event_log.append(DischargeCompleted(occurred_at=executor.now(), patient=pid))


def _nearest(world: World, src: NodeId, candidates: tuple[NodeId, ...]) -> NodeId:
    """Deterministic nearest node by route time, id as the total-order tail."""
    if not candidates:
        raise LayoutError("no candidate nodes on this floor for a required visit")
    return min(candidates, key=lambda n: (world.route(src, n).total.root, n.root))


def _self_walk(
    world: World, executor: TaskExecutor, patient: Patient, dest: NodeId
) -> Generator[simpy.Event, object]:
    pos = world.patient_at(patient.id)
    if pos != dest:
        yield from executor.walk(patient.id, world.route(pos, dest))


def _transport(world: World, patient: Patient, dest: NodeId) -> Generator[simpy.Event, object]:
    """An escorted move: a porter is dispatched, walks over, and escorts (🟡 A9)."""
    if world.patient_at(patient.id) == dest:
        return
    task = world.add_task(
        kind="transport",
        patient=patient.id,
        at=world.patient_at(patient.id),
        required_role=StaffRole.PORTER,
        activity=Activity.TRANSPORT,
        duration=Duration(0),  # transport time is the walk itself, not a service
        esi=patient.esi,
        destination=dest,
    )
    world.request_decision()
    yield task.done


def _bedside_service(
    world: World,
    patient: Patient,
    bay: Bay,
    kind: TaskKind,
    activity: Activity,
    role: StaffRole,
    index: int,
    service_times: ServiceTimes,
) -> Generator[simpy.Event, object]:
    """A staff service at the bay: enqueue the task, wake the policy, await done.

    The duration is drawn HERE, keyed ``(patient, activity, index)`` with the
    index derived from the pre-sampled workup — never from dispatch order — so
    CRN holds across arms. The duration rides the sim-side task record, hidden
    from policies.
    """
    duration = service_times.sample(
        activity, patient.esi, patient.complaint, patient=patient.id, index=index
    )
    task = world.add_task(
        kind=kind,
        patient=patient.id,
        at=bay.node,
        required_role=role,
        activity=activity,
        duration=duration,
        esi=patient.esi,
        bay=bay.id,
    )
    world.request_decision()
    yield task.done


def _imaging(
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    patient: Patient,
    bay: Bay,
    index: int,
    priority: int,
    service_times: ServiceTimes,
) -> Generator[simpy.Event, object]:
    """One imaging round trip: order -> escorted transport -> scan -> return -> result."""
    pid = patient.id
    event_log.append(
        TestOrdered(occurred_at=executor.now(), patient=pid, activity=Activity.IMAGING)
    )
    suite = _nearest(world, bay.node, tuple(world.resources.imaging))
    yield from _transport(world, patient, suite)  # ALL imaging transports are escorted
    resource = world.resources.imaging[suite]
    req = yield from executor.acquire(resource, priority=priority)
    try:
        scan = service_times.sample(
            Activity.IMAGING, patient.esi, patient.complaint, patient=pid, index=index
        )
        yield executor.delay(scan, PriorityTier.COMPLETION)
    finally:
        resource.release(req)
    yield from _transport(world, patient, bay.node)
    yield executor.delay(
        service_times.result_delay(Activity.IMAGING, patient=pid, index=index),
        PriorityTier.COMPLETION,
    )
    event_log.append(
        TestResulted(occurred_at=executor.now(), patient=pid, activity=Activity.IMAGING)
    )
    world.request_decision()


def _lab(
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    patient: Patient,
    bay: Bay,
    index: int,
    priority: int,
    service_times: ServiceTimes,
) -> Generator[simpy.Event, object]:
    """One lab round: order -> bedside draw -> analyzer run -> off-machine result.

    The patient never moves: the *sample* goes to the lab. The analyzer station
    is held only for the run itself; the reporting/validation tail
    (``result_delay``) elapses off-machine — holding a station through it would
    make the small lab the floor's binding constraint by construction. The draw
    is genuine nurse direct care (NurseVisit* pair — the schema has no Lab*
    events).
    """
    pid = patient.id
    event_log.append(TestOrdered(occurred_at=executor.now(), patient=pid, activity=Activity.LAB))
    yield from _bedside_service(
        world, patient, bay, "lab", Activity.LAB, StaffRole.NURSE, index, service_times
    )
    station = _nearest(world, bay.node, tuple(world.resources.lab))
    resource = world.resources.lab[station]
    req = yield from executor.acquire(resource, priority=priority)
    try:
        yield executor.delay(
            service_times.analyzer_time(patient=pid, index=index), PriorityTier.COMPLETION
        )
    finally:
        resource.release(req)
    yield executor.delay(
        service_times.result_delay(Activity.LAB, patient=pid, index=index),
        PriorityTier.COMPLETION,
    )
    event_log.append(TestResulted(occurred_at=executor.now(), patient=pid, activity=Activity.LAB))
    world.request_decision()


def _enqueue_cleaning(
    world: World, patient: Patient, bay: Bay, service_times: ServiceTimes
) -> None:
    """A dirty bay is a physics fact: the cleaning task exists the instant it is vacated.

    The turnaround policy *prioritizes* it (a ``clean`` item boosts it) and the
    dispatch policy assigns housekeeping; duration keys on the vacating patient
    so the draw is identical across arms.
    """
    duration = service_times.sample(
        Activity.CLEANING, patient.esi, patient.complaint, patient=patient.id
    )
    world.add_task(
        kind="cleaning",
        patient=None,
        at=bay.node,
        required_role=StaffRole.HOUSEKEEPING,
        activity=Activity.CLEANING,
        duration=duration,
        bay=bay.id,
    )


__all__ = ["TRIAGE_TEAM", "patient_process"]
