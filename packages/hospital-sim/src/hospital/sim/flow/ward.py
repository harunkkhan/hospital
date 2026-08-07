"""ED boarding into an inpatient ward, and the transport that gets them there (M4).

Through M3 an admitted patient held their ED bay for a **sampled** boarding delay and
then vanished off the floor. That is an honest abstraction of a single-floor model — there
was no upstairs for them to go to — but it hard-codes the answer to the question a
hospital-wide model exists to ask. A two-hour mean boarding time drawn from a distribution
cannot get worse when the ICU fills, because nothing in it knows the ICU exists.

Here boarding becomes a *consequence*: the patient holds their ED bay until a ward bed is
actually free, a porter escorts them to it — over the same route graph, whose elevator
edges make the trip upstairs cost what it costs — and the bed is occupied for a ward stay
before it returns to the pool. ED crowding is then downstream of ward capacity, which is
the mechanism this milestone is about.

**Opt-in by construction.** A floor plan with no ward beds keeps the sampled delay
exactly, so every ED-only scenario — and every committed golden — is byte-identical. There
is no flag: the presence of an inpatient bed *is* the switch.

Two pieces are deliberately simple for now, and both are marked where they sit:

* **Which ward** an admitted patient belongs in is a clinical routing question, answered
  here by an explicit acuity table rather than by the ED's acuity→zone rules. Reusing
  those would be wrong in a specific way: an ESI-3 who may occupy a GENERAL ED bay is not
  thereby a candidate for an ICU bed.
* **Which free bed** they take is the nearest one, chosen greedily. That is a placement
  decision and it belongs in the solver alongside ED placement — hospital-wide placement
  is the next increment, and this module is written so that swapping ``_claim_bed`` for a
  solver grant is the only change it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from hospital.core import (
    WARD_ZONE_TYPES,
    BayAssigned,
    BayId,
    BayRequested,
    BayStatus,
    DischargeCompleted,
    DischargeStarted,
    Duration,
    EsiAcuity,
    ZoneType,
    minutes,
)
from hospital.sim.physics.executor import PriorityTier

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import simpy

    from hospital.core import Bay, EventLog, NodeId, Patient, RandomStreams
    from hospital.sim.physics.executor import TaskExecutor
    from hospital.sim.physics.service_times import ServiceTimes
    from hospital.sim.physics.world import World

    # The two `flow.patient` helpers this module borrows. Typed rather than imported:
    # `flow.patient` calls into here, so importing them back would be a cycle.
    Escort = Callable[[World, Patient, NodeId], Generator[simpy.Event, object]]
    EnqueueCleaning = Callable[[World, Patient, Bay, ServiceTimes], None]

# Where an admitted patient is sent, most-preferred first. A 🟡-tunable clinical routing
# heuristic in the same family as the travel proxy in `solver.placement`, NOT a rule: it
# picks the ward a patient belongs in, and never overrides a capacity or isolation
# constraint. The fallbacks matter more than the first entry — a full ICU boards its
# patients in the ED rather than sending them to maternity.
WARD_PREFERENCE: Final[dict[EsiAcuity, tuple[ZoneType, ...]]] = {
    EsiAcuity.ESI1: (ZoneType.ICU, ZoneType.SURGERY, ZoneType.MED_SURG),
    EsiAcuity.ESI2: (ZoneType.ICU, ZoneType.MED_SURG, ZoneType.SURGERY),
    EsiAcuity.ESI3: (ZoneType.MED_SURG, ZoneType.SURGERY, ZoneType.ICU),
    EsiAcuity.ESI4: (ZoneType.MED_SURG, ZoneType.SURGERY),
    EsiAcuity.ESI5: (ZoneType.MED_SURG,),
}

# How often a boarding patient re-checks for a bed. Interim, and the reason it is
# acceptable: it only quantizes the *reported* boarding time, never the bed's
# availability, and five minutes is far below the hours that boarding actually runs to.
# It disappears when placement grants the bed instead of the patient asking for it.
BED_POLL: Final[Duration] = minutes(5)


def ward_beds(world: World) -> tuple[BayId, ...]:
    """Every inpatient bed in the building, in deterministic id order."""
    return tuple(
        sorted(
            (bay.id for bay in world.layout.bays if bay.zone_type in WARD_ZONE_TYPES),
            key=lambda b: b.root,
        )
    )


def has_ward_beds(world: World) -> bool:
    """Whether this floor plan admits anyone at all.

    The switch for the whole module: false for every ED-only scenario, which is what
    keeps their runs byte-identical to the milestones that preceded wards.
    """
    return bool(ward_beds(world))


def _claim_bed(world: World, patient: Patient) -> BayId | None:
    """The nearest free bed in the most-preferred ward that has one, else ``None``.

    Preference is exhausted in order before distance is consulted, so a full ICU sends an
    ESI-1 to surgery rather than keeping them in the ED — a hospital would rather have the
    patient in a bed than in a corridor. Distance breaks ties within a ward type, id
    breaks ties within a distance, so the choice is deterministic under CRN.
    """
    origin = world.patient_at(patient.id)
    for zone_type in WARD_PREFERENCE.get(patient.esi, ()):
        free = [
            bay
            for bay in world.layout.bays
            if bay.zone_type is zone_type and world.bay_status(bay.id) is BayStatus.FREE
        ]
        if patient.isolation_required:
            free = [bay for bay in free if bay.isolation_capable]
        if not free:
            continue
        routed = [
            (path.total.root, bay.id.root, bay.id)
            for bay in free
            if (path := world.try_route(origin, bay.node)) is not None
        ]
        if routed:
            return min(routed)[2]
    return None


def admit_to_ward(
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    patient: Patient,
    ed_bay: Bay,
    *,
    streams: RandomStreams,
    service_times: ServiceTimes,
    transport: Escort,
    enqueue_cleaning: EnqueueCleaning,
) -> Generator[simpy.Event, object]:
    """Board in the ED, move upstairs, occupy the bed, and eventually leave it.

    ``transport`` and ``enqueue_cleaning`` are passed in rather than imported: they live in
    ``flow.patient``, which calls this, and importing them back would be a cycle. They are
    the same helpers the ED path uses, so an escort upstairs is an escort like any other —
    the elevator is in the route, not in this code.
    """
    from hospital.sim.physics.service_times import sample_ward_stay

    pid = patient.id
    # The second bay request of this patient's stay. Same event as the first: from the
    # log's point of view a patient needing a bed is a patient needing a bed, and a
    # reader distinguishes them by what came before (a disposition of ADMIT).
    event_log.append(BayRequested(occurred_at=executor.now(), patient=pid))

    while (bed_id := _claim_bed(world, patient)) is None:
        # Boarding: the ED bay is still theirs, and still unavailable to anyone else.
        # That blockage is the entire point — it is how a full ward becomes an ED wait.
        yield executor.delay(BED_POLL, PriorityTier.COMPLETION)

    world.assign_bay(bed_id, pid)
    event_log.append(
        BayAssigned(occurred_at=executor.now(), patient=pid, bay=bed_id, by="baseline")
    )

    # Claimed before the move, released after it: the bed cannot be taken out from under a
    # patient already in the elevator, and the ED bay is not free until they have left it.
    bed = world.bay(bed_id)
    world.vacate_bay(ed_bay.id)
    enqueue_cleaning(world, patient, ed_bay, service_times)
    world.request_decision()

    yield from transport(world, patient, bed.node)

    yield executor.delay(sample_ward_stay(streams, patient), PriorityTier.COMPLETION)

    event_log.append(DischargeStarted(occurred_at=executor.now(), patient=pid))
    world.vacate_bay(bed_id)
    enqueue_cleaning(world, patient, bed, service_times)
    world.request_decision()
    event_log.append(DischargeCompleted(occurred_at=executor.now(), patient=pid))


__all__ = [
    "BED_POLL",
    "WARD_PREFERENCE",
    "admit_to_ward",
    "has_ward_beds",
    "ward_beds",
]
