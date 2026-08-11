"""ED boarding into an inpatient ward, and the transport that gets them there (M4).

Through M3 an admitted patient held their ED bay for a **sampled** boarding delay and
then vanished off the floor. That is an honest abstraction of a single-floor model — there
was no upstairs for them to go to — but it hard-codes the answer to the question a
hospital-wide model exists to ask. A two-hour mean boarding time drawn from a distribution
cannot get worse when the ICU fills, because nothing in it knows the ICU exists.

Here boarding becomes a *consequence*: the patient holds their ED bay until a ward bed is
actually granted, a porter escorts them to it — over the same route graph, whose elevator
edges make the trip upstairs cost what it costs — and the bed is occupied for a ward stay
before it returns to the pool. ED crowding is then downstream of ward capacity, which is
the mechanism this milestone is about.

**Opt-in by construction.** A floor plan with no ward beds keeps the sampled delay
exactly, so every ED-only scenario — and every committed golden — is byte-identical. There
is no flag: the presence of an inpatient bed *is* the switch.

**Which bed is a placement decision, and this module does not make it (M4 §3).** An
admitted patient re-enters the *same* bay queue their ED placement came from, under the
:data:`~hospital.core.AWAITING_ADMISSION` stage, and waits for the decision layer to grant
one — solver or operator, through the one validated seam, as an ordinary ``assign_bay``.
The earlier version claimed the nearest free bed itself on a five-minute poll, which meant
the single scarcest resource in a hospital-wide model was the one thing no policy could
decide and no override could touch. What survives of it is the *preference* ordering, now
``solver.placement.WARD_PREFERENCE`` — a bias on the solver's side of the seam, where the
permission question (``AdmissionRule``) is a rule and the preference question is not.

The stage does the rest by itself: it selects the ward whitelist in the validator and in
every backend's ``compat`` derivation, so an admitted patient cannot be handed an ED bay
and a waiting one cannot be handed an ICU bed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from hospital.core import (
    AWAITING_ADMISSION,
    WARD_ZONE_TYPES,
    BayId,
    BayRequested,
    DischargeCompleted,
    DischargeStarted,
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
    # The care phase changes here and stays changed: from this instant every placement
    # naming this patient is judged against the ward whitelist, whether they are still
    # boarding, queued, or asleep in an ICU bed. Marked before the request, so no
    # decision tick can see them queued without knowing what they are queued for.
    world.mark_admitted(pid)
    # The second bay request of this patient's stay. Same event as the first: from the
    # log's point of view a patient needing a bed is a patient needing a bed, and a
    # reader distinguishes them by what came before (a disposition of ADMIT).
    event_log.append(BayRequested(occurred_at=executor.now(), patient=pid))

    # Boarding is the wait on this event. The ED bay stays theirs and stays unavailable
    # to everyone else for its whole duration — that blockage is the entire point, and
    # it is how a full ward becomes an ED wait. Infinite patience, like the ED queue:
    # there is no timeout branch, because an admitted patient does not give up and go
    # home. `BayAssigned` is emitted by the seam on grant, as it is for an ED placement.
    wake = world.request_bay(patient, stage=AWAITING_ADMISSION)
    world.request_decision()
    bed_id = cast("BayId", (yield wake))

    # Granted before the move, the ED bay released after it: the bed cannot be taken out
    # from under a patient already in the elevator, and the ED bay is not free until they
    # have actually left it.
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
    "admit_to_ward",
    "has_ward_beds",
    "ward_beds",
]
