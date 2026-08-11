"""Frozen static descriptors of patients, bays, staff, and the floor.

Everything here is immutable *static* data. The canonical example of the
"one mutable owner" rule (nuance 1.6): ``Bay`` has **no** ``status`` field —
dynamic bay status lives only in ``sim.physics.world``. If status crept into the
frozen ``Bay`` there would be two sources of truth.

``WorkupNeeds`` is **pre-sampled** at generation time (v1 simplification): the
whole workup is drawn up front and fixed, rather than endogenously growing as
results come back. The placement cost uses these expected visit counts.
"""

from __future__ import annotations

from pydantic import Field

from hospital.core.enums import ArrivalMode, EsiAcuity, StaffRole, ZoneType
from hospital.core.graph import RouteGraph
from hospital.core.ids import BayId, NodeId, PatientId, StaffId, ZoneId
from hospital.core.models import FrozenModel
from hospital.core.time import Duration, SimTime, TimeWindow, minutes

# Door-to-provider targets by acuity — the **soft** SLA a ``care_deadline`` encodes.
# 🟡-tunable data, in the same family as the acuity urgency curve: these are the
# conventional ESI door-to-provider expectations (ESI-1 immediate, then 10/30/60/120
# minutes), stated as data so a scenario can disagree with them without anyone
# re-deriving the acuity inversion.
#
# **Soft, and nothing enforces it.** Missing a deadline is not a violation the plan
# validator can refuse — the engine cannot conjure a physician — and no lever prices it,
# because pricing it would silently change every existing decision. It is a *measurement
# reference*: the acuity-relative answer to "was this patient seen in time", available to
# analysis and to an operator without either having to restate the table.
CARE_SLA_BY_ACUITY: dict[EsiAcuity, Duration] = {
    EsiAcuity.ESI1: minutes(0),
    EsiAcuity.ESI2: minutes(10),
    EsiAcuity.ESI3: minutes(30),
    EsiAcuity.ESI4: minutes(60),
    EsiAcuity.ESI5: minutes(120),
}


def care_deadline_for(arrival: SimTime, esi: EsiAcuity) -> SimTime:
    """When a patient arriving at ``arrival`` with acuity ``esi`` should have been seen.

    The ONE place the deadline formula lives. :attr:`Patient.care_deadline` is this function
    applied to its own fields; ``analysis`` calls it directly, because the KPI fold works
    from an event-log index that has an arrival and an acuity but no ``Patient`` to ask.
    Two callers, one formula — the alternative is ``arrival + SLA[esi]`` written twice and
    a rate table that only half the codebase honours.
    """
    return arrival + CARE_SLA_BY_ACUITY[esi]


class WorkupNeeds(FrozenModel):
    """The pre-sampled diagnostic/care workup for a patient."""

    provider_visits: int
    nurse_visits: int
    imaging: tuple[ZoneType, ...]
    labs: int
    procedures: int


class Patient(FrozenModel):
    """A patient's immutable arrival-time descriptor."""

    id: PatientId
    arrival_time: SimTime
    arrival_mode: ArrivalMode
    esi: EsiAcuity
    complaint: str
    isolation_required: bool
    workup: WorkupNeeds

    @property
    def care_deadline(self) -> SimTime:
        """When this patient should have reached a provider — ``arrival + SLA(esi)``.

        A **property, not a field**, because the spec calls it *derived* and because a
        stored copy would be a second source of truth for something already implied by
        two values that cannot change: the patient is frozen, so the deadline is a pure
        function of the record. It also keeps it off every wire and out of every
        serialized byte, so no event log, golden trace, or generated TypeScript contract
        moves for a quantity nobody transmits.

        Absolute sim-time rather than a remaining-time countdown, for the same reason
        every other instant here is: a countdown would need a "now" to be meaningful and
        would therefore be a different number every time it was read.
        """
        return care_deadline_for(self.arrival_time, self.esi)


class Bay(FrozenModel):
    """A static bay descriptor. Deliberately has **no** ``status`` field."""

    id: BayId
    zone: ZoneId
    zone_type: ZoneType
    node: NodeId
    serving_station: NodeId
    isolation_capable: bool
    equipment: frozenset[str]


class Zone(FrozenModel):
    """A care zone with a static capacity, on a floor.

    ``floor`` defaults to ``0`` so a single-floor ER scenario — every scenario before the
    multi-floor hospital — describes itself exactly as it did before. It is an index into
    the hospital's floors, not a storey number: routing never reads it, because vertical
    movement is edges in the graph like any other movement. It is here so a *decision*
    can be about a floor (is this ward upstairs from the ED?) without re-deriving that
    from node ids.
    """

    id: ZoneId
    zone_type: ZoneType
    capacity: int
    floor: int = Field(default=0, ge=0)


class StaffMember(FrozenModel):
    """A static staff descriptor (role, home station, skills, and a shift schedule).

    ``shifts`` is the schedule §3.3's "``on_shift`` flag driven by the schedule" refers
    to — the *flag* is derived from it by :meth:`on_shift`, for the same reason
    ``care_deadline`` is derived: a stored boolean would need re-writing at every shift
    boundary and would be a second source of truth for something the windows already say.

    **Empty means always on duty, and that is the pre-existing behaviour.** Through M4 the
    engine realized one static roster for the whole horizon — shift blocks were collapsed
    to their per-role maximum, so a scenario declaring 07:00-19:00 cover ran that headcount
    at 03:00 too. That is what ``()`` reproduces exactly, so every scenario and golden
    written before shift-awareness is byte-identical. A scenario opts in by asking
    ``realize_staff`` for shift-aware staff, which fills this in.
    """

    id: StaffId
    role: StaffRole
    home_station: NodeId
    skills: frozenset[str]
    shifts: tuple[TimeWindow, ...] = ()

    def on_shift(self, t: SimTime) -> bool:
        """Whether this member is rostered at ``t`` — always true with no schedule."""
        return not self.shifts or any(shift.contains(t) for shift in self.shifts)

    def on_duty_until(self, t: SimTime) -> SimTime | None:
        """When the shift covering ``t`` ends, or ``None`` when off duty or unscheduled."""
        for shift in self.shifts:
            if shift.contains(t):
                return shift.end
        return None

    def next_shift_start(self, t: SimTime) -> SimTime | None:
        """The start of the earliest shift beginning at or after ``t``, if any.

        What an off-duty member is "busy until": the engine projects unavailability as a
        ``busy_until``, the same channel a staff-absence disruption uses, so every existing
        dispatch path skips an off-shift member without knowing shifts exist. ``None`` means
        they have no further shift in the horizon and are unavailable for the rest of it.
        """
        upcoming = [shift.start for shift in self.shifts if shift.start >= t]
        return min(upcoming) if upcoming else None


class FloorLayout(FrozenModel):
    """The static built environment: route graph plus zones/bays/stations/entrances.

    Named for the single ER floor it described through M3, and still exactly that for an
    ED-only scenario. A multi-floor hospital is the *same* type with a graph that spans
    floors, joined by ``elevators`` — which is why nothing downstream had to change to
    gain vertical movement: an elevator is an edge whose ``seconds`` greatly exceed its
    distance, and :class:`~hospital.core.graph.RouteEdge` was built to allow that.

    ``elevators`` lists the boarding node on each floor. Empty for a single floor: there
    is nowhere to go.
    """

    graph: RouteGraph
    zones: tuple[Zone, ...]
    bays: tuple[Bay, ...]
    stations: tuple[NodeId, ...]
    entrances: tuple[NodeId, ...]
    imaging_nodes: tuple[NodeId, ...]
    lab_nodes: tuple[NodeId, ...]
    elevators: tuple[NodeId, ...] = ()


__all__ = [
    "CARE_SLA_BY_ACUITY",
    "Bay",
    "FloorLayout",
    "Patient",
    "StaffMember",
    "WorkupNeeds",
    "Zone",
    "care_deadline_for",
]
