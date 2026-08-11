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
from hospital.core.time import SimTime


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
    """A static staff descriptor (role, home station, skills)."""

    id: StaffId
    role: StaffRole
    home_station: NodeId
    skills: frozenset[str]


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
    "Bay",
    "FloorLayout",
    "Patient",
    "StaffMember",
    "WorkupNeeds",
    "Zone",
]
