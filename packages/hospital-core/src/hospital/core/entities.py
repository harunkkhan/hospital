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
    """A care zone with a static capacity."""

    id: ZoneId
    zone_type: ZoneType
    capacity: int


class StaffMember(FrozenModel):
    """A static staff descriptor (role, home station, skills)."""

    id: StaffId
    role: StaffRole
    home_station: NodeId
    skills: frozenset[str]


class FloorLayout(FrozenModel):
    """The static floor: route graph plus zones/bays/stations/entrances."""

    graph: RouteGraph
    zones: tuple[Zone, ...]
    bays: tuple[Bay, ...]
    stations: tuple[NodeId, ...]
    entrances: tuple[NodeId, ...]
    imaging_nodes: tuple[NodeId, ...]
    lab_nodes: tuple[NodeId, ...]


__all__ = [
    "Bay",
    "FloorLayout",
    "Patient",
    "StaffMember",
    "WorkupNeeds",
    "Zone",
]
