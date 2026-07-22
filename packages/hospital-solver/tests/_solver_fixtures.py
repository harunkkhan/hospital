"""Importable test builders for the solver suite (no ``conftest``; doc 00 §5.10).

A tiny but fully-connected ER floor, a representative rule set, an
``ObjectiveConfig``, and ``DecisionInput`` builders -- reused across every test
module so scenario construction is never copy-pasted.
"""

from __future__ import annotations

from hospital.core import (
    Bay,
    BayId,
    BayState,
    BayStatus,
    CapacityRule,
    CompatibilityRule,
    CompiledRules,
    DecisionInput,
    Distance,
    Duration,
    EsiAcuity,
    FloorLayout,
    NodeId,
    Patient,
    PatientId,
    RouteEdge,
    RouteGraph,
    RouteNode,
    Rule,
    SimTime,
    SkillRule,
    StaffId,
    StaffMember,
    StaffRole,
    StaffState,
    TaskId,
    TaskSpec,
    WaitingPatient,
    WalkSpeed,
    WorkupNeeds,
    Zone,
    ZoneId,
    ZoneType,
    compile_rules,
    seconds,
    walk_duration,
)
from hospital.core.enums import ArrivalMode
from hospital.core.seam import TaskKind
from hospital.solver.objective import ObjectiveConfig

WALK_SPEED = WalkSpeed(100)  # cm/s


def node(nid: str, x: int = 0, y: int = 0) -> RouteNode:
    return RouteNode(id=NodeId(nid), label=nid, x_cm=x, y_cm=y)


def edge(a: str, b: str, dist_cm: int, *, bidirectional: bool = True) -> RouteEdge:
    d = Distance(dist_cm)
    return RouteEdge(
        a=NodeId(a),
        b=NodeId(b),
        distance=d,
        seconds=walk_duration(d, WALK_SPEED),
        bidirectional=bidirectional,
    )


def tiny_graph() -> RouteGraph:
    """A star hub (``gstat``) with bays, imaging, and lab hanging off it."""
    nodes = (
        node("gstat", 0, 0),
        node("b1", 100, 0),
        node("b2", 200, 0),
        node("b3", 300, 0),
        node("b4", 400, 0),
        node("img", 0, 100),
        node("lab", 0, 200),
    )
    edges = (
        edge("gstat", "b1", 3000),
        edge("gstat", "b2", 6000),
        edge("gstat", "b3", 9000),
        edge("gstat", "b4", 4000),
        edge("gstat", "img", 7000),
        edge("img", "lab", 4000),
    )
    return RouteGraph(nodes=nodes, edges=edges)


def tiny_layout() -> FloorLayout:
    """General (2 bays), resus (1), fast-track (1); one imaging + one lab node."""
    bays = (
        Bay(
            id=BayId("bay-1"),
            zone=ZoneId("z-gen"),
            zone_type=ZoneType.GENERAL,
            node=NodeId("b1"),
            serving_station=NodeId("gstat"),
            isolation_capable=False,
            equipment=frozenset(),
        ),
        Bay(
            id=BayId("bay-2"),
            zone=ZoneId("z-gen"),
            zone_type=ZoneType.GENERAL,
            node=NodeId("b2"),
            serving_station=NodeId("gstat"),
            isolation_capable=True,
            equipment=frozenset({"monitor"}),
        ),
        Bay(
            id=BayId("bay-3"),
            zone=ZoneId("z-resus"),
            zone_type=ZoneType.RESUS_TRAUMA,
            node=NodeId("b3"),
            serving_station=NodeId("gstat"),
            isolation_capable=True,
            equipment=frozenset({"monitor", "vent"}),
        ),
        Bay(
            id=BayId("bay-4"),
            zone=ZoneId("z-fast"),
            zone_type=ZoneType.FAST_TRACK,
            node=NodeId("b4"),
            serving_station=NodeId("gstat"),
            isolation_capable=False,
            equipment=frozenset(),
        ),
    )
    zones = (
        Zone(id=ZoneId("z-gen"), zone_type=ZoneType.GENERAL, capacity=2),
        Zone(id=ZoneId("z-resus"), zone_type=ZoneType.RESUS_TRAUMA, capacity=1),
        Zone(id=ZoneId("z-fast"), zone_type=ZoneType.FAST_TRACK, capacity=1),
    )
    return FloorLayout(
        graph=tiny_graph(),
        zones=zones,
        bays=bays,
        stations=(NodeId("gstat"),),
        entrances=(NodeId("gstat"),),
        imaging_nodes=(NodeId("img"),),
        lab_nodes=(NodeId("lab"),),
    )


def demo_rules() -> tuple[Rule, ...]:
    """A representative rule set: acuity->zone, equipment, capacity, skill."""
    return (
        CompatibilityRule(
            allowed_zone_types=frozenset(
                {
                    (EsiAcuity.ESI1, ZoneType.RESUS_TRAUMA),
                    (EsiAcuity.ESI2, ZoneType.RESUS_TRAUMA),
                    (EsiAcuity.ESI2, ZoneType.GENERAL),
                    (EsiAcuity.ESI3, ZoneType.GENERAL),
                    (EsiAcuity.ESI4, ZoneType.FAST_TRACK),
                    (EsiAcuity.ESI5, ZoneType.FAST_TRACK),
                }
            ),
            isolation_enforced=True,
            required_equipment=frozenset({(EsiAcuity.ESI1, "monitor")}),
        ),
        CapacityRule(zone_type=ZoneType.GENERAL, max_occupancy=2),
        CapacityRule(zone_type=ZoneType.RESUS_TRAUMA, max_occupancy=1),
        CapacityRule(zone_type=ZoneType.FAST_TRACK, max_occupancy=1),
        SkillRule(task_kind="provider_visit", required_skills=frozenset({"md"})),
    )


def demo_compiled() -> CompiledRules:
    return compile_rules(demo_rules())


def default_config(**overrides: object) -> ObjectiveConfig:
    return ObjectiveConfig(**overrides)  # type: ignore[arg-type]


def make_patient(
    pid: str,
    esi: EsiAcuity = EsiAcuity.ESI3,
    *,
    isolation: bool = False,
    arrival_us: int = 0,
    provider_visits: int = 1,
    nurse_visits: int = 1,
    imaging: tuple[ZoneType, ...] = (),
    labs: int = 0,
) -> Patient:
    return Patient(
        id=PatientId(pid),
        arrival_time=SimTime(arrival_us),
        arrival_mode=ArrivalMode.WALK_IN,
        esi=esi,
        complaint="chest_pain",
        isolation_required=isolation,
        workup=WorkupNeeds(
            provider_visits=provider_visits,
            nurse_visits=nurse_visits,
            imaging=imaging,
            labs=labs,
            procedures=0,
        ),
    )


def waiting(patient: Patient, waited_s: float, stage: str = "needs_bay") -> WaitingPatient:
    return WaitingPatient(patient=patient, waited=seconds(waited_s), stage=stage)


def bay_state(
    bay: str, status: BayStatus = BayStatus.FREE, occupant: str | None = None
) -> BayState:
    return BayState(
        bay=BayId(bay),
        status=status,
        occupant=PatientId(occupant) if occupant is not None else None,
    )


def all_free_bays() -> tuple[BayState, ...]:
    return tuple(bay_state(f"bay-{i}") for i in (1, 2, 3, 4))


def staff_member(
    sid: str, role: StaffRole, *, skills: frozenset[str] = frozenset(), home: str = "gstat"
) -> StaffMember:
    return StaffMember(id=StaffId(sid), role=role, home_station=NodeId(home), skills=skills)


def staff_state(sid: str, at: str = "gstat") -> StaffState:
    return StaffState(staff=StaffId(sid), at=NodeId(at))


def task(
    tid: str,
    kind: TaskKind,
    *,
    at: str,
    role: StaffRole,
    patient: str | None = None,
    skills: frozenset[str] = frozenset(),
) -> TaskSpec:
    return TaskSpec(
        id=TaskId(tid),
        kind=kind,
        patient=PatientId(patient) if patient is not None else None,
        at=NodeId(at),
        required_role=role,
        required_skills=skills,
        ready_at=SimTime(0),
    )


def decision_input(
    *,
    waiting_patients: tuple[WaitingPatient, ...] = (),
    bays: tuple[BayState, ...] | None = None,
    staff: tuple[StaffState, ...] = (),
    tasks: tuple[TaskSpec, ...] = (),
    layout: FloorLayout | None = None,
    now_us: int = 0,
) -> DecisionInput:
    resolved_layout = layout if layout is not None else tiny_layout()
    return DecisionInput(
        now=SimTime(now_us),
        layout=resolved_layout,
        waiting=waiting_patients,
        bays=bays if bays is not None else all_free_bays(),
        staff=staff,
        pending_tasks=tasks,
        events_since=(),
    )


def zero() -> Duration:
    return Duration(0)
