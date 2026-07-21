"""Importable test builders (no ``conftest``; helpers are imported by name).

Shared, reused across the core test modules so scenario construction is never
copy-pasted (doc 00 §5.10 / doc 08 §1).
"""

from __future__ import annotations

from hospital.core import (
    KPI_KEYS,
    Bay,
    BayId,
    BayState,
    BayStatus,
    CapacityRule,
    CompatibilityRule,
    CompiledRules,
    Distance,
    EsiAcuity,
    FloorLayout,
    NodeId,
    Patient,
    PatientId,
    PrecedenceRule,
    RouteEdge,
    RouteGraph,
    RouteNode,
    Rule,
    SkillRule,
    StaffId,
    StaffMember,
    StaffRole,
    StaffState,
    TaskId,
    TaskSpec,
    WalkSpeed,
    WorkupNeeds,
    Zone,
    ZoneId,
    ZoneType,
    compile_rules,
    walk_duration,
)
from hospital.core.enums import Activity, ArrivalMode
from hospital.core.time import SimTime

WALK_SPEED = WalkSpeed(100)  # cm/s


def node(nid: str, x: int = 0, y: int = 0) -> RouteNode:
    return RouteNode(id=NodeId(nid), label=nid, x_cm=x, y_cm=y)


def edge(a: str, b: str, dist_cm: int, *, bidirectional: bool = True) -> RouteEdge:
    """An edge whose ``seconds`` is derived from distance at the shared walk speed."""
    d = Distance(dist_cm)
    return RouteEdge(
        a=NodeId(a),
        b=NodeId(b),
        distance=d,
        seconds=walk_duration(d, WALK_SPEED),
        bidirectional=bidirectional,
    )


def diamond_graph() -> RouteGraph:
    """Two equal-cost routes ``src->{b,c}->dst`` plus a maskable long shortcut.

    The two branches are symmetric, so the deterministic tie-break
    ``(seconds, distance, node_id)`` decides the route (``b`` < ``c``). Node
    ``closed_only`` hangs off ``dst`` for closed-node tests.
    """
    nodes = (
        node("src"),
        node("b"),
        node("c"),
        node("dst"),
        node("shortcut_mid"),
        node("closed_only"),
    )
    edges = (
        edge("src", "b", 1000),
        edge("src", "c", 1000),
        edge("b", "dst", 1000),
        edge("c", "dst", 1000),
        # A longer detour via shortcut_mid (total 4000 > 2000) — never chosen unless masked.
        edge("src", "shortcut_mid", 1500),
        edge("shortcut_mid", "dst", 2500),
        edge("dst", "closed_only", 1000),
    )
    return RouteGraph(nodes=nodes, edges=edges)


def tiny_er_layout() -> FloorLayout:
    """A minimal but fully-connected ER floor: triage + general + resus bays."""
    nodes = (
        node("entrance", 0, 0),
        node("triage", 100, 0),
        node("hub", 200, 0),
        node("gstat", 300, 0),
        node("b1", 300, 100),
        node("b2", 300, 200),
        node("b3", 300, 300),
        node("img", 400, 0),
        node("lab", 400, 100),
    )
    edges = (
        edge("entrance", "triage", 5000),
        edge("triage", "hub", 5000),
        edge("hub", "gstat", 5000),
        edge("gstat", "b1", 3000),
        edge("gstat", "b2", 6000),
        edge("gstat", "b3", 9000),
        edge("hub", "img", 7000),
        edge("img", "lab", 4000),
    )
    graph = RouteGraph(nodes=nodes, edges=edges)
    zones = (
        Zone(id=ZoneId("z-gen"), zone_type=ZoneType.GENERAL, capacity=2),
        Zone(id=ZoneId("z-resus"), zone_type=ZoneType.RESUS_TRAUMA, capacity=1),
    )
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
    )
    return FloorLayout(
        graph=graph,
        zones=zones,
        bays=bays,
        stations=(NodeId("gstat"),),
        entrances=(NodeId("entrance"),),
        imaging_nodes=(NodeId("img"),),
        lab_nodes=(NodeId("lab"),),
    )


def patient(
    pid: str,
    esi: EsiAcuity = EsiAcuity.ESI3,
    *,
    isolation: bool = False,
    arrival_us: int = 0,
) -> Patient:
    return Patient(
        id=PatientId(pid),
        arrival_time=SimTime(arrival_us),
        arrival_mode=ArrivalMode.WALK_IN,
        esi=esi,
        complaint="chest_pain",
        isolation_required=isolation,
        workup=WorkupNeeds(provider_visits=1, nurse_visits=1, imaging=(), labs=0, procedures=0),
    )


def demo_rules() -> tuple[Rule, ...]:
    """A representative rule set exercising every rule kind."""
    return (
        CompatibilityRule(
            allowed_zone_types=frozenset(
                {
                    (EsiAcuity.ESI1, ZoneType.RESUS_TRAUMA),
                    (EsiAcuity.ESI1, ZoneType.GENERAL),
                    (EsiAcuity.ESI3, ZoneType.GENERAL),
                    (EsiAcuity.ESI3, ZoneType.FAST_TRACK),
                    (EsiAcuity.ESI5, ZoneType.FAST_TRACK),
                }
            ),
            isolation_enforced=True,
            required_equipment=frozenset({(EsiAcuity.ESI1, "monitor")}),
        ),
        CapacityRule(zone_type=ZoneType.GENERAL, max_occupancy=2),
        CapacityRule(zone_type=ZoneType.RESUS_TRAUMA, max_occupancy=1),
        SkillRule(task_kind="provider_visit", required_skills=frozenset({"md"})),
        PrecedenceRule(before=Activity.TRIAGE, after=Activity.PROVIDER_VISIT),
    )


def demo_compiled() -> CompiledRules:
    return compile_rules(demo_rules())


def bay_state(bay: str, status: BayStatus, occupant: str | None = None) -> BayState:
    return BayState(
        bay=BayId(bay),
        status=status,
        occupant=PatientId(occupant) if occupant is not None else None,
    )


def staff_member(sid: str, role: StaffRole, skills: frozenset[str]) -> StaffMember:
    return StaffMember(id=StaffId(sid), role=role, home_station=NodeId("gstat"), skills=skills)


def staff_state(sid: str, at: str = "gstat") -> StaffState:
    return StaffState(staff=StaffId(sid), at=NodeId(at))


def provider_task(tid: str, patient_id: str) -> TaskSpec:
    return TaskSpec(
        id=TaskId(tid),
        kind="provider_visit",
        patient=PatientId(patient_id),
        at=NodeId("b1"),
        required_role=StaffRole.PHYSICIAN,
        ready_at=SimTime(0),
    )


def full_kpi_values(**overrides: float) -> dict[str, float]:
    """A complete, contract-valid KPI dict (staff_frac_* sum to 1.0), with overrides."""
    values = dict.fromkeys(KPI_KEYS, 0.0)
    values["staff_frac_idle"] = 1.0  # residual makes fractions sum to 1.0
    values.update(overrides)
    return values
