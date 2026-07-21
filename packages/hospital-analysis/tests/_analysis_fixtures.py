"""Importable test builders (no ``conftest``; helpers are imported by name).

Reused across this package's test modules so scenario/log construction is
never copy-pasted (doc 00 §5.10 / doc 08 §1) — the contract test for when
``hospital-sim`` later feeds real logs: fabricate a small ``EventLog`` by hand
from ``hospital.core.events`` constructors and assert the fold/decomposition/
utilization outputs against hand-computed expectations.
"""

from __future__ import annotations

from hospital.core import (
    KPI_KEYS,
    Bay,
    BayAssigned,
    BayCleaningCompleted,
    BayCleaningStarted,
    BayId,
    BayRequested,
    DischargeCompleted,
    DischargeStarted,
    DispositionDecided,
    DispositionKind,
    Distance,
    DocumentationCompleted,
    DocumentationStarted,
    EsiAcuity,
    EventLog,
    FloorLayout,
    NodeId,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientArrived,
    PatientId,
    PatientMoved,
    ProviderVisitCompleted,
    ProviderVisitStarted,
    RouteEdge,
    RouteGraph,
    RouteNode,
    SimTime,
    StaffId,
    StaffMember,
    StaffMoved,
    StaffRole,
    TriageCompleted,
    TriageStarted,
    WalkSpeed,
    Zone,
    ZoneId,
    ZoneType,
    seconds,
    walk_duration,
)
from hospital.core.enums import ArrivalMode

PHYSICIAN = StaffId("phys-1")
NURSE = StaffId("nurse-1")
HOUSEKEEPER = StaffId("hk-1")

_WALK_SPEED = WalkSpeed(100)  # cm/s, arbitrary — analysis never routes


def t(sec: float) -> SimTime:
    """Seconds -> ``SimTime`` (µs), via the one banker's-rounding conversion in core."""
    return SimTime(seconds(sec).root)


def node(nid: str, x: int = 0, y: int = 0) -> RouteNode:
    return RouteNode(id=NodeId(nid), label=nid, x_cm=x, y_cm=y)


def edge(a: str, b: str, dist_cm: int = 100) -> RouteEdge:
    d = Distance(dist_cm)
    return RouteEdge(a=NodeId(a), b=NodeId(b), distance=d, seconds=walk_duration(d, _WALK_SPEED))


def tiny_layout() -> FloorLayout:
    """A minimal 3-bay floor: two GENERAL bays + one RESUS_TRAUMA bay."""
    nodes = (
        node("entrance"),
        node("triage"),
        node("gstat"),
        node("b1"),
        node("b2"),
        node("b3"),
    )
    edges = (
        edge("entrance", "triage"),
        edge("triage", "gstat"),
        edge("gstat", "b1"),
        edge("gstat", "b2"),
        edge("gstat", "b3"),
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
            isolation_capable=False,
            equipment=frozenset(),
        ),
        Bay(
            id=BayId("bay-3"),
            zone=ZoneId("z-resus"),
            zone_type=ZoneType.RESUS_TRAUMA,
            node=NodeId("b3"),
            serving_station=NodeId("gstat"),
            isolation_capable=True,
            equipment=frozenset({"monitor"}),
        ),
    )
    return FloorLayout(
        graph=graph,
        zones=zones,
        bays=bays,
        stations=(NodeId("gstat"),),
        entrances=(NodeId("entrance"),),
        imaging_nodes=(),
        lab_nodes=(),
    )


def tiny_roster() -> tuple[StaffMember, ...]:
    return (
        StaffMember(
            id=PHYSICIAN, role=StaffRole.PHYSICIAN, home_station=NodeId("gstat"), skills=frozenset()
        ),
        StaffMember(
            id=NURSE, role=StaffRole.NURSE, home_station=NodeId("gstat"), skills=frozenset()
        ),
        StaffMember(
            id=HOUSEKEEPER,
            role=StaffRole.HOUSEKEEPING,
            home_station=NodeId("gstat"),
            skills=frozenset(),
        ),
    )


def build_sample_log() -> EventLog:
    """Two patients: P1 (ESI3) completes discharge->clean; P2 (ESI5) stays WIP.

    Hand-picked µs timings (in seconds below) so every KPI/decomposition has an
    exact, hand-computable expectation:

    P1: a=0, ts0=60, ts1=300, br=360, ba=480, pv=600 (svc 300s), nurse visit
    950-1000 (50s), dd=1200, documentation 1210-1240, ds=1250, de=1500,
    clean_start=1560, clean_end=1860.
      -> stages: wait_triage=60, svc_triage=240, wait_bay=180, wait_provider=120,
         workup_service=350, workup_wait=250, paperwork_or_boarding=300 (sum=1500=los_s).
      -> bay turnaround: hold_to_vacate=300, wait_housekeeper=60, cleaning=300 (sum=660).

    P2: a=2000, ts0=2050, ts1=2100 (esi=ESI5), br=2150, ba=2200, pv=2300 — no
    disposition/discharge (still WIP at the horizon): excluded from LOS/tiling,
    counted only in wip_end_of_week.
    """
    log = EventLog()
    p1, p2 = PatientId("p1"), PatientId("p2")

    # --- P1: full flow through discharge + bay cleaning ---
    log.append(PatientArrived(occurred_at=t(0), patient=p1, mode=ArrivalMode.WALK_IN))
    log.append(TriageStarted(occurred_at=t(60), patient=p1, staff=NURSE))
    log.append(TriageCompleted(occurred_at=t(300), patient=p1, esi=EsiAcuity.ESI3))
    log.append(BayRequested(occurred_at=t(300), patient=p1))
    log.append(BayAssigned(occurred_at=t(360), patient=p1, bay=BayId("bay-1"), by="baseline"))
    log.append(
        PatientMoved(
            occurred_at=t(480),
            patient=p1,
            edge=(NodeId("gstat"), NodeId("b1")),
            seconds=seconds(60),
        )
    )
    log.append(ProviderVisitStarted(occurred_at=t(600), patient=p1, staff=PHYSICIAN))
    log.append(ProviderVisitCompleted(occurred_at=t(900), patient=p1, staff=PHYSICIAN))
    log.append(NurseVisitStarted(occurred_at=t(950), patient=p1, staff=NURSE))
    log.append(NurseVisitCompleted(occurred_at=t(1000), patient=p1, staff=NURSE))
    log.append(
        DispositionDecided(occurred_at=t(1200), patient=p1, disposition=DispositionKind.DISCHARGE)
    )
    log.append(DocumentationStarted(occurred_at=t(1210), patient=p1, staff=NURSE))
    log.append(DocumentationCompleted(occurred_at=t(1240), patient=p1, staff=NURSE))
    log.append(DischargeStarted(occurred_at=t(1250), patient=p1))
    log.append(DischargeCompleted(occurred_at=t(1500), patient=p1))
    log.append(BayCleaningStarted(occurred_at=t(1560), bay=BayId("bay-1"), staff=HOUSEKEEPER))
    log.append(BayCleaningCompleted(occurred_at=t(1860), bay=BayId("bay-1"), staff=HOUSEKEEPER))

    # --- staff walk events around P1's care ---
    log.append(
        StaffMoved(
            occurred_at=t(590),
            staff=PHYSICIAN,
            edge=(NodeId("gstat"), NodeId("b1")),
            seconds=seconds(20),
        )
    )
    log.append(
        StaffMoved(
            occurred_at=t(945),
            staff=NURSE,
            edge=(NodeId("gstat"), NodeId("b1")),
            seconds=seconds(15),
        )
    )
    log.append(
        StaffMoved(
            occurred_at=t(1555),
            staff=HOUSEKEEPER,
            edge=(NodeId("b1"), NodeId("gstat")),
            seconds=seconds(25),
        )
    )

    # --- P2: still WIP at the horizon (never disposed/discharged) ---
    log.append(PatientArrived(occurred_at=t(2000), patient=p2, mode=ArrivalMode.WALK_IN))
    log.append(TriageStarted(occurred_at=t(2050), patient=p2, staff=NURSE))
    log.append(TriageCompleted(occurred_at=t(2100), patient=p2, esi=EsiAcuity.ESI5))
    log.append(BayRequested(occurred_at=t(2100), patient=p2))
    log.append(BayAssigned(occurred_at=t(2150), patient=p2, bay=BayId("bay-2"), by="baseline"))
    log.append(
        PatientMoved(
            occurred_at=t(2200),
            patient=p2,
            edge=(NodeId("gstat"), NodeId("b2")),
            seconds=seconds(50),
        )
    )
    log.append(
        StaffMoved(
            occurred_at=t(2290),
            staff=PHYSICIAN,
            edge=(NodeId("b1"), NodeId("b2")),
            seconds=seconds(30),
        )
    )
    log.append(ProviderVisitStarted(occurred_at=t(2300), patient=p2, staff=PHYSICIAN))

    return log


def build_bottleneck_log() -> EventLog:
    """Two patients queueing for bays of different zone types.

    Patient ``qa`` waits 60s for a GENERAL bay; patient ``qb`` waits 600s for
    the RESUS_TRAUMA bay — an engineered 10x queue so ``bay:resus_trauma`` is
    unambiguously the binding constraint.
    """
    log = EventLog()
    qa, qb = PatientId("qa"), PatientId("qb")

    log.append(PatientArrived(occurred_at=t(0), patient=qa, mode=ArrivalMode.WALK_IN))
    log.append(BayRequested(occurred_at=t(0), patient=qa))
    log.append(BayAssigned(occurred_at=t(60), patient=qa, bay=BayId("bay-1"), by="baseline"))

    log.append(PatientArrived(occurred_at=t(0), patient=qb, mode=ArrivalMode.WALK_IN))
    log.append(BayRequested(occurred_at=t(0), patient=qb))
    log.append(BayAssigned(occurred_at=t(600), patient=qb, bay=BayId("bay-3"), by="baseline"))

    return log


def full_kpi_values(**overrides: float) -> dict[str, float]:
    """A complete, contract-valid KPI dict (staff_frac_* sum to 1.0), with overrides."""
    values = dict.fromkeys(KPI_KEYS, 0.0)
    values["staff_frac_idle"] = 1.0  # residual makes fractions sum to 1.0
    values.update(overrides)
    return values
