"""Importable test builders for ``hospital-sim`` (no ``conftest``; DECISIONS D10).

Shared by every test module in this package so scenario/harness construction is
never copy-pasted (doc 00 §5.10). The tiny scenario is a real, runnable ER —
a handful of bays, one imaging suite, one lab, two triage rooms, a small
roster — sized so a full ``run_replication`` finishes in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import simpy

from hospital.core import (
    CompatibilityRule,
    CompiledRules,
    Duration,
    EsiAcuity,
    EventLog,
    FloorLayout,
    NodeId,
    OperatingWeek,
    Patient,
    PatientId,
    RandomStreams,
    Rule,
    SimTime,
    StaffMember,
    StaffRole,
    TimeWindow,
    WorkupNeeds,
    ZoneType,
    compile_rules,
    hours,
    seconds,
)
from hospital.core.enums import ArrivalMode
from hospital.data.layout import generate_floor
from hospital.data.scenario import (
    FacilitySpec,
    Scenario,
    StaffingSpec,
    WorkloadSpec,
    WorkupProfile,
    ZoneQuota,
    realize_staff,
)
from hospital.sim.physics.executor import TaskExecutor
from hospital.sim.physics.resources import ResourcePool, build_resources
from hospital.sim.physics.service_times import ServiceTimes, default_service_table
from hospital.sim.physics.world import World


def tiny_facility() -> FacilitySpec:
    """A small but fully-featured floor: 8 bays across three zones + specials."""
    return FacilitySpec(
        target_area_sqft=30_000,
        zones=(
            ZoneQuota(zone_type=ZoneType.GENERAL, bays=4, isolation_bays=1),
            ZoneQuota(zone_type=ZoneType.RESUS_TRAUMA, bays=2, isolation_bays=2),
            ZoneQuota(zone_type=ZoneType.FAST_TRACK, bays=2),
        ),
        imaging_suites=1,
        lab_stations=1,
        triage_rooms=2,
    )


def tiny_workload(*, rate_per_hour: float = 2.0, horizon_hours: int = 8) -> WorkloadSpec:
    return WorkloadSpec(
        horizon=OperatingWeek(start=SimTime(0), end=SimTime(hours(horizon_hours).root)),
        base_rate_per_hour=rate_per_hour,
        hourly_profile=tuple([1.0] * 24),
        dow_profile=tuple([1.0] * 7),
        esi_mix={
            EsiAcuity.ESI1: 0.05,
            EsiAcuity.ESI2: 0.20,
            EsiAcuity.ESI3: 0.45,
            EsiAcuity.ESI4: 0.20,
            EsiAcuity.ESI5: 0.10,
        },
        complaint_mix={"chest_pain": 1.0},
        ambulance_fraction=0.2,
        isolation_fraction=0.1,
        workups={
            "chest_pain": WorkupProfile(
                provider_visits_mean=1.2,
                nurse_visits_mean=1.0,
                imaging_prob={ZoneType.IMAGING: 0.3},
                labs_mean=0.5,
                procedure_prob=0.0,
            )
        },
    )


def tiny_staffing() -> StaffingSpec:
    return StaffingSpec(
        default_counts={
            StaffRole.PHYSICIAN: 2,
            StaffRole.NURSE: 3,
            StaffRole.TECH: 1,
            StaffRole.PORTER: 2,
            StaffRole.HOUSEKEEPING: 1,
        }
    )


def tiny_scenario(*, seed: int = 7, rate_per_hour: float = 2.0, horizon_hours: int = 8) -> Scenario:
    return Scenario(
        name="tiny_er",
        seed=seed,
        facility=tiny_facility(),
        workload=tiny_workload(rate_per_hour=rate_per_hour, horizon_hours=horizon_hours),
        staffing=tiny_staffing(),
    )


def tiny_rules() -> CompiledRules:
    """A small compat kernel matching the tiny floor's zones."""
    rules: tuple[Rule, ...] = (
        CompatibilityRule(
            allowed_zone_types=frozenset(
                {
                    (EsiAcuity.ESI1, ZoneType.RESUS_TRAUMA),
                    (EsiAcuity.ESI2, ZoneType.RESUS_TRAUMA),
                    (EsiAcuity.ESI2, ZoneType.GENERAL),
                    (EsiAcuity.ESI3, ZoneType.GENERAL),
                    (EsiAcuity.ESI3, ZoneType.FAST_TRACK),
                    (EsiAcuity.ESI4, ZoneType.FAST_TRACK),
                    (EsiAcuity.ESI4, ZoneType.GENERAL),
                    (EsiAcuity.ESI5, ZoneType.FAST_TRACK),
                    (EsiAcuity.ESI5, ZoneType.GENERAL),
                }
            ),
        ),
    )
    return compile_rules(rules)


def make_patient(
    pid: str,
    *,
    esi: EsiAcuity = EsiAcuity.ESI3,
    arrival_s: float = 0.0,
    complaint: str = "chest_pain",
    isolation: bool = False,
    mode: ArrivalMode = ArrivalMode.WALK_IN,
    provider_visits: int = 1,
    nurse_visits: int = 0,
    imaging: tuple[ZoneType, ...] = (),
    labs: int = 0,
) -> Patient:
    return Patient(
        id=PatientId(pid),
        arrival_time=SimTime(seconds(arrival_s).root),
        arrival_mode=mode,
        esi=esi,
        complaint=complaint,
        isolation_required=isolation,
        workup=WorkupNeeds(
            provider_visits=provider_visits,
            nurse_visits=nurse_visits,
            imaging=imaging,
            labs=labs,
            procedures=0,
        ),
    )


def ring_layout() -> FloorLayout:
    """A 4-node ring (two equal-cost routes) — the reroute/mask test surface.

    ``generate_floor`` builds a *tree* (spine + leaf stubs), where any block
    disconnects the graph; mask tests need an alternate path, so this layout is
    hand-built. Route ``a -> c`` ties between ``a-b-c`` and ``a-d-c``; core's
    deterministic tie-break picks ``b`` (lower node id).
    """
    from hospital.core import Distance, RouteEdge, RouteGraph, RouteNode, WalkSpeed, walk_duration

    speed = WalkSpeed(100)

    def node(nid: str, x: int, y: int, label: str = "corridor") -> RouteNode:
        return RouteNode(id=NodeId(nid), label=label, x_cm=x, y_cm=y)

    def edge(a: str, b: str, dist_cm: int) -> RouteEdge:
        d = Distance(dist_cm)
        return RouteEdge(a=NodeId(a), b=NodeId(b), distance=d, seconds=walk_duration(d, speed))

    graph = RouteGraph(
        nodes=(
            node("a", 0, 0),
            node("b", 100, 0),
            node("c", 200, 0),
            node("d", 100, 100),
            node("t", 0, 100, label="triage"),
        ),
        edges=(
            edge("a", "b", 1000),
            edge("b", "c", 1000),
            edge("a", "d", 1000),
            edge("d", "c", 1000),
            edge("a", "t", 500),
        ),
    )
    return FloorLayout(
        graph=graph,
        zones=(),
        bays=(),
        stations=(NodeId("a"),),
        entrances=(NodeId("a"), NodeId("c")),
        imaging_nodes=(),
        lab_nodes=(),
    )


@dataclass(frozen=True)
class RingHarness:
    env: simpy.Environment
    layout: FloorLayout
    world: World


def ring_world() -> RingHarness:
    layout = ring_layout()
    env = simpy.Environment()
    log = EventLog()
    resources = build_resources(env, layout, ())
    world = World(env, layout, resources, log)
    return RingHarness(env=env, layout=layout, world=world)


@dataclass(frozen=True)
class PhysicsHarness:
    """Everything a physics-level test needs, wired once (no policies, no ticks)."""

    env: simpy.Environment
    layout: FloorLayout
    log: EventLog
    resources: ResourcePool
    world: World
    executor: TaskExecutor
    streams: RandomStreams
    service_times: ServiceTimes
    roster: tuple[StaffMember, ...]


def build_physics(*, seed: int = 7, horizon_hours: int = 8) -> PhysicsHarness:
    """Construct the physics stack for the tiny floor (mirrors the composition root)."""
    scenario = tiny_scenario(seed=seed, horizon_hours=horizon_hours)
    layout = generate_floor(scenario.facility)
    window = TimeWindow(start=SimTime(0), end=SimTime(hours(horizon_hours).root))
    roster = realize_staff(scenario.staffing, layout, window)
    env = simpy.Environment()
    log = EventLog()
    resources = build_resources(env, layout, roster)
    world = World(env, layout, resources, log)
    for member in roster:
        world.register_staff(member)
    streams = RandomStreams(seed)
    service_times = ServiceTimes(streams, default_service_table())
    executor = TaskExecutor(env, world, log)
    return PhysicsHarness(
        env=env,
        layout=layout,
        log=log,
        resources=resources,
        world=world,
        executor=executor,
        streams=streams,
        service_times=service_times,
        roster=roster,
    )


def total_seconds(*durations: Duration) -> int:
    return sum(d.root for d in durations)
