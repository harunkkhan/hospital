"""``run_replication`` — the composition root (doc 04 §3.12 / §4.4).

The ONLY place components are constructed and wired: exactly one
``RandomStreams(seed)`` (the single CRN lineage), one integer-µs
``simpy.Environment``, one ``EventLog``, one ``World``, one oracle, one
compiled-rules kernel. A component constructing its own streams or clock would
fork determinism.

The horizon is ONE continuous week (``scenario.workload.horizon``): queues,
occupied bays, in-flight patients, and staff positions carry across day
boundaries because it is one ``Environment`` — there is no per-day reset.
``env.run(until=end)`` never executes an event at exactly ``end``, which *is*
the half-open ``[start, end)`` convention: a patient process still alive at the
cutoff is WIP, never a completion, and conservation
``arrivals == completions + wip`` is the property-test tripwire.

Workload comes from ``data.workload.generate_workload`` — ``sim`` has no
arrivals generator of its own (the cardinal anti-dup rule), and surge arrivals
ride the same call so both arms face identical weather.

The decision tick (doc 04 §4.1): flow steps call ``world.request_decision()``;
coalesced ``DECISION``-tier callbacks run :func:`_make_tick`'s closure — build
the ``DecisionInput`` (with the events accumulated since the previous tick),
ask the ``PolicySet``, validate-then-apply. An ``InfeasiblePlan`` (stale or
buggy plan) applies nothing and requests a fresh solve against the now-current
state; the per-instant tick bound turns a non-converging retry loop into a
``ZeroTimeCycle`` instead of a hang.

``Replication`` carries the byte-stable JSONL (folding re-parses it, so the
fold sees exactly the persisted bytes) plus the ``Scenario`` itself, so
``fold_scorecard`` can rebuild the layout/roster deterministically without a
side channel (deviation from the doc's field list, noted in the build report).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import simpy

from hospital.core import (
    CompatibilityRule,
    CompiledRules,
    Duration,
    EsiAcuity,
    Event,
    EventEnvelope,
    EventLog,
    FrozenModel,
    InfeasiblePlan,
    OperatingWeek,
    PatientId,
    RandomStreams,
    RiskMonitor,
    Rule,
    RunId,
    TimeWindow,
    ZoneType,
    compile_rules,
)
from hospital.data.hospital import generate_hospital
from hospital.data.scenario import Scenario, realize_staff
from hospital.data.workload import generate_workload
from hospital.sim.experiment.disruptions import schedule_disruptions
from hospital.sim.flow.patient import patient_process
from hospital.sim.flow.staff import staff_process
from hospital.sim.flow.vitals import VitalsWatch, vitals_process
from hospital.sim.physics.executor import PriorityTier, TaskExecutor
from hospital.sim.physics.resources import build_resources
from hospital.sim.physics.service_times import ServiceTimes, default_service_table
from hospital.sim.physics.world import World
from hospital.sim.policies.factory import Arm, make_policies
from hospital.sim.policies.optimized import SolverPlacement
from hospital.sim.seam_adapter import apply_plan, build_decision_input, validation_context
from hospital.solver import GraphRoutingOracle, ObjectiveConfig, SolverStatus, config_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from hospital.data.workload import PatientArrival
    from hospital.sim.policies.protocols import PolicySet
    from hospital.solver import RoutingOracle

# Acuity-weighted patient time must dominate staff travel (PLAN §5.1): at 1:1 the
# solver trades seconds of door-to-provider for staff-walk savings (M1 stress
# fit). Two experiment-level tunings of the one weight set:
#
# * ``unplaced_wait_penalty`` — the deferral horizon dispatch prices an
#   unserved task at (placement's use of it is a lexicographic reward scale,
#   invariant to its magnitude). 150 s sits just above the stressed floor's
#   maximum walk (~94 s): far tasks stay protectable, while the travel an
#   acuity step can buy stays on the scale of real walk gaps. At 200-1000 s
#   every acuity gap dominated all travel — tier-like matching, and the
#   door-to-provider regression of the lexicographic formulation returned.
# * ``acuity_urgency`` — convex at the top, compressed at the bottom.
#   ``priority_weight``'s linear curve couples "ESI-1 always outbids a nearby
#   task" (needs a large urgency-x-horizon product) to "every ESI step
#   dominates all travel" (starves the ESI-3/4/5 majority and re-creates the
#   baseline service order, destroying the walking savings). ESI-1 is priced
#   as categorical (resuscitation); the ambulatory classes are compressed so
#   travel decides among them.
DEFAULT_OBJECTIVE = ObjectiveConfig(
    w_time=10,
    w_travel=1,
    unplaced_wait_penalty=150,
    acuity_urgency=(
        (EsiAcuity.ESI1, 12),
        (EsiAcuity.ESI2, 3),
        (EsiAcuity.ESI3, 2),
        (EsiAcuity.ESI4, 2),
        (EsiAcuity.ESI5, 2),
    ),
)


class Replication(FrozenModel):
    """One finished run: the byte-stable log plus everything needed to fold it.

    ``solver_status`` is the optimized arm's worst-observed placement solve
    claim over the whole run (``None`` for the baseline arm): a week that ever
    fell back below OPTIMAL is distinguishable downstream (``Scorecard.status``,
    CLI/comparison) — recorded, never hidden (PLAN §5).
    """

    run_id: RunId
    arm: Arm
    seed: int
    scenario: Scenario
    event_log_jsonl: str
    objective_hash: str
    horizon: OperatingWeek
    solver_status: SolverStatus | None = None


class _TapEventLog(EventLog):
    """The run's ``EventLog`` plus a drainable since-last-tick buffer.

    ``DecisionInput.events_since`` must carry exactly the envelopes appended
    since the previous decision tick; re-slicing the whole log every tick would
    be quadratic over a week, so the sim-side writer taps its own appends. The
    serialized bytes are untouched — ``to_jsonl`` is inherited verbatim.
    """

    def __init__(self) -> None:
        super().__init__()
        self._since_tick: list[EventEnvelope] = []

    def append(self, e: Event, *, caused_by: int | None = None) -> int:
        sequence = super().append(e, caused_by=caused_by)
        self._since_tick.append(EventEnvelope(event=e, sequence=sequence, caused_by=caused_by))
        return sequence

    def drain_since_tick(self) -> tuple[EventEnvelope, ...]:
        out = tuple(self._since_tick)
        self._since_tick.clear()
        return out


# The clinically-sane compatibility whitelist used when a scenario declares no
# rules of its own (the reference er_floor.yaml ships ``rules: []``, and the
# validator's whitelist semantics read an EMPTY compat rule as "may go
# nowhere"). A scenario with explicit rules always wins.
_DEFAULT_ESI_ZONES: dict[EsiAcuity, tuple[ZoneType, ...]] = {
    EsiAcuity.ESI1: (ZoneType.RESUS_TRAUMA,),
    EsiAcuity.ESI2: (ZoneType.RESUS_TRAUMA, ZoneType.GENERAL, ZoneType.OBSERVATION),
    EsiAcuity.ESI3: (ZoneType.GENERAL, ZoneType.OBSERVATION, ZoneType.FAST_TRACK),
    EsiAcuity.ESI4: (ZoneType.FAST_TRACK, ZoneType.GENERAL, ZoneType.OBSERVATION),
    EsiAcuity.ESI5: (ZoneType.FAST_TRACK, ZoneType.GENERAL, ZoneType.OBSERVATION),
}


def default_rules() -> tuple[Rule, ...]:
    """The fallback rule set for scenarios that carry none (judgment call, in report)."""
    allowed = frozenset(
        (esi, zone_type)
        for esi, zone_types in _DEFAULT_ESI_ZONES.items()
        for zone_type in zone_types
    )
    return (CompatibilityRule(allowed_zone_types=allowed),)


def _make_tick(
    world: World,
    oracle: RoutingOracle,
    policies: PolicySet,
    rules: CompiledRules,
    executor: TaskExecutor,
    log: _TapEventLog,
) -> Callable[[], None]:
    def tick() -> None:
        di = build_decision_input(world, world.now(), log.drain_since_tick())
        response = policies.decide(di, oracle)
        if response.mode == "replace" and response.plan is not None:
            ctx = validation_context(world, rules)
            try:
                apply_plan(world, response.plan, ctx, executor, log, origin=policies.origin)
            except InfeasiblePlan:
                # reject-then-re-solve: nothing was applied; ask again against
                # the now-current state (the per-instant tick bound guards a
                # deterministic non-converging loop with ZeroTimeCycle).
                world.request_decision()
        if response.wake.kind == "schedule" and response.wake.at is not None:
            world.schedule_decision_at(response.wake.at)

    return tick


def _spawn_arrivals(
    env: simpy.Environment,
    world: World,
    executor: TaskExecutor,
    log: EventLog,
    arrivals: tuple[PatientArrival, ...],
    service_times: ServiceTimes,
    streams: RandomStreams,
    watch: VitalsWatch | None = None,
    monitor: RiskMonitor | None = None,
) -> Generator[simpy.Event, object]:
    """Start each patient process at its arrival instant, in workload order.

    A vitals process is started beside it only when ``watch`` is set and the
    patient is acute enough to be monitored. Without a watch nothing extra is
    scheduled and no ``VitalsSampled`` is written, so the run stays byte-identical
    to the M1/M2 engine.
    """
    for arrival in arrivals:
        dt = arrival.patient.arrival_time.root - int(env.now)
        if dt > 0:
            yield executor.delay(Duration(dt), PriorityTier.COMPLETION)
        stay = env.process(
            patient_process(
                env,
                world,
                executor,
                log,
                arrival.patient,
                service_times=service_times,
                streams=streams,
            )
        )
        if watch is not None and int(arrival.patient.esi) <= int(watch.monitor_at_or_above):
            env.process(
                vitals_process(
                    env,
                    world,
                    executor,
                    log,
                    arrival.patient,
                    streams=streams,
                    watch=watch,
                    stay=stay,
                    monitor=monitor,
                )
            )


def run_replication(
    scenario: Scenario,
    arm: Arm,
    seed: int,
    *,
    objective: ObjectiveConfig = DEFAULT_OBJECTIVE,
    watch: VitalsWatch | None = None,
    monitor: RiskMonitor | None = None,
    expected_stay: Mapping[PatientId, Duration] | None = None,
) -> Replication:
    """Run one full horizon of ``scenario`` under ``arm`` — deterministic in ``seed``.

    ``objective`` is the ONE weight set for the run: it drives the optimized
    arm's decisions (through ``make_policies``) AND is the config whose hash is
    recorded on the ``Replication``. A caller that scores the run under some
    objective must pass that same objective here — otherwise the reported
    ``objective_hash`` would describe weights that never drove a decision.
    """
    # 1-2: the single CRN source; randomness-free floor construction
    streams = RandomStreams(seed)
    # One code path for one floor or ten: `generate_hospital` over a scenario with no
    # upper floors returns exactly what `generate_floor` does, ids included.
    layout = generate_hospital(scenario.hospital())
    horizon = scenario.workload.horizon

    # 3-5: one clock, one log, resources, the one mutable World, physics wiring
    env = simpy.Environment()
    log = _TapEventLog()
    window = TimeWindow(start=horizon.start, end=horizon.end)
    roster = realize_staff(scenario.staffing, layout, window)
    resources = build_resources(env, layout, roster)
    world = World(env, layout, resources, log)
    for member in roster:
        world.register_staff(member)
    service_times = ServiceTimes(streams, default_service_table())
    executor = TaskExecutor(env, world, log)

    # 6-8: oracle + the caller's objective + compiled rules + the chosen arm
    oracle = GraphRoutingOracle(layout.graph)
    rules = compile_rules(scenario.rules if scenario.rules else default_rules())
    # `expected_stay` is the prediction port (doc 06 §3). `None` -> the arm decides
    # exactly as it did before predictions existed, which is what keeps the M1 goldens
    # a check rather than a re-baseline. Only the optimized arm consumes it.
    policies = make_policies(
        arm,
        oracle=oracle,
        rules=rules,
        roster=roster,
        objective=objective,
        expected_stay=expected_stay,
    )
    world.set_decision_hook(_make_tick(world, oracle, policies, rules, executor, log))

    # 9-11: the one workload generator (surges included), agents, disruptions
    arrivals = generate_workload(scenario.workload, streams, disruptions=scenario.disruptions)
    env.process(
        _spawn_arrivals(
            env,
            world,
            executor,
            log,
            arrivals,
            service_times,
            streams,
            watch=watch,
            monitor=monitor,
        )
    )
    staff_processes = {
        member.id: env.process(
            staff_process(env, world, executor, log, member, resources.mailboxes[member.id])
        )
        for member in roster
    }
    schedule_disruptions(
        env,
        world,
        executor,
        streams,
        scenario.disruptions.events,
        staff_processes=staff_processes,
        event_log=log,
    )

    # 12-13: one continuous horizon, half-open [start, end); return the record
    env.run(until=horizon.end.root)
    placement = policies.placement
    solver_status = placement.worst_status if isinstance(placement, SolverPlacement) else None
    return Replication(
        run_id=RunId(f"{scenario.name}-{arm}-{seed}"),
        arm=arm,
        seed=seed,
        scenario=scenario,
        event_log_jsonl=log.to_jsonl(),
        objective_hash=config_hash(objective),
        horizon=horizon,
        solver_status=solver_status,
    )


__all__ = ["Replication", "default_rules", "run_replication"]
