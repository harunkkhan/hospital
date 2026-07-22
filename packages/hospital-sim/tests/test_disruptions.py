"""Disruption injectors — markers, physical effects, exact restore, reroutes."""

from __future__ import annotations

import json

import simpy
from _sim_fixtures import build_physics, ring_layout, tiny_scenario

from hospital.core import (
    BayStatus,
    DisruptionInjected,
    EventLog,
    NodeId,
    PatientId,
    RandomStreams,
    SimTime,
    StaffRole,
    hours,
)
from hospital.data.scenario import DisruptionEvent, DisruptionSpec
from hospital.sim.experiment.disruptions import schedule_disruptions
from hospital.sim.experiment.replication import run_replication
from hospital.sim.flow.staff import staff_process
from hospital.sim.physics.executor import TaskExecutor
from hospital.sim.physics.resources import build_resources
from hospital.sim.physics.world import World


def _markers(log: EventLog, kind: str) -> list[DisruptionInjected]:
    return [
        env.event
        for env in log
        if isinstance(env.event, DisruptionInjected) and env.event.disruption == kind
    ]


class TestSurge:
    def test_surge_adds_arrivals_and_emits_marker(self) -> None:
        base = tiny_scenario()
        surge = base.model_copy(
            update={
                "disruptions": DisruptionSpec(
                    events=(
                        DisruptionEvent(
                            kind="surge",
                            at=SimTime(hours(2).root),
                            duration=hours(2),
                            magnitude=3.0,
                        ),
                    )
                )
            }
        )
        rep_base = run_replication(base, "baseline", 7)
        rep_surge = run_replication(surge, "baseline", 7)
        n_base = rep_base.event_log_jsonl.count('"patient_arrived"')
        n_surge = rep_surge.event_log_jsonl.count('"patient_arrived"')
        assert n_surge > n_base
        log = EventLog.from_jsonl(rep_surge.event_log_jsonl)
        markers = _markers(log, "surge")
        assert len(markers) == 1
        assert markers[0].occurred_at == SimTime(hours(2).root)
        # additive CRN: the base week's arrivals are untouched by the overlay
        base_ids = {
            json.loads(line)["event"]["patient"]
            for line in rep_base.event_log_jsonl.splitlines()
            if json.loads(line)["event"]["kind"] == "patient_arrived"
        }
        surge_ids = {
            json.loads(line)["event"]["patient"]
            for line in rep_surge.event_log_jsonl.splitlines()
            if json.loads(line)["event"]["kind"] == "patient_arrived"
        }
        assert base_ids <= surge_ids


class TestAbsence:
    def test_absence_pauses_the_named_staff_for_the_window(self) -> None:
        h = build_physics()
        nurse = next(m for m in h.roster if m.role is StaffRole.NURSE)
        procs = {
            m.id: h.env.process(
                staff_process(h.env, h.world, h.executor, h.log, m, h.resources.mailboxes[m.id])
            )
            for m in h.roster
        }
        at, duration = SimTime(hours(1).root), hours(1)
        schedule_disruptions(
            h.env,
            h.world,
            h.executor,
            h.streams,
            (
                DisruptionEvent(
                    kind="staff_absence", at=at, duration=duration, target=nurse.id.root
                ),
            ),
            staff_processes=procs,
            event_log=h.log,
        )
        h.env.run(until=hours(3).root)

        markers = _markers(h.log, "staff_absence")
        assert len(markers) == 1
        assert markers[0].detail == nurse.id.root
        # the agent returns to duty at exactly the window end
        idles = [
            env.event.occurred_at.root
            for env in h.log
            if env.event.kind == "staff_idle" and env.event.staff == nurse.id
        ]
        assert idles == [0, (at + duration).root]
        assert h.world.absent_until(nurse.id) is None


class TestClosure:
    def test_node_closure_forces_a_reroute_then_restores(self) -> None:
        layout = ring_layout()
        env = simpy.Environment()
        log = EventLog()
        resources = build_resources(env, layout, ())
        world = World(env, layout, resources, log)
        executor = TaskExecutor(env, world, log)
        src, dst = NodeId("a"), NodeId("c")
        before = world.route(src, dst)

        schedule_disruptions(
            env,
            world,
            executor,
            RandomStreams(0),
            (
                DisruptionEvent(
                    kind="zone_closure",
                    at=SimTime(hours(1).root),
                    duration=hours(1),
                    target="node:b",
                ),
            ),
            staff_processes={},
            event_log=log,
        )
        env.run(until=hours(1).root + 1)  # mid-window
        during = world.route(src, dst)
        assert during != before
        assert NodeId("b") not in during.nodes  # the live reroute avoids the closure
        env.run(until=hours(3).root)  # past the window
        assert world.closed_nodes == frozenset()
        assert world.route(src, dst) == before  # exact restore

    def test_zone_closure_closes_only_free_bays_and_reopens(self) -> None:
        h = build_physics()
        zone = h.layout.bays[0].zone
        zone_bays = [b.id for b in h.layout.bays if b.zone == zone]
        occupied = zone_bays[0]
        h.world.assign_bay(occupied, PatientId("px"))
        schedule_disruptions(
            h.env,
            h.world,
            h.executor,
            h.streams,
            (
                DisruptionEvent(
                    kind="zone_closure",
                    at=SimTime(hours(1).root),
                    duration=hours(1),
                    target=zone.root,
                ),
            ),
            staff_processes={},
            event_log=h.log,
        )
        h.env.run(until=hours(1).root + 1)
        assert h.world.bay_status(occupied) is BayStatus.OCCUPIED  # never yanked
        for bay in zone_bays[1:]:
            assert h.world.bay_status(bay) is BayStatus.CLOSED
        h.env.run(until=hours(3).root)
        for bay in zone_bays[1:]:
            assert h.world.bay_status(bay) is BayStatus.FREE


class TestImagingOutage:
    def test_outage_seizes_capacity_for_the_window_then_releases(self) -> None:
        h = build_physics()
        node = next(iter(sorted(h.resources.imaging, key=lambda n: n.root)))
        resource = h.resources.imaging[node]
        schedule_disruptions(
            h.env,
            h.world,
            h.executor,
            h.streams,
            (
                DisruptionEvent(
                    kind="imaging_outage",
                    at=SimTime(hours(1).root),
                    duration=hours(2),
                    target=node.root,
                ),
            ),
            staff_processes={},
            event_log=h.log,
        )
        h.env.run(until=hours(2).root)  # mid-window
        assert resource.count == resource.capacity  # fully seized
        h.env.run(until=hours(4).root)  # past the window
        assert resource.count == 0  # fully released
        assert len(_markers(h.log, "imaging_outage")) == 1


class TestIdenticalAcrossArms:
    def test_injection_schedule_is_pure_spec_no_rng(self) -> None:
        # two different CRN seeds -> the same markers at the same instants
        logs: list[EventLog] = []
        for seed in (1, 2):
            h = build_physics(seed=seed)
            schedule_disruptions(
                h.env,
                h.world,
                h.executor,
                RandomStreams(seed),
                (
                    DisruptionEvent(
                        kind="staff_absence", at=SimTime(hours(1).root), duration=hours(1)
                    ),
                ),
                staff_processes={},
                event_log=h.log,
            )
            h.env.run(until=hours(2).root)
            logs.append(h.log)
        first = [
            (m.occurred_at, m.disruption, m.detail) for m in _markers(logs[0], "staff_absence")
        ]
        second = [
            (m.occurred_at, m.disruption, m.detail) for m in _markers(logs[1], "staff_absence")
        ]
        assert first == second
