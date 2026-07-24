"""``TaskExecutor`` — no teleport, tier ordering, paired events, zero-time guard."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import simpy
from _sim_fixtures import build_physics, make_patient

from hospital.core import (
    Activity,
    Duration,
    LayoutError,
    NodeId,
    PatientMoved,
    StaffMoved,
    ZeroTimeCycle,
    minutes,
)
from hospital.sim.physics.executor import PriorityTier


class TestWalk:
    def test_no_teleport_one_edge_per_hop(self) -> None:
        h = build_physics()
        staff = h.roster[0]
        src = h.world.staff_at(staff.id)
        dst = h.layout.entrances[-1]
        path = h.world.route(src, dst)
        assert len(path.nodes) >= 3

        h.env.process(h.executor.walk(staff.id, path))
        h.env.run()

        moves = [env.event for env in h.log if isinstance(env.event, StaffMoved)]
        assert len(moves) == len(path.nodes) - 1
        # each hop is exactly one edge, in path order
        hops = list(zip(path.nodes, path.nodes[1:], strict=False))
        assert [m.edge for m in moves] == hops
        # graph-walk integrity: emitted edge seconds sum to the dijkstra total
        assert sum(m.seconds.root for m in moves) == path.total.root
        # the clock advanced by exactly the traversal total
        assert int(h.env.now) == path.total.root
        # every intermediate event was stamped at its own hop-arrival instant
        arrival = 0
        for m, (u, v) in zip(moves, hops, strict=True):
            arrival += h.world.edge_seconds(u, v).root
            assert m.occurred_at.root == arrival
        assert h.world.staff_at(staff.id) == dst

    def test_escorted_walk_moves_patient_and_staff_together(self) -> None:
        h = build_physics()
        staff = h.roster[0]
        p = make_patient("p1")
        h.world.register_patient(p)
        start = h.world.staff_at(staff.id)
        h.world.set_patient_position(p.id, start)
        dst = h.layout.entrances[0]
        path = h.world.route(start, dst)

        h.env.process(h.executor.walk(p.id, path, escort=staff.id))
        h.env.run()

        patient_moves = [e.event for e in h.log if isinstance(e.event, PatientMoved)]
        staff_moves = [e.event for e in h.log if isinstance(e.event, StaffMoved)]
        assert len(patient_moves) == len(staff_moves) == len(path.nodes) - 1
        assert h.world.patient_at(p.id) == dst
        assert h.world.staff_at(staff.id) == dst

    def test_walk_from_wrong_position_is_refused(self) -> None:
        h = build_physics()
        staff = h.roster[0]
        elsewhere = h.layout.entrances[0]
        assert h.world.staff_at(staff.id) != elsewhere
        path = h.world.route(elsewhere, h.layout.entrances[-1])

        def proc() -> Generator[simpy.Event, object]:
            yield from h.executor.walk(staff.id, path)

        h.env.process(proc())
        with pytest.raises(ValueError, match="walk path starts at"):
            h.env.run()


class TestTierOrdering:
    def test_same_instant_completion_before_decision_before_disruption(self) -> None:
        h = build_physics()
        fired: list[str] = []

        def waiter(tier: PriorityTier, name: str) -> Generator[simpy.Event, object]:
            yield h.executor.delay(minutes(1), tier)
            fired.append(name)

        # deliberately create in reverse tier order so eid cannot save us
        h.env.process(waiter(PriorityTier.DISRUPTION, "disruption"))
        h.env.process(waiter(PriorityTier.DECISION, "decision"))
        h.env.process(waiter(PriorityTier.COMPLETION, "completion"))
        h.env.run()
        assert fired == ["completion", "decision", "disruption"]

    def test_within_tier_fifo_by_creation_order(self) -> None:
        h = build_physics()
        fired: list[str] = []

        def waiter(name: str) -> Generator[simpy.Event, object]:
            yield h.executor.delay(minutes(1), PriorityTier.COMPLETION)
            fired.append(name)

        h.env.process(waiter("first"))
        h.env.process(waiter("second"))
        h.env.run()
        assert fired == ["first", "second"]


class TestRunService:
    def test_started_completed_pair_carries_caused_by(self) -> None:
        h = build_physics()
        staff = h.roster[0]
        p = make_patient("p1")
        h.world.register_patient(p)

        def proc() -> Generator[simpy.Event, object]:
            yield from h.executor.run_service(
                Activity.PROVIDER_VISIT,
                duration=minutes(10),
                patient=p.id,
                staff=staff.id,
            )

        h.env.process(proc())
        h.env.run()
        envelopes = list(h.log)
        assert len(envelopes) == 2
        started, completed = envelopes
        assert started.event.kind == "provider_visit_started"
        assert completed.event.kind == "provider_visit_completed"
        assert completed.caused_by == started.sequence
        assert completed.event.occurred_at - started.event.occurred_at == minutes(10)

    def test_unsupported_activity_is_refused(self) -> None:
        h = build_physics()

        def proc() -> Generator[simpy.Event, object]:
            yield from h.executor.run_service(
                Activity.IMAGING,
                duration=minutes(10),
                patient=make_patient("p1").id,
                staff=h.roster[0].id,
            )

        h.env.process(proc())
        with pytest.raises(ValueError, match="no started/completed event pair"):
            h.env.run()


class TestZeroTimeGuard:
    def test_unbounded_zero_delays_at_one_instant_raise(self) -> None:
        h = build_physics()

        def spinner() -> Generator[simpy.Event, object]:
            while True:
                yield h.executor.delay(Duration(0), PriorityTier.COMPLETION)

        h.env.process(spinner())
        with pytest.raises(ZeroTimeCycle):
            h.env.run(until=1)

    def test_negative_delay_is_refused(self) -> None:
        h = build_physics()
        with pytest.raises(ValueError, match="negative delay"):
            h.executor.delay(Duration(-1), PriorityTier.COMPLETION)


class TestAcquire:
    def test_priority_queue_serves_most_critical_first(self) -> None:
        h = build_physics()
        node = next(iter(h.resources.imaging))
        res = h.resources.imaging[node]
        served: list[str] = []

        def holder() -> Generator[simpy.Event, object]:
            req = yield from h.executor.acquire(res, priority=0)
            yield h.executor.delay(minutes(5), PriorityTier.COMPLETION)
            res.release(req)

        def queued(name: str, priority: int) -> Generator[simpy.Event, object]:
            req = yield from h.executor.acquire(res, priority=priority)
            served.append(name)
            res.release(req)

        h.env.process(holder())

        def enqueue_later() -> Generator[simpy.Event, object]:
            yield h.executor.delay(minutes(1), PriorityTier.COMPLETION)
            # ESI5 asks first, ESI1 second — priority must win over arrival order
            h.env.process(queued("esi5", -1))
            h.env.process(queued("esi1", -5))

        h.env.process(enqueue_later())
        h.env.run()
        assert served == ["esi1", "esi5"]


def test_edge_seconds_unknown_hop_raises() -> None:
    h = build_physics()
    with pytest.raises(LayoutError):
        h.world.edge_seconds(NodeId("nope"), NodeId("also_nope"))
