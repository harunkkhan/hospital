"""``World`` — sole mutator, mask rerouting, queue discipline, tick coalescing."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import simpy
from _sim_fixtures import build_physics, make_patient, ring_world, tiny_rules

from hospital.core import (
    BayId,
    BayStatus,
    EsiAcuity,
    LayoutError,
    NodeId,
    PatientId,
    UnknownEntity,
    ZeroTimeCycle,
)
from hospital.sim.physics.world import World


def _first_free_bay(world: World) -> BayId:
    return world.layout.bays[0].id


class TestRouting:
    def test_route_delegates_to_core_dijkstra(self) -> None:
        h = build_physics()
        src = h.layout.entrances[0]
        dst = h.layout.bays[0].node
        direct = h.layout.graph.dijkstra(src, dst)
        assert h.world.route(src, dst) == direct

    def test_block_edge_forces_reroute_and_bumps_memo(self) -> None:
        r = ring_world()
        src, dst = r.layout.entrances[0], r.layout.entrances[1]
        base = r.world.route(src, dst)
        assert [n.root for n in base.nodes] == ["a", "b", "c"]
        u, v = base.nodes[0], base.nodes[1]
        r.world.block_edge(u, v)
        rerouted = r.world.route(src, dst)
        hops = list(zip(rerouted.nodes, rerouted.nodes[1:], strict=False))
        assert (u, v) not in hops
        assert (v, u) not in hops  # a bidirectional block closes both orientations
        # graph-walk integrity: the path total is the sum of its edge seconds
        assert rerouted.total.root == sum(r.world.edge_seconds(a, b).root for a, b in hops)
        r.world.unblock_edge(u, v)
        assert r.world.route(src, dst) == base

    def test_close_node_is_never_traversed(self) -> None:
        r = ring_world()
        src, dst = r.layout.entrances[0], r.layout.entrances[1]
        base = r.world.route(src, dst)
        blocked = base.nodes[1]
        r.world.close_node(blocked)
        rerouted = r.world.route(src, dst)
        assert blocked not in rerouted.nodes
        assert [n.root for n in rerouted.nodes] == ["a", "d", "c"]

    def test_route_allows_egress_from_a_closed_src(self) -> None:
        # Finding 4: an actor standing ON a closed node must still be routable
        # out; only entering/traversing a closure is forbidden (nuance 4.1).
        r = ring_world()
        r.world.close_node(NodeId("a"))
        path = r.world.route(NodeId("a"), NodeId("c"))
        assert path.nodes[0] == NodeId("a")
        assert r.world.try_route(NodeId("c"), NodeId("a")) is None  # no ingress

    def test_try_route_distinguishes_closures_from_layout_bugs(self) -> None:
        # Finding 4: closure-severed routes are recoverable (None -> wait);
        # a genuinely broken route still raises even while masks are active.
        r = ring_world()
        r.world.close_node(NodeId("b"))
        r.world.close_node(NodeId("d"))
        assert r.world.try_route(NodeId("a"), NodeId("c")) is None
        with pytest.raises(LayoutError):
            r.world.try_route(NodeId("a"), NodeId("ghost"))

    def test_await_route_waits_for_a_reopening(self) -> None:
        # Finding 4: an actor needing a closed/severed destination parks until
        # the closure lifts, then routes — the run never sees a LayoutError.
        r = ring_world()
        r.world.close_node(NodeId("b"))
        r.world.close_node(NodeId("d"))
        resumed: list[tuple[int, tuple[str, ...]]] = []

        def traveler() -> Generator[simpy.Event, object]:
            path = yield from r.world.await_route(NodeId("a"), NodeId("c"))
            resumed.append((int(r.env.now), tuple(n.root for n in path.nodes)))

        def reopener() -> Generator[simpy.Event, object]:
            yield r.env.timeout(10)
            r.world.open_node(NodeId("b"))

        r.env.process(traveler())
        r.env.process(reopener())
        r.env.run(until=100)
        assert resumed == [(10, ("a", "b", "c"))]

    def test_overlapping_closures_are_refcounted(self) -> None:
        # Finding 10: the first window's reopen must not undo the second's.
        r = ring_world()
        r.world.close_node(NodeId("b"))
        r.world.close_node(NodeId("b"))
        r.world.open_node(NodeId("b"))
        assert NodeId("b") in r.world.closed_nodes
        r.world.open_node(NodeId("b"))
        assert r.world.closed_nodes == frozenset()
        u, v = NodeId("a"), NodeId("b")
        r.world.block_edge(u, v)
        r.world.block_edge(u, v)
        r.world.unblock_edge(u, v)
        assert (u, v) in r.world.blocked_edges
        r.world.unblock_edge(u, v)
        assert r.world.blocked_edges == frozenset()


class TestBayLifecycle:
    def test_four_state_machine(self) -> None:
        h = build_physics()
        bay = _first_free_bay(h.world)
        pid = PatientId("p1")
        assert h.world.bay_status(bay) is BayStatus.FREE
        h.world.assign_bay(bay, pid)
        assert h.world.bay_status(bay) is BayStatus.OCCUPIED
        assert h.world.occupant(bay) == pid
        h.world.vacate_bay(bay)
        assert h.world.bay_status(bay) is BayStatus.CLEANING
        assert h.world.occupant(bay) is None
        h.world.free_bay(bay)
        assert h.world.bay_status(bay) is BayStatus.FREE

    def test_transitions_have_exactly_one_path(self) -> None:
        h = build_physics()
        bay = _first_free_bay(h.world)
        with pytest.raises(ValueError, match="cannot vacate"):
            h.world.vacate_bay(bay)
        with pytest.raises(ValueError, match="cannot free"):
            h.world.free_bay(bay)
        h.world.assign_bay(bay, PatientId("p1"))
        with pytest.raises(ValueError, match="cannot assign"):
            h.world.assign_bay(bay, PatientId("p2"))

    def test_close_and_reopen_restores_prior(self) -> None:
        h = build_physics()
        bay = _first_free_bay(h.world)
        h.world.close_bay(bay)
        assert h.world.bay_status(bay) is BayStatus.CLOSED
        h.world.reopen_bay(bay)
        assert h.world.bay_status(bay) is BayStatus.FREE
        h.world.assign_bay(bay, PatientId("p1"))
        with pytest.raises(ValueError, match="cannot close occupied"):
            h.world.close_bay(bay)

    def test_unknown_bay_raises(self) -> None:
        h = build_physics()
        with pytest.raises(UnknownEntity):
            h.world.bay_status(BayId("nope"))


class TestBayQueue:
    def test_acuity_tiers_then_fifo(self) -> None:
        h = build_physics()
        late_critical = make_patient("p_esi1", esi=EsiAcuity.ESI1, arrival_s=100.0)
        early_routine = make_patient("p_esi3_a", esi=EsiAcuity.ESI3, arrival_s=0.0)
        later_routine = make_patient("p_esi3_b", esi=EsiAcuity.ESI3, arrival_s=50.0)
        for p in (early_routine, later_routine, late_critical):
            h.world.register_patient(p)
            h.world.request_bay(p, stage="triage->bay")
        order = [w.patient.id.root for w in h.world.waiting_for_bay()]
        assert order == ["p_esi1", "p_esi3_a", "p_esi3_b"]

    def test_grant_bay_succeeds_the_wake_event(self) -> None:
        h = build_physics()
        p = make_patient("p1")
        h.world.register_patient(p)
        bay = _first_free_bay(h.world)
        got: list[object] = []

        def proc() -> Generator[simpy.Event, object]:
            wake = h.world.request_bay(p, stage="triage->bay")
            granted = yield wake
            got.append(granted)

        h.env.process(proc())
        h.env.run(until=1)
        h.world.grant_bay(p.id, bay)
        h.env.run(until=2)
        assert got == [bay]
        assert h.world.occupant(bay) == p.id
        assert h.world.waiting_for_bay() == ()

    def test_grant_for_non_waiting_patient_raises(self) -> None:
        h = build_physics()
        with pytest.raises(UnknownEntity):
            h.world.grant_bay(PatientId("ghost"), _first_free_bay(h.world))

    def test_resequence_overrides_within_tier(self) -> None:
        h = build_physics()
        a = make_patient("pa", arrival_s=0.0)
        b = make_patient("pb", arrival_s=10.0)
        for p in (a, b):
            h.world.register_patient(p)
            h.world.request_bay(p, stage="triage->bay")
        h.world.resequence_waiting(("pb", "pa"))
        order = [w.patient.id.root for w in h.world.waiting_for_bay()]
        assert order == ["pb", "pa"]


class TestCompatibility:
    def test_free_compatible_bays_filters_zone_and_isolation(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        esi1 = make_patient("p1", esi=EsiAcuity.ESI1)
        bays = h.world.free_compatible_bays(esi1, rules)
        assert bays
        assert all(h.world.bay(b).zone_type.value == "resus_trauma" for b in bays)

        iso = make_patient("p2", esi=EsiAcuity.ESI3, isolation=True)
        assert all(
            h.world.bay(b).isolation_capable for b in h.world.free_compatible_bays(iso, rules)
        )

        # occupied bays drop out
        general = h.world.free_compatible_bays(make_patient("p3"), rules)
        h.world.assign_bay(general[0], PatientId("px"))
        assert general[0] not in h.world.free_compatible_bays(make_patient("p4"), rules)


class TestStaff:
    def test_positions_are_world_owned(self) -> None:
        h = build_physics()
        member = h.roster[0]
        assert h.world.staff_at(member.id) == member.home_station
        target = h.layout.entrances[0]
        h.world.set_staff_position(member.id, target)
        assert h.world.staff_at(member.id) == target


class TestDecisionTicks:
    def test_simultaneous_triggers_coalesce_to_one_tick(self) -> None:
        h = build_physics()
        calls: list[int] = []
        h.world.set_decision_hook(lambda: calls.append(int(h.env.now)))
        h.world.request_decision()
        h.world.request_decision()
        h.world.request_decision()
        h.env.run(until=1)
        assert calls == [0]

    def test_re_request_after_tick_runs_again_same_instant(self) -> None:
        h = build_physics()
        calls: list[int] = []

        def hook() -> None:
            calls.append(int(h.env.now))
            if len(calls) == 1:
                h.world.request_decision()

        h.world.set_decision_hook(hook)
        h.world.request_decision()
        h.env.run(until=1)
        assert calls == [0, 0]

    def test_unbounded_same_instant_ticks_raise_zero_time_cycle(self) -> None:
        h = build_physics()
        h.world.set_decision_hook(h.world.request_decision)
        h.world.request_decision()
        with pytest.raises(ZeroTimeCycle):
            h.env.run(until=1)

    def test_without_hook_request_decision_is_a_no_op(self) -> None:
        h = build_physics()
        h.world.request_decision()
        h.env.run(until=1)  # nothing scheduled, nothing raises
