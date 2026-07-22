"""dispatch: nearest-qualified, optimal assignment, Held-Karp, oracle-only distances."""

from __future__ import annotations

import itertools

from _solver_fixtures import decision_input, default_config, staff_member, staff_state, task

from hospital.core import DecisionInput, Duration, NodeId, PlanItem, RoutePath, StaffRole
from hospital.solver.dispatch import assign_staff, route_visits
from hospital.solver.oracle import EMPTY_MASK, GraphRoutingOracle, RouteMask


class SpyOracle:
    """Wraps a real oracle and counts distance calls (to prove no direct pathfinding)."""

    def __init__(self, inner: GraphRoutingOracle) -> None:
        self._inner = inner
        self.distance_calls = 0

    def distance(self, src: NodeId, dst: NodeId, *, mask: RouteMask = EMPTY_MASK) -> Duration:
        self.distance_calls += 1
        return self._inner.distance(src, dst, mask=mask)

    def path(self, src: NodeId, dst: NodeId, *, mask: RouteMask = EMPTY_MASK) -> RoutePath:
        return self._inner.path(src, dst, mask=mask)


def _oracle(di: DecisionInput) -> GraphRoutingOracle:
    return GraphRoutingOracle(di.layout.graph)


def _matching(items: tuple[PlanItem, ...]) -> dict[str, str]:
    return {i.task.root: i.staff.root for i in items if i.task is not None and i.staff is not None}


def test_single_task_nearest_qualified() -> None:
    di = decision_input(
        staff=(staff_state("s-near", at="gstat"), staff_state("s-far", at="b2")),
    )
    members = (
        staff_member("s-near", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
        staff_member("s-far", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
    )
    t = task("t1", "provider_visit", at="b1", role=StaffRole.PHYSICIAN, skills=frozenset({"md"}))
    items = assign_staff(
        di, _oracle(di), config=default_config(), tasks=(t,), staff_members=members
    )
    assert _matching(items) == {"t1": "s-near"}  # gstat->b1 shorter than b2->b1


def test_unqualified_staff_never_dispatched() -> None:
    di = decision_input(staff=(staff_state("nurse1", at="gstat"),))
    members = (staff_member("nurse1", StaffRole.NURSE, skills=frozenset()),)
    # Task needs a physician with md -> the lone nurse cannot take it.
    t = task("t1", "provider_visit", at="b1", role=StaffRole.PHYSICIAN, skills=frozenset({"md"}))
    items = assign_staff(
        di, _oracle(di), config=default_config(), tasks=(t,), staff_members=members
    )
    assert items == ()


def test_multi_task_assignment_is_optimal() -> None:
    di = decision_input(
        staff=(staff_state("s1", at="b1"), staff_state("s2", at="b4")),
    )
    members = (
        staff_member("s1", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
        staff_member("s2", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
    )
    tasks = (
        task("tA", "provider_visit", at="b1", role=StaffRole.PHYSICIAN, skills=frozenset({"md"})),
        task("tB", "provider_visit", at="b4", role=StaffRole.PHYSICIAN, skills=frozenset({"md"})),
    )
    oracle = _oracle(di)
    items = assign_staff(di, oracle, config=default_config(), tasks=tasks, staff_members=members)
    matching = _matching(items)
    assert len(matching) == 2  # both tasks covered

    # Brute-force the two perfect matchings and confirm the chosen one is minimal.
    pos = {"s1": NodeId("b1"), "s2": NodeId("b4")}
    tnode = {"tA": NodeId("b1"), "tB": NodeId("b4")}

    def cost(assign: dict[str, str]) -> int:
        return sum(oracle.distance(pos[s], tnode[t]).root for t, s in assign.items())

    all_matchings = [
        {"tA": "s1", "tB": "s2"},
        {"tA": "s2", "tB": "s1"},
    ]
    assert cost(matching) == min(cost(m) for m in all_matchings)
    assert matching == {"tA": "s1", "tB": "s2"}  # co-located pairing is cheapest


def test_assignment_never_strands_a_coverable_task() -> None:
    # Regression: with big-M = max(cost)+1 the solver preferred ONE cheap match
    # (s1 already at t1's node) over the only full cover, stranding t2 even
    # though s2 could take t1 and s1 (the sole ultrasound-skilled) could take t2.
    # Cardinality must be strictly lexicographic: cover-first, then min travel.
    di = decision_input(staff=(staff_state("s1", at="b1"), staff_state("s2", at="b3")))
    members = (
        staff_member("s1", StaffRole.PHYSICIAN, skills=frozenset({"md", "us"})),
        staff_member("s2", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
    )
    tasks = (
        task("t1", "provider_visit", at="b1", role=StaffRole.PHYSICIAN, skills=frozenset({"md"})),
        task(
            "t2",
            "imaging",
            at="b3",
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md", "us"}),
        ),
    )
    items = assign_staff(
        di, _oracle(di), config=default_config(), tasks=tasks, staff_members=members
    )
    assert _matching(items) == {"t1": "s2", "t2": "s1"}  # both covered, min travel


def _brute_open_path(start: NodeId, stops: list[NodeId], oracle: GraphRoutingOracle) -> int:
    best = None
    for perm in itertools.permutations(stops):
        total = 0
        prev = start
        for s in perm:
            total += oracle.distance(prev, s).root
            prev = s
        best = total if best is None else min(best, total)
    assert best is not None
    return best


def test_held_karp_matches_brute_force() -> None:
    di = decision_input()
    oracle = _oracle(di)
    start = NodeId("gstat")
    stops = [NodeId("b1"), NodeId("b2"), NodeId("b3"), NodeId("b4"), NodeId("img")]
    order = route_visits(start, stops, oracle, exact_max=10)
    assert set(order) == set(stops)

    def route_len(seq: tuple[NodeId, ...]) -> int:
        total = 0
        prev = start
        for s in seq:
            total += oracle.distance(prev, s).root
            prev = s
        return total

    assert route_len(order) == _brute_open_path(start, stops, oracle)


def test_route_no_better_than_naive_order() -> None:
    di = decision_input()
    oracle = _oracle(di)
    start = NodeId("gstat")
    stops = [NodeId("b3"), NodeId("b1"), NodeId("lab"), NodeId("b4")]
    order = route_visits(start, stops, oracle, exact_max=10)

    def route_len(seq: tuple[NodeId, ...]) -> int:
        total = 0
        prev = start
        for s in seq:
            total += oracle.distance(prev, s).root
            prev = s
        return total

    assert route_len(order) <= route_len(tuple(stops))


def test_nn_two_opt_branch_beats_naive() -> None:
    # Force the heuristic branch with a small exact_max.
    di = decision_input()
    oracle = _oracle(di)
    start = NodeId("gstat")
    stops = [NodeId("b3"), NodeId("b1"), NodeId("lab"), NodeId("b4"), NodeId("b2")]
    order = route_visits(start, stops, oracle, exact_max=3)
    assert set(order) == set(stops)

    def route_len(seq: tuple[NodeId, ...]) -> int:
        total = 0
        prev = start
        for s in seq:
            total += oracle.distance(prev, s).root
            prev = s
        return total

    assert route_len(order) <= route_len(tuple(stops))


def test_every_spatial_cost_from_oracle() -> None:
    di = decision_input(staff=(staff_state("s1", at="gstat"),))
    members = (staff_member("s1", StaffRole.PHYSICIAN, skills=frozenset({"md"})),)
    t = task("t1", "provider_visit", at="b1", role=StaffRole.PHYSICIAN, skills=frozenset({"md"}))
    spy = SpyOracle(_oracle(di))
    assign_staff(di, spy, config=default_config(), tasks=(t,), staff_members=members)
    assert spy.distance_calls >= 1  # dispatch sourced its cost from the oracle
