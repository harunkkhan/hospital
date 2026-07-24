"""dispatch: nearest-qualified, optimal assignment, Held-Karp, oracle-only distances."""

from __future__ import annotations

import itertools

from _solver_fixtures import (
    decision_input,
    default_config,
    demo_compiled,
    staff_member,
    staff_state,
    task,
)

from hospital.core import (
    DecisionInput,
    Duration,
    EsiAcuity,
    NodeId,
    Plan,
    PlanItem,
    RoutePath,
    StaffRole,
    ValidationContext,
    validate,
)
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
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=(t,),
        staff_members=members,
    )
    assert _matching(items) == {"t1": "s-near"}  # gstat->b1 shorter than b2->b1


def test_unqualified_staff_never_dispatched() -> None:
    di = decision_input(staff=(staff_state("nurse1", at="gstat"),))
    members = (staff_member("nurse1", StaffRole.NURSE, skills=frozenset()),)
    # Task needs a physician with md -> the lone nurse cannot take it.
    t = task("t1", "provider_visit", at="b1", role=StaffRole.PHYSICIAN, skills=frozenset({"md"}))
    items = assign_staff(
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=(t,),
        staff_members=members,
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
    items = assign_staff(
        di,
        oracle,
        config=default_config(),
        rules=demo_compiled(),
        tasks=tasks,
        staff_members=members,
    )
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
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=tasks,
        staff_members=members,
    )
    assert _matching(items) == {"t1": "s2", "t2": "s1"}  # both covered, min travel


def test_scarce_staff_serves_critical_before_nearer_low_acuity() -> None:
    # Regression (M1 stress finding): with travel-only costs the matched SUBSET
    # under scarcity followed distance alone — the far resus-zone ESI-1 task
    # lost to nearer low-acuity work tick after tick (ESI-1 LOS +22 min while
    # the solver "saved walking"). Priority is strict acuity tiers: the one
    # idle physician takes the just-created ESI-1 task at the FAR node over a
    # nearer ESI-4 task that has already waited 600 s.
    di = decision_input(staff=(staff_state("doc", at="gstat"),), now_us=600_000_000)
    members = (staff_member("doc", StaffRole.PHYSICIAN, skills=frozenset({"md"})),)
    tasks = (
        task(
            "t-low",
            "provider_visit",
            at="b1",  # 3000 cm from gstat — the travel-cheap choice
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI4,
            ready_at_us=0,  # waited 600 s
        ),
        task(
            "t-crit",
            "provider_visit",
            at="b3",  # 9000 cm from gstat — the resus bay
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI1,
            ready_at_us=600_000_000,  # just created
        ),
    )
    items = assign_staff(
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=tasks,
        staff_members=members,
    )
    assert _matching(items) == {"t-crit": "doc"}


def test_same_tier_dispatch_is_fifo_not_travel() -> None:
    # Within one acuity tier the longest-waiting task wins over a nearer fresh
    # one — aging is bounded (no starvation within a tier), and travel cannot
    # buy its way past a longer wait.
    di = decision_input(staff=(staff_state("doc", at="gstat"),), now_us=600_000_000)
    members = (staff_member("doc", StaffRole.PHYSICIAN, skills=frozenset({"md"})),)
    tasks = (
        task(
            "t-near-fresh",
            "provider_visit",
            at="b1",
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI3,
            ready_at_us=600_000_000,
        ),
        task(
            "t-far-aged",
            "provider_visit",
            at="b3",
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI3,
            ready_at_us=0,
        ),
    )
    items = assign_staff(
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=tasks,
        staff_members=members,
    )
    assert _matching(items) == {"t-far-aged": "doc"}


def test_equal_priority_ties_break_on_travel() -> None:
    # Same tier, same wait -> the third lexicographic level (min travel) decides.
    di = decision_input(staff=(staff_state("doc", at="gstat"),))
    members = (staff_member("doc", StaffRole.PHYSICIAN, skills=frozenset({"md"})),)
    tasks = (
        task(
            "t-far",
            "provider_visit",
            at="b4",  # 4000 cm
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI3,
        ),
        task(
            "t-near",
            "provider_visit",
            at="b1",  # 3000 cm
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI3,
        ),
    )
    items = assign_staff(
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=tasks,
        staff_members=members,
    )
    assert _matching(items) == {"t-near": "doc"}


def test_priority_never_reduces_cardinality() -> None:
    # Serve-first still dominates: taking the ESI-1 with the only us-skilled
    # physician would strand the other task, so the full cover wins even though
    # it hands the critical task to the farther staff member.
    di = decision_input(staff=(staff_state("s1", at="b3"), staff_state("s2", at="b1")))
    members = (
        staff_member("s1", StaffRole.PHYSICIAN, skills=frozenset({"md", "us"})),
        staff_member("s2", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
    )
    tasks = (
        task(
            "t-crit",
            "provider_visit",
            at="b3",
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md"}),
            esi=EsiAcuity.ESI1,
        ),
        task(
            "t-us",
            "imaging",
            at="b1",
            role=StaffRole.PHYSICIAN,
            skills=frozenset({"md", "us"}),
            esi=EsiAcuity.ESI5,
        ),
    )
    items = assign_staff(
        di,
        _oracle(di),
        config=default_config(),
        rules=demo_compiled(),
        tasks=tasks,
        staff_members=members,
    )
    assert _matching(items) == {"t-crit": "s2", "t-us": "s1"}


def test_rule_skills_unioned_with_task_skills() -> None:
    # Regression (review finding 2): qualification used only TaskSpec.required_skills,
    # while the validator unions rules.skills_for(task.kind) — so dispatch could
    # emit plans the validator rejects. demo rules require {"md"} for
    # provider_visit; the task itself asks for nothing, and the NEAREST idle
    # physician lacks "md". Dispatch must skip them, exactly as validate() would.
    di = decision_input(staff=(staff_state("s-md", at="b2"), staff_state("s-plain", at="gstat")))
    members = (
        staff_member("s-md", StaffRole.PHYSICIAN, skills=frozenset({"md"})),
        staff_member("s-plain", StaffRole.PHYSICIAN, skills=frozenset()),
    )
    t = task("t1", "provider_visit", at="b1", role=StaffRole.PHYSICIAN)  # no explicit skills
    rules = demo_compiled()
    items = assign_staff(
        di, _oracle(di), config=default_config(), rules=rules, tasks=(t,), staff_members=members
    )
    assert _matching(items) == {"t1": "s-md"}  # nearer-but-unqualified s-plain skipped
    # And the emitted plan passes the one validator (the drift this guards against).
    ctx = ValidationContext(
        layout=di.layout,
        bays=di.bays,
        staff=di.staff,
        rules=rules,
        staff_members=members,
        tasks=(t,),
    )
    assert validate(Plan(items=items), ctx) == ()


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
    assign_staff(
        di, spy, config=default_config(), rules=demo_compiled(), tasks=(t,), staff_members=members
    )
    assert spy.distance_calls >= 1  # dispatch sourced its cost from the oracle
