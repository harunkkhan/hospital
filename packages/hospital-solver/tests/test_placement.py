"""placement (CP-SAT): feasibility property, compat/capacity, scarcity, determinism."""

from __future__ import annotations

import math

import pytest
from _solver_fixtures import (
    bay_state,
    decision_input,
    default_config,
    demo_compiled,
    make_patient,
    tiny_layout,
    waiting,
)

from hospital.core import (
    BayId,
    BayStatus,
    DecisionInput,
    Duration,
    EsiAcuity,
    FloorLayout,
    Plan,
    Zone,
    ZoneId,
    ZoneType,
    hours,
)
from hospital.core.time import seconds
from hospital.core.validation import ValidationContext, validate
from hospital.solver import SolveResult, SolverStatus, get_backend
from hospital.solver.oracle import GraphRoutingOracle


def _ctx(di: DecisionInput) -> ValidationContext:
    return ValidationContext(
        layout=di.layout,
        bays=di.bays,
        staff=di.staff,
        rules=demo_compiled(),
        patients=tuple(wp.patient for wp in di.waiting),
    )


def _solve(
    di: DecisionInput, *, time_cap: Duration | None = None, warm_start: Plan | None = None
) -> SolveResult:
    backend = get_backend("placement_cpsat")
    oracle = GraphRoutingOracle(di.layout.graph)
    return backend.solve(
        di,
        oracle,
        config=default_config(),
        rules=demo_compiled(),
        time_cap=time_cap,
        warm_start=warm_start,
    )


def _assignments(plan: Plan) -> dict[str, str]:
    return {i.patient.root: i.bay.root for i in plan.items if i.patient and i.bay}


def test_result_always_validates() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p3", EsiAcuity.ESI3), 30),
            waiting(make_patient("p4", EsiAcuity.ESI4), 20),
        )
    )
    result = _solve(di)
    assert validate(result.plan, _ctx(di)) == ()
    assert result.status is SolverStatus.OPTIMAL


def test_compat_respected_esi1_to_resus_never_fast() -> None:
    di = decision_input(waiting_patients=(waiting(make_patient("p1", EsiAcuity.ESI1), 60),))
    result = _solve(di)
    assert _assignments(result.plan) == {"p1": "bay-3"}  # resus, has monitor+vent


def test_prefers_nearer_compatible_bay() -> None:
    # One ESI-3: both general bays compatible; bay-1 (3000cm) is nearer than bay-2.
    di = decision_input(
        waiting_patients=(waiting(make_patient("p3", EsiAcuity.ESI3), 30),),
        bays=(bay_state("bay-1"), bay_state("bay-2")),
    )
    result = _solve(di)
    assert _assignments(result.plan) == {"p3": "bay-1"}


def test_zone_capacity_leaves_lowest_priority_unplaced() -> None:
    # 3 ESI-3 patients, only 2 GENERAL bays -> exactly one unplaced (the shortest wait).
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p-long", EsiAcuity.ESI3), 300),
            waiting(make_patient("p-mid", EsiAcuity.ESI3), 100),
            waiting(make_patient("p-short", EsiAcuity.ESI3), 10),
        ),
        bays=(bay_state("bay-1"), bay_state("bay-2")),
    )
    result = _solve(di)
    assigned = _assignments(result.plan)
    assert len(assigned) == 2
    assert "p-short" not in assigned  # lowest sequencing score dropped under scarcity
    assert validate(result.plan, _ctx(di)) == ()


def test_one_bay_scarcity_follows_the_sequencing_score() -> None:
    # Regression (M1 review finding 2): the sequencing lever ranks by the
    # additive anti-starvation score u(esi) + alpha*waited, but placement
    # priced scarcity by the multiplicative u(esi)*(waited+1) — so the enacted
    # queue order (long-waiting ESI-3 first) never won the one free bay; the
    # fresher ESI-2 always did, and the sequencing rank was a placement no-op.
    # Scores (default urgency u2=4, u3=3, alpha=1): ESI-3 waited 600 s -> 603
    # beats ESI-2 waited 500 s -> 504; the old pricing had 3*601=1803 < 4*501.
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p-esi2-fresh", EsiAcuity.ESI2), 500),
            waiting(make_patient("p-esi3-starved", EsiAcuity.ESI3), 600),
        ),
        bays=(bay_state("bay-1"),),  # ONE free bay, compatible with both
    )
    result = _solve(di)
    assert _assignments(result.plan) == {"p-esi3-starved": "bay-1"}
    assert validate(result.plan, _ctx(di)) == ()


def test_incompatible_only_patient_left_unplaced_not_infeasible() -> None:
    # Isolation-required ESI-4 can only go to FAST_TRACK, but bay-4 is not iso-capable.
    iso = make_patient("p-iso", EsiAcuity.ESI4, isolation=True)
    di = decision_input(waiting_patients=(waiting(iso, 50),))
    result = _solve(di)
    assert result.plan.items == ()  # left unplaced (penalized), never a crash
    assert validate(result.plan, _ctx(di)) == ()


def test_occupied_zone_reduces_remaining_capacity() -> None:
    # bay-1 already OCCUPIED -> GENERAL cap 2 leaves room for only one more.
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("pa", EsiAcuity.ESI3), 100),
            waiting(make_patient("pb", EsiAcuity.ESI3), 90),
        ),
        bays=(
            bay_state("bay-1", BayStatus.OCCUPIED, occupant="prior"),
            bay_state("bay-2"),
        ),
    )
    result = _solve(di)
    assert len(_assignments(result.plan)) == 1  # only bay-2 free within cap
    assert validate(result.plan, _ctx(di)) == ()


def test_deterministic_same_input_same_plan_and_objective() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p2", EsiAcuity.ESI3), 40),
            waiting(make_patient("p3", EsiAcuity.ESI4), 20),
        )
    )
    r1 = _solve(di)
    r2 = _solve(di)
    assert r1.plan == r2.plan
    assert r1.objective_value == r2.objective_value


def test_time_cap_returns_valid_plan() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p2", EsiAcuity.ESI3), 40),
        )
    )
    result = _solve(di, time_cap=seconds(0.02))
    assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert validate(result.plan, _ctx(di)) == ()


def test_warm_start_produces_valid_plan() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p3", EsiAcuity.ESI3), 40),
        )
    )
    cold = _solve(di)
    warm = _solve(di, warm_start=cold.plan)
    assert validate(warm.plan, _ctx(di)) == ()
    assert warm.plan == cold.plan  # optimum is unique here; warm start doesn't change it


def test_objective_value_matches_brute_force_small_instance() -> None:
    # 1 ESI-3, two general bays: optimal = the cheaper travel assignment.
    di = decision_input(
        waiting_patients=(waiting(make_patient("p3", EsiAcuity.ESI3), 0),),
        bays=(bay_state("bay-1"), bay_state("bay-2")),
    )
    result = _solve(di)
    # Must place (place-first) and pick bay-1 (nearer) -> lowest travel objective.
    assert _assignments(result.plan) == {"p3": "bay-1"}
    assert result.objective_value is not None


def test_place_first_even_when_travel_exceeds_wait_penalty() -> None:
    # Regression (review finding 1): with a small unplaced penalty the travel
    # proxy (360 here) exceeded wait_penalty*(waited+1) (3 here), so the
    # all-unplaced solution was cheaper and CP-SAT left the free bay idle.
    # Place-first must hold for ANY config: the placement reward is scaled by
    # an instance-derived big-B that provably dominates every travel sum.
    di = decision_input(
        waiting_patients=(waiting(make_patient("p3", EsiAcuity.ESI3), 0),),
        bays=(bay_state("bay-1"),),
    )
    backend = get_backend("placement_cpsat")
    oracle = GraphRoutingOracle(di.layout.graph)
    result = backend.solve(
        di, oracle, config=default_config(unplaced_wait_penalty=1), rules=demo_compiled()
    )
    assert _assignments(result.plan) == {"p3": "bay-1"}
    assert result.status is SolverStatus.OPTIMAL


def test_place_first_even_when_occupancy_exceeds_the_wait_penalty() -> None:
    """The occupancy term must not make leaving a patient unplaced look cheap.

    Same failure mode as its travel-term sibling above, but occupancy is the term with
    room to get large: it scales with a *predicted* stay, so one bad prediction could
    dwarf any travel proxy. Place-first survives only because `big_b` is derived from
    the finished `weight` dict -- occupancy already folded in. Compute it from travel
    alone and this test fails while every other placement test still passes.
    """
    patient = make_patient("p3", EsiAcuity.ESI3)
    di = decision_input(
        waiting_patients=(waiting(patient, 0),),
        bays=(bay_state("bay-1"),),
    )
    backend = get_backend("placement_cpsat")
    oracle = GraphRoutingOracle(di.layout.graph)
    result = backend.solve(
        di,
        oracle,
        # A tiny unplaced penalty against an absurd predicted stay and a heavy weight:
        # the assignment cost here is orders of magnitude above the cost of refusing.
        config=default_config(unplaced_wait_penalty=1, w_occupancy=50),
        rules=demo_compiled(),
        expected_stay={patient.id: hours(500)},
    )
    assert _assignments(result.plan) == {"p3": "bay-1"}
    assert result.status is SolverStatus.OPTIMAL


def test_search_budget_is_deterministic_time_not_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (review finding 5): max_time_in_seconds let OS scheduling pick
    # the incumbent, so the same DecisionInput could yield different plans
    # (breaking CRN / byte-identity). The budget must be CP-SAT deterministic
    # time, with the wall-clock limit left at its infinite default.
    from ortools.sat.python import cp_model

    captured: list[tuple[float, float]] = []
    original = cp_model.CpSolver.solve

    def spy(
        self: cp_model.CpSolver,
        model: cp_model.CpModel,
        solution_callback: cp_model.CpSolverSolutionCallback | None = None,
    ):
        captured.append(
            (self.parameters.max_time_in_seconds, self.parameters.max_deterministic_time)
        )
        return original(self, model, solution_callback)

    monkeypatch.setattr(cp_model.CpSolver, "solve", spy)
    di = decision_input(waiting_patients=(waiting(make_patient("p3", EsiAcuity.ESI3), 30),))
    _solve(di, time_cap=seconds(0.02))
    [(wall, deterministic)] = captured
    assert math.isinf(wall)  # wall clock no longer truncates the search
    assert deterministic == pytest.approx(0.02)


def test_repeated_capped_solves_are_identical() -> None:
    # Regression (review finding 5): under a search cap, repeated solves of the
    # same DecisionInput must return byte-identical plans and objective values.
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p2", EsiAcuity.ESI2), 50),
            waiting(make_patient("p3", EsiAcuity.ESI3), 40),
            waiting(make_patient("p4", EsiAcuity.ESI3), 30),
            waiting(make_patient("p5", EsiAcuity.ESI4), 20),
        )
    )
    results = [_solve(di, time_cap=seconds(0.001)) for _ in range(3)]
    first = results[0]
    for other in results[1:]:
        assert other.plan.model_dump_json() == first.plan.model_dump_json()
        assert other.objective_value == first.objective_value
        assert other.status == first.status


def test_no_incumbent_falls_back_to_deterministic_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (review finding 8): CP-SAT UNKNOWN (no incumbent under a tiny
    # budget) was reported as FEASIBLE with objective_value=None, contradicting
    # the status contract. It must return the greedy backend's plan, labeled
    # HEURISTIC — never FEASIBLE without an incumbent.
    from ortools.sat.python import cp_model

    def unknown(
        self: cp_model.CpSolver,
        model: cp_model.CpModel,
        solution_callback: cp_model.CpSolverSolutionCallback | None = None,
    ):
        return cp_model.UNKNOWN

    monkeypatch.setattr(cp_model.CpSolver, "solve", unknown)
    di = decision_input(waiting_patients=(waiting(make_patient("p3", EsiAcuity.ESI3), 30),))
    result = _solve(di)
    assert result.status is SolverStatus.HEURISTIC
    assert result.objective_value is None
    assert result.backend == "placement_greedy"
    assert _assignments(result.plan) == {"p3": "bay-1"}  # the validated greedy plan
    assert validate(result.plan, _ctx(di)) == ()


def _two_general_zones() -> FloorLayout:
    """A one-bay GENERAL zone next to a three-bay one, the nearer bay in the tight zone.

    ``tiny_layout`` gives each acuity exactly one eligible zone, so no patient there ever
    faces a cross-zone choice and scarcity can never break a tie. Scarcity is measured
    from *assignable* bays, so the roomy zone needs more than one of them — hence the two
    extra bays, cloned from bay-2 onto its own node so travel stays identical between
    them and bay-1 keeps its distance advantage.
    """
    base = tiny_layout()
    roomy, tight = ZoneId("z-gen"), ZoneId("z-gen-tight")
    original = {b.id: b for b in base.bays}
    clones = tuple(
        original[BayId("bay-2")].model_copy(update={"id": BayId(f"bay-2{suffix}")})
        for suffix in ("b", "c")
    )
    bays = (
        *(b.model_copy(update={"zone": tight}) if b.id == BayId("bay-1") else b for b in base.bays),
        *clones,
    )
    zones = (
        Zone(id=roomy, zone_type=ZoneType.GENERAL, capacity=4),
        Zone(id=tight, zone_type=ZoneType.GENERAL, capacity=1),
        *(z for z in base.zones if z.id != roomy),
    )
    return base.model_copy(update={"bays": bays, "zones": zones})


def test_the_fallback_still_prices_the_predicted_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegating to the greedy backend must not silently drop the predictions.

    The fallback fires exactly when the instance is hardest, so an arm that quietly
    reverts to travel-only placement there is optimizing something other than what it
    reports — and ``HEURISTIC`` names *which backend* answered, never which terms it
    read, so nothing downstream would show it.
    """
    from ortools.sat.python import cp_model

    def unknown(
        self: cp_model.CpSolver,
        model: cp_model.CpModel,
        solution_callback: cp_model.CpSolverSolutionCallback | None = None,
    ):
        return cp_model.UNKNOWN

    monkeypatch.setattr(cp_model.CpSolver, "solve", unknown)
    patient = make_patient("p3", EsiAcuity.ESI3)
    di = decision_input(
        waiting_patients=(waiting(patient, 30),),
        bays=(
            bay_state("bay-1"),
            bay_state("bay-2"),
            bay_state("bay-2b"),
            bay_state("bay-2c"),
        ),
        layout=_two_general_zones(),
    )
    backend = get_backend("placement_cpsat")
    oracle = GraphRoutingOracle(di.layout.graph)
    config = default_config(w_occupancy=40)

    priced = backend.solve(
        di, oracle, config=config, rules=demo_compiled(), expected_stay={patient.id: hours(8)}
    )
    unpriced = backend.solve(di, oracle, config=config, rules=demo_compiled())
    assert priced.status is SolverStatus.HEURISTIC
    assert unpriced.status is SolverStatus.HEURISTIC
    # Travel alone takes the nearer bay-1; it is also the last bay in its zone, so a
    # long predicted stay must send the patient to the roomy zone instead.
    assert _assignments(unpriced.plan) == {"p3": "bay-1"}
    assert _assignments(priced.plan) == {"p3": "bay-2"}
