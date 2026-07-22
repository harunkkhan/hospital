"""heuristic: valid plans, matches CP-SAT on forced instances, faster, registered."""

from __future__ import annotations

from _solver_fixtures import (
    bay_state,
    decision_input,
    default_config,
    demo_compiled,
    make_patient,
    waiting,
)

from hospital.core import DecisionInput, EsiAcuity, Plan
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


def _solve(name: str, di: DecisionInput) -> SolveResult:
    oracle = GraphRoutingOracle(di.layout.graph)
    return get_backend(name).solve(di, oracle, config=default_config(), rules=demo_compiled())


def _assignments(plan: Plan) -> dict[str, str]:
    return {i.patient.root: i.bay.root for i in plan.items if i.patient and i.bay}


def test_heuristic_plan_validates_and_claims_heuristic() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p3", EsiAcuity.ESI3), 30),
            waiting(make_patient("p4", EsiAcuity.ESI4), 20),
        )
    )
    result = _solve("placement_greedy", di)
    assert result.status is SolverStatus.HEURISTIC
    assert result.objective_value is None
    assert validate(result.plan, _ctx(di)) == ()


def test_matches_cpsat_on_forced_instance() -> None:
    # Each patient has exactly one compatible bay -> both backends must agree.
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),  # -> resus bay-3
            waiting(make_patient("p4", EsiAcuity.ESI4), 20),  # -> fast bay-4
        ),
        bays=(bay_state("bay-3"), bay_state("bay-4")),
    )
    greedy = _solve("placement_greedy", di)
    exact = _solve("placement_cpsat", di)
    assert _assignments(greedy.plan) == _assignments(exact.plan)
    assert _assignments(greedy.plan) == {"p1": "bay-3", "p4": "bay-4"}


def test_heuristic_respects_capacity() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("pa", EsiAcuity.ESI3), 100),
            waiting(make_patient("pb", EsiAcuity.ESI3), 90),
            waiting(make_patient("pc", EsiAcuity.ESI3), 80),
        ),
        bays=(bay_state("bay-1"), bay_state("bay-2")),
    )
    result = _solve("placement_greedy", di)
    assert len(_assignments(result.plan)) == 2
    assert validate(result.plan, _ctx(di)) == ()


def test_heuristic_faster_than_cpsat() -> None:
    di = decision_input(
        waiting_patients=tuple(
            waiting(make_patient(f"p{i}", EsiAcuity.ESI3), 10 * i) for i in range(4)
        )
    )
    greedy = _solve("placement_greedy", di)
    exact = _solve("placement_cpsat", di)
    assert greedy.solve_wall_us <= exact.solve_wall_us
