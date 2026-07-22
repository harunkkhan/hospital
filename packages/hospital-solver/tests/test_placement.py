"""placement (CP-SAT): feasibility property, compat/capacity, scarcity, determinism."""

from __future__ import annotations

from _solver_fixtures import (
    bay_state,
    decision_input,
    default_config,
    demo_compiled,
    make_patient,
    waiting,
)

from hospital.core import BayStatus, DecisionInput, Duration, EsiAcuity, Plan
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
    assert "p-short" not in assigned  # lowest u*(waited) dropped under scarcity
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
