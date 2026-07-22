"""stamping: the provenance choke point — pure attribution, no mutation, no upgrade."""

from __future__ import annotations

from _solver_fixtures import (
    decision_input,
    default_config,
    demo_compiled,
    make_patient,
    waiting,
)

from hospital.core import EsiAcuity, Plan, SimTime
from hospital.solver import SolverStatus, get_backend, stamp
from hospital.solver.objective import config_hash
from hospital.solver.oracle import GraphRoutingOracle
from hospital.solver.protocol import SolveResult
from hospital.solver.stamping import SOLVER_VERSION, StampedPlan


def _result(backend_name: str = "placement_greedy") -> SolveResult:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),
            waiting(make_patient("p3", EsiAcuity.ESI3), 30),
        )
    )
    oracle = GraphRoutingOracle(di.layout.graph)
    return get_backend(backend_name).solve(
        di, oracle, config=default_config(), rules=demo_compiled()
    )


def test_stamp_attaches_full_provenance() -> None:
    config = default_config()
    rules = demo_compiled()
    result = _result()
    stamped = stamp(result, config, rules_hash=rules.rules_hash, now=SimTime(123))
    assert stamped.backend == "placement_greedy"
    assert stamped.backend_version == "1.0.0"  # resolved through the one registry
    assert stamped.objective_config_hash == config_hash(config)
    assert stamped.rules_hash == rules.rules_hash
    assert stamped.solver_version == SOLVER_VERSION
    assert stamped.stamped_at == SimTime(123)


def test_plan_is_byte_unchanged_by_stamping() -> None:
    result = _result()
    stamped = stamp(result, default_config())
    assert stamped.plan == result.plan
    assert stamped.plan.model_dump_json() == result.plan.model_dump_json()


def test_status_and_objective_value_carried_never_recomputed() -> None:
    # A FEASIBLE claim must survive stamping untouched (never upgraded to OPTIMAL).
    result = SolveResult(
        plan=Plan(items=()),
        status=SolverStatus.FEASIBLE,
        objective_value=42,
        solve_wall_us=10,
        backend="placement_cpsat",
    )
    stamped = stamp(result, default_config())
    assert stamped.status is SolverStatus.FEASIBLE
    assert stamped.objective_value == 42
    assert stamped.backend_version == "1.0.0"


def test_rules_hash_is_optional() -> None:
    # A lever with no compiled-rules dependency stamps None; consumers must cope.
    stamped = stamp(_result(), default_config())
    assert stamped.rules_hash is None


def test_stamped_plan_round_trips() -> None:
    rules = demo_compiled()
    stamped = stamp(_result(), default_config(), rules_hash=rules.rules_hash, now=SimTime(7))
    revived = StampedPlan.model_validate_json(stamped.model_dump_json())
    assert revived == stamped


def test_config_drift_is_detectable_on_the_stamp() -> None:
    result = _result()
    a = stamp(result, default_config())
    b = stamp(result, default_config(w_travel=99))
    same = stamp(result, default_config())
    assert a.objective_config_hash != b.objective_config_hash
    assert a.objective_config_hash == same.objective_config_hash


def test_single_choke_point_serves_every_backend() -> None:
    # The same doorway attributes both registry backends — no per-backend stamping.
    greedy = stamp(_result("placement_greedy"), default_config())
    cpsat = stamp(_result("placement_cpsat"), default_config())
    assert greedy.backend == "placement_greedy"
    assert cpsat.backend == "placement_cpsat"
    assert {greedy.backend_version, cpsat.backend_version} == {"1.0.0"}


def test_non_registry_producer_gets_explicit_or_empty_version() -> None:
    result = SolveResult(
        plan=Plan(items=()),
        status=SolverStatus.HEURISTIC,
        objective_value=None,
        solve_wall_us=1,
        backend="human_override",
    )
    assert stamp(result, default_config()).backend_version == ""
    override = stamp(result, default_config(), backend_version="2.0.0")
    assert override.backend_version == "2.0.0"
