"""Validation: each Violation kind is reproducible; feasible <=> empty tuple."""

from __future__ import annotations

from _fixtures import (
    bay_state,
    demo_compiled,
    patient,
    provider_task,
    staff_member,
    tiny_er_layout,
)

from hospital.core import (
    BayId,
    BayState,
    BayStatus,
    CapacityRule,
    CompatibilityRule,
    CompiledRules,
    EsiAcuity,
    Patient,
    PatientId,
    Plan,
    PlanItem,
    StaffMember,
    StaffRole,
    TaskId,
    TaskSpec,
    ValidationContext,
    Violation,
    ZoneType,
    compile_rules,
    validate,
)


def _ctx(
    *,
    bays: tuple[BayState, ...] = (),
    patients: tuple[Patient, ...] = (),
    staff_members: tuple[StaffMember, ...] = (),
    tasks: tuple[TaskSpec, ...] = (),
    rules: CompiledRules | None = None,
) -> ValidationContext:
    return ValidationContext(
        layout=tiny_er_layout(),
        bays=bays,
        staff=(),
        rules=rules if rules is not None else demo_compiled(),
        patients=patients,
        staff_members=staff_members,
        tasks=tasks,
    )


def _assign(stable_id: str, patient_id: str, bay: str) -> PlanItem:
    return PlanItem(
        stable_id=stable_id, kind="assign_bay", patient=PatientId(patient_id), bay=BayId(bay)
    )


def _kinds(violations: tuple[Violation, ...]) -> set[str]:
    return {v.kind for v in violations}


def test_feasible_plan_returns_empty_tuple() -> None:
    ctx = _ctx(patients=(patient("p1", EsiAcuity.ESI3),))
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    assert validate_ok(plan, ctx)


def validate_ok(plan: Plan, ctx: ValidationContext) -> bool:
    return validate(plan, ctx) == ()


def test_unknown_entity() -> None:
    ctx = _ctx(patients=(patient("p1"),))
    plan = Plan(items=(_assign("a", "p1", "ghost-bay"),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


def test_bay_incompatible_zone_mismatch() -> None:
    # ESI5 is allowed only in FAST_TRACK; bay-1 is GENERAL.
    ctx = _ctx(patients=(patient("p5", EsiAcuity.ESI5),))
    plan = Plan(items=(_assign("a", "p5", "bay-1"),))
    assert _kinds(validate(plan, ctx)) == {"bay_incompatible"}


def test_bay_incompatible_closed_bay() -> None:
    ctx = _ctx(
        patients=(patient("p1", EsiAcuity.ESI3),),
        bays=(bay_state("bay-1", BayStatus.CLOSED),),
    )
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    assert "bay_incompatible" in _kinds(validate(plan, ctx))


def test_capacity_exceeded() -> None:
    rules = compile_rules(
        (
            CompatibilityRule(allowed_zone_types=frozenset({(EsiAcuity.ESI3, ZoneType.GENERAL)})),
            CapacityRule(zone_type=ZoneType.GENERAL, max_occupancy=1),
        )
    )
    ctx = _ctx(
        patients=(patient("p1", EsiAcuity.ESI3), patient("p2", EsiAcuity.ESI3)),
        rules=rules,
    )
    plan = Plan(items=(_assign("a", "p1", "bay-1"), _assign("b", "p2", "bay-2")))
    assert "capacity_exceeded" in _kinds(validate(plan, ctx))


def test_isolation_violated() -> None:
    ctx = _ctx(patients=(patient("piso", EsiAcuity.ESI3, isolation=True),))
    plan = Plan(items=(_assign("a", "piso", "bay-1"),))  # bay-1 is not isolation-capable
    assert _kinds(validate(plan, ctx)) == {"isolation_violated"}


def test_staff_lacks_skill() -> None:
    nurse = staff_member("n1", StaffRole.NURSE, frozenset({"rn"}))
    task = provider_task("t1", "p1")  # requires PHYSICIAN + "md"
    ctx = _ctx(staff_members=(nurse,), tasks=(task,))
    plan = Plan(
        items=(PlanItem(stable_id="d", kind="dispatch", staff=nurse.id, task=TaskId("t1")),)
    )
    assert "staff_lacks_skill" in _kinds(validate(plan, ctx))


def test_double_booked() -> None:
    ctx = _ctx(patients=(patient("p1", EsiAcuity.ESI3), patient("p2", EsiAcuity.ESI3)))
    plan = Plan(items=(_assign("a", "p1", "bay-1"), _assign("b", "p2", "bay-1")))
    assert "double_booked" in _kinds(validate(plan, ctx))


def test_precedence_violated() -> None:
    ctx = _ctx()  # demo rules: TRIAGE must precede PROVIDER_VISIT
    plan = Plan(
        items=(
            PlanItem(
                stable_id="seq",
                kind="sequence",
                patient=PatientId("p1"),
                order=("provider_visit", "triage"),
            ),
        )
    )
    assert "precedence_violated" in _kinds(validate(plan, ctx))


def test_precedence_satisfied_is_clean() -> None:
    ctx = _ctx()
    plan = Plan(
        items=(
            PlanItem(
                stable_id="seq",
                kind="sequence",
                patient=PatientId("p1"),
                order=("triage", "provider_visit"),
            ),
        )
    )
    assert validate(plan, ctx) == ()
