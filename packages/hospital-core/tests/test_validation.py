"""Validation: each Violation kind is reproducible; feasible <=> empty tuple."""

from __future__ import annotations

import os
import subprocess
import sys

from _fixtures import (
    bay_state,
    demo_compiled,
    demo_rules,
    patient,
    provider_task,
    staff_member,
    tiny_er_layout,
)

from hospital.core import (
    AdmissionRule,
    Bay,
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
    StaffId,
    StaffMember,
    StaffRole,
    TaskId,
    TaskSpec,
    ValidationContext,
    Violation,
    Zone,
    ZoneId,
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


# --- Finding #3: an empty acuity whitelist means INCOMPATIBLE (whitelist semantics) ---


def test_empty_acuity_whitelist_is_incompatible() -> None:
    # demo_rules has no entry for ESI2, so its whitelist is empty -> nowhere is allowed.
    ctx = _ctx(patients=(patient("p2", EsiAcuity.ESI2),))
    plan = Plan(items=(_assign("a", "p2", "bay-1"),))
    assert "bay_incompatible" in _kinds(validate(plan, ctx))


# --- Finding #4: assignment to a bay not FREE at apply time is rejected ---


def test_cleaning_bay_assignment_rejected() -> None:
    ctx = _ctx(
        patients=(patient("p1", EsiAcuity.ESI3),),
        bays=(bay_state("bay-1", BayStatus.CLEANING),),
    )
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    assert "bay_incompatible" in _kinds(validate(plan, ctx))


def test_occupied_by_other_patient_rejected() -> None:
    ctx = _ctx(
        patients=(patient("p1", EsiAcuity.ESI3),),
        bays=(bay_state("bay-1", BayStatus.OCCUPIED, "pX"),),
    )
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    assert "bay_incompatible" in _kinds(validate(plan, ctx))


def test_occupied_by_same_patient_is_idempotent() -> None:
    # Re-affirming a patient already in their own bay is not a breach.
    ctx = _ctx(
        patients=(patient("p1", EsiAcuity.ESI3),),
        bays=(bay_state("bay-1", BayStatus.OCCUPIED, "p1"),),
    )
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    assert validate(plan, ctx) == ()


# --- Finding #5: an entity absent from the context is ALWAYS unknown (never suppressed) ---


def test_unknown_patient_not_suppressed_by_empty_context() -> None:
    ctx = _ctx(patients=())  # empty: does NOT exempt a referenced patient
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


def test_unknown_staff_not_suppressed_by_empty_context() -> None:
    ctx = _ctx(staff_members=(), tasks=())
    plan = Plan(
        items=(PlanItem(stable_id="d", kind="dispatch", staff=StaffId("s1"), task=TaskId("t1")),)
    )
    assert "unknown_entity" in _kinds(validate(plan, ctx))


def test_unknown_task_not_suppressed_by_empty_context() -> None:
    doc = staff_member("d1", StaffRole.PHYSICIAN, frozenset({"md"}))
    ctx = _ctx(staff_members=(doc,), tasks=())  # task referenced but tasks empty
    plan = Plan(items=(PlanItem(stable_id="d", kind="dispatch", staff=doc.id, task=TaskId("t1")),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


# --- Finding #6: clean/discharge/staffing payloads are validated ---


def test_clean_unknown_bay_flagged() -> None:
    ctx = _ctx()
    plan = Plan(items=(PlanItem(stable_id="c", kind="clean", bay=BayId("ghost-bay")),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


def test_clean_known_bay_is_clean() -> None:
    ctx = _ctx()
    plan = Plan(items=(PlanItem(stable_id="c", kind="clean", bay=BayId("bay-1")),))
    assert validate(plan, ctx) == ()


def test_discharge_unknown_patient_flagged() -> None:
    ctx = _ctx(patients=(patient("p1", EsiAcuity.ESI3),))
    plan = Plan(items=(PlanItem(stable_id="x", kind="discharge", patient=PatientId("ghost")),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


def test_staffing_unknown_staff_flagged() -> None:
    ctx = _ctx(staff_members=(staff_member("s1", StaffRole.NURSE, frozenset()),))
    plan = Plan(items=(PlanItem(stable_id="s", kind="staffing", staff=StaffId("ghost")),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


# --- Finding #7: one patient assigned to two distinct bays is double-booked ---


def test_patient_in_two_bays_is_double_booked() -> None:
    ctx = _ctx(patients=(patient("p1", EsiAcuity.ESI3),))
    plan = Plan(items=(_assign("a", "p1", "bay-1"), _assign("b", "p1", "bay-2")))
    assert "double_booked" in _kinds(validate(plan, ctx))


# --- Finding #10: a dispatch without a task is a violation ---


def test_dispatch_without_task_flagged() -> None:
    doc = staff_member("d1", StaffRole.PHYSICIAN, frozenset({"md"}))
    ctx = _ctx(staff_members=(doc,))
    plan = Plan(items=(PlanItem(stable_id="d", kind="dispatch", staff=doc.id, task=None),))
    assert "unknown_entity" in _kinds(validate(plan, ctx))


# --- Finding #2: with several CapacityRules per zone, the effective cap is the MINIMUM ---
#
# Run under several PYTHONHASHSEED values: frozenset iteration order is salted by
# the seed, so returning the *first* entry rather than the minimum would flip the
# outcome across processes. The minimum must always be enforced.

_TESTS_DIR = os.path.dirname(__file__)

_CAPACITY_SCRIPT = """
from _fixtures import patient, tiny_er_layout

from hospital.core import (
    AdmissionRule,
    Bay,
    BayId, CompiledRules, EsiAcuity, PatientId, Plan, PlanItem, ValidationContext, ZoneType,
    validate,
)

rules = CompiledRules(
    allowed_zone_types=frozenset({(EsiAcuity.ESI3, ZoneType.GENERAL)}),
    capacities=frozenset({(ZoneType.GENERAL, 1), (ZoneType.GENERAL, 5)}),
)
ctx = ValidationContext(
    layout=tiny_er_layout(),
    bays=(),
    staff=(),
    rules=rules,
    patients=(patient("p1", EsiAcuity.ESI3), patient("p2", EsiAcuity.ESI3)),
)
plan = Plan(items=(
    PlanItem(stable_id="a", kind="assign_bay", patient=PatientId("p1"), bay=BayId("bay-1")),
    PlanItem(stable_id="b", kind="assign_bay", patient=PatientId("p2"), bay=BayId("bay-2")),
))
print("capacity_exceeded" in {v.kind for v in validate(plan, ctx)})
"""


def _run(script: str, seed: int) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONPATH"] = _TESTS_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def test_capacity_uses_minimum_across_hash_seeds() -> None:
    # Two GENERAL bays occupied; min cap is 1, so it is always exceeded. Picking
    # the first frozenset entry (5) would miss it under some hash seeds.
    results = {_run(_CAPACITY_SCRIPT, seed) for seed in range(8)}
    assert results == {"True"}, f"effective cap was not the minimum under some seed: {results}"


# --- M4 §3: the care phase selects which whitelist judges a placement ---


def _ward_ctx(
    *, inpatients: frozenset[PatientId] = frozenset(), bays: tuple[BayState, ...] = ()
) -> ValidationContext:
    """`tiny_er_layout` with one ICU bed bolted on, and a rule set that permits it."""
    base = tiny_er_layout()
    icu = Bay(
        id=BayId("icu-1"),
        zone=ZoneId("z-icu"),
        zone_type=ZoneType.ICU,
        node=base.bays[0].node,
        serving_station=base.bays[0].serving_station,
        isolation_capable=True,
        equipment=frozenset({"monitor"}),
    )
    layout = base.model_copy(
        update={
            "bays": (*base.bays, icu),
            "zones": (*base.zones, Zone(id=ZoneId("z-icu"), zone_type=ZoneType.ICU, capacity=1)),
        }
    )
    rules = compile_rules(
        (
            *demo_rules(),
            AdmissionRule(allowed_zone_types=frozenset({(EsiAcuity.ESI3, ZoneType.ICU)})),
        )
    )
    return ValidationContext(
        layout=layout,
        bays=bays,
        staff=(),
        rules=rules,
        patients=(patient("p3", EsiAcuity.ESI3),),
        inpatients=inpatients,
    )


def test_a_patient_being_worked_up_may_not_be_placed_in_a_ward_bed() -> None:
    """The ward whitelist is not reachable before the disposition that unlocks it.

    ESI-3 is permitted in the ICU *as an admission*, and this is the same patient and
    the same bed — only the phase differs. Judged under the ED rule, the ICU is not on
    the list, so the placement is refused.
    """
    plan = Plan(items=(_assign("a", "p3", "icu-1"),))
    assert "bay_incompatible" in _kinds(validate(plan, _ward_ctx()))


def test_the_same_placement_is_feasible_once_the_patient_is_an_inpatient() -> None:
    ctx = _ward_ctx(inpatients=frozenset({PatientId("p3")}))
    plan = Plan(items=(_assign("a", "p3", "icu-1"),))
    assert validate(plan, ctx) == ()


def test_an_inpatient_may_not_be_placed_back_in_an_ed_bay() -> None:
    """The wall runs both ways — an admitted patient does not re-enter the ED."""
    ctx = _ward_ctx(inpatients=frozenset({PatientId("p3")}))
    plan = Plan(items=(_assign("a", "p3", "bay-1"),))
    assert "bay_incompatible" in _kinds(validate(plan, ctx))


def test_a_housed_inpatient_keeps_their_bed_when_a_standing_plan_is_rejudged() -> None:
    """The phase outlives the bay queue, which is what an operator override depends on.

    ``api.overrides`` re-validates every standing assignment when a bay is closed. A
    patient asleep in an ICU bed left the queue hours ago, so a phase read from the
    queue alone would judge their bed by the ED whitelist and refuse the override —
    the reason ``inpatients`` spans the whole care phase rather than just the wait.
    """
    ctx = _ward_ctx(
        inpatients=frozenset({PatientId("p3")}),
        bays=(bay_state("icu-1", BayStatus.OCCUPIED, "p3"),),
    )
    standing = Plan(items=(_assign("standing:icu-1", "p3", "icu-1"),))
    assert validate(standing, ctx) == ()
