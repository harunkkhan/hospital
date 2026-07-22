"""turnaround: highest-demand bay cleaned first; one-per assignment; kind=clean."""

from __future__ import annotations

from _solver_fixtures import (
    bay_state,
    decision_input,
    default_config,
    demo_compiled,
    demo_rules,
    make_patient,
    staff_member,
    staff_state,
    waiting,
)

from hospital.core import (
    BayStatus,
    DecisionInput,
    EsiAcuity,
    SkillRule,
    StaffRole,
    compile_rules,
)
from hospital.solver.oracle import GraphRoutingOracle
from hospital.solver.turnaround import prioritize_cleaning

# Demand (acuity units) must outweigh travel seconds for this lever to be
# demand-driven; a large w_time supplies that balance (assumption 6).
CONFIG = default_config(w_time=1000, w_travel=1)


def _oracle(di: DecisionInput) -> GraphRoutingOracle:
    return GraphRoutingOracle(di.layout.graph)


def test_highest_demand_bay_cleaned_first() -> None:
    # bay-3 (resus) has a compatible waiting ESI-1; bay-4 (fast) has none.
    di = decision_input(
        waiting_patients=(waiting(make_patient("p1", EsiAcuity.ESI1), 60),),
        bays=(
            bay_state("bay-3", BayStatus.CLEANING),
            bay_state("bay-4", BayStatus.CLEANING),
        ),
        staff=(staff_state("hk1", at="gstat"),),
    )
    members = (staff_member("hk1", StaffRole.HOUSEKEEPING),)
    items = prioritize_cleaning(
        di, _oracle(di), config=CONFIG, rules=demo_compiled(), staff_members=members
    )
    assert len(items) == 1
    item = items[0]
    assert item.kind == "clean"
    assert item.bay is not None and item.bay.root == "bay-3"  # unblocks the ESI-1 demand
    assert item.staff is not None and item.staff.root == "hk1"


def test_no_housekeepers_no_cleaning() -> None:
    di = decision_input(
        waiting_patients=(waiting(make_patient("p1", EsiAcuity.ESI1), 60),),
        bays=(bay_state("bay-3", BayStatus.CLEANING),),
        staff=(),
    )
    assert prioritize_cleaning(di, _oracle(di), config=CONFIG, rules=demo_compiled()) == ()


def test_one_housekeeper_per_bay_and_one_bay_per_housekeeper() -> None:
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p1", EsiAcuity.ESI1), 60),  # -> bay-3
            waiting(make_patient("p3", EsiAcuity.ESI3), 60),  # -> bay-1/2 (general)
        ),
        bays=(
            bay_state("bay-1", BayStatus.CLEANING),
            bay_state("bay-3", BayStatus.CLEANING),
        ),
        staff=(staff_state("hk1", at="gstat"), staff_state("hk2", at="gstat")),
    )
    members = (
        staff_member("hk1", StaffRole.HOUSEKEEPING),
        staff_member("hk2", StaffRole.HOUSEKEEPING),
    )
    items = prioritize_cleaning(
        di, _oracle(di), config=CONFIG, rules=demo_compiled(), staff_members=members
    )
    bays = [i.bay.root for i in items if i.bay is not None]
    staff = [i.staff.root for i in items if i.staff is not None]
    assert len(bays) == len(set(bays))  # each bay cleaned at most once
    assert len(staff) == len(set(staff))  # each housekeeper assigned at most once
    assert set(bays) == {"bay-1", "bay-3"}


def test_housekeeper_without_compiled_cleaning_skills_not_assigned() -> None:
    # Regression (review finding 6): the candidate filter was role-only,
    # ignoring rules.skills_for("cleaning") — so a hazmat-requiring rule set
    # could emit a clean the validator would reject. The NEARER housekeeper
    # here lacks hazmat; only the qualified one may be assigned.
    rules = compile_rules(
        (*demo_rules(), SkillRule(task_kind="cleaning", required_skills=frozenset({"hazmat"})))
    )
    di = decision_input(
        waiting_patients=(waiting(make_patient("p1", EsiAcuity.ESI1), 60),),
        bays=(bay_state("bay-3", BayStatus.CLEANING),),
        staff=(staff_state("hk-haz", at="b2"), staff_state("hk-plain", at="gstat")),
    )
    plain = staff_member("hk-plain", StaffRole.HOUSEKEEPING)
    haz = staff_member("hk-haz", StaffRole.HOUSEKEEPING, skills=frozenset({"hazmat"}))
    items = prioritize_cleaning(
        di, _oracle(di), config=CONFIG, rules=rules, staff_members=(plain, haz)
    )
    assert [
        (i.staff.root, i.bay.root) for i in items if i.staff is not None and i.bay is not None
    ] == [("hk-haz", "bay-3")]
    # An unqualified-only pool cleans nothing (feasible partial, never a breach).
    alone = prioritize_cleaning(di, _oracle(di), config=CONFIG, rules=rules, staff_members=(plain,))
    assert alone == ()


def test_unblock_demand_counts_only_bay_waiting_stages() -> None:
    # Regression (review finding 9): value(b) summed ALL waiting patients,
    # including post-placement stages (waiting for providers/labs/docs) that a
    # clean does not unblock. A lone ESI-1 waiting for labs is NOT bay demand:
    # the clean has no net value and no assignment is emitted.
    di = decision_input(
        waiting_patients=(waiting(make_patient("p1", EsiAcuity.ESI1), 60, stage="awaiting_labs"),),
        bays=(bay_state("bay-3", BayStatus.CLEANING),),
        staff=(staff_state("hk1", at="gstat"),),
    )
    members = (staff_member("hk1", StaffRole.HOUSEKEEPING),)
    items = prioritize_cleaning(
        di, _oracle(di), config=CONFIG, rules=demo_compiled(), staff_members=members
    )
    assert items == ()


def test_only_housekeeping_role_assigned() -> None:
    di = decision_input(
        waiting_patients=(waiting(make_patient("p1", EsiAcuity.ESI1), 60),),
        bays=(bay_state("bay-3", BayStatus.CLEANING),),
        staff=(staff_state("nurse1", at="gstat"),),
    )
    members = (staff_member("nurse1", StaffRole.NURSE),)
    assert (
        prioritize_cleaning(
            di, _oracle(di), config=CONFIG, rules=demo_compiled(), staff_members=members
        )
        == ()
    )
