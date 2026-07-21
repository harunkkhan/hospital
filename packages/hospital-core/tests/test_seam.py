"""Seam: Plan.diff computes added/removed/changed by stable_id."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hospital.core import (
    BayId,
    DecisionResponse,
    PatientId,
    Plan,
    PlanItem,
    SeamViolation,
    StaffId,
    TaskId,
    WakeDirective,
)


def _assign(stable_id: str, patient: str, bay: str) -> PlanItem:
    return PlanItem(
        stable_id=stable_id, kind="assign_bay", patient=PatientId(patient), bay=BayId(bay)
    )


def test_plan_diff_by_stable_id() -> None:
    old = Plan(items=(_assign("a", "p1", "bay-1"), _assign("b", "p2", "bay-2")))
    new = Plan(
        items=(
            _assign("a", "p1", "bay-1"),  # unchanged
            _assign("b", "p2", "bay-3"),  # changed: bay-2 -> bay-3
            PlanItem(stable_id="c", kind="dispatch", staff=StaffId("s1"), task=TaskId("t1")),  # new
        )
    )
    diff = new.diff(old)

    assert {i.stable_id for i in diff.added} == {"c"}
    assert diff.removed == ()
    assert len(diff.changed) == 1
    old_item, new_item = diff.changed[0]
    assert old_item.stable_id == "b" == new_item.stable_id
    assert old_item.bay == BayId("bay-2")
    assert new_item.bay == BayId("bay-3")


def test_plan_diff_detects_removed() -> None:
    old = Plan(items=(_assign("a", "p1", "bay-1"), _assign("b", "p2", "bay-2")))
    new = Plan(items=(_assign("a", "p1", "bay-1"),))
    diff = new.diff(old)
    assert {i.stable_id for i in diff.removed} == {"b"}
    assert diff.added == ()
    assert diff.changed == ()


def test_identical_plans_have_empty_diff() -> None:
    plan = Plan(items=(_assign("a", "p1", "bay-1"),))
    diff = plan.diff(plan)
    assert diff.added == ()
    assert diff.removed == ()
    assert diff.changed == ()


# --- Finding #9: duplicate stable_id makes diff lossy -> rejected at construction ---


def test_duplicate_stable_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Plan(items=(_assign("dup", "p1", "bay-1"), _assign("dup", "p2", "bay-2")))


def test_unique_stable_ids_accepted() -> None:
    plan = Plan(items=(_assign("a", "p1", "bay-1"), _assign("b", "p2", "bay-2")))
    assert len(plan.items) == 2


# --- Finding #11: DecisionResponse mode and plan must agree ---


def _plan() -> Plan:
    return Plan(items=(_assign("a", "p1", "bay-1"),))


def test_replace_without_plan_is_seam_violation() -> None:
    with pytest.raises(SeamViolation):
        DecisionResponse(mode="replace", plan=None, wake=WakeDirective(kind="keep"))


def test_keep_with_plan_is_seam_violation() -> None:
    with pytest.raises(SeamViolation):
        DecisionResponse(mode="keep", plan=_plan(), wake=WakeDirective(kind="keep"))


def test_well_formed_decision_responses_accepted() -> None:
    keep = DecisionResponse(mode="keep", plan=None, wake=WakeDirective(kind="keep"))
    replace = DecisionResponse(mode="replace", plan=_plan(), wake=WakeDirective(kind="keep"))
    assert keep.plan is None
    assert replace.plan is not None
