"""Seam: Plan.diff computes added/removed/changed by stable_id."""

from __future__ import annotations

from hospital.core import BayId, PatientId, Plan, PlanItem, StaffId, TaskId


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
