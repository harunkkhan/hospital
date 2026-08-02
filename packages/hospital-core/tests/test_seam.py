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
    RiskAssessment,
    RiskMonitor,
    SeamViolation,
    SimTime,
    StaffId,
    TaskId,
    VitalsReading,
    VitalsSampled,
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


# --------------------------------------------------------------- RiskMonitor
_READING = VitalsReading(hr=120, spo2=90, sbp=95, dbp=60, temp_c_x10=385, rr=28)


class _AlwaysEscalates:
    """A minimal monitor: structural conformance is the whole contract."""

    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        del reading
        return RiskAssessment(
            patient=event.patient,
            at=event.occurred_at,
            probability=1.0,
            news2=event.news2,
            escalate=True,
        )


class _Undecided:
    def observe(self, event: VitalsSampled, reading: VitalsReading) -> RiskAssessment | None:
        del event, reading
        return None


def test_risk_monitor_is_satisfied_structurally() -> None:
    """`forecast` supplies a monitor without `sim` importing it (doc 06 §3).

    The Protocol is the entire seam, so it must match on shape alone — nothing in
    `forecast` subclasses a `core` base to opt in.
    """
    assert isinstance(_AlwaysEscalates(), RiskMonitor)
    assert isinstance(_Undecided(), RiskMonitor)
    assert not isinstance(object(), RiskMonitor)


def test_a_monitor_may_decline_to_decide() -> None:
    """`None` is a normal answer — usually "the rolling window is not full yet"."""
    event = VitalsSampled(occurred_at=SimTime(10), patient=PatientId("p1"), news2=3)
    assert _Undecided().observe(event, _READING) is None


def test_risk_assessment_carries_a_decided_verdict() -> None:
    """`escalate` is decided by the monitor, never re-derived by the engine.

    The threshold is chosen on validation to hit a target sensitivity; an engine
    that compared `probability` against a constant of its own would silently
    override that choice.
    """
    event = VitalsSampled(occurred_at=SimTime(10), patient=PatientId("p1"), news2=7)
    assessment = _AlwaysEscalates().observe(event, _READING)
    assert assessment is not None
    assert assessment.escalate is True
    assert assessment.patient == event.patient
    assert assessment.at == event.occurred_at


def test_risk_assessment_rejects_an_impossible_probability() -> None:
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            RiskAssessment(
                patient=PatientId("p1"),
                at=SimTime(0),
                probability=bad,
                news2=0,
                escalate=False,
            )
