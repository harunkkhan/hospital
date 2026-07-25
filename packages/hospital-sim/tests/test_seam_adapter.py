"""Seam adapter — pure projection, validate-then-apply, reject-never-repair."""

from __future__ import annotations

import pytest
from _sim_fixtures import PhysicsHarness, build_physics, make_patient, tiny_rules

from hospital.core import (
    Activity,
    BayAssigned,
    BayStatus,
    DecisionInput,
    EsiAcuity,
    InfeasiblePlan,
    Plan,
    PlanItem,
    SimTime,
    StaffRole,
    minutes,
)
from hospital.sim.seam_adapter import apply_plan, build_decision_input, validation_context


def _free_bay(h: PhysicsHarness, zone_type: str) -> str:
    for b in h.layout.bays:
        if b.zone_type.value == zone_type:
            return b.id.root
    raise AssertionError(f"no {zone_type} bay on the tiny floor")


class TestProjection:
    def test_decision_input_mirrors_world_with_no_hidden_fields(self) -> None:
        h = build_physics()
        p = make_patient("p1")
        h.world.register_patient(p)
        h.world.request_bay(p, stage="waiting_for_bay")
        task = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=h.layout.bays[0].node,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )

        di = build_decision_input(h.world, SimTime(0), ())

        assert di.bays == h.world.snapshot_bays()
        assert di.staff == h.world.snapshot_staff()
        assert di.waiting == h.world.waiting_for_bay()
        assert di.pending_tasks == h.world.pending_tasks()
        assert di.pending_tasks == (task.spec,)
        # the projection is exactly the seam contract — no extra fields smuggled in
        assert set(DecisionInput.model_fields) == {
            "now",
            "layout",
            "waiting",
            "bays",
            "staff",
            "pending_tasks",
            "events_since",
        }
        # physics-only task facts (duration/destination/bay) never reach the
        # spec; acuity does (observable post-triage — dispatch's urgency signal)
        assert set(type(task.spec).model_fields) == {
            "id",
            "kind",
            "patient",
            "at",
            "required_role",
            "required_skills",
            "ready_at",
            "esi",
        }


class TestRejectNeverRepair:
    def test_infeasible_plan_raises_and_mutates_nothing(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1", esi=EsiAcuity.ESI5)  # ESI5 may not enter resus
        h.world.register_patient(p)
        wake = h.world.request_bay(p, stage="waiting_for_bay")
        resus_bay = _free_bay(h, "resus_trauma")
        plan = Plan(
            items=(
                PlanItem(
                    stable_id=f"assign:{p.id.root}",
                    kind="assign_bay",
                    patient=p.id,
                    bay=h.world.bay(next(b.id for b in h.layout.bays if b.id.root == resus_bay)).id,
                ),
            )
        )
        bays_before = h.world.snapshot_bays()
        waiting_before = h.world.waiting_for_bay()

        ctx = validation_context(h.world, rules)
        with pytest.raises(InfeasiblePlan) as exc:
            apply_plan(h.world, plan, ctx, h.executor, h.log)

        assert exc.value.violations  # concrete violations ride the exception
        assert h.world.snapshot_bays() == bays_before
        assert h.world.waiting_for_bay() == waiting_before
        assert not wake.triggered
        assert len(h.log) == 0

    def test_unknown_patient_is_always_unknown(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        plan = Plan(
            items=(
                PlanItem(
                    stable_id="assign:ghost",
                    kind="assign_bay",
                    patient=make_patient("ghost").id,
                    bay=h.layout.bays[0].id,
                ),
            )
        )
        ctx = validation_context(h.world, rules)
        with pytest.raises(InfeasiblePlan):
            apply_plan(h.world, plan, ctx, h.executor, h.log)


class TestPreflight:
    """Finding 1: dynamic dispatchability is judged BEFORE the first mutation."""

    def test_busy_staff_dispatch_rejects_whole_plan_before_any_mutation(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1", esi=EsiAcuity.ESI3)
        h.world.register_patient(p)
        wake = h.world.request_bay(p, stage="waiting_for_bay")
        bay = h.world.free_compatible_bays(p, rules)[0]
        nurse = next(m for m in h.roster if m.role is StaffRole.NURSE)
        first = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=h.layout.bays[0].node,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )
        h.world.dispatch_task(first.spec.id, nurse.id)  # the nurse is now mid-task
        second = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=h.layout.bays[1].node,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )
        # A feasible assign_bay FOLLOWED by a dispatch to the busy nurse: the
        # validator cannot see mid-task staff, so before the fix the bay was
        # granted (mutation!) and then _enact_dispatch raised a bare ValueError.
        plan = Plan(
            items=(
                PlanItem(stable_id=f"assign:{p.id.root}", kind="assign_bay", patient=p.id, bay=bay),
                PlanItem(
                    stable_id=f"dispatch:{second.spec.id.root}",
                    kind="dispatch",
                    staff=nurse.id,
                    task=second.spec.id,
                ),
            )
        )
        with pytest.raises(InfeasiblePlan) as exc:
            apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert any(v.kind == "double_booked" for v in exc.value.violations)
        # nothing was applied: the earlier assign_bay item did NOT land
        assert h.world.bay_status(bay) is BayStatus.FREE
        assert not wake.triggered
        assert h.world.is_waiting(p.id)
        assert h.world.is_pending(second.spec.id)
        assert not any(isinstance(e.event, BayAssigned) for e in h.log)

    def test_no_longer_pending_task_rejects_as_infeasible(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1")
        h.world.register_patient(p)
        nurses = [m for m in h.roster if m.role is StaffRole.NURSE]
        assert len(nurses) >= 2
        task = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=h.layout.bays[0].node,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )
        h.world.dispatch_task(task.spec.id, nurses[0].id)  # taken between solve and apply
        plan = Plan(
            items=(
                PlanItem(
                    stable_id=f"dispatch:{task.spec.id.root}",
                    kind="dispatch",
                    staff=nurses[1].id,
                    task=task.spec.id,
                ),
            )
        )
        # Before the fix this raised a bare ValueError ("task not pending"),
        # which _make_tick does not catch — the whole run aborted.
        with pytest.raises(InfeasiblePlan) as exc:
            apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert any(v.kind == "double_booked" for v in exc.value.violations)
        assert len(h.resources.mailboxes[nurses[1].id].items) == 0

    def test_patient_no_longer_waiting_rejects_as_infeasible(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1", esi=EsiAcuity.ESI3)
        h.world.register_patient(p)
        free = h.world.free_compatible_bays(p, rules)
        h.world.request_bay(p, stage="waiting_for_bay")
        h.world.grant_bay(p.id, free[0])  # granted between solve and apply
        plan = Plan(
            items=(
                PlanItem(
                    stable_id=f"assign:{p.id.root}", kind="assign_bay", patient=p.id, bay=free[1]
                ),
            )
        )
        with pytest.raises(InfeasiblePlan) as exc:
            apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert any(v.kind == "unknown_entity" for v in exc.value.violations)
        assert h.world.bay_status(free[1]) is BayStatus.FREE


class TestHappyPath:
    def test_assign_bay_grants_and_emits_with_origin(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1", esi=EsiAcuity.ESI3)
        h.world.register_patient(p)
        wake = h.world.request_bay(p, stage="waiting_for_bay")
        bay = h.world.free_compatible_bays(p, rules)[0]
        plan = Plan(
            items=(
                PlanItem(stable_id=f"assign:{p.id.root}", kind="assign_bay", patient=p.id, bay=bay),
            )
        )

        apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)

        assert h.world.bay_status(bay) is BayStatus.OCCUPIED
        assert h.world.occupant(bay) == p.id
        assert wake.triggered
        events = [e.event for e in h.log if isinstance(e.event, BayAssigned)]
        assert len(events) == 1
        assert events[0].by == "baseline"

        # idempotent re-affirmation: validates (same-patient carve-out), no churn
        apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert len([e for e in h.log if isinstance(e.event, BayAssigned)]) == 1

    def test_dispatch_puts_task_in_the_named_mailbox(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1")
        h.world.register_patient(p)
        nurse = next(m for m in h.roster if m.role is StaffRole.NURSE)
        task = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=h.layout.bays[0].node,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )
        plan = Plan(
            items=(
                PlanItem(
                    stable_id=f"dispatch:{task.spec.id.root}",
                    kind="dispatch",
                    staff=nurse.id,
                    task=task.spec.id,
                ),
            )
        )

        apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)

        assert list(h.resources.mailboxes[nurse.id].items) == [task]
        assert h.world.pending_tasks() == ()
        busy = next(s for s in h.world.snapshot_staff() if s.staff == nurse.id)
        assert busy.current_task == task.spec.id

        # re-applying the identical dispatch is a no-op, not a double-put
        apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert len(h.resources.mailboxes[nurse.id].items) == 1

    def test_wrong_role_dispatch_is_rejected(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        p = make_patient("p1")
        h.world.register_patient(p)
        porter = next(m for m in h.roster if m.role is StaffRole.PORTER)
        task = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=h.layout.bays[0].node,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )
        plan = Plan(
            items=(
                PlanItem(
                    stable_id=f"dispatch:{task.spec.id.root}",
                    kind="dispatch",
                    staff=porter.id,
                    task=task.spec.id,
                ),
            )
        )
        with pytest.raises(InfeasiblePlan):
            apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert h.world.pending_tasks() == (task.spec,)

    def test_clean_item_boosts_the_bay_cleaning_task(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        bay_a, bay_b = h.layout.bays[0].id, h.layout.bays[1].id
        first = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=h.layout.bays[0].node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=bay_a,
        )
        second = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=h.layout.bays[1].node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=bay_b,
        )
        plan = Plan(items=(PlanItem(stable_id=f"clean:{bay_b.root}", kind="clean", bay=bay_b),))
        apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert h.world.pending_tasks() == (second.spec, first.spec)

    def test_staffing_item_validates_but_enacts_nothing(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        staff = h.roster[0]
        before = h.world.snapshot_staff()
        plan = Plan(
            items=(
                PlanItem(stable_id=f"staffing:{staff.id.root}", kind="staffing", staff=staff.id),
            )
        )
        apply_plan(h.world, plan, validation_context(h.world, rules), h.executor, h.log)
        assert h.world.snapshot_staff() == before
