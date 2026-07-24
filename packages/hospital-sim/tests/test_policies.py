"""Baseline policies — deterministic, myopic, competent; factory arm selection."""

from __future__ import annotations

import pytest
from _sim_fixtures import build_physics, make_patient, tiny_rules

from hospital.core import (
    Activity,
    CapacityRule,
    CompatibilityRule,
    EsiAcuity,
    Plan,
    Rule,
    SimTime,
    SkillRule,
    StaffId,
    StaffMember,
    StaffRole,
    ZoneType,
    compile_rules,
    minutes,
    validate,
)
from hospital.sim.policies.factory import make_policies
from hospital.sim.seam_adapter import build_decision_input, validation_context
from hospital.solver import GraphRoutingOracle, ObjectiveConfig


class TestFirstAvailablePlacement:
    def test_fixed_bay_order_and_acuity_priority(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)

        routine = make_patient("p_routine", esi=EsiAcuity.ESI3, arrival_s=0.0)
        critical = make_patient("p_critical", esi=EsiAcuity.ESI1, arrival_s=60.0)
        for p in (routine, critical):
            h.world.register_patient(p)
            h.world.request_bay(p, stage="triage->bay")

        di = build_decision_input(h.world, SimTime(0), ())
        items = policies.placement.place(di, oracle)

        by_patient = {i.patient.root: i for i in items if i.patient is not None}
        assert set(by_patient) == {"p_routine", "p_critical"}
        # the critical patient placed in resus, the routine one in the first
        # free compatible bay in fixed BayId order
        critical_bay = by_patient["p_critical"].bay
        routine_bay = by_patient["p_routine"].bay
        assert critical_bay is not None and routine_bay is not None
        assert h.world.bay(critical_bay).zone_type.value == "resus_trauma"
        assert routine_bay == h.world.free_compatible_bays(routine, rules)[0]

    def test_no_compatible_capacity_leaves_patient_waiting(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)

        # occupy every bay an ESI3 may enter
        p = make_patient("p_blocked", esi=EsiAcuity.ESI3)
        for i, bay in enumerate(h.world.free_compatible_bays(p, rules)):
            h.world.assign_bay(bay, make_patient(f"occ{i}").id)
        h.world.register_patient(p)
        h.world.request_bay(p, stage="triage->bay")

        di = build_decision_input(h.world, SimTime(0), ())
        assert policies.placement.place(di, oracle) == ()

    def test_isolation_patient_only_gets_isolation_capable_bay(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)
        p = make_patient("p_iso", esi=EsiAcuity.ESI3, isolation=True)
        h.world.register_patient(p)
        h.world.request_bay(p, stage="triage->bay")
        di = build_decision_input(h.world, SimTime(0), ())
        items = policies.placement.place(di, oracle)
        assert len(items) == 1
        assert items[0].bay is not None
        assert h.world.bay(items[0].bay).isolation_capable


class TestCapacityRule:
    """Finding 2: placement must honor a zone cap below the physical bay count."""

    def test_capacity_rule_bounds_placements_including_occupants(self) -> None:
        h = build_physics()
        rules_tuple: tuple[Rule, ...] = (
            CompatibilityRule(allowed_zone_types=frozenset({(EsiAcuity.ESI3, ZoneType.GENERAL)})),
            CapacityRule(zone_type=ZoneType.GENERAL, max_occupancy=2),
        )
        rules = compile_rules(rules_tuple)
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)

        general = [b.id for b in h.layout.bays if b.zone_type is ZoneType.GENERAL]
        assert len(general) >= 3  # the cap binds below the physical count
        # one occupant already in the zone -> only ONE cap slot remains
        h.world.assign_bay(general[0], make_patient("occ").id)
        waiting = [make_patient(f"p{i}", esi=EsiAcuity.ESI3) for i in range(3)]
        for p in waiting:
            h.world.register_patient(p)
            h.world.request_bay(p, stage="triage->bay")

        di = build_decision_input(h.world, SimTime(0), ())
        items = policies.placement.place(di, oracle)

        # Before the fix: 3 items -> validator capacity_exceeded -> identical
        # deterministic retry -> ZeroTimeCycle. The occupant AND earlier items
        # in the same plan both count against the cap.
        assert len(items) == 1
        plan = Plan(items=items)
        assert validate(plan, validation_context(h.world, rules)) == ()


class TestSkillRule:
    """Finding 3: dispatch must union compiled rule skills into qualification."""

    def _skill_rules(self) -> tuple[Rule, ...]:
        return (
            CompatibilityRule(allowed_zone_types=frozenset({(EsiAcuity.ESI3, ZoneType.GENERAL)})),
            SkillRule(task_kind="cleaning", required_skills=frozenset({"biohazard"})),
        )

    def test_rule_unqualified_staff_is_never_dispatched(self) -> None:
        h = build_physics()
        rules = compile_rules(self._skill_rules())
        oracle = GraphRoutingOracle(h.layout.graph)
        station = h.layout.stations[0]
        plain = StaffMember(
            id=StaffId("hk_plain"),
            role=StaffRole.HOUSEKEEPING,
            home_station=station,
            skills=frozenset(),
        )
        skilled = StaffMember(
            id=StaffId("hk_bio"),
            role=StaffRole.HOUSEKEEPING,
            home_station=station,
            skills=frozenset({"biohazard"}),
        )
        h.world.register_staff(plain)
        h.world.register_staff(skilled)
        task = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=h.layout.bays[0].node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=h.layout.bays[0].id,
        )
        # the unqualified housekeeper is NEARER — the old skill check would pick it
        h.world.set_staff_position(plain.id, task.spec.at)

        policies = make_policies(
            "baseline", oracle=oracle, rules=rules, roster=(plain, skilled)
        )
        di = build_decision_input(h.world, SimTime(0), ())
        items = policies.dispatch.dispatch(di, oracle)

        assert len(items) == 1
        assert items[0].staff == skilled.id  # rule skills won, not raw distance
        plan = Plan(items=items)
        assert validate(plan, validation_context(h.world, rules)) == ()

    def test_nobody_rule_qualified_leaves_the_task_pending(self) -> None:
        h = build_physics()
        rules = compile_rules(self._skill_rules())
        oracle = GraphRoutingOracle(h.layout.graph)
        plain = StaffMember(
            id=StaffId("hk_plain"),
            role=StaffRole.HOUSEKEEPING,
            home_station=h.layout.stations[0],
            skills=frozenset(),
        )
        h.world.register_staff(plain)
        task = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=h.layout.bays[0].node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=h.layout.bays[0].id,
        )
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=(plain,))
        di = build_decision_input(h.world, SimTime(0), ())
        # before the fix: a rule-unqualified dispatch -> validator reject -> retry loop
        assert policies.dispatch.dispatch(di, oracle) == ()
        assert task.spec in di.pending_tasks


class TestNearestIdleDispatch:
    def test_nearest_idle_qualified_staff_wins(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)

        p = make_patient("p1")
        h.world.register_patient(p)
        target = h.layout.bays[0].node
        task = h.world.add_task(
            kind="nurse_visit",
            patient=p.id,
            at=target,
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=minutes(5),
        )
        nurses = [m for m in h.roster if m.role is StaffRole.NURSE]
        assert len(nurses) >= 2
        # park one nurse right at the target, the rest far away
        near, far = nurses[0], nurses[1:]
        h.world.set_staff_position(near.id, target)
        for m in far:
            h.world.set_staff_position(m.id, h.layout.entrances[-1])

        di = build_decision_input(h.world, SimTime(0), ())
        items = policies.dispatch.dispatch(di, oracle)
        assert len(items) == 1
        assert items[0].staff == near.id
        assert items[0].task == task.spec.id

    def test_busy_and_wrong_role_staff_are_skipped(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)

        p = make_patient("p1")
        h.world.register_patient(p)
        task = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=h.layout.bays[0].node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=h.layout.bays[0].id,
        )
        housekeeper = next(m for m in h.roster if m.role is StaffRole.HOUSEKEEPING)
        # the only housekeeper is dispatched elsewhere -> nobody qualifies
        h.world.dispatch_task(task.spec.id, housekeeper.id)
        another = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=h.layout.bays[1].node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=h.layout.bays[1].id,
        )
        di = build_decision_input(h.world, SimTime(0), ())
        assert policies.dispatch.dispatch(di, oracle) == ()
        assert another.spec in di.pending_tasks

    def test_one_task_per_staff_per_tick(self) -> None:
        h = build_physics()
        rules = tiny_rules()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=rules, roster=h.roster)
        p = make_patient("p1")
        h.world.register_patient(p)
        for i in range(3):
            h.world.add_task(
                kind="nurse_visit",
                patient=p.id,
                at=h.layout.bays[i].node,
                required_role=StaffRole.NURSE,
                activity=Activity.NURSE_VISIT,
                duration=minutes(5),
            )
        di = build_decision_input(h.world, SimTime(0), ())
        items = policies.dispatch.dispatch(di, oracle)
        staff_ids = [i.staff for i in items]
        assert len(staff_ids) == len(set(staff_ids))  # never two tasks to one staff


class TestComposition:
    def test_empty_levers_yield_keep(self) -> None:
        h = build_physics()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=tiny_rules(), roster=h.roster)
        di = build_decision_input(h.world, SimTime(0), ())
        resp = policies.decide(di, oracle)
        assert resp.mode == "keep"
        assert resp.plan is None

    def test_items_compose_into_one_replace_plan(self) -> None:
        h = build_physics()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=tiny_rules(), roster=h.roster)
        p = make_patient("p1")
        h.world.register_patient(p)
        h.world.request_bay(p, stage="triage->bay")
        di = build_decision_input(h.world, SimTime(0), ())
        resp = policies.decide(di, oracle)
        assert resp.mode == "replace"
        assert resp.plan is not None
        assert [i.kind for i in resp.plan.items] == ["assign_bay"]

    def test_fifo_levers_emit_no_churn(self) -> None:
        h = build_physics()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=tiny_rules(), roster=h.roster)
        di = build_decision_input(h.world, SimTime(0), ())
        assert policies.sequencing.sequence(di) == ()
        assert policies.turnaround.turnaround(di) == ()
        assert policies.discharge.discharge(di) == ()
        assert policies.staffing.staffing(di) == ()


class TestFactory:
    def test_baseline_arm_is_wired(self) -> None:
        h = build_physics()
        oracle = GraphRoutingOracle(h.layout.graph)
        policies = make_policies("baseline", oracle=oracle, rules=tiny_rules(), roster=h.roster)
        assert policies.origin == "baseline"

    def test_optimized_arm_is_next_phase(self) -> None:
        h = build_physics()
        oracle = GraphRoutingOracle(h.layout.graph)
        with pytest.raises(NotImplementedError, match="optimized arm"):
            make_policies(
                "optimized",
                oracle=oracle,
                rules=tiny_rules(),
                roster=h.roster,
                objective=ObjectiveConfig(),
            )

    def test_optimized_arm_requires_an_objective(self) -> None:
        h = build_physics()
        oracle = GraphRoutingOracle(h.layout.graph)
        with pytest.raises(ValueError, match="requires an ObjectiveConfig"):
            make_policies("optimized", oracle=oracle, rules=tiny_rules(), roster=h.roster)
