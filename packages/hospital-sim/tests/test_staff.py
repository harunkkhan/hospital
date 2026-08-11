"""Staff agent — idle/travel/serve/idle, escorted transport, absence interrupts."""

from __future__ import annotations

from collections.abc import Generator

import simpy
from _sim_fixtures import PhysicsHarness, build_physics, make_patient

from hospital.core import (
    Activity,
    BayCleaningCompleted,
    BayCleaningStarted,
    BayStatus,
    Duration,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientMoved,
    SimTime,
    StaffIdle,
    StaffMember,
    StaffMoved,
    StaffRole,
    TimeWindow,
    compile_rules,
    hours,
    minutes,
)
from hospital.sim.flow.staff import staff_process
from hospital.sim.physics.executor import PriorityTier
from hospital.sim.physics.world import SimTask
from hospital.sim.policies.baseline import NearestIdleDispatch
from hospital.sim.seam_adapter import build_decision_input
from hospital.solver.oracle import GraphRoutingOracle


def _start_staff(h: PhysicsHarness, member: StaffMember) -> simpy.Process:
    return h.env.process(
        staff_process(h.env, h.world, h.executor, h.log, member, h.resources.mailboxes[member.id])
    )


def _nurse(h: PhysicsHarness) -> StaffMember:
    return next(m for m in h.roster if m.role is StaffRole.NURSE)


def _bedside_task(h: PhysicsHarness, pid: str = "p1") -> SimTask:
    p = make_patient(pid)
    h.world.register_patient(p)
    bay = h.layout.bays[0]
    return h.world.add_task(
        kind="nurse_visit",
        patient=p.id,
        at=bay.node,
        required_role=StaffRole.NURSE,
        activity=Activity.NURSE_VISIT,
        duration=minutes(5),
        esi=p.esi,
        bay=bay.id,
    )


class TestServeCycle:
    def test_idle_travel_serve_idle(self) -> None:
        h = build_physics()
        nurse = _nurse(h)
        _start_staff(h, nurse)
        task = _bedside_task(h)
        h.env.run(until=1)
        h.world.dispatch_task(task.spec.id, nurse.id)
        h.env.run(until=hours(2).root)

        assert task.completed
        assert task.done.triggered
        # travelled hop-by-hop to the bedside, then stayed there idle
        expected = h.layout.graph.dijkstra(nurse.home_station, task.spec.at)
        moves = [e.event for e in h.log if isinstance(e.event, StaffMoved)]
        assert [m.edge for m in moves] == list(
            zip(expected.nodes, expected.nodes[1:], strict=False)
        )
        assert h.world.staff_at(nurse.id) == task.spec.at
        # service pair with caused_by linkage
        started = next(e for e in h.log if isinstance(e.event, NurseVisitStarted))
        completed = next(e for e in h.log if isinstance(e.event, NurseVisitCompleted))
        assert completed.caused_by == started.sequence
        assert completed.event.occurred_at - started.event.occurred_at == minutes(5)
        # idle on entry AND idle again after the serve
        idles = [e.event for e in h.log if isinstance(e.event, StaffIdle)]
        assert len(idles) == 2
        assert idles[-1].at == task.spec.at

    def test_transport_escorts_the_patient(self) -> None:
        h = build_physics()
        porter = next(m for m in h.roster if m.role is StaffRole.PORTER)
        _start_staff(h, porter)
        p = make_patient("p1")
        h.world.register_patient(p)
        start = h.layout.entrances[0]
        dest = h.layout.bays[0].node
        h.world.set_patient_position(p.id, start)
        task = h.world.add_task(
            kind="transport",
            patient=p.id,
            at=start,
            required_role=StaffRole.PORTER,
            activity=Activity.TRANSPORT,
            duration=minutes(0),
            esi=p.esi,
            destination=dest,
        )
        h.env.run(until=1)
        h.world.dispatch_task(task.spec.id, porter.id)
        h.env.run(until=hours(2).root)

        assert task.completed
        assert h.world.patient_at(p.id) == dest
        assert h.world.staff_at(porter.id) == dest  # the escort physically walked too
        escort_leg = h.layout.graph.dijkstra(start, dest)
        patient_moves = [e.event for e in h.log if isinstance(e.event, PatientMoved)]
        assert [m.edge for m in patient_moves] == list(
            zip(escort_leg.nodes, escort_leg.nodes[1:], strict=False)
        )


class TestAbsence:
    def test_idle_absence_pauses_then_resumes(self) -> None:
        h = build_physics()
        nurse = _nurse(h)
        proc = _start_staff(h, nurse)
        h.env.run(until=1)
        until = SimTime(hours(1).root)
        h.world.set_absent(nurse.id, until)
        proc.interrupt()
        h.env.run(until=hours(1).root + 1)

        idles = [e.event for e in h.log if isinstance(e.event, StaffIdle)]
        assert [i.occurred_at.root for i in idles] == [0, until.root]
        assert h.world.absent_until(nurse.id) is None
        # while absent, the snapshot advertises busy_until so dispatch skips them
        h.world.set_absent(nurse.id, SimTime(hours(2).root))
        state = next(s for s in h.world.snapshot_staff() if s.staff == nurse.id)
        assert state.busy_until == SimTime(hours(2).root)

    def test_mid_serve_interrupt_closes_the_pair_and_requeues(self) -> None:
        h = build_physics()
        nurse = _nurse(h)
        proc = _start_staff(h, nurse)
        task = _bedside_task(h)
        h.env.run(until=1)
        h.world.dispatch_task(task.spec.id, nurse.id)

        walk = h.layout.graph.dijkstra(nurse.home_station, task.spec.at).total
        interrupt_at = walk.root + minutes(2).root  # two minutes into the 5-min visit
        absence_end = interrupt_at + minutes(30).root

        def interrupter() -> Generator[simpy.Event, object]:
            yield h.executor.delay(
                SimTime(interrupt_at) - h.executor.now(), PriorityTier.DISRUPTION
            )
            h.world.set_absent(nurse.id, SimTime(absence_end))
            proc.interrupt()

        h.env.process(interrupter())
        h.env.run(until=absence_end - 1)

        # the started visit was truthfully closed at the interruption instant
        started = next(e for e in h.log if isinstance(e.event, NurseVisitStarted))
        completed = next(e for e in h.log if isinstance(e.event, NurseVisitCompleted))
        assert completed.caused_by == started.sequence
        assert completed.event.occurred_at.root == interrupt_at
        # the unfinished work went back to pending — never silently dropped
        assert not task.completed
        assert h.world.pending_tasks() == (task.spec,)

        # after the absence the agent returns; re-dispatch completes the task
        h.env.run(until=absence_end + 1)
        h.world.dispatch_task(task.spec.id, nurse.id)
        h.env.run(until=absence_end + hours(1).root)
        assert task.completed
        pairs = [e.event for e in h.log if isinstance(e.event, NurseVisitCompleted)]
        assert len(pairs) == 2  # the cut-short visit and the full re-serve

    def test_interrupted_cleaning_emits_no_terminal_event(self) -> None:
        # Finding 7: an absence mid-clean must NOT emit BayCleaningCompleted —
        # the bay is still CLEANING and the task requeues; the terminal event
        # is the bay-cycle closer downstream, so it belongs to the re-clean.
        h = build_physics()
        hk = next(m for m in h.roster if m.role is StaffRole.HOUSEKEEPING)
        proc = _start_staff(h, hk)
        bay = h.layout.bays[0]
        h.world.assign_bay(bay.id, make_patient("p1").id)
        h.world.vacate_bay(bay.id)  # OCCUPIED -> CLEANING: a dirty bay
        task = h.world.add_task(
            kind="cleaning",
            patient=None,
            at=bay.node,
            required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING,
            duration=minutes(10),
            bay=bay.id,
        )
        h.env.run(until=1)
        h.world.dispatch_task(task.spec.id, hk.id)

        walk = h.layout.graph.dijkstra(hk.home_station, task.spec.at).total
        interrupt_at = walk.root + minutes(2).root  # two minutes into the clean
        absence_end = interrupt_at + minutes(30).root

        def interrupter() -> Generator[simpy.Event, object]:
            yield h.executor.delay(
                SimTime(interrupt_at) - h.executor.now(), PriorityTier.DISRUPTION
            )
            h.world.set_absent(hk.id, SimTime(absence_end))
            proc.interrupt()

        h.env.process(interrupter())
        h.env.run(until=absence_end - 1)

        # the started clean is left open — no early bay-cycle close
        assert any(isinstance(e.event, BayCleaningStarted) for e in h.log)
        assert not any(isinstance(e.event, BayCleaningCompleted) for e in h.log)
        assert h.world.bay_status(bay.id) is BayStatus.CLEANING
        assert h.world.pending_tasks() == (task.spec,)  # requeued, never dropped

        # the re-clean emits the ONE terminal event and frees the bay
        h.env.run(until=absence_end + 1)
        h.world.dispatch_task(task.spec.id, hk.id)
        h.env.run(until=absence_end + hours(1).root)
        assert task.completed
        completions = [e.event for e in h.log if isinstance(e.event, BayCleaningCompleted)]
        assert len(completions) == 1
        assert h.world.bay_status(bay.id) is BayStatus.FREE


class TestShiftAwareness:
    """Off-shift rides the absence channel, so no policy learned about schedules."""

    @staticmethod
    def _rostered(h: PhysicsHarness, member: StaffMember, *shifts: TimeWindow) -> StaffMember:
        """Re-register ``member`` carrying a schedule, replacing the unscheduled one."""
        scheduled = member.model_copy(update={"shifts": shifts})
        h.world.register_staff(scheduled)
        return scheduled

    def test_an_unscheduled_member_is_available_exactly_as_before(self) -> None:
        """The default path: no schedule, no busy_until, byte-identical behaviour."""
        h = build_physics()
        nurse = _nurse(h)
        assert not nurse.shifts
        assert h.world.unavailable_until(nurse.id) is None
        state = next(s for s in h.world.snapshot_staff() if s.staff == nurse.id)
        assert state.busy_until is None

    def test_off_shift_surfaces_as_busy_until_the_next_shift_starts(self) -> None:
        """The whole mechanism: dispatch already skips a future busy_until, so it skips
        an off-shift member without knowing shifts exist."""
        h = build_physics()
        nurse = _nurse(h)
        self._rostered(
            h,
            nurse,
            TimeWindow(start=SimTime(hours(4).root), end=SimTime(hours(6).root)),
        )
        # now == 0, before the shift: unavailable until it starts.
        assert h.world.unavailable_until(nurse.id) == SimTime(hours(4).root)
        state = next(s for s in h.world.snapshot_staff() if s.staff == nurse.id)
        assert state.busy_until == SimTime(hours(4).root)

    def test_on_shift_is_available(self) -> None:
        h = build_physics()
        nurse = _nurse(h)
        self._rostered(h, nurse, TimeWindow(start=SimTime(0), end=SimTime(hours(6).root)))
        assert h.world.unavailable_until(nurse.id) is None

    def test_a_member_with_no_remaining_shift_stays_unavailable(self) -> None:
        """ "No more shifts" must not read as "available" — hence the far-future sentinel."""
        h = build_physics()
        nurse = _nurse(h)
        self._rostered(h, nurse, TimeWindow(start=SimTime(0), end=SimTime(hours(1).root)))
        h.env.run(until=hours(2).root)
        resumes = h.world.unavailable_until(nurse.id)
        assert resumes is not None
        assert resumes.root > hours(24 * 7).root

    def test_absence_and_off_shift_take_the_later_instant(self) -> None:
        """Someone off shift *and* off sick is not back at the earlier of the two."""
        h = build_physics()
        nurse = _nurse(h)
        self._rostered(
            h,
            nurse,
            TimeWindow(start=SimTime(hours(2).root), end=SimTime(hours(6).root)),
        )
        h.world.set_absent(nurse.id, SimTime(hours(5).root))
        # Shift starts at 2h but the absence runs to 5h: available at 5h, not 2h.
        assert h.world.unavailable_until(nurse.id) == SimTime(hours(5).root)

    def test_the_baseline_dispatch_lever_skips_an_off_shift_member(self) -> None:
        """The end of the argument for reusing busy_until: an unmodified policy honours it."""
        h = build_physics()
        nurse = _nurse(h)
        self._rostered(
            h,
            nurse,
            TimeWindow(start=SimTime(hours(4).root), end=SimTime(hours(6).root)),
        )
        roster = tuple(
            m.model_copy(
                update={
                    "shifts": (
                        TimeWindow(start=SimTime(hours(4).root), end=SimTime(hours(6).root)),
                    )
                }
            )
            if m.id == nurse.id
            else m
            for m in h.roster
        )
        lever = NearestIdleDispatch(rules=compile_rules(()), roster=roster)
        h.world.add_task(
            kind="nurse_visit",
            patient=None,
            at=h.layout.stations[0],
            required_role=StaffRole.NURSE,
            activity=Activity.NURSE_VISIT,
            duration=Duration(0),
        )
        di = build_decision_input(h.world, h.world.now(), ())
        items = lever.dispatch(di, GraphRoutingOracle(h.layout.graph))
        assert items, "nobody was dispatched at all — the fixture would prove nothing"
        assert all(item.staff != nurse.id for item in items), "an off-shift nurse was dispatched"
