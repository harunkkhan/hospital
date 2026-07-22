"""The staff agent: idle -> travel -> serve -> idle (doc 04 §3.11 / §4.8).

A staff member is a *positioned* agent, moved only via ``executor.walk`` —
"nurse walking too far" is a first-order measurable cost because the nurse
physically traverses edges. The idle block is **directed dispatch**: the agent
blocks on its own mailbox until the decision layer names it; there is no
"any free nurse grabs the task" race.

Absence (`simpy.Interrupt`, the only in-service interruption in v1):

* interrupted while idle -> the pending ``get`` is cancelled;
* interrupted mid-travel -> the position is truthfully the last completed hop
  (the executor only updates position after a hop finishes);
* interrupted mid-serve -> ``run_service`` closes the ``*_started`` with a
  ``*_completed`` at the interruption instant (the pair discipline holds);
* in every case the unfinished task is **re-queued to pending** (the patient is
  still blocked on ``task.done`` — dropping it would deadlock the flow) and a
  decision is requested so another staff can be dispatched;
* the agent then sleeps out ``world.absent_until`` and returns to idle.

Staff idle in place where they finish (no walk back to ``home_station``);
dispatch may re-target them from wherever they stand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import simpy

from hospital.core import Activity, EventLog, StaffIdle, StaffMember
from hospital.sim.physics.executor import PriorityTier, TaskExecutor
from hospital.sim.physics.world import SimTask

if TYPE_CHECKING:
    from collections.abc import Generator

    from hospital.sim.physics.world import World


def staff_process(
    env: simpy.Environment,
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    staff: StaffMember,
    mailbox: simpy.Store,
) -> Generator[simpy.Event, object]:
    """One SimPy process per staff member, interruptible by absence."""
    sid = staff.id
    while True:
        world.set_staff_idle(sid)
        event_log.append(StaffIdle(occurred_at=executor.now(), staff=sid, at=world.staff_at(sid)))
        world.request_decision()  # an idle staff is dispatchable capacity

        get_ev = mailbox.get()
        try:
            item = yield get_ev
        except simpy.Interrupt:
            if get_ev.triggered:
                # the task raced the interrupt: it is ours on paper — hand it back
                world.requeue_task(cast("SimTask", get_ev.value))
                world.request_decision()
            else:
                get_ev.cancel()
            yield from _absence_leave(world, executor, staff, mailbox)
            continue

        task = cast("SimTask", item)
        try:
            yield from _serve(world, executor, staff, task)
        except simpy.Interrupt:
            world.requeue_task(task)
            world.request_decision()
            yield from _absence_leave(world, executor, staff, mailbox)
            continue
        world.complete_task(task)
        world.request_decision()


def _serve(
    world: World,
    executor: TaskExecutor,
    staff: StaffMember,
    task: SimTask,
) -> Generator[simpy.Event, object]:
    """Travel to the task, then perform it (escorted walk or a paired service)."""
    sid = staff.id
    if task.spec.kind == "transport":
        patient = task.spec.patient
        destination = task.destination
        assert patient is not None and destination is not None
        # walk to the patient's CURRENT position (spec.at can be stale after a
        # requeue), then escort — both actors emit per-edge movement events
        pickup = world.patient_at(patient)
        if world.staff_at(sid) != pickup:
            yield from executor.walk(sid, world.route(world.staff_at(sid), pickup))
        if world.patient_at(patient) != destination:
            yield from executor.walk(
                patient, world.route(world.patient_at(patient), destination), escort=sid
            )
        return
    if world.staff_at(sid) != task.spec.at:
        yield from executor.walk(sid, world.route(world.staff_at(sid), task.spec.at))
    yield from executor.run_service(
        task.activity,
        duration=task.duration,
        patient=task.spec.patient,
        staff=sid,
        bay=task.bay,
        esi=task.esi,
    )
    if task.activity is Activity.CLEANING and task.bay is not None:
        world.free_bay(task.bay)  # CLEANING -> FREE; wakes the placement policy


def _absence_leave(
    world: World,
    executor: TaskExecutor,
    staff: StaffMember,
    mailbox: simpy.Store,
) -> Generator[simpy.Event, object]:
    """Drain the mailbox, sleep out the absence window, return to duty."""
    sid = staff.id
    items = cast("list[object]", mailbox.items)
    while items:
        world.requeue_task(cast("SimTask", items.pop(0)))
        world.request_decision()
    while True:
        until = world.absent_until(sid)
        now = int(executor.env.now)
        if until is None or until.root <= now:
            break
        try:
            yield executor.delay(until - executor.now(), PriorityTier.COMPLETION)
        except simpy.Interrupt:
            continue  # a second absence merely extends the window; keep sleeping
    world.clear_absent(sid)


__all__ = ["staff_process"]
