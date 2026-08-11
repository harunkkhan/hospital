"""``RunSession``/``SessionRegistry``, the playback driver, and ``POST /runs/{id}/control``.

A :class:`RunSession` bundles one arm's live run by composing **exactly** the
``hospital.sim`` primitives the headless ``run_replication`` composition root
wires (doc 07 §5.1): one ``RandomStreams(seed)``, one integer-µs SimPy ``env``,
one ``EventLog``, one ``World`` (the only mutable-state owner), the arm's
``PolicySet``, the seam adapter, and the same default-rules/objective choices.
It reimplements no engine mechanics — the ONE genuinely new thing here is the
playback clock (doc 07 §10): the headless engine calls ``env.run()`` to
completion, while the driver advances the env one scheduled instant at a time
and paces the *human-visible* wall-clock between instants.

The central invariant — **determinism = pacing, not sampling** (doc 07 §5.2):
the ``await``-sleep between instants is outside SimPy, so the realized event
sequence is fixed by ``(scenario, seed, arm)`` and ``speed`` only stretches
wall time. A run at 1x and at 16x yields byte-identical ``EventLog``s; with no
overrides the log is byte-identical to ``run_replication``'s for the same cell
(both are asserted in tests).

Concurrency: one async driver task per session; control, override, and frame
builds are single-writer via ``session.lock`` — the driver and a handler never
advance/mutate the env concurrently (doc 07 nuances 7.5).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from typing import TYPE_CHECKING, Annotated, Literal, cast

import simpy
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from hospital.api.stream import StreamFrame, build_frame, coalesce_frames
from hospital.core import (
    Duration,
    Event,
    EventEnvelope,
    EventLog,
    FrozenModel,
    InfeasiblePlan,
    RandomStreams,
    RunId,
    SimTime,
    TimeWindow,
    compile_rules,
)
from hospital.data.hospital import generate_hospital
from hospital.data.scenario import Scenario, realize_staff
from hospital.data.workload import generate_workload
from hospital.sim.experiment.disruptions import schedule_disruptions
from hospital.sim.experiment.replication import DEFAULT_OBJECTIVE, default_rules
from hospital.sim.flow.patient import patient_process
from hospital.sim.flow.staff import staff_process
from hospital.sim.physics.executor import PriorityTier, TaskExecutor
from hospital.sim.physics.resources import build_resources
from hospital.sim.physics.service_times import ServiceTimes, default_service_table
from hospital.sim.physics.world import World
from hospital.sim.policies.factory import Arm, make_policies
from hospital.sim.seam_adapter import apply_plan, build_decision_input, validation_context
from hospital.solver import GraphRoutingOracle, ObjectiveConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from fastapi import FastAPI

    from hospital.api.overrides import PinRegistry
    from hospital.data.workload import PatientArrival

router = APIRouter()

RunState = Literal["created", "playing", "paused", "stepping", "finished"]
StepGranularity = Literal["decision", "event", "tick"]

# Default playback speed: sim-seconds advanced per wall-second (60 = one sim
# minute per wall second, a watchable week). Pacing only — never sampling.
DEFAULT_SPEED = 60.0

# Frames are coalesced to at most this wall-clock rate while playing so a high
# `speed` cannot flood the socket (doc 07 assumption 8.2). The cadence cap is a
# coalescing rate, not a sampling rate.
MIN_FRAME_INTERVAL_S = 1.0 / 30.0

# Below this wall delay the driver yields to the loop instead of arming a timer.
_MIN_SLEEP_S = 0.002

# Bounded per-subscriber frame queue; overflow coalesces (merging event tails).
_SUBSCRIBER_QUEUE_MAX = 64


# A pacing multiplier divides the wall-clock delay, so it must be finite and
# strictly positive: `inf` collapses every delay to 0 (the driver spins the whole
# horizon with no pacing at all) and `NaN` poisons the comparison
# `delay >= _MIN_SLEEP_S` into False, which does the same thing silently. Both
# are rejected at the boundary rather than sanitized inside the driver.
_SpeedMultiplier = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class ControlCommand(FrozenModel):
    """``POST /runs/{id}/control`` body (doc 07 §3.3)."""

    action: Literal["play", "pause", "step", "speed"]
    multiplier: _SpeedMultiplier | None = None
    granularity: StepGranularity = "decision"
    count: int = Field(default=1, ge=1)


class SessionState(FrozenModel):
    """The authoritative playback state returned by every control call."""

    run: RunId
    state: RunState
    sim_time: SimTime
    speed: float
    horizon: SimTime


class TappedEventLog(EventLog):
    """The run's ``EventLog`` plus two drainable tails (tick + frame).

    ``DecisionInput.events_since`` needs exactly the envelopes appended since
    the previous decision tick, and ``StreamFrame.events`` needs the envelopes
    since the previous delta frame; re-slicing the whole log would be quadratic
    over a week, so the writer taps its own appends (the same pattern as the
    composition root's tap). Serialized bytes are untouched — ``to_jsonl`` is
    inherited verbatim.
    """

    def __init__(self) -> None:
        super().__init__()
        self._since_tick: list[EventEnvelope] = []
        self._since_frame: list[EventEnvelope] = []

    def append(self, e: Event, *, caused_by: int | None = None) -> int:
        sequence = super().append(e, caused_by=caused_by)
        envelope = EventEnvelope(event=e, sequence=sequence, caused_by=caused_by)
        self._since_tick.append(envelope)
        self._since_frame.append(envelope)
        return sequence

    def drain_since_tick(self) -> tuple[EventEnvelope, ...]:
        out = tuple(self._since_tick)
        self._since_tick.clear()
        return out

    def drain_since_frame(self) -> tuple[EventEnvelope, ...]:
        out = tuple(self._since_frame)
        self._since_frame.clear()
        return out


class RunSession:
    """One arm's live run: composed engine + the playback clock (doc 07 §5)."""

    def __init__(
        self,
        run_id: RunId,
        scenario: Scenario,
        arm: Arm,
        seed: int,
        *,
        pins: PinRegistry,
        objective: ObjectiveConfig = DEFAULT_OBJECTIVE,
    ) -> None:
        self.run_id = run_id
        self.scenario = scenario
        self.arm: Arm = arm
        self.seed = seed
        self.pins = pins
        self.horizon = scenario.workload.horizon
        self.state: RunState = "created"
        self.speed = DEFAULT_SPEED
        self.shadow: RunId | None = None
        self.shadow_of: RunId | None = None
        self.lock = asyncio.Lock()
        self.decision_count = 0
        self.frame_seq = 0
        # The displayed clock while coasting the idle tail (see `coast`). None
        # whenever the env's own clock is the authority, which is almost always.
        self._coast_us: int | None = None
        # Set once by `close()` at teardown. A stream transport is a long-lived
        # loop that only ever learns about its OWN peer hanging up; without this
        # it would keep heartbeating a deleted run's world forever (doc 07 §3.2).
        self.closed = False
        self._subscribers: list[asyncio.Queue[StreamFrame | None]] = []
        self._driver: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()

        # --- engine composition: the same wiring, in the same order, as the
        # headless run_replication composition root (steps 1-11) — minus the
        # terminal `env.run(until=end)`, which the driver replaces. ---
        self.streams = RandomStreams(seed)
        # The whole building (see scorecard.fold_scorecard) — a session must render
        # and route over the same floors the engine is simulating.
        self.layout = generate_hospital(scenario.hospital())
        self.env = simpy.Environment()
        self.log = TappedEventLog()
        window = TimeWindow(start=self.horizon.start, end=self.horizon.end)
        self.roster = realize_staff(scenario.staffing, self.layout, window)
        resources = build_resources(self.env, self.layout, self.roster)
        self.world = World(self.env, self.layout, resources, self.log)
        for member in self.roster:
            self.world.register_staff(member)
        service_times = ServiceTimes(self.streams, default_service_table())
        self.executor = TaskExecutor(self.env, self.world, self.log)
        self.oracle = GraphRoutingOracle(self.layout.graph)
        self.rules = compile_rules(scenario.rules if scenario.rules else default_rules())
        self.policies = make_policies(
            arm, oracle=self.oracle, rules=self.rules, roster=self.roster, objective=objective
        )
        self.world.set_decision_hook(self._tick)
        arrivals = generate_workload(
            scenario.workload, self.streams, disruptions=scenario.disruptions
        )
        self.env.process(self._spawn_arrivals(arrivals, service_times))
        staff_processes = {
            member.id: self.env.process(
                staff_process(
                    self.env,
                    self.world,
                    self.executor,
                    self.log,
                    member,
                    resources.mailboxes[member.id],
                )
            )
            for member in self.roster
        }
        schedule_disruptions(
            self.env,
            self.world,
            self.executor,
            self.streams,
            scenario.disruptions.events,
            staff_processes=staff_processes,
            event_log=self.log,
        )

    # ------------------------------------------------------------- projections
    @property
    def finished(self) -> bool:
        """Whether the run has reached its horizon (a fresh read of the clock's end).

        Read through this predicate — never a bare ``state != "finished"`` after
        an ``advance`` — so the finish an advance may have just triggered is
        actually observed rather than narrowed away by the earlier assignment.
        """
        return self.state == "finished"

    @property
    def sim_time(self) -> SimTime:
        if self.state == "finished":
            return self.horizon.end
        if self._coast_us is not None:
            return SimTime(self._coast_us)
        return SimTime(int(self.env.now))

    @property
    def frame_state(self) -> Literal["playing", "paused", "stepping", "finished"]:
        return "paused" if self.state == "created" else self.state

    def session_state(self) -> SessionState:
        return SessionState(
            run=self.run_id,
            state=self.state,
            sim_time=self.sim_time,
            speed=self.speed,
            horizon=self.horizon.end,
        )

    def next_frame_seq(self) -> int:
        self.frame_seq += 1
        return self.frame_seq

    # -------------------------------------------------------- engine plumbing
    def _spawn_arrivals(
        self,
        arrivals: tuple[PatientArrival, ...],
        service_times: ServiceTimes,
    ) -> Generator[simpy.Event, object]:
        """Start each patient process at its arrival instant (composition wiring)."""
        for arrival in arrivals:
            dt = arrival.patient.arrival_time.root - int(self.env.now)
            if dt > 0:
                yield self.executor.delay(Duration(dt), PriorityTier.COMPLETION)
            self.env.process(
                patient_process(
                    self.env,
                    self.world,
                    self.executor,
                    self.log,
                    arrival.patient,
                    service_times=service_times,
                    streams=self.streams,
                )
            )

    def _tick(self) -> None:
        """The decision tick: DecisionInput -> PolicySet -> pin merge -> validate-then-apply.

        Identical to the engine's tick when the pin registry is empty (the
        merge is a pass-through), which is what keeps a no-override API run
        byte-identical to ``run_replication``. Pinned operator decisions are
        merged as pre-fixed ``PlanItem``s so the ONE ``validate()`` still
        governs the whole plan (doc 07 §4.4).
        """
        self.decision_count += 1
        di = build_decision_input(self.world, self.world.now(), self.log.drain_since_tick())
        response = self.policies.decide(di, self.oracle)
        plan = response.plan if response.mode == "replace" else None
        merged = self.pins.merge(plan, self.world)
        if merged is not None:
            ctx = validation_context(self.world, self.rules)
            try:
                apply_plan(
                    self.world, merged, ctx, self.executor, self.log, origin=self.policies.origin
                )
            except InfeasiblePlan as exc:
                # reject-then-re-solve, exactly like the engine tick; a pin that
                # itself went stale is dropped so the retry can converge.
                self.pins.drop_conflicting(exc.violations)
                self.world.request_decision()
        if response.wake.kind == "schedule" and response.wake.at is not None:
            self.world.schedule_decision_at(response.wake.at)

    # ------------------------------------------------------------- advancing
    def _settle_next_instant(self) -> bool:
        """Advance the env to its next scheduled instant and settle it fully.

        Returns ``False`` when the next instant would be at/after the horizon
        end (the half-open ``[start, end)`` convention — an event at exactly
        ``end`` is never executed) or the schedule is exhausted.
        """
        peek = self.env.peek()
        if peek >= self.horizon.end.root:
            return False
        while self.env.peek() == peek:
            self.env.step()
        return True

    def _step_one_event(self) -> bool:
        """Process exactly one scheduled SimPy event (may not advance time)."""
        if self.env.peek() >= self.horizon.end.root:
            return False
        self.env.step()
        return True

    # --------------------------------------------------------- the idle tail
    @property
    def coasting(self) -> bool:
        """Whether the clock is walking out an exhausted schedule's remaining time."""
        return self._coast_us is not None

    def coast_chunk_us(self) -> int:
        """One frame's worth of sim time at the current ``speed``."""
        return max(1, int(MIN_FRAME_INTERVAL_S * self.speed * 1_000_000))

    def coast(self, chunk_us: int) -> int:
        """Walk the displayed clock over an exhausted schedule; returns µs moved (0 at the end).

        A schedule can run dry well before the horizon — arrivals are done and
        every staff member is blocked on an empty mailbox — and that does NOT mean
        the run is over: the *horizon* ends it, and the remainder is real idle time
        the week paid for. Declaring ``finished`` right there teleports the clock
        from, say, hour 3 to hour 168 inside a single frame, which the console
        renders as the playhead jumping the whole week at once, and which (now that
        the fold window follows ``sim_time``) makes every KPI denominator jump with
        it. So the gap is paced like any other stretch of the run.

        The SimPy env is deliberately NOT advanced. ``env.run(until=...)`` schedules
        its own stop event at ``until``, and a ``PriorityTier.COMPLETION`` event
        sitting at exactly ``horizon.end`` shares that priority — it could be
        executed on the way, which the half-open ``[start, end)`` convention says
        must never happen. The idle tail has nothing to execute by definition, so
        only the clock needs to move.
        """
        position = self._coast_us if self._coast_us is not None else int(self.env.now)
        moved = min(chunk_us, self.horizon.end.root - position)
        self._coast_us = position + moved
        return moved

    def advance(self, granularity: StepGranularity, count: int) -> None:
        """Advance ``count`` units of ``granularity`` (caller holds the lock).

        Stepping past the last scheduled event ends the run at its horizon rather
        than coasting: a step asks for the next unit of work, and an explicit "no
        more work exists" is an answer. Pacing the idle tail is playback's job.
        """
        for _ in range(count):
            if self.state == "finished":
                return
            if granularity == "decision":
                target = self.decision_count + 1
                while self.decision_count < target:
                    if not self._settle_next_instant():
                        self.state = "finished"
                        return
            elif granularity == "event":
                if not self._step_one_event():
                    self.state = "finished"
                    return
            else:  # "tick" — the next scheduled instant, settled fully
                if not self._settle_next_instant():
                    self.state = "finished"
                    return

    # ---------------------------------------------------------------- driver
    def play(self) -> None:
        """Start/resume the driver loop (caller holds the lock)."""
        if self.state == "finished":
            return
        self.state = "playing"
        self._wake.set()
        if self._driver is None or self._driver.done():
            self._driver = asyncio.get_running_loop().create_task(self._drive())

    def pause(self) -> None:
        """Freeze ``sim_time`` (caller holds the lock); the driver exits its loop."""
        if self.state in ("created", "playing", "stepping"):
            self.state = "paused"
        self._wake.set()

    def set_speed(self, multiplier: float) -> None:
        """Scale wall-clock pacing only — results are identical at any speed."""
        self.speed = multiplier
        self._wake.set()

    async def _drive(self) -> None:
        """The playback loop: settle one instant (or coast), emit a frame, pace wall-clock.

        The pacing sleep is OUTSIDE SimPy — `speed` stretches or compresses the
        human-visible clock and never touches what the env runs (doc 07 §5.2).
        Pacing covers the idle tail too (see :meth:`coast`), so the clock reaches
        the horizon by running there rather than by jumping.
        """
        loop = asyncio.get_running_loop()
        last_emit = float("-inf")
        try:
            while True:
                frame: StreamFrame | None = None
                delta_us = 0
                async with self.lock:
                    if self.state != "playing":
                        break
                    before = self.sim_time.root
                    # Nothing left to execute before the horizon: walk the remaining
                    # idle time instead of teleporting to the end. Once the walk has
                    # no gap left to cover, the horizon is reached and the run ends.
                    exhausted = self.coasting or not self._settle_next_instant()
                    if exhausted and self.coast(self.coast_chunk_us()) == 0:
                        self.state = "finished"
                        frame = build_frame(self, kind="delta")
                        self.broadcast(frame)
                        break
                    delta_us = self.sim_time.root - before
                    now_wall = loop.time()
                    if now_wall - last_emit >= MIN_FRAME_INTERVAL_S:
                        frame = build_frame(self, kind="delta")
                        last_emit = now_wall
                if frame is not None:
                    self.broadcast(frame)
                delay = delta_us / 1_000_000 / self.speed
                if delay >= _MIN_SLEEP_S:
                    self._wake.clear()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._wake.wait(), timeout=delay)
                else:
                    await asyncio.sleep(0)
        finally:
            self._driver = None

    async def stop_driver(self) -> None:
        """Cancel the driver task (teardown path)."""
        task = self._driver
        self._wake.set()
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._driver = None

    # ------------------------------------------------------------- streaming
    def subscribe(self) -> asyncio.Queue[StreamFrame | None]:
        """A per-subscriber frame queue. ``None`` is the end-of-stream sentinel."""
        queue: asyncio.Queue[StreamFrame | None] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        if self.closed:
            # Subscribing to an already-torn-down session yields an immediately
            # ended stream rather than a loop with nothing to wake it.
            queue.put_nowait(None)
            return queue
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StreamFrame | None]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def close(self) -> None:
        """End every subscriber's stream (teardown path).

        The sentinel is what makes this prompt: a transport parked in
        ``queue.get()`` wakes at once instead of after another heartbeat, and one
        parked between heartbeats sees ``closed``. Queued frames of a deleted run
        are dropped as needed to make room — the run they describe is gone.
        """
        self.closed = True
        for queue in self._subscribers:
            while queue.full():
                queue.get_nowait()
            queue.put_nowait(None)
        self._subscribers.clear()

    def broadcast(self, frame: StreamFrame) -> None:
        """Fan a frame out to every subscriber; overflow coalesces, never grows."""
        if self.closed:
            return
        for queue in self._subscribers:
            if queue.full():
                backlog: list[StreamFrame] = []
                while not queue.empty():
                    queued = queue.get_nowait()
                    if queued is not None:
                        backlog.append(queued)
                queue.put_nowait(coalesce_frames((*backlog, frame)))
            else:
                queue.put_nowait(frame)


class SessionRegistry:
    """Lifespan-scoped ``RunId -> RunSession`` map (one per app instance)."""

    def __init__(self) -> None:
        self._sessions: dict[str, RunSession] = {}
        self._counter = itertools.count(1)

    def mint_run_id(self, scenario_name: str, arm: Arm, seed: int) -> RunId:
        return RunId(f"run-{next(self._counter):04d}-{scenario_name}-{arm}-{seed}")

    def add(self, session: RunSession) -> None:
        self._sessions[session.run_id.root] = session

    def get(self, run_id: str) -> RunSession | None:
        return self._sessions.get(run_id)

    def run_ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    async def teardown(self, run_id: str) -> bool:
        """Remove a session (and its paired shadow) and cancel their drivers."""
        session = self._sessions.pop(run_id, None)
        if session is None:
            return False
        doomed = [session]
        if session.shadow is not None:
            shadow = self._sessions.pop(session.shadow.root, None)
            if shadow is not None:
                doomed.append(shadow)
        if session.shadow_of is not None:
            primary = self._sessions.get(session.shadow_of.root)
            if primary is not None:
                primary.shadow = None
        for doomed_session in doomed:
            doomed_session.close()
            await doomed_session.stop_driver()
        return True

    async def shutdown(self) -> None:
        for run_id in list(self._sessions):
            await self.teardown(run_id)


def require_session(app: FastAPI, run_id: str) -> RunSession:
    """Resolve a run id against the app's registry or raise a 404."""
    registry = cast("SessionRegistry", app.state.registry)
    session = registry.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return session


async def execute_control(session: RunSession, command: ControlCommand) -> SessionState:
    """Apply one control command to one session and return its new state."""
    if command.action == "play":
        async with session.lock:
            session.play()
    elif command.action == "pause":
        frame: StreamFrame | None = None
        async with session.lock:
            was_advancing = session.state in ("playing", "stepping")
            session.pause()
            if was_advancing:
                frame = build_frame(session, kind="delta")
        if frame is not None:
            session.broadcast(frame)
    elif command.action == "step":
        step_frame: StreamFrame | None = None
        async with session.lock:
            if session.state == "playing":
                raise HTTPException(status_code=409, detail="pause the run before stepping")
            if not session.finished:
                session.state = "stepping"
                session.advance(command.granularity, command.count)
                if not session.finished:
                    session.state = "paused"
                step_frame = build_frame(session, kind="delta")
        if step_frame is not None:
            session.broadcast(step_frame)
    else:  # "speed"
        # Range/finiteness is the model's job (`_SpeedMultiplier`); what is left
        # here is the cross-field requirement that `speed` carry one at all.
        if command.multiplier is None:
            raise HTTPException(status_code=422, detail="speed requires a positive `multiplier`")
        async with session.lock:
            session.set_speed(command.multiplier)
    return session.session_state()


@router.post("/runs/{run_id}/control", response_model=SessionState)
async def post_control(run_id: str, command: ControlCommand, request: Request) -> SessionState:
    """Playback control. Commands mirror to the paired shadow arm so a live
    comparison advances both arms under the same operator clock."""
    app = cast("FastAPI", request.app)
    session = require_session(app, run_id)
    state = await execute_control(session, command)
    if session.shadow is not None:
        registry = cast("SessionRegistry", app.state.registry)
        shadow = registry.get(session.shadow.root)
        if shadow is not None:
            await execute_control(shadow, command)
    return state


__all__ = [
    "DEFAULT_SPEED",
    "ControlCommand",
    "RunSession",
    "SessionRegistry",
    "SessionState",
    "TappedEventLog",
    "execute_control",
    "require_session",
    "router",
]
