"""``GET /runs/{id}/stream`` — the ``StreamFrame`` schema, builder, and transports.

Server -> client only (doc 07 §3.2/§7): WebSocket primary, SSE (``text/event-stream``)
fallback. Control and overrides are POST endpoints — never smuggled up the socket —
so they can return a synchronous ``Violation`` list and stay individually testable.

The frame types are a **presentation projection** of ``sim.physics.world`` — close
cousins of ``core.seam.BayState``/``WaitingPatient`` with render-only fields, and
deliberately NOT the seam types: they carry no authority and never feed a decision.
The one exception is ``events``: ``core.events.EventEnvelope`` verbatim, the single
place a core type crosses the wire unprojected, so the browser sees exactly the log
the sim wrote.

Framing protocol:

* On connect the server sends one ``snapshot`` frame (full mutable state, empty
  event tail), then ``delta`` frames as the driver advances.
* ``seq`` increments once per built ``delta`` frame; a ``snapshot`` carries the
  current ``seq`` without consuming one. A gap in ``seq`` means "re-snapshot",
  which is also the reconnect protocol — there is no per-client replay buffer.
* A paused session emits only heartbeat ``snapshot`` frames so the client can tell
  "paused" from "socket stalled".
* Backpressure: a slow subscriber's queue is coalesced, never grown unbounded —
  :func:`coalesce_frames` keeps the newest world projection and MERGES the event
  tails (a stale bay fill is recoverable from the next frame; a dropped event is a
  silent hole in the log-faithful ``events`` field, so tails are never dropped).

Positions: the ``World`` tracks actor positions at node resolution (the executor
updates position per completed hop), so ``StaffKinematic.at_node`` is always the
last completed node and ``edge``/``edge_progress`` are reserved for a finer-grained
kinematics source — the client dead-reckons between frames and snaps on arrival.
"""

from __future__ import annotations

import asyncio
from itertools import chain
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from hospital.core import (
    Activity,
    BayId,
    BayStatus,
    Duration,
    EsiAcuity,
    EventEnvelope,
    FrozenModel,
    KpiVector,
    NodeId,
    PatientId,
    RunId,
    SimTime,
    StaffId,
    StaffRole,
    TaskId,
    UnknownEntity,
)
from hospital.core.seam import TaskKind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from fastapi import FastAPI

    from hospital.api.sessions import RunSession, SessionRegistry
    from hospital.sim.physics.world import World

router = APIRouter()

# A paused session emits a heartbeat snapshot at this cadence so a client can
# distinguish "paused" from "stalled socket".
HEARTBEAT_SECONDS = 0.5

# First-N waiting patients surfaced per queue for chip rendering.
_QUEUE_HEAD_N = 8


class StaffKinematic(FrozenModel):
    """Rendering-shaped staff state — NOT a domain type (doc 07 §7.2)."""

    staff: StaffId
    role: StaffRole
    at_node: NodeId | None
    edge: tuple[NodeId, NodeId] | None = None
    edge_progress: float = 0.0
    activity: Activity | Literal["idle"]
    current_task: TaskId | None = None


class BayFrame(FrozenModel):
    """Per-frame mutable bay state (the static ``Bay`` geometry is fetched once)."""

    bay: BayId
    status: BayStatus
    occupant: PatientId | None = None
    cleaning_eta: SimTime | None = None


class QueueFrame(FrozenModel):
    """One waiting stage: its depth and the first-N patients for chip rendering."""

    stage: str
    depth: int
    head: tuple[PatientId, ...]


class PatientChip(FrozenModel):
    """A patient marker: acuity color key, position, stage, and time waited."""

    patient: PatientId
    esi: EsiAcuity
    at_node: NodeId | None
    stage: str
    waited: Duration


class PendingTask(FrozenModel):
    """A pending unit of work carrying its REAL ``TaskSpec.id`` (doc 07 §7.2).

    The sim mints opaque task ids (``task_000001``); an operator ``reroute`` names
    one verbatim, and a fabricated id is rejected by ``validate()`` — so the
    console must see the true id. Render-only like every frame field: the id
    crosses the wire as a reference the operator echoes back, never as authority.
    ``role`` is the role the task requires, so the console can offer only staff
    who could actually serve it.

    The field is ``id`` — the *task's own* identity, the same spelling the
    console's ``PendingTask`` reads. Naming it ``task`` here would read as a
    reference to some other task and, worse, silently leave every reroute
    picker empty: the console would find ``undefined`` and could only ever
    echo back a fabricated id.
    """

    id: TaskId
    kind: TaskKind
    at: NodeId
    patient: PatientId | None = None
    role: StaffRole


class StreamFrame(FrozenModel):
    """One streamed projection of live world state (doc 07 §7.2)."""

    run: RunId
    sim_time: SimTime
    seq: int
    kind: Literal["snapshot", "delta"]
    state: Literal["playing", "paused", "stepping", "finished"]
    speed: float
    staff: tuple[StaffKinematic, ...]
    bays: tuple[BayFrame, ...]
    queues: tuple[QueueFrame, ...]
    patients: tuple[PatientChip, ...]
    pending_tasks: tuple[PendingTask, ...]
    events: tuple[EventEnvelope, ...]
    kpi_preview: KpiVector | None = None


def _patient_node(world: World, patient: PatientId) -> NodeId | None:
    try:
        return world.patient_at(patient)
    except UnknownEntity:
        return None


def _staff_frames(world: World) -> tuple[StaffKinematic, ...]:
    out: list[StaffKinematic] = []
    for state in world.snapshot_staff():
        member = world.staff_member(state.staff)
        activity: Activity | Literal["idle"] = (
            world.task(state.current_task).activity if state.current_task is not None else "idle"
        )
        out.append(
            StaffKinematic(
                staff=state.staff,
                role=member.role,
                at_node=state.at,
                activity=activity,
                current_task=state.current_task,
            )
        )
    return tuple(out)


def _queue_frames(world: World) -> tuple[QueueFrame, ...]:
    by_stage: dict[str, list[PatientId]] = {}
    for waiting in world.waiting_for_bay():
        by_stage.setdefault(waiting.stage, []).append(waiting.patient.id)
    return tuple(
        QueueFrame(stage=stage, depth=len(ids), head=tuple(ids[:_QUEUE_HEAD_N]))
        for stage, ids in sorted(by_stage.items())
    )


def _patient_chips(world: World) -> tuple[PatientChip, ...]:
    chips: list[PatientChip] = []
    for waiting in world.waiting_for_bay():
        chips.append(
            PatientChip(
                patient=waiting.patient.id,
                esi=waiting.patient.esi,
                at_node=_patient_node(world, waiting.patient.id),
                stage=waiting.stage,
                waited=waiting.waited,
            )
        )
    for bay_state in world.snapshot_bays():
        if bay_state.occupant is not None:
            chips.append(
                PatientChip(
                    patient=bay_state.occupant,
                    esi=world.patient(bay_state.occupant).esi,
                    at_node=_patient_node(world, bay_state.occupant),
                    stage="bay",
                    waited=Duration(0),
                )
            )
    return tuple(chips)


def _pending_task_frames(world: World) -> tuple[PendingTask, ...]:
    """Project the world's real pending ``TaskSpec``s — ids the operator reroutes by."""
    return tuple(
        PendingTask(
            id=spec.id,
            kind=spec.kind,
            at=spec.at,
            patient=spec.patient,
            role=spec.required_role,
        )
        for spec in world.pending_tasks()
    )


def build_frame(session: RunSession, *, kind: Literal["snapshot", "delta"]) -> StreamFrame:
    """Project the session's ``World`` into one frame (caller holds the session lock).

    A ``delta`` frame consumes the event tail accumulated since the previous
    ``delta`` frame and a fresh ``seq``; a ``snapshot`` reads full state without
    consuming either, so heartbeats and late joiners never steal events from the
    delta stream.
    """
    world = session.world
    events: tuple[EventEnvelope, ...] = ()
    if kind == "delta":
        events = session.log.drain_since_frame()
        seq = session.next_frame_seq()
    else:
        seq = session.frame_seq
    return StreamFrame(
        run=session.run_id,
        sim_time=session.sim_time,
        seq=seq,
        kind=kind,
        state=session.frame_state,
        speed=session.speed,
        staff=_staff_frames(world),
        bays=tuple(
            BayFrame(bay=bs.bay, status=bs.status, occupant=bs.occupant)
            for bs in world.snapshot_bays()
        ),
        queues=_queue_frames(world),
        patients=_patient_chips(world),
        pending_tasks=_pending_task_frames(world),
        events=events,
    )


def coalesce_frames(frames: Sequence[StreamFrame]) -> StreamFrame:
    """Collapse a backlog: newest world projection, MERGED event tails.

    Intermediate world projections are recoverable (the next frame carries the
    current state) but the ``events`` field is log-faithful, so tails are
    concatenated, never dropped (doc 07 nuances 7.3).
    """
    newest = frames[-1]
    merged = tuple(chain.from_iterable(f.events for f in frames))
    if len(merged) == len(newest.events):
        return newest
    return newest.model_copy(update={"events": merged})


def _registry(app: FastAPI) -> SessionRegistry:
    return cast("SessionRegistry", app.state.registry)


async def _next_frame(
    session: RunSession, queue: asyncio.Queue[StreamFrame | None]
) -> StreamFrame | None:
    """The next delta frame, a heartbeat snapshot when quiet, or ``None`` once closed.

    ``None`` means the session was torn down: both the queue sentinel (a
    subscriber parked in ``get()``) and the ``closed`` check on the heartbeat path
    (a subscriber between frames) map onto it, so neither can be left
    heartbeating a run that no longer exists.
    """
    try:
        return await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
    except TimeoutError:
        if session.closed:
            return None
        async with session.lock:
            return build_frame(session, kind="snapshot")


@router.websocket("/runs/{run_id}/stream")
async def stream_websocket(websocket: WebSocket, run_id: str) -> None:
    """WebSocket transport (primary): one snapshot on connect, then deltas."""
    session = _registry(cast("FastAPI", websocket.app)).get(run_id)
    if session is None:
        await websocket.close(code=4404, reason=f"unknown run: {run_id}")
        return
    await websocket.accept()
    queue = session.subscribe()
    try:
        async with session.lock:
            snapshot = build_frame(session, kind="snapshot")
        await websocket.send_text(snapshot.model_dump_json())
        while True:
            frame = await _next_frame(session, queue)
            if frame is None:
                # The run was deleted under us: say so in the close code rather
                # than leaving the socket open on a session that no longer exists.
                await websocket.close(code=4404, reason=f"run ended: {run_id}")
                return
            await websocket.send_text(frame.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)


def _sse(frame: StreamFrame) -> str:
    return f"data: {frame.model_dump_json()}\n\n"


async def sse_frames(session: RunSession, request: Request) -> AsyncIterator[str]:
    """SSE ``data:`` lines: one snapshot, then deltas/heartbeats until the peer goes.

    The loop ends on :meth:`Request.is_disconnected` so a browser that closes its
    ``EventSource`` releases its subscriber (and stops its heartbeat) at once — a
    plain ``while True`` would leak both, since this ASGI path is told a client
    left only through the disconnect signal, never a raised ``send``. It ends on a
    ``None`` frame for the other direction: the *run* going away rather than the
    peer (``DELETE /runs/{id}``), which no disconnect signal ever reports.
    """
    queue = session.subscribe()
    try:
        async with session.lock:
            snapshot = build_frame(session, kind="snapshot")
        yield _sse(snapshot)
        while not await request.is_disconnected():
            frame = await _next_frame(session, queue)
            if frame is None:
                return
            yield _sse(frame)
    finally:
        session.unsubscribe(queue)


@router.get("/runs/{run_id}/stream")
async def stream_sse(run_id: str, request: Request) -> StreamingResponse:
    """SSE fallback (read-only): the same frames as ``data:`` lines."""
    session = _registry(cast("FastAPI", request.app)).get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return StreamingResponse(sse_frames(session, request), media_type="text/event-stream")


__all__ = [
    "HEARTBEAT_SECONDS",
    "BayFrame",
    "PatientChip",
    "PendingTask",
    "QueueFrame",
    "StaffKinematic",
    "StreamFrame",
    "build_frame",
    "coalesce_frames",
    "router",
    "sse_frames",
]
