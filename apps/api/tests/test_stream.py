"""Streaming: snapshot-then-deltas, monotonic ``seq``, log-faithful event tails,
paused heartbeats, and backpressure coalescing (doc 07 §7 / nuances 7.3)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from _api_fixtures import create_run, make_app, quiet_scenario, run_to_finish, session_of, step
from hospital.api.overrides import PinRegistry
from hospital.api.sessions import RunSession
from hospital.api.stream import StreamFrame, coalesce_frames, sse_frames
from hospital.core import RunId

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.testclient import WebSocketTestSession


def _recv(ws: WebSocketTestSession) -> dict[str, Any]:
    return cast("dict[str, Any]", ws.receive_json())


def test_snapshot_then_deltas_with_gapless_seq(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        run_id = handle["run"]
        with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
            snapshot = _recv(ws)
            assert snapshot["kind"] == "snapshot"
            assert snapshot["state"] == "paused"
            assert snapshot["seq"] == 0
            assert snapshot["events"] == []
            assert snapshot["run"] == run_id

            seqs: list[int] = []
            for _ in range(3):
                step(client, run_id, granularity="tick")
                frame = _recv(ws)
                assert frame["kind"] == "delta"
                seqs.append(frame["seq"])
            assert seqs == [1, 2, 3], "delta seq must be monotonic and gapless"


def test_delta_event_tails_reconstruct_the_log(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        run_id = handle["run"]
        session = session_of(app, handle["run"])
        with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
            _recv(ws)  # snapshot
            sequences: list[int] = []
            for _ in range(4):
                step(client, run_id, granularity="decision")
                frame = _recv(ws)
                sequences.extend(e["sequence"] for e in frame["events"])
        # The concatenated tails are exactly the log prefix: nothing dropped,
        # nothing duplicated.
        assert sequences == list(range(len(sequences)))
        assert len(sequences) == len(session.log)


def test_paused_session_emits_heartbeat_snapshots(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        with client.websocket_connect(f"/runs/{handle['run']}/stream") as ws:
            first = _recv(ws)
            heartbeat = _recv(ws)  # arrives after HEARTBEAT_SECONDS, no state change
            assert heartbeat["kind"] == "snapshot"
            assert heartbeat["seq"] == first["seq"]
            assert heartbeat["state"] == "paused"


def test_finished_run_streams_a_finished_snapshot(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        run_to_finish(client, handle["run"])
        with client.websocket_connect(f"/runs/{handle['run']}/stream") as ws:
            snapshot = _recv(ws)
            assert snapshot["state"] == "finished"
            assert snapshot["sim_time"] == handle["horizon"]


def test_unknown_run_websocket_is_closed(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/runs/nope/stream"):
                pass
        assert excinfo.value.code == 4404


class _StubRequest:
    """A request that reports connected for the snapshot, then disconnected.

    The bundled ``TestClient`` transport buffers a response to completion before
    returning, so it cannot drive an unbounded SSE stream; this exercises the
    real :func:`sse_frames` generator directly and asserts it ends on disconnect.
    """

    def __init__(self, *, disconnect_after: int) -> None:
        self._polls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        connected = self._polls < self._disconnect_after
        self._polls += 1
        return not connected


def test_sse_fallback_serves_a_snapshot_then_ends_on_disconnect() -> None:
    session = RunSession(RunId("t-sse"), quiet_scenario(), "baseline", 7, pins=PinRegistry())

    async def collect() -> list[str]:
        request = cast("Any", _StubRequest(disconnect_after=0))
        return [chunk async for chunk in sse_frames(session, request)]

    chunks = asyncio.run(collect())
    # The snapshot is always sent; the loop then ends at the first disconnect
    # poll (the generator's `finally` unsubscribes), so nothing is left dangling.
    assert len(chunks) == 1
    assert chunks[0].startswith("data: ")
    assert chunks[0].endswith("\n\n")
    snapshot = StreamFrame.model_validate_json(chunks[0].removeprefix("data: "))
    assert snapshot.kind == "snapshot"
    assert snapshot.run == session.run_id


def test_coalesce_frames_merges_event_tails(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        session = session_of(app, handle["run"])
        with client.websocket_connect(f"/runs/{handle['run']}/stream") as ws:
            _recv(ws)
            step(client, handle["run"], granularity="decision")
            first = StreamFrame.model_validate(_recv(ws))
            step(client, handle["run"], granularity="decision")
            second = StreamFrame.model_validate(_recv(ws))
        merged = coalesce_frames((first, second))
        # Newest projection, concatenated tails — the backpressure contract.
        assert merged.seq == second.seq
        assert merged.sim_time == second.sim_time
        assert merged.bays == second.bays
        assert merged.events == (*first.events, *second.events)
        assert session.frame_seq == second.seq
