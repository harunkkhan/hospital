"""Streaming: snapshot-then-deltas, monotonic ``seq``, log-faithful event tails,
paused heartbeats, and backpressure coalescing (doc 07 §7 / nuances 7.3)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from _api_fixtures import (
    create_run,
    make_app,
    make_patient,
    quiet_scenario,
    run_to_finish,
    session_of,
    step,
)
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hospital.api.overrides import PinRegistry
from hospital.api.sessions import RunSession
from hospital.api.stream import StreamFrame, coalesce_frames, sse_frames
from hospital.core import Activity, EsiAcuity, RunId, StaffRole, seconds

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
        with (
            pytest.raises(WebSocketDisconnect) as excinfo,
            client.websocket_connect("/runs/nope/stream"),
        ):
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


def test_pending_tasks_carry_real_ids_the_operator_reroutes_by(tmp_path: Path) -> None:
    app = make_app(tmp_path, {"quiet": quiet_scenario()})
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        run_id = handle["run"]
        session = session_of(app, run_id)

        patient = make_patient("pt_task", esi=EsiAcuity.ESI3)
        session.world.register_patient(patient)
        session.world.set_patient_position(patient.id, session.layout.entrances[0])
        task = session.world.add_task(
            kind="provider_visit",
            patient=patient.id,
            at=session.layout.entrances[0],
            required_role=StaffRole.PHYSICIAN,
            activity=Activity.PROVIDER_VISIT,
            duration=seconds(120),
        )
        real_id = task.spec.id.root

        with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
            frame = cast("dict[str, Any]", ws.receive_json())
        # The console reads `PendingTask.id` -- keyed on that field name here so a
        # rename back to `task` fails as the empty reroute picker it would cause.
        pending = {t["id"]: t for t in frame["pending_tasks"]}
        assert "task" not in frame["pending_tasks"][0], "the wire field is `id`, not `task`"
        assert real_id in pending, "the frame must expose the real TaskSpec id"
        assert pending[real_id]["kind"] == "provider_visit"
        assert pending[real_id]["at"] == session.layout.entrances[0].root
        assert StaffRole(pending[real_id]["role"]) is StaffRole.PHYSICIAN

        # The operator reroutes using the id the frame just exposed -> accepted,
        # stamped by the operator, threaded straight through the one validate().
        physician = next(m for m in session.roster if m.role == StaffRole.PHYSICIAN)
        accepted = client.post(
            f"/runs/{run_id}/override",
            json={"action": {"kind": "reroute", "staff": physician.id.root, "task": real_id}},
        )
        assert accepted.status_code == 200, accepted.text
        assert session.world.staff_task(physician.id) == task.spec.id

        # A fabricated id the frame never showed is rejected -- ids are not minted
        # by the operator, only echoed back.
        bogus = client.post(
            f"/runs/{run_id}/override",
            json={"action": {"kind": "reroute", "staff": physician.id.root, "task": "task_999999"}},
        )
        assert bogus.status_code == 422
