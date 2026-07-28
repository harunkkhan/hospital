"""Playback control: pause freezes, step is exact, speed never touches sampling.

The crown-jewel invariant (doc 07 §5.2): ``speed`` scales wall-clock pacing
only, so runs at wildly different speeds produce byte-identical ``EventLog``s —
and the API-driven incremental playback is byte-identical to the headless
``run_replication`` for the same ``(scenario, seed, arm)`` cell.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, NoReturn

from _api_fixtures import (
    api_scenario,
    control,
    create_run,
    make_app,
    run_to_finish,
    session_of,
    step,
)
from fastapi.testclient import TestClient

from hospital.api.sessions import DEFAULT_SPEED
from hospital.sim import run_replication

if TYPE_CHECKING:
    from pathlib import Path


def test_speed_invariance_and_engine_equality(tmp_path: Path) -> None:
    scenario = api_scenario()
    headless = run_replication(scenario, "baseline", 7).event_log_jsonl

    app = make_app(tmp_path)
    logs: dict[float, str] = {}
    with TestClient(app) as client:
        for speed in (1e9, 3e8):
            handle = create_run(client, seed=7)
            run_to_finish(client, handle["run"], speed=speed)
            logs[speed] = session_of(app, handle["run"]).log.to_jsonl()

    assert logs[1e9] == logs[3e8], "pacing must never touch sampling"
    assert logs[1e9] == headless, "incremental playback must equal the headless engine"


def test_pause_freezes_sim_time(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        run_id = handle["run"]
        control(client, run_id, "speed", multiplier=5e5)
        control(client, run_id, "play")
        time.sleep(0.05)
        paused = control(client, run_id, "pause")
        assert paused["state"] in ("paused", "finished")

        first = client.get(f"/runs/{run_id}").json()["sim_time"]
        time.sleep(0.05)
        second = client.get(f"/runs/{run_id}").json()["sim_time"]
        assert first == second


def test_step_advances_exactly_count_decision_ticks(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        session = session_of(app, handle["run"])
        assert session.decision_count == 0

        state = step(client, handle["run"], granularity="decision", count=3)
        assert session.decision_count == 3
        assert state["state"] == "paused"
        assert state["sim_time"] > 0

        step(client, handle["run"], granularity="decision", count=2)
        assert session.decision_count == 5


def test_step_granularities_advance_monotonically(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        run_id = handle["run"]
        t_prev = 0
        for granularity in ("event", "tick", "decision"):
            state = step(client, run_id, granularity=granularity)
            assert state["sim_time"] >= t_prev
            t_prev = state["sim_time"]


def test_step_while_playing_is_a_409(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, start="playing")
        response = client.post(f"/runs/{handle['run']}/control", json={"action": "step"})
        assert response.status_code == 409
        control(client, handle["run"], "pause")


def _no_non_finite_literals(literal: str) -> NoReturn:
    raise AssertionError(f"422 body carries a non-JSON literal: {literal}")


def test_speed_requires_a_positive_multiplier(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        url = f"/runs/{handle['run']}/control"
        for bad in ({"action": "speed"}, {"action": "speed", "multiplier": -2.0}):
            response = client.post(url, json=bad)
            assert response.status_code == 422, bad
        for bad in ({"action": "speed", "multiplier": 0.0},):
            response = client.post(url, json=bad)
            assert response.status_code == 422, bad

        # A multiplier divides the pacing delay, so a non-finite one runs the whole
        # horizon unpaced: `Infinity` zeroes every delay outright and `NaN` makes
        # `delay >= _MIN_SLEEP_S` False, which does the same thing silently. Sent as
        # raw bytes because no conforming JSON encoder will emit these literals --
        # `json.loads` accepts them, so the boundary is what has to reject them.
        for literal in ("Infinity", "-Infinity", "NaN"):
            response = client.post(
                url,
                content=f'{{"action": "speed", "multiplier": {literal}}}',
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422, literal
            # ...and the rejection body is itself conforming JSON: the offending
            # value must not be echoed back as a bare `Infinity`/`NaN` literal
            # (starlette's dumps would raise, turning the 422 into a 500).
            json.loads(response.text, parse_constant=_no_non_finite_literals)

        # The rejections are total: the session keeps the speed it had.
        assert session_of(app, handle["run"]).speed == DEFAULT_SPEED


def test_finished_run_ignores_further_control(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        run_to_finish(client, handle["run"])
        assert step(client, handle["run"])["state"] == "finished"
        assert control(client, handle["run"], "play")["state"] == "finished"
        assert client.get(f"/runs/{handle['run']}").json()["sim_time"] == handle["horizon"]
