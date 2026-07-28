"""``POST /runs`` lifecycle: handles, CRN-paired shadows, teardown (doc 07 §11)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _api_fixtures import (
    DEFAULT_SCENARIO_ID,
    create_run,
    make_app,
    run_to_finish,
    session_of,
)
from fastapi.testclient import TestClient

from hospital.core import FloorLayout

if TYPE_CHECKING:
    from pathlib import Path

    from hospital.api.sessions import RunSession


def test_create_run_builds_a_paused_session_at_t0(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, seed=11)
        assert handle["state"] == "created"
        assert handle["sim_time"] == 0
        assert handle["arm"] == "baseline"
        assert handle["seed"] == 11
        assert handle["shadow"] is None
        assert handle["stream_url"] == f"/runs/{handle['run']}/stream"
        assert handle["horizon"] > 0

        fetched = client.get(f"/runs/{handle['run']}")
        assert fetched.status_code == 200
        assert fetched.json() == handle


def test_unknown_scenario_is_a_404(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/runs", json={"scenario": {"id": "nope"}, "seed": 1, "arm": "baseline"}
        )
        assert response.status_code == 404


def test_compare_to_must_name_the_other_arm(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/runs",
            json={
                "scenario": {"id": DEFAULT_SCENARIO_ID},
                "seed": 1,
                "arm": "baseline",
                "compare_to": "baseline",
            },
        )
        assert response.status_code == 422


def _arrivals(session: RunSession) -> list[tuple[int, str]]:
    return [
        (envelope.event.occurred_at.root, envelope.event.patient.root)
        for envelope in session.log
        if envelope.event.kind == "patient_arrived"
    ]


def test_compare_to_spins_a_same_seed_shadow_and_crn_holds(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, arm="optimized", compare_to="baseline", seed=7)
        assert handle["shadow"] is not None

        primary = session_of(app, handle["run"])
        shadow = session_of(app, handle["shadow"])
        assert shadow.arm == "baseline"
        assert shadow.seed == primary.seed == 7
        assert shadow.shadow_of == primary.run_id

        # Control mirrors to the shadow, so both arms finish under one clock.
        run_to_finish(client, handle["run"])
        assert client.get(f"/runs/{handle['shadow']}").json()["state"] == "finished"

        # CRN: the two arms saw the identical realized arrival stream.
        assert _arrivals(primary) == _arrivals(shadow)
        assert len(_arrivals(primary)) > 0


def test_delete_tears_down_the_run_and_its_shadow(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, arm="optimized", compare_to="baseline")
        shadow_id = handle["shadow"]

        response = client.delete(f"/runs/{handle['run']}")
        assert response.status_code == 204
        assert client.get(f"/runs/{handle['run']}").status_code == 404
        assert client.get(f"/runs/{shadow_id}").status_code == 404
        assert client.delete(f"/runs/{handle['run']}").status_code == 404


def test_start_playing_launches_the_driver(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, start="playing")
        assert handle["state"] == "playing"
        run_to_finish(client, handle["run"])


def test_layout_returns_the_static_geometry_verbatim(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        response = client.get(f"/runs/{handle['run']}/layout")
        assert response.status_code == 200
        # The API owns no geometry model: the body re-validates as the core model,
        # equal to the session's own layout (fetched once, never per frame).
        layout = FloorLayout.model_validate(response.json())
        assert layout == session_of(app, handle["run"]).layout
        assert layout.bays, "geometry must carry the static bays"
        assert layout.graph.nodes, "geometry must carry the route graph"

        assert client.get("/runs/nope/layout").status_code == 404
