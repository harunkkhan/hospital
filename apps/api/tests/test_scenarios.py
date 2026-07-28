"""``/scenarios``: list, verbatim round-trip, and data-validated slider overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from _api_fixtures import DEFAULT_SCENARIO_ID, api_scenario, create_run, make_app
from fastapi.testclient import TestClient

from hospital.data.scenario import Scenario

if TYPE_CHECKING:
    from pathlib import Path


def test_list_scenarios_includes_the_seeded_store(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        response = client.get("/scenarios")
        assert response.status_code == 200
        summaries: list[dict[str, Any]] = response.json()
        by_id = {s["id"]: s for s in summaries}
        assert DEFAULT_SCENARIO_ID in by_id
        assert by_id[DEFAULT_SCENARIO_ID]["name"] == "api_tiny"
        assert by_id[DEFAULT_SCENARIO_ID]["horizon"] > 0
        assert "loaded from" in by_id[DEFAULT_SCENARIO_ID]["note"]


def test_get_scenario_round_trips_the_data_model(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        response = client.get(f"/scenarios/{DEFAULT_SCENARIO_ID}")
        assert response.status_code == 200
        # The body re-validates as the frozen data model, byte-for-byte equal.
        fetched = Scenario.model_validate(response.json())
        assert fetched == api_scenario()

        assert client.get("/scenarios/nope").status_code == 404


def test_post_scenario_derives_and_stores_a_validated_variant(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/scenarios",
            json={
                "base": DEFAULT_SCENARIO_ID,
                "overrides": {"workload.base_rate_per_hour": 9.5},
            },
        )
        assert response.status_code == 201, response.text
        derived_id = response.json()["id"]
        assert derived_id != DEFAULT_SCENARIO_ID

        derived = Scenario.model_validate(client.get(f"/scenarios/{derived_id}").json())
        assert derived.workload.base_rate_per_hour == 9.5
        # Only the addressed leaf changed.
        assert derived.facility == api_scenario().facility

        # ... and a run launches from the derived scenario.
        handle = create_run(client, scenario_id=derived_id)
        assert handle["state"] == "created"


def test_invalid_overrides_surface_the_data_layer_rejection(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        # A negative rate violates the data model's own bounds (not an API rule).
        response = client.post(
            "/scenarios",
            json={
                "base": DEFAULT_SCENARIO_ID,
                "overrides": {"workload.base_rate_per_hour": -1.0},
            },
        )
        assert response.status_code == 422

        assert client.post("/scenarios", json={"base": "nope", "overrides": {}}).status_code == 404


def test_inline_scenario_launches_a_run(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/runs",
            json={
                "scenario": {
                    "base": DEFAULT_SCENARIO_ID,
                    "overrides": {"workload.base_rate_per_hour": 3.0},
                },
                "seed": 3,
                "arm": "baseline",
            },
        )
        assert response.status_code == 201, response.text
