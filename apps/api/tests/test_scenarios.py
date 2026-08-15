"""``/scenarios``: list, verbatim round-trip, and data-validated slider overrides."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from _api_fixtures import (
    DEFAULT_SCENARIO_ID,
    api_facility,
    api_scenario,
    api_workload,
    create_run,
    make_app,
)
from fastapi.testclient import TestClient

from hospital.core import SimTime, StaffRole, TimeWindow, ZoneType, hours
from hospital.data.layout import generate_floor
from hospital.data.scenario import Scenario, ShiftBlock, StaffingSpec, realize_staff

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


def _derive(client: TestClient, overrides: dict[str, float]) -> Scenario:
    """POST the overrides, then GET the derived scenario back as the data model."""
    created = client.post("/scenarios", json={"base": DEFAULT_SCENARIO_ID, "overrides": overrides})
    assert created.status_code == 201, created.text
    fetched = client.get(f"/scenarios/{created.json()['id']}")
    assert fetched.status_code == 200, fetched.text
    return Scenario.model_validate(fetched.json())


def test_console_slider_keys_reach_the_fields_they_name(tmp_path: Path) -> None:
    """A named knob must land on a real ``Scenario`` field, over the wire.

    None of these is a literal path, so before the slider vocabulary existed
    ``extra="forbid"`` rejected all of them — the whole panel could only 422.
    Asserted one slider at a time, on the *effect* rather than the overlay, so a
    key that validates but changes nothing still fails. (The vocabulary itself is
    exercised knob by knob in ``test_sliders``; this is the ``/scenarios`` path.)
    """
    base = api_scenario()
    with TestClient(make_app(tmp_path)) as client:
        doubled = _derive(client, {"workload.arrival_rate_multiplier": 2.0})
        assert doubled.workload.base_rate_per_hour == base.workload.base_rate_per_hour * 2.0

        ambulances = _derive(client, {"workload.ambulance_share": 0.75})
        assert ambulances.workload.ambulance_fraction == 0.75

        fast_track = _derive(client, {"facility.fast_track_bays": 5})
        quotas = {q.zone_type: q for q in fast_track.facility.zones}
        assert quotas[ZoneType.FAST_TRACK].bays == 5
        # Only that zone moved.
        assert (
            quotas[ZoneType.GENERAL]
            == {q.zone_type: q for q in base.facility.zones}[ZoneType.GENERAL]
        )


def test_headcount_sliders_change_the_realized_roster(tmp_path: Path) -> None:
    """A headcount slider is judged by ``realize_staff``, not by the spec it wrote.

    ``default_counts`` alone is a no-op wherever explicit ``blocks`` supply the
    role (as the shipped ``scenarios/er_floor*.yaml`` do for all five), so the
    scenario used here has both -- writing only one of the two would pass a
    spec-shaped assertion while realizing the base's staffing.
    """
    window = TimeWindow(start=SimTime(0), end=SimTime(hours(2).root))
    shifted = Scenario(
        name="api_shifted",
        seed=7,
        facility=api_facility(),
        workload=api_workload(rate_per_hour=6.0, horizon_hours=2),
        staffing=StaffingSpec(
            blocks=(
                ShiftBlock(
                    window=window,
                    role_counts={
                        StaffRole.PHYSICIAN: 2,
                        StaffRole.NURSE: 2,
                        StaffRole.TECH: 1,
                        StaffRole.PORTER: 1,
                        StaffRole.HOUSEKEEPING: 1,
                    },
                ),
            ),
            default_counts={StaffRole.NURSE: 2, StaffRole.PHYSICIAN: 2},
        ),
    )
    with TestClient(make_app(tmp_path, {DEFAULT_SCENARIO_ID: shifted})) as client:
        derived = _derive(client, {"staffing.nurse_count": 6, "staffing.physician_count": 4})
        layout = generate_floor(derived.facility)
        roster = realize_staff(derived.staffing, layout, window)
        realized = Counter(member.role for member in roster)
        # Both sliders survive together: each rebuilds the whole `blocks` tuple, and
        # a tuple replaces wholesale -- merged as independent fragments one would
        # have silently overwritten the other.
        assert realized[StaffRole.NURSE] == 6
        assert realized[StaffRole.PHYSICIAN] == 4
        # Untouched roles keep the base's counts.
        assert realized[StaffRole.TECH] == 1
        assert realized[StaffRole.HOUSEKEEPING] == 1


def test_slider_overrides_launch_a_run_and_stay_data_validated(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        # The console's own re-run path: inline sliders straight into POST /runs.
        response = client.post(
            "/runs",
            json={
                "scenario": {
                    "base": DEFAULT_SCENARIO_ID,
                    "overrides": {
                        "workload.arrival_rate_multiplier": 1.5,
                        "workload.ambulance_share": 0.4,
                        "staffing.nurse_count": 3,
                        "staffing.physician_count": 3,
                        "facility.fast_track_bays": 2,
                    },
                },
                "seed": 11,
                "arm": "optimized",
            },
        )
        assert response.status_code == 201, response.text

        # Bounds stay the data layer's: a share above 1.0 is rejected by
        # WorkloadSpec.ambulance_fraction, not by an API-invented rule.
        rejected = client.post(
            "/scenarios",
            json={"base": DEFAULT_SCENARIO_ID, "overrides": {"workload.ambulance_share": 1.4}},
        )
        assert rejected.status_code == 422

        # A fractional headcount is rejected rather than truncated.
        fractional = client.post(
            "/scenarios",
            json={"base": DEFAULT_SCENARIO_ID, "overrides": {"staffing.nurse_count": 2.5}},
        )
        assert fractional.status_code == 422

        # An alias and a literal path onto the same leaf is ambiguous, so it is
        # refused rather than resolved by dict ordering.
        ambiguous = client.post(
            "/scenarios",
            json={
                "base": DEFAULT_SCENARIO_ID,
                "overrides": {
                    "workload.arrival_rate_multiplier": 2.0,
                    "workload.base_rate_per_hour": 9.0,
                },
            },
        )
        assert ambiguous.status_code == 422


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
