"""Importable test builders for ``hospital-api`` (no ``conftest``; DECISIONS D10).

Tiny, fully-featured scenarios (a real runnable ER, seconds not minutes), the
app/TestClient wiring, and typed white-box accessors into the lifespan-scoped
``SessionRegistry`` for assertions that need the engine's own state (log bytes,
world snapshots) rather than the wire projection.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from hospital.api.main import create_app
from hospital.core import (
    EsiAcuity,
    OperatingWeek,
    Patient,
    PatientId,
    SimTime,
    StaffRole,
    WorkupNeeds,
    ZoneType,
    hours,
    seconds,
)
from hospital.core.enums import ArrivalMode
from hospital.data.scenario import (
    FacilitySpec,
    Scenario,
    StaffingSpec,
    WorkloadSpec,
    WorkupProfile,
    ZoneQuota,
    dump_scenario,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hospital.api.sessions import RunSession, SessionRegistry

DEFAULT_SCENARIO_ID = "tiny"


def api_facility() -> FacilitySpec:
    """A small but fully-featured floor: 4 bays across three zones + specials."""
    return FacilitySpec(
        target_area_sqft=25_000,
        zones=(
            ZoneQuota(zone_type=ZoneType.GENERAL, bays=2, isolation_bays=1),
            ZoneQuota(zone_type=ZoneType.RESUS_TRAUMA, bays=1, isolation_bays=1),
            ZoneQuota(zone_type=ZoneType.FAST_TRACK, bays=1),
        ),
        imaging_suites=1,
        lab_stations=1,
        triage_rooms=1,
    )


def api_workload(*, rate_per_hour: float, horizon_hours: int) -> WorkloadSpec:
    return WorkloadSpec(
        horizon=OperatingWeek(start=SimTime(0), end=SimTime(hours(horizon_hours).root)),
        base_rate_per_hour=rate_per_hour,
        hourly_profile=tuple([1.0] * 24),
        dow_profile=tuple([1.0] * 7),
        esi_mix={
            EsiAcuity.ESI1: 0.05,
            EsiAcuity.ESI2: 0.20,
            EsiAcuity.ESI3: 0.45,
            EsiAcuity.ESI4: 0.20,
            EsiAcuity.ESI5: 0.10,
        },
        complaint_mix={"chest_pain": 1.0},
        ambulance_fraction=0.2,
        isolation_fraction=0.0,
        workups={
            "chest_pain": WorkupProfile(
                provider_visits_mean=1.2,
                nurse_visits_mean=1.0,
                imaging_prob={ZoneType.IMAGING: 0.2},
                labs_mean=0.3,
                procedure_prob=0.0,
            )
        },
    )


def api_staffing() -> StaffingSpec:
    return StaffingSpec(
        default_counts={
            StaffRole.PHYSICIAN: 2,
            StaffRole.NURSE: 2,
            StaffRole.TECH: 1,
            StaffRole.PORTER: 1,
            StaffRole.HOUSEKEEPING: 1,
        }
    )


def api_scenario(*, seed: int = 7, rate_per_hour: float = 6.0, horizon_hours: int = 2) -> Scenario:
    """A tiny live scenario: real arrivals, finishes in well under a second."""
    return Scenario(
        name="api_tiny",
        seed=seed,
        facility=api_facility(),
        workload=api_workload(rate_per_hour=rate_per_hour, horizon_hours=horizon_hours),
        staffing=api_staffing(),
    )


def quiet_scenario(*, seed: int = 7) -> Scenario:
    """Zero arrivals — override tests arrange the world by hand, deterministically."""
    return Scenario(
        name="api_quiet",
        seed=seed,
        facility=api_facility(),
        workload=api_workload(rate_per_hour=0.0, horizon_hours=1),
        staffing=api_staffing(),
    )


def make_patient(
    pid: str,
    *,
    esi: EsiAcuity = EsiAcuity.ESI3,
    arrival_s: float = 0.0,
    isolation: bool = False,
) -> Patient:
    return Patient(
        id=PatientId(pid),
        arrival_time=SimTime(seconds(arrival_s).root),
        arrival_mode=ArrivalMode.WALK_IN,
        esi=esi,
        complaint="chest_pain",
        isolation_required=isolation,
        workup=WorkupNeeds(provider_visits=1, nurse_visits=0, imaging=(), labs=0, procedures=0),
    )


def make_app(tmp_path: Path, scenarios: Mapping[str, Scenario] | None = None) -> FastAPI:
    """An app whose scenario store is seeded from YAML files written to ``tmp_path``.

    Round-tripping through ``dump_scenario`` keeps the store on the exact
    ``data.scenario`` codec path the real server uses.
    """
    if scenarios is None:
        scenarios = {DEFAULT_SCENARIO_ID: api_scenario()}
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    for scenario_id, scenario in scenarios.items():
        dump_scenario(scenario, scenario_dir / f"{scenario_id}.yaml")
    return create_app(scenario_dir=scenario_dir)


# ------------------------------------------------------ white-box accessors
def registry_of(app: FastAPI) -> SessionRegistry:
    return cast("SessionRegistry", app.state.registry)


def session_of(app: FastAPI, run_id: str) -> RunSession:
    session = registry_of(app).get(run_id)
    assert session is not None, f"unknown run: {run_id}"
    return session


def world_fingerprint(session: RunSession) -> tuple[object, ...]:
    """Everything an override could touch — for byte-level atomicity checks."""
    world = session.world
    return (
        session.log.to_jsonl(),
        world.snapshot_bays(),
        world.snapshot_staff(),
        world.waiting_for_bay(),
        world.pending_tasks(),
        world.blocked_edges,
        world.closed_nodes,
        len(session.pins),
    )


# ------------------------------------------------------------- HTTP helpers
def create_run(
    client: TestClient,
    *,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    seed: int = 7,
    arm: str = "baseline",
    compare_to: str | None = None,
    start: str = "paused",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "scenario": {"id": scenario_id},
        "seed": seed,
        "arm": arm,
        "start": start,
    }
    if compare_to is not None:
        body["compare_to"] = compare_to
    response = client.post("/runs", json=body)
    assert response.status_code == 201, response.text
    return cast("dict[str, Any]", response.json())


def control(client: TestClient, run_id: str, action: str, **extra: object) -> dict[str, Any]:
    response = client.post(f"/runs/{run_id}/control", json={"action": action, **extra})
    assert response.status_code == 200, response.text
    return cast("dict[str, Any]", response.json())


def step(
    client: TestClient, run_id: str, *, granularity: str = "decision", count: int = 1
) -> dict[str, Any]:
    return control(client, run_id, "step", granularity=granularity, count=count)


def run_to_finish(
    client: TestClient, run_id: str, *, speed: float = 1e9, timeout_s: float = 60.0
) -> None:
    """Play at high speed and poll until the session reports ``finished``.

    ``speed`` scales wall-clock pacing only — the realized run is identical at
    any value (the invariance the control tests assert).
    """
    control(client, run_id, "speed", multiplier=speed)
    control(client, run_id, "play")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        handle = client.get(f"/runs/{run_id}")
        assert handle.status_code == 200, handle.text
        if handle.json()["state"] == "finished":
            return
        time.sleep(0.005)
    raise AssertionError(f"run {run_id} did not finish within {timeout_s}s")
