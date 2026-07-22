"""Importable test builders (no ``conftest``; helpers are imported by name).

Shared across the ``hospital-data`` test modules so scenario/spec construction
is never copy-pasted (doc 00 §5.10 / doc 08 §1) — mirrors the ``hospital-core``
tests' ``_fixtures.py`` convention. Named ``_data_fixtures`` rather than the
bare ``_fixtures`` used elsewhere: pytest's default "prepend" import mode
caches same-named rootless modules by bare name in ``sys.modules``, so a
whole-repo ``pytest`` run would otherwise have this module shadowed by (or
shadow) ``hospital-core``'s own ``tests/_fixtures.py`` depending on collection
order — a real collision, not a hypothetical one (reproduced while building
this package). A package-qualified name sidesteps it without touching
``hospital-core``'s tests or introducing a ``conftest.py``.
"""

from __future__ import annotations

from hospital.core import EsiAcuity, StaffRole, ZoneType
from hospital.data.scenario import (
    DisruptionSpec,
    FacilitySpec,
    Scenario,
    StaffingSpec,
    WorkloadSpec,
    WorkupProfile,
    ZoneQuota,
)


def small_facility(**overrides: object) -> FacilitySpec:
    """A tiny but fully-connected 3-zone facility, cheap for property tests."""
    params: dict[str, object] = {
        "target_area_sqft": 20_000,
        "zones": (
            ZoneQuota(zone_type=ZoneType.FAST_TRACK, bays=4),
            ZoneQuota(zone_type=ZoneType.GENERAL, bays=6, isolation_bays=2),
            ZoneQuota(zone_type=ZoneType.RESUS_TRAUMA, bays=2, isolation_bays=2),
        ),
        "imaging_suites": 2,
        "lab_stations": 1,
        "triage_rooms": 3,
    }
    params.update(overrides)
    return FacilitySpec(**params)  # type: ignore[arg-type]


def small_workups() -> dict[str, WorkupProfile]:
    return {
        "chest_pain": WorkupProfile(
            provider_visits_mean=1.5,
            nurse_visits_mean=2.0,
            imaging_prob={ZoneType.IMAGING: 0.3},
            labs_mean=1.0,
            procedure_prob=0.1,
        ),
        "laceration": WorkupProfile(
            provider_visits_mean=1.2,
            nurse_visits_mean=1.0,
            imaging_prob={},
            labs_mean=0.1,
            procedure_prob=0.6,
        ),
    }


def small_esi_mix() -> dict[EsiAcuity, float]:
    return {
        EsiAcuity.ESI1: 0.05,
        EsiAcuity.ESI2: 0.2,
        EsiAcuity.ESI3: 0.5,
        EsiAcuity.ESI4: 0.2,
        EsiAcuity.ESI5: 0.05,
    }


def small_workload(**overrides: object) -> WorkloadSpec:
    """A small but fully-valid week-horizon workload spec."""
    params: dict[str, object] = {
        "base_rate_per_hour": 4.0,
        "hourly_profile": tuple([1.0] * 24),
        "dow_profile": tuple([1.0] * 7),
        "esi_mix": small_esi_mix(),
        "complaint_mix": {"chest_pain": 0.6, "laceration": 0.4},
        "ambulance_fraction": 0.2,
        "isolation_fraction": 0.05,
        "workups": small_workups(),
    }
    params.update(overrides)
    return WorkloadSpec(**params)  # type: ignore[arg-type]


def small_staffing() -> StaffingSpec:
    return StaffingSpec(
        default_counts={StaffRole.NURSE: 4, StaffRole.PHYSICIAN: 2, StaffRole.TECH: 1}
    )


def small_scenario(**overrides: object) -> Scenario:
    params: dict[str, object] = {
        "name": "test-scenario",
        "seed": 1234,
        "facility": small_facility(),
        "workload": small_workload(),
        "staffing": small_staffing(),
        "disruptions": DisruptionSpec(),
    }
    params.update(overrides)
    return Scenario(**params)  # type: ignore[arg-type]
