"""KPI: the closed contract — exact key set, and staff_frac_* sum to 1.0."""

from __future__ import annotations

import math
from typing import cast

import pytest
from _fixtures import full_kpi_values

from hospital.core import KPI_KEYS, KpiContractError, KpiVector


def test_kpi_keys_are_the_27_unique_closed_set() -> None:
    assert len(KPI_KEYS) == 27
    assert len(set(KPI_KEYS)) == 27
    # Spot-check representative keys, including per-ESI strata.
    for key in ("completions_per_week", "los_s_mean_by_esi_1", "los_s_p90_by_esi_5"):
        assert key in KPI_KEYS


def test_full_valid_vector_accepted() -> None:
    vec = KpiVector(values=full_kpi_values(door_to_provider_s_mean=1234.0))
    assert vec.values["door_to_provider_s_mean"] == 1234.0


def test_extra_key_rejected() -> None:
    bad = full_kpi_values()
    bad["not_a_kpi"] = 1.0
    with pytest.raises(KpiContractError):
        KpiVector(values=bad)


def test_missing_key_rejected() -> None:
    bad = full_kpi_values()
    del bad["wip_end_of_week"]
    with pytest.raises(KpiContractError):
        KpiVector(values=bad)


def test_staff_frac_must_sum_to_one() -> None:
    bad = full_kpi_values()
    bad["staff_frac_idle"] = 0.5  # now fractions sum to 0.5, not 1.0
    with pytest.raises(KpiContractError):
        KpiVector(values=bad)


def test_staff_frac_sum_within_epsilon_accepted() -> None:
    values = full_kpi_values(
        staff_frac_walk=0.2,
        staff_frac_direct_care=0.3,
        staff_frac_cleaning=0.1,
        staff_frac_documentation=0.15,
        staff_frac_idle=0.25,
    )
    vec = KpiVector(values=values)
    total = math.fsum(vec.values[k] for k in KPI_KEYS if k.startswith("staff_frac_"))
    assert abs(total - 1.0) < 1e-9


def test_empty_strata_nan_is_allowed() -> None:
    # Empty ESI strata are reported as NaN (never omitted) — the vector stays complete.
    vec = KpiVector(values=full_kpi_values(los_s_mean_by_esi_1=float("nan")))
    assert math.isnan(vec.values["los_s_mean_by_esi_1"])


# --- Finding #8: the validated values mapping is immutable after construction ---


def test_values_mapping_is_immutable() -> None:
    vec = KpiVector(values=full_kpi_values())
    mutable = cast("dict[str, float]", vec.values)  # runtime type is a read-only mapping
    with pytest.raises(TypeError):
        mutable["staff_frac_idle"] = 0.5
    with pytest.raises(TypeError):
        mutable["injected_key"] = 1.0
    # Mutating the caller's original dict must not leak into the validated vector.
    original = full_kpi_values()
    vec2 = KpiVector(values=original)
    original["staff_frac_idle"] = 99.0
    assert vec2.values["staff_frac_idle"] == 1.0


def test_values_still_serializes() -> None:
    vec = KpiVector(values=full_kpi_values(door_to_provider_s_mean=12.0))
    dumped = vec.model_dump()
    assert dumped["values"]["door_to_provider_s_mean"] == 12.0
    restored = KpiVector.model_validate_json(vec.model_dump_json())
    assert restored.values["door_to_provider_s_mean"] == 12.0


# --- Finding #12: each proportion KPI must lie in [0, 1], even if fractions sum to 1 ---


def test_staff_frac_out_of_range_rejected_even_if_sum_is_one() -> None:
    # -1 + 0 + 0 + 0 + 2 == 1.0 passes the sum check but is out of range.
    bad = full_kpi_values(staff_frac_walk=-1.0, staff_frac_idle=2.0)
    assert math.isclose(sum(bad[k] for k in KPI_KEYS if k.startswith("staff_frac_")), 1.0)
    with pytest.raises(KpiContractError):
        KpiVector(values=bad)


def test_utilization_out_of_range_rejected() -> None:
    with pytest.raises(KpiContractError):
        KpiVector(values=full_kpi_values(bay_utilization=1.5))
