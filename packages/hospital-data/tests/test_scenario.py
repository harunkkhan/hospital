"""Scenario schema: YAML round-trip, overlay merge, cross-field validators."""

from __future__ import annotations

from pathlib import Path

import pytest
from _data_fixtures import small_facility, small_scenario, small_workload
from pydantic import ValidationError

from hospital.core import EsiAcuity, SimTime, StaffRole, TimeWindow, ZoneType
from hospital.data.layout import generate_floor
from hospital.data.scenario import (
    CostSpec,
    ShiftBlock,
    StaffingSpec,
    ZoneQuota,
    apply_overlay,
    dump_scenario,
    load_arm,
    load_scenario,
    realize_staff,
)


def test_yaml_round_trip_is_identity(tmp_path: Path) -> None:
    scenario = small_scenario()
    path = tmp_path / "s.yaml"
    dump_scenario(scenario, path)
    loaded = load_scenario(path)
    assert loaded == scenario


def test_dump_load_dump_is_idempotent(tmp_path: Path) -> None:
    scenario = small_scenario()
    p1, p2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    dump_scenario(scenario, p1)
    dump_scenario(load_scenario(p1), p2)
    assert p1.read_text() == p2.read_text()


def test_overlay_deep_merges_mappings_and_replaces_sequences() -> None:
    base = small_scenario()
    overlay = {
        "staffing": {"default_counts": {"nurse": 99}},  # merges into existing mapping
        "disruptions": {
            "events": [{"kind": "surge", "at": 0, "duration": 3_600_000_000, "magnitude": 2.0}]
        },
    }
    merged = apply_overlay(base, overlay)
    # Mapping merge: default_counts.nurse overridden, but sibling keys (physician, tech)
    # from the base survive because dict keys merge recursively.
    assert merged.staffing.default_counts[StaffRole.NURSE] == 99
    assert (
        merged.staffing.default_counts[StaffRole.PHYSICIAN]
        == base.staffing.default_counts[StaffRole.PHYSICIAN]
    )
    # Sequence replace: events is the new tuple wholesale, not positionally patched.
    assert len(merged.disruptions.events) == 1
    assert merged.disruptions.events[0].kind == "surge"
    # seed/workload untouched by the overlay -> CRN-preserving arm semantics.
    assert merged.seed == base.seed
    assert merged.workload == base.workload


def test_load_arm_applies_overlay_onto_base(tmp_path: Path) -> None:
    base = small_scenario()
    base_path = tmp_path / "base.yaml"
    dump_scenario(base, base_path)
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text("staffing:\n  default_counts:\n    nurse: 42\n")
    arm = load_arm(base_path, overlay_path)
    assert arm.staffing.default_counts[StaffRole.NURSE] == 42
    assert arm.workload == base.workload
    assert arm.seed == base.seed


def test_hourly_profile_wrong_length_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(hourly_profile=tuple([1.0] * 23))


def test_dow_profile_wrong_length_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(dow_profile=tuple([1.0] * 6))


def test_esi_mix_not_summing_to_one_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(esi_mix={EsiAcuity.ESI1: 0.5, EsiAcuity.ESI2: 0.6})


def test_complaint_mix_not_summing_to_one_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(complaint_mix={"chest_pain": 0.9, "laceration": 0.2})


def test_complaint_not_in_workups_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(complaint_mix={"chest_pain": 0.6, "unknown_complaint": 0.4})


def test_ambulance_fraction_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(ambulance_fraction=1.5)


def test_isolation_fraction_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(isolation_fraction=-0.1)


def test_isolation_bays_exceeding_bays_rejected() -> None:
    with pytest.raises(ValidationError):
        ZoneQuota(zone_type=ZoneType.GENERAL, bays=2, isolation_bays=3)


def test_facility_requires_at_least_one_bay() -> None:
    with pytest.raises(ValidationError):
        small_facility(zones=(ZoneQuota(zone_type=ZoneType.GENERAL, bays=0),))


def test_cost_spec_rejects_any_key() -> None:
    with pytest.raises(ValidationError):
        CostSpec.model_validate({"dollars_per_minute": 1.0})


def test_realize_staff_round_robins_over_stations_deterministically() -> None:
    facility = small_facility()
    layout = generate_floor(facility)
    window = TimeWindow(start=SimTime(0), end=SimTime(24 * 3_600_000_000))
    staffing = StaffingSpec(default_counts={StaffRole.NURSE: len(layout.stations) * 2 + 1})
    staff = realize_staff(staffing, layout, window)
    assert len(staff) == len(layout.stations) * 2 + 1
    for k, member in enumerate(staff):
        assert member.home_station == layout.stations[k % len(layout.stations)]
    # deterministic across calls
    again = realize_staff(staffing, layout, window)
    assert staff == again


def test_realize_staff_uses_overlapping_block_over_default() -> None:
    facility = small_facility()
    layout = generate_floor(facility)
    day_window = TimeWindow(start=SimTime(0), end=SimTime(12 * 3_600_000_000))
    staffing = StaffingSpec(
        blocks=(
            ShiftBlock(
                window=TimeWindow(start=SimTime(0), end=SimTime(6 * 3_600_000_000)),
                role_counts={StaffRole.NURSE: 5},
            ),
        ),
        default_counts={StaffRole.NURSE: 1},
    )
    staff = realize_staff(staffing, layout, day_window)
    assert len(staff) == 5
