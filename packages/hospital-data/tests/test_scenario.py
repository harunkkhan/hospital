"""Scenario schema: YAML round-trip, overlay merge, cross-field validators."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from _data_fixtures import small_facility, small_scenario, small_workload, small_workups
from pydantic import ValidationError

from hospital.core import Duration, EsiAcuity, SimTime, StaffRole, TimeWindow, ZoneType
from hospital.data.layout import generate_floor
from hospital.data.scenario import (
    CostSpec,
    DisruptionEvent,
    ShiftBlock,
    StaffingSpec,
    WorkupProfile,
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


# Finding 13: a list/scalar overlay root must raise, never silently run the
# baseline in place of the requested arm. Only null/empty-mapping means "no
# changes".
def test_load_arm_rejects_non_mapping_overlay_root(tmp_path: Path) -> None:
    base = small_scenario()
    base_path = tmp_path / "base.yaml"
    dump_scenario(base, base_path)

    list_overlay = tmp_path / "list.yaml"
    list_overlay.write_text("- kind: surge\n")
    with pytest.raises(ValueError, match="overlay root must be a mapping"):
        load_arm(base_path, list_overlay)

    scalar_overlay = tmp_path / "scalar.yaml"
    scalar_overlay.write_text("42\n")
    with pytest.raises(ValueError, match="overlay root must be a mapping"):
        load_arm(base_path, scalar_overlay)

    empty_overlay = tmp_path / "empty.yaml"
    empty_overlay.write_text("")
    assert load_arm(base_path, empty_overlay) == base

    null_overlay = tmp_path / "null.yaml"
    null_overlay.write_text("null\n")
    assert load_arm(base_path, null_overlay) == base

    empty_mapping_overlay = tmp_path / "mapping.yaml"
    empty_mapping_overlay.write_text("{}\n")
    assert load_arm(base_path, empty_mapping_overlay) == base


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


# Finding 1: non-finite rates/profiles must be rejected at load — an ``.inf``
# arrival rate makes the Poisson sampler's exponential scale 0 and spins
# ``generate_workload`` forever.
def test_non_finite_rates_and_profiles_rejected() -> None:
    with pytest.raises(ValidationError):
        small_workload(base_rate_per_hour=float("inf"))
    with pytest.raises(ValidationError):
        small_workload(hourly_profile=(float("nan"), *[1.0] * 23))
    with pytest.raises(ValidationError):
        small_workload(dow_profile=(float("inf"), *[1.0] * 6))
    with pytest.raises(ValidationError):
        small_workload(esi_workup_scale={EsiAcuity.ESI1: float("nan")})
    with pytest.raises(ValidationError):
        DisruptionEvent(
            kind="surge", at=SimTime(0), duration=Duration(3_600_000_000), magnitude=float("inf")
        )
    with pytest.raises(ValidationError):
        WorkupProfile(
            provider_visits_mean=float("inf"),
            nurse_visits_mean=1.0,
            labs_mean=1.0,
            procedure_prob=0.1,
        )
    with pytest.raises(ValidationError):
        WorkupProfile(
            provider_visits_mean=1.0,
            nurse_visits_mean=1.0,
            labs_mean=float("nan"),
            procedure_prob=0.1,
        )
    with pytest.raises(ValidationError):
        small_facility(aspect_ratio=float("inf"))


# Finding 9: every categorical weight must be finite and >= 0 individually —
# a {-0.1, 1.1} pair sums to 1.0 and a NaN weight poisons the total silently.
def test_mix_weights_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValidationError):
        small_workload(complaint_mix={"chest_pain": -0.1, "laceration": 1.1})
    with pytest.raises(ValidationError):
        small_workload(
            esi_mix={
                EsiAcuity.ESI1: float("nan"),
                EsiAcuity.ESI2: 0.2,
                EsiAcuity.ESI3: 0.5,
                EsiAcuity.ESI4: 0.2,
                EsiAcuity.ESI5: 0.05,
            }
        )


# Finding 10: imaging probabilities are probabilities — finite and in [0, 1].
def test_imaging_prob_values_bounded_to_unit_interval() -> None:
    for bad in (1.2, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            WorkupProfile(
                provider_visits_mean=1.0,
                nurse_visits_mean=1.0,
                imaging_prob={ZoneType.IMAGING: bad},
                labs_mean=1.0,
                procedure_prob=0.1,
            )


# Finding 12: negative staffing counts silently realize zero staff (range(-n)).
def test_negative_staffing_counts_rejected() -> None:
    window = TimeWindow(start=SimTime(0), end=SimTime(3_600_000_000))
    with pytest.raises(ValidationError):
        ShiftBlock(window=window, role_counts={StaffRole.NURSE: -1})
    with pytest.raises(ValidationError):
        StaffingSpec(default_counts={StaffRole.NURSE: -2})


# Finding 5: a multi-item frozenset iterates in hash-table order, which varies
# with PYTHONHASHSEED — serialization must emit sorted order to stay byte-stable.
def test_equipment_serializes_in_sorted_order() -> None:
    equipment = frozenset({"vent", "ct_adjacent", "defib", "monitor", "us"})
    quota = ZoneQuota(zone_type=ZoneType.RESUS_TRAUMA, bays=2, equipment=equipment)
    assert quota.model_dump(mode="json")["equipment"] == sorted(equipment)
    assert quota.model_dump()["equipment"] == sorted(equipment)
    # Round-trips back to the same frozenset.
    assert ZoneQuota.model_validate(quota.model_dump(mode="json")) == quota


# Finding 8: Mapping fields inside a FrozenModel must be read-only copies, not
# mutable dicts (same treatment as core.kpi.KpiVector.values).
def test_mapping_fields_are_read_only() -> None:
    scenario = small_scenario()
    workload = scenario.workload
    with pytest.raises(TypeError):
        workload.esi_mix[EsiAcuity.ESI1] = 0.9  # type: ignore[index]
    with pytest.raises(TypeError):
        workload.complaint_mix["chest_pain"] = 0.9  # type: ignore[index]
    with pytest.raises(TypeError):
        workload.workups["chest_pain"] = small_workups()["laceration"]  # type: ignore[index]
    with pytest.raises(TypeError):
        workload.esi_workup_scale[EsiAcuity.ESI1] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        workload.workups["chest_pain"].imaging_prob[ZoneType.IMAGING] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        scenario.staffing.default_counts[StaffRole.NURSE] = 99  # type: ignore[index]
    block = ShiftBlock(
        window=TimeWindow(start=SimTime(0), end=SimTime(3_600_000_000)),
        role_counts={StaffRole.NURSE: 1},
    )
    with pytest.raises(TypeError):
        block.role_counts[StaffRole.NURSE] = 99  # type: ignore[index]


def test_frozen_mappings_still_round_trip_through_yaml(tmp_path: Path) -> None:
    scenario = small_scenario()
    path = tmp_path / "frozen.yaml"
    dump_scenario(scenario, path)
    assert load_scenario(path) == scenario
    assert math.isclose(sum(scenario.workload.esi_mix.values()), 1.0)


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


# Finding 6: defaults fill in ONLY roles that no overlapping block supplies —
# a block explicitly scheduling 2 nurses realizes 2, never a higher default.
def test_realize_staff_defaults_only_fill_roles_absent_from_blocks() -> None:
    facility = small_facility()
    layout = generate_floor(facility)
    day_window = TimeWindow(start=SimTime(0), end=SimTime(12 * 3_600_000_000))
    staffing = StaffingSpec(
        blocks=(
            ShiftBlock(
                window=TimeWindow(start=SimTime(0), end=SimTime(6 * 3_600_000_000)),
                role_counts={StaffRole.NURSE: 2},
            ),
        ),
        default_counts={StaffRole.NURSE: 10, StaffRole.TECH: 3},
    )
    staff = realize_staff(staffing, layout, day_window)
    nurses = [m for m in staff if m.role is StaffRole.NURSE]
    techs = [m for m in staff if m.role is StaffRole.TECH]
    # The overlapping block supplies nurses, so the default of 10 must not win.
    assert len(nurses) == 2
    # No block supplies techs, so the default fills that role in.
    assert len(techs) == 3
