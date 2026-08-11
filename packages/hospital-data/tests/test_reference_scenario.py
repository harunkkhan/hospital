"""The committed reference scenario: loads, and drives a valid floor + week."""

from __future__ import annotations

from pathlib import Path

import pytest

from hospital.core import (
    WARD_ZONE_TYPES,
    LayoutError,
    RandomStreams,
    RouteGraph,
    SimTime,
    StaffRole,
    TimeWindow,
    hours,
)
from hospital.data.hospital import generate_hospital
from hospital.data.layout import generate_floor
from hospital.data.scenario import (
    ShiftBlock,
    StaffingSpec,
    dump_scenario,
    load_arm,
    load_scenario,
    realize_staff,
)
from hospital.data.workload import generate_workload

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCENARIOS = _REPO_ROOT / "scenarios"


def test_er_floor_yaml_loads() -> None:
    scenario = load_scenario(_SCENARIOS / "er_floor.yaml")
    assert scenario.name == "er_floor"
    assert sum(q.bays for q in scenario.facility.zones) > 0


def test_er_floor_yaml_produces_a_valid_connected_floor() -> None:
    scenario = load_scenario(_SCENARIOS / "er_floor.yaml")
    layout = generate_floor(scenario.facility)
    assert len(layout.bays) > 0
    entrance = layout.entrances[0]
    for bay in layout.bays:
        assert layout.graph.dijkstra(entrance, bay.node).total.root >= 0


def test_er_floor_yaml_produces_a_full_week_of_arrivals() -> None:
    scenario = load_scenario(_SCENARIOS / "er_floor.yaml")
    streams = RandomStreams(scenario.seed)
    arrivals = generate_workload(scenario.workload, streams, disruptions=scenario.disruptions)
    assert len(arrivals) > 0
    week_end_us = scenario.workload.horizon.end.root
    assert all(0 <= a.patient.arrival_time.root < week_end_us for a in arrivals)
    # deterministic given the scenario's own seed
    again = generate_workload(
        scenario.workload, RandomStreams(scenario.seed), disruptions=scenario.disruptions
    )
    assert arrivals == again


def test_er_floor_stressed_yaml_loads_and_round_trips(tmp_path: Path) -> None:
    """The committed M1 operating point (the scenario the comparison/goldens pin).

    It must be exactly the reference floor + demand under a leaner roster, and
    it must survive the codec byte-identically (``dump_scenario`` is canonical,
    so re-dumping the loaded model reproduces the committed file).
    """
    src = _SCENARIOS / "er_floor_stressed.yaml"
    scenario = load_scenario(src)
    assert scenario.name == "er_floor_stressed"

    reference = load_scenario(_SCENARIOS / "er_floor.yaml")
    assert scenario.seed == reference.seed
    assert scenario.facility == reference.facility
    assert scenario.workload == reference.workload  # identical demand (CRN-comparable)
    assert scenario.staffing != reference.staffing  # the stress is staffing only

    out = tmp_path / "er_floor_stressed.yaml"
    dump_scenario(scenario, out)
    assert out.read_text() == src.read_text()
    assert load_scenario(out) == scenario


def test_baseline_and_surge_arms_share_the_identical_base_week() -> None:
    baseline = load_arm(_SCENARIOS / "er_floor.yaml", _SCENARIOS / "arms" / "baseline.yaml")
    surge = load_arm(_SCENARIOS / "er_floor.yaml", _SCENARIOS / "arms" / "surge.yaml")
    assert baseline.workload == surge.workload
    assert baseline.seed == surge.seed
    assert len(surge.disruptions.events) == 1

    base_arrivals = generate_workload(
        baseline.workload, RandomStreams(baseline.seed), disruptions=baseline.disruptions
    )
    surge_arrivals = generate_workload(
        surge.workload, RandomStreams(surge.seed), disruptions=surge.disruptions
    )
    base_by_id = {a.patient.id.root: a.patient for a in base_arrivals}
    surge_base_by_id = {
        a.patient.id.root: a.patient
        for a in surge_arrivals
        if not a.patient.id.root.startswith("s")
    }
    assert base_by_id.keys() == surge_base_by_id.keys()
    assert all(base_by_id[pid] == surge_base_by_id[pid] for pid in base_by_id)
    assert len(surge_arrivals) > len(base_arrivals)


def test_hospital_yaml_is_the_reference_floor_in_a_building(tmp_path: Path) -> None:
    """The committed M4 operating point: same ED, same week, floors above it.

    Held to the same discipline as ``er_floor_stressed`` — exactly one thing differs
    from the reference, so the two are CRN-comparable and any measured difference is
    the building rather than the demand. Here that one thing is ``upper_floors``:
    seed, facility, workload, and staffing are identical, which is what lets an ED-only
    run and a hospital run of the same week be set beside each other at all.
    """
    src = _SCENARIOS / "hospital.yaml"
    scenario = load_scenario(src)
    assert scenario.name == "hospital"

    reference = load_scenario(_SCENARIOS / "er_floor.yaml")
    assert scenario.seed == reference.seed
    assert scenario.facility == reference.facility  # the ED half is untouched
    assert scenario.workload == reference.workload
    assert scenario.staffing == reference.staffing
    assert not reference.upper_floors
    assert [f.name for f in scenario.upper_floors] == ["icu", "surgery", "med_surg"]

    dump_scenario(scenario, out := tmp_path / "hospital.yaml")
    assert out.read_text() == src.read_text()
    assert load_scenario(out) == scenario


def test_hospital_yaml_builds_one_connected_building_reachable_by_elevator() -> None:
    """Every inpatient bed is reachable from the ED entrance, and only via a shaft.

    ``generate_hospital`` already refuses a disconnected stack, so the reachability
    half would pass on a building with a stray inter-floor corridor too. Cutting the
    shafts and requiring an upstairs bed to become unreachable is what proves the
    elevators are the only vertical path in the *committed* scenario, not merely in
    the generator's unit tests.
    """
    scenario = load_scenario(_SCENARIOS / "hospital.yaml")
    layout = generate_hospital(scenario.hospital())
    ward_bays = [bay for bay in layout.bays if bay.zone_type in WARD_ZONE_TYPES]
    assert len(ward_bays) == 10 + 16 + 48
    assert layout.elevators

    entrance = layout.entrances[0]
    for bay in ward_bays:
        assert layout.graph.dijkstra(entrance, bay.node).total.root > 0

    shafts = set(layout.elevators)
    severed = RouteGraph(
        nodes=layout.graph.nodes,
        edges=tuple(e for e in layout.graph.edges if not (e.a in shafts and e.b in shafts)),
    )
    with pytest.raises(LayoutError):
        severed.dijkstra(entrance, ward_bays[-1].node)


def test_shift_aware_is_opt_in_and_the_default_is_the_old_behaviour() -> None:
    """Same people either way; only their schedules differ.

    Ids are deliberately unchanged by the mode, so flipping the flag does not renumber
    anyone — and with the flag off nobody carries a schedule at all, which is what keeps
    every scenario and golden written before shift-awareness byte-identical.
    """
    scenario = load_scenario(_SCENARIOS / "er_floor.yaml")
    assert scenario.staffing.shift_aware is False
    layout = generate_hospital(scenario.hospital())
    horizon = scenario.workload.horizon
    window = TimeWindow(start=horizon.start, end=horizon.end)

    flat = realize_staff(scenario.staffing, layout, window)
    aware = realize_staff(
        scenario.staffing.model_copy(update={"shift_aware": True}), layout, window
    )
    assert [m.id.root for m in flat] == [m.id.root for m in aware]
    assert all(not m.shifts for m in flat)
    assert all(m.shifts for m in aware)


def test_the_reference_ed_declares_no_night_cover_which_is_why_it_is_opt_in() -> None:
    """The finding that made this a mode rather than a fix.

    The committed reference scenario schedules 07:00-19:00 blocks only, leaving 84 of the
    week's 168 hours in no block at all. Collapsed, that runs day headcount at 03:00;
    enacted, it runs *nobody*. Flipping the default would therefore un-staff every
    committed ED overnight — a re-siting of every scenario, not a bug fix.
    """
    scenario = load_scenario(_SCENARIOS / "er_floor.yaml")
    layout = generate_hospital(scenario.hospital())
    horizon = scenario.workload.horizon
    window = TimeWindow(start=horizon.start, end=horizon.end)
    aware = realize_staff(
        scenario.staffing.model_copy(update={"shift_aware": True}), layout, window
    )
    noon = SimTime(hours(12).root)
    three_am = SimTime(hours(3).root)
    assert any(m.on_shift(noon) for m in aware)
    assert not any(m.on_shift(three_am) for m in aware)


def test_member_k_works_every_block_that_schedules_more_than_k_of_their_role() -> None:
    """A tapering headcount rosters the low-numbered members for more blocks."""
    layout = generate_hospital(load_scenario(_SCENARIOS / "er_floor.yaml").hospital())
    window = TimeWindow(start=SimTime(0), end=SimTime(hours(48).root))
    spec = StaffingSpec(
        shift_aware=True,
        blocks=(
            ShiftBlock(
                window=TimeWindow(start=SimTime(0), end=SimTime(hours(12).root)),
                role_counts={StaffRole.NURSE: 3},
            ),
            ShiftBlock(
                window=TimeWindow(start=SimTime(hours(24).root), end=SimTime(hours(36).root)),
                role_counts={StaffRole.NURSE: 1},
            ),
        ),
    )
    nurses = sorted(realize_staff(spec, layout, window), key=lambda m: m.id.root)
    assert len(nurses) == 3
    assert len(nurses[0].shifts) == 2  # scheduled by both blocks
    assert len(nurses[1].shifts) == 1  # only the block asking for 3
    assert len(nurses[2].shifts) == 1


def test_shifts_are_clipped_to_the_run_window() -> None:
    """A block running past the horizon must not roster anyone beyond it."""
    layout = generate_hospital(load_scenario(_SCENARIOS / "er_floor.yaml").hospital())
    window = TimeWindow(start=SimTime(0), end=SimTime(hours(10).root))
    spec = StaffingSpec(
        shift_aware=True,
        blocks=(
            ShiftBlock(
                window=TimeWindow(start=SimTime(0), end=SimTime(hours(24).root)),
                role_counts={StaffRole.NURSE: 1},
            ),
        ),
    )
    member = realize_staff(spec, layout, window)[0]
    assert member.shifts == (TimeWindow(start=SimTime(0), end=SimTime(hours(10).root)),)
