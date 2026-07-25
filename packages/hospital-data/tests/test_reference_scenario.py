"""The committed reference scenario: loads, and drives a valid floor + week."""

from __future__ import annotations

from pathlib import Path

from hospital.core import RandomStreams
from hospital.data.layout import generate_floor
from hospital.data.scenario import dump_scenario, load_arm, load_scenario
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
