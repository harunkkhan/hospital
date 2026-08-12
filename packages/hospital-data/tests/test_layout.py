"""``generate_floor``: determinism, area, coverage, geometry, reachability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from _data_fixtures import small_facility
from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import LayoutError, WalkSpeed, ZoneType, walk_duration
from hospital.data.layout import generate_floor
from hospital.data.scenario import FacilitySpec, ZoneQuota, load_scenario

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQFT_TO_CM2 = 929.0304
_GOLDEN_ER_FLOOR_GRAPH_SHA256 = "55471c97946a260edfe8fce783349422ba2b13c5e6fd4327f9c2c804d7a35c0d"

# The hash this golden held until `RouteNode` gained its viz-only `floor`. Kept, and still
# asserted below against the graph with that field projected out, because the two together
# say something the new hash alone cannot: the *geometry* did not move, only its
# serialization grew a key. A re-baseline that could not show that would be indistinguishable
# from a generator change nobody noticed.
_PRE_FLOOR_ER_FLOOR_GRAPH_SHA256 = (
    "3859dacaa4511d6496ea7794a4d891c242d9c7e31ce10994719a74d8865d76ec"
)


def _graph_hash(facility: FacilitySpec) -> str:
    layout = generate_floor(facility)
    return hashlib.sha256(layout.graph.model_dump_json().encode("utf-8")).hexdigest()


def test_generate_floor_is_deterministic() -> None:
    facility = small_facility()
    a = generate_floor(facility)
    b = generate_floor(facility)
    assert a.graph.model_dump() == b.graph.model_dump()
    assert a.bays == b.bays
    assert a.zones == b.zones
    assert a.stations == b.stations
    assert a.entrances == b.entrances


def test_reference_er_floor_graph_matches_golden_hash() -> None:
    scenario = load_scenario(_REPO_ROOT / "scenarios" / "er_floor.yaml")
    assert _graph_hash(scenario.facility) == _GOLDEN_ER_FLOOR_GRAPH_SHA256


def test_the_floor_field_changed_the_serialization_and_not_the_geometry() -> None:
    """Justifies the one re-baseline this golden has had.

    ``RouteNode.floor`` is viz-only and defaults to 0, so adding it moved the hash without
    moving a single coordinate. Projecting the field back out must reproduce the pre-M4b
    digest exactly — which is the difference between "we added a key" and "the floor
    generator quietly changed", two things a bare new constant cannot tell apart.
    """
    scenario = load_scenario(_REPO_ROOT / "scenarios" / "er_floor.yaml")
    graph = generate_floor(scenario.facility).graph
    assert {node.floor for node in graph.nodes} == {0}, "a single floor is all floor 0"

    dumped = graph.model_dump(mode="json")
    for node in dumped["nodes"]:
        node.pop("floor")
    legacy = hashlib.sha256(json.dumps(dumped, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert legacy == _PRE_FLOOR_ER_FLOOR_GRAPH_SHA256


def test_reference_er_floor_area_within_tolerance() -> None:
    scenario = load_scenario(_REPO_ROOT / "scenarios" / "er_floor.yaml")
    facility = scenario.facility
    layout = generate_floor(facility)
    xs = [n.x_cm for n in layout.graph.nodes]
    ys = [n.y_cm for n in layout.graph.nodes]
    footprint_sqft = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / _SQFT_TO_CM2
    # The bounding box of placed nodes is a lower bound on the true footprint
    # (entrances/specials sit inside the modeled building, not past its walls);
    # it should be within a generous band of the requested target.
    assert 0.3 * facility.target_area_sqft <= footprint_sqft <= facility.target_area_sqft


def test_every_bay_serving_station_is_a_station_inside_its_own_zone() -> None:
    facility = small_facility()
    layout = generate_floor(facility)
    stations = set(layout.stations)
    for bay in layout.bays:
        assert bay.serving_station in stations
        # Station ids are minted as f"station_{zone_id}_{s:02d}" — the zone id is
        # embedded, so the covering station must belong to the bay's own zone.
        assert bay.serving_station.root.startswith(f"station_{bay.zone.root}_")


def test_every_edge_seconds_and_distance_are_consistent() -> None:
    facility = small_facility()
    layout = generate_floor(facility)
    speed = WalkSpeed(facility.walk_speed_cm_s)
    node_by_id = {n.id: n for n in layout.graph.nodes}
    for edge in layout.graph.edges:
        a, b = node_by_id[edge.a], node_by_id[edge.b]
        assert edge.distance.root == abs(a.x_cm - b.x_cm) + abs(a.y_cm - b.y_cm)
        assert edge.distance.root > 0
        assert edge.seconds == walk_duration(edge.distance, speed)


def test_no_two_nodes_share_coordinates() -> None:
    facility = small_facility()
    layout = generate_floor(facility)
    coords = [(n.x_cm, n.y_cm) for n in layout.graph.nodes]
    assert len(coords) == len(set(coords))


_FUZZ_ZONE_TYPES = (
    ZoneType.GENERAL,
    ZoneType.FAST_TRACK,
    ZoneType.OBSERVATION,
    ZoneType.RESUS_TRAUMA,
)


@st.composite
def _facility_spec(draw: st.DrawFn) -> FacilitySpec:
    n_zones = draw(st.integers(min_value=1, max_value=4))
    zones: list[ZoneQuota] = []
    for _ in range(n_zones):
        bays = draw(st.integers(min_value=0, max_value=8))
        iso = draw(st.integers(min_value=0, max_value=bays))
        mbps = draw(st.integers(min_value=1, max_value=6))
        zt = draw(st.sampled_from(_FUZZ_ZONE_TYPES))
        zones.append(
            ZoneQuota(zone_type=zt, bays=bays, isolation_bays=iso, max_bays_per_station=mbps)
        )
    if sum(z.bays for z in zones) < 1:
        zones[0] = zones[0].model_copy(update={"bays": 1, "isolation_bays": 0})
    area = draw(st.sampled_from([5_000, 20_000, 50_000, 100_000]))
    margin = draw(st.sampled_from([100, 300, 600]))
    room_depth = draw(st.sampled_from([100, 210, 420]))
    return FacilitySpec(
        zones=tuple(zones),
        target_area_sqft=area,
        corridor_margin_cm=margin,
        room_depth_cm=room_depth,
        imaging_suites=draw(st.integers(min_value=0, max_value=3)),
        lab_stations=draw(st.integers(min_value=0, max_value=2)),
        triage_rooms=draw(st.integers(min_value=0, max_value=4)),
    )


@settings(max_examples=40, deadline=None)
@given(_facility_spec())
def test_every_entrance_reaches_every_bay_and_imaging_and_lab(facility: FacilitySpec) -> None:
    layout = generate_floor(facility)
    entrance = layout.entrances[0]
    for bay in layout.bays:
        path = layout.graph.dijkstra(entrance, bay.node)
        assert path.total.root >= 0
    for node_id in (*layout.imaging_nodes, *layout.lab_nodes):
        path = layout.graph.dijkstra(entrance, node_id)
        assert path.total.root >= 0


# Finding 7: max_bays_per_station is a hard cap. A zone with more required
# stations than spine columns (2 bays, cap 1 -> 1 column) must still get its
# full station count — at distinct coordinates — and no station may ever
# serve more bays than the cap.
def test_max_bays_per_station_cap_is_honored() -> None:
    facility = small_facility(
        zones=(ZoneQuota(zone_type=ZoneType.GENERAL, bays=2, max_bays_per_station=1),),
    )
    layout = generate_floor(facility)
    assert len(layout.stations) == 2
    served: dict[str, int] = {}
    for bay in layout.bays:
        served[bay.serving_station.root] = served.get(bay.serving_station.root, 0) + 1
    assert all(count <= 1 for count in served.values())
    # The doubled-up column still yields distinct node coordinates.
    coords = [(n.x_cm, n.y_cm) for n in layout.graph.nodes]
    assert len(coords) == len(set(coords))


def test_station_coverage_never_exceeds_the_cap_even_with_nearest_ties() -> None:
    # 6 bays over 3 columns with cap 1 -> 6 stations, each serving exactly one bay.
    facility = small_facility(
        zones=(ZoneQuota(zone_type=ZoneType.GENERAL, bays=6, max_bays_per_station=1),),
    )
    layout = generate_floor(facility)
    assert len(layout.stations) == 6
    served = [bay.serving_station for bay in layout.bays]
    assert len(served) == len(set(served))


def test_malformed_facility_spec_raises_layout_error() -> None:
    # An absurdly tiny footprint with many bays drives the derived pitch to <= 0.
    facility = small_facility(
        target_area_sqft=1,
        zones=(ZoneQuota(zone_type=ZoneType.GENERAL, bays=5000),),
    )
    with pytest.raises(LayoutError):
        generate_floor(facility)
