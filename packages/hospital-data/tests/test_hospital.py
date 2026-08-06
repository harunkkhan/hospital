"""``generate_hospital``: stacking, namespacing, and the elevator as the only way up.

The load-bearing tests here are :func:`test_one_floor_is_byte_identical_to_the_er_floor`
and :func:`test_the_elevators_are_the_only_way_between_floors`. The first is what lets the
committed M1 goldens keep validating instead of being re-baselined; the second is the claim
the whole vertical model rests on, and it is verified by cutting the shafts and checking
the building falls apart rather than by asserting it in a docstring.
"""

from __future__ import annotations

import pytest

from hospital.core import ED_ZONE_TYPES, WARD_ZONE_TYPES, LayoutError, NodeId, ZoneType
from hospital.data.hospital import FloorSpec, HospitalSpec, generate_hospital
from hospital.data.layout import generate_floor
from hospital.data.scenario import FacilitySpec, ZoneQuota


def _ed() -> FacilitySpec:
    return FacilitySpec(
        target_area_sqft=40_000,
        zones=(
            ZoneQuota(zone_type=ZoneType.GENERAL, bays=6, isolation_bays=1),
            ZoneQuota(zone_type=ZoneType.RESUS_TRAUMA, bays=2, isolation_bays=2),
            ZoneQuota(zone_type=ZoneType.FAST_TRACK, bays=4),
        ),
        imaging_suites=2,
        lab_stations=1,
        triage_rooms=3,
    )


def _ward(zone_type: ZoneType, bays: int = 8) -> FacilitySpec:
    """A ward floor: beds and a station, no imaging/lab/triage of its own."""
    return FacilitySpec(
        target_area_sqft=30_000,
        zones=(ZoneQuota(zone_type=zone_type, bays=bays, isolation_bays=2),),
        imaging_suites=0,
        lab_stations=0,
        triage_rooms=0,
    )


def _tower() -> HospitalSpec:
    return HospitalSpec(
        floors=(
            FloorSpec(name="ed", facility=_ed()),
            FloorSpec(name="icu", facility=_ward(ZoneType.ICU, bays=6)),
            FloorSpec(name="med_surg", facility=_ward(ZoneType.MED_SURG, bays=10)),
        ),
        elevator_shafts=2,
    )


def test_one_floor_is_byte_identical_to_the_er_floor() -> None:
    """A single-floor hospital must be the ER floor exactly, ids and all.

    This is what keeps the committed M1/M2 goldens a check rather than a re-baseline: the
    multi-floor code path exists, and an ED-only scenario is untouched by it.
    """
    facility = _ed()
    solo = generate_hospital(HospitalSpec(floors=(FloorSpec(name="ed", facility=facility),)))
    assert solo == generate_floor(facility)
    assert solo.elevators == ()


def test_stacking_is_deterministic() -> None:
    assert generate_hospital(_tower()) == generate_hospital(_tower())


def test_every_id_is_unique_across_the_stack() -> None:
    """Floor-local ids would silently merge two floors into one.

    ``generate_floor`` numbers bays from zero on every floor, so without namespacing the
    two ``bay_gen_00``s would collapse to a single node with corridors on both storeys —
    and a patient could walk between floors without ever boarding a car.
    """
    layout = generate_hospital(_tower())
    for label, ids in (
        ("nodes", [n.id.root for n in layout.graph.nodes]),
        ("bays", [b.id.root for b in layout.bays]),
        ("zones", [z.id.root for z in layout.zones]),
    ):
        assert len(ids) == len(set(ids)), f"duplicate {label}"


def test_every_bay_and_zone_knows_its_floor() -> None:
    layout = generate_hospital(_tower())
    by_zone = {zone.id: zone for zone in layout.zones}
    assert {zone.floor for zone in layout.zones} == {0, 1, 2}
    # The ED's zones are all on the ground floor; wards are all above it.
    for zone in layout.zones:
        if zone.zone_type in WARD_ZONE_TYPES:
            assert zone.floor > 0, f"{zone.id.root} is a ward on the ground floor"
        elif zone.zone_type in ED_ZONE_TYPES:
            assert zone.floor == 0, f"{zone.id.root} is an ED zone upstairs"
    for bay in layout.bays:
        assert bay.zone in by_zone, f"{bay.id.root} references an unknown zone"


def test_the_stack_is_fully_reachable_from_the_door() -> None:
    layout = generate_hospital(_tower())
    origin = layout.entrances[0]
    for node in layout.graph.nodes:
        layout.graph.dijkstra(origin, node.id)  # raises if unreachable


def test_the_elevators_are_the_only_way_between_floors() -> None:
    """Cut the shafts and the building must fall apart.

    Asserted by removing every vertical edge and checking an upstairs bay becomes
    unreachable. Without this, a stray edge between floors would leave every other test
    passing while patients strolled up a corridor to the ICU.
    """
    layout = generate_hospital(_tower())
    shafts = frozenset(layout.elevators)
    vertical = frozenset(
        (edge.a, edge.b) for edge in layout.graph.edges if edge.a in shafts and edge.b in shafts
    )
    assert vertical, "the stack has no vertical edges at all"

    upstairs = next(bay for bay in layout.bays if bay.id.root.startswith("f01_"))
    origin = layout.entrances[0]
    layout.graph.dijkstra(origin, upstairs.node)  # fine with the shafts intact

    blocked = vertical | frozenset((b, a) for a, b in vertical)
    with pytest.raises(LayoutError):
        layout.graph.dijkstra(origin, upstairs.node, blocked_edges=blocked)


def test_going_upstairs_costs_more_than_crossing_a_floor() -> None:
    """An elevator's cost is its dwell and travel, not the four metres it moves."""
    layout = generate_hospital(_tower())
    origin = layout.entrances[0]
    ground = next(bay for bay in layout.bays if not bay.id.root.startswith("f"))
    first = next(bay for bay in layout.bays if bay.id.root.startswith("f01_"))
    second = next(bay for bay in layout.bays if bay.id.root.startswith("f02_"))

    across = layout.graph.dijkstra(origin, ground.node).total
    up_one = layout.graph.dijkstra(origin, first.node).total
    up_two = layout.graph.dijkstra(origin, second.node).total
    assert up_one.root > across.root, "a storey up cost no more than a walk across"
    assert up_two.root > up_one.root, "two storeys cost no more than one"


def test_a_slower_car_makes_the_upper_floors_farther() -> None:
    """The spec's seconds must actually reach routing, not just be stored."""

    def cost(seconds_per_floor: float) -> int:
        spec = _tower().model_copy(update={"seconds_per_floor": seconds_per_floor})
        layout = generate_hospital(spec)
        upstairs = next(bay for bay in layout.bays if bay.id.root.startswith("f02_"))
        return layout.graph.dijkstra(layout.entrances[0], upstairs.node).total.root

    assert cost(60.0) > cost(6.0)


def test_only_the_ground_floor_has_entrances() -> None:
    """An upstairs entrance would let arrivals teleport past the shafts."""
    layout = generate_hospital(_tower())
    assert layout.entrances
    for node in layout.entrances:
        assert not node.root.startswith("f"), f"{node.root} is an entrance above ground"


def test_each_floor_gets_a_car_on_every_shaft() -> None:
    spec = _tower()
    layout = generate_hospital(spec)
    assert len(layout.elevators) == spec.elevator_shafts * len(spec.floors)
    assert len(set(layout.elevators)) == len(layout.elevators)
    assert set(layout.elevators) <= {node.id for node in layout.graph.nodes}


def test_wards_contribute_their_beds_as_placeable_bays() -> None:
    """A ward floor is a placement target, so its beds have to be real bays."""
    layout = generate_hospital(_tower())
    icu = [bay for bay in layout.bays if bay.zone_type is ZoneType.ICU]
    med_surg = [bay for bay in layout.bays if bay.zone_type is ZoneType.MED_SURG]
    assert len(icu) == 6
    assert len(med_surg) == 10
    # Each is served by a station on its own floor, not by the ED's.
    for bay in [*icu, *med_surg]:
        prefix = bay.id.root.split("_", 1)[0]
        assert bay.serving_station.root.startswith(prefix), (
            f"{bay.id.root} is served from another floor"
        )


def test_duplicate_floor_names_are_refused() -> None:
    """Two floors called the same thing is a spec error, not something to resolve."""
    with pytest.raises(ValueError, match="duplicate names"):
        HospitalSpec(
            floors=(
                FloorSpec(name="icu", facility=_ward(ZoneType.ICU)),
                FloorSpec(name="icu", facility=_ward(ZoneType.MED_SURG)),
            )
        )


def test_an_empty_stack_is_refused() -> None:
    with pytest.raises(ValueError):
        HospitalSpec(floors=())


def test_a_floor_with_no_lobby_is_refused() -> None:
    """Every floor needs a node the cars can open onto."""
    lobbyless = FacilitySpec(
        target_area_sqft=20_000,
        zones=(ZoneQuota(zone_type=ZoneType.MED_SURG, bays=1),),
        imaging_suites=0,
        lab_stations=0,
        triage_rooms=0,
    )
    # The generator always lays a station, so this spec *does* have a lobby -- the check
    # is that `_lobby` finds it rather than depending on entrances an upper floor lacks.
    layout = generate_hospital(
        HospitalSpec(
            floors=(
                FloorSpec(name="ed", facility=_ed()),
                FloorSpec(name="ward", facility=lobbyless),
            )
        )
    )
    upstairs = next(bay for bay in layout.bays if bay.id.root.startswith("f01_"))
    layout.graph.dijkstra(layout.entrances[0], upstairs.node)


def test_the_shafts_are_the_only_nodes_shared_between_floors() -> None:
    """Namespacing check from the other side: no floor's nodes leak into another's."""
    layout = generate_hospital(_tower())
    shafts = set(layout.elevators)
    by_floor: dict[str, set[NodeId]] = {}
    for node in layout.graph.nodes:
        if node.id in shafts:
            continue
        prefix = node.id.root.split("_")[0] if node.id.root.startswith("f") else "ground"
        by_floor.setdefault(prefix, set()).add(node.id)
    assert len(by_floor) == 3
    seen: set[NodeId] = set()
    for nodes in by_floor.values():
        assert not (seen & nodes)
        seen |= nodes
