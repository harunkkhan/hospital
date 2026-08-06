"""``generate_hospital`` — many floors joined by elevators (doc 02, M4).

A hospital is **the same** :class:`~hospital.core.FloorLayout` an ED-only scenario uses,
with a graph that spans floors. That is the whole design, and it is why nothing
downstream needed to change to gain vertical movement: an elevator is a
:class:`~hospital.core.RouteEdge` whose ``seconds`` greatly exceed its ``distance``, which
:mod:`hospital.core.graph` was written to allow, and the routing oracle already minimizes
seconds over whatever graph it is handed. Placement, dispatch, and physics see one
building.

Each floor is produced by the *unmodified* :func:`~hospital.data.layout.generate_floor`
and then re-namespaced. Two reasons for composing rather than generalizing that function:

* its geometry is pinned by committed goldens, so a single-floor hospital must still
  produce byte-identical ids — and :func:`generate_hospital` over one floor does, because
  floor 0 takes the empty prefix;
* floor geometry and floor *stacking* are separate concerns, and keeping them in separate
  functions means a change to bay pitch cannot quietly move an elevator.

**Randomness-free**, like the floor generator: pure integer arithmetic over a spec, with
canonical (sorted) node and edge ordering, so the same :class:`HospitalSpec` always yields
the byte-identical layout.

Elevators are the only vertical path, deliberately: stairs would be a second one with
different physics (no dwell, no capacity, refused by non-ambulatory patients), and
modelling one route honestly beats modelling two badly. :func:`generate_hospital` verifies
that claim rather than asserting it — see
:func:`~hospital.data.tests.test_hospital` for the cut-the-elevators probe.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from hospital.core import (
    BayId,
    Distance,
    Duration,
    FloorLayout,
    FrozenModel,
    LayoutError,
    NodeId,
    RouteEdge,
    RouteGraph,
    RouteNode,
    WalkSpeed,
    ZoneId,
    seconds,
    walk_duration,
)
from hospital.data.layout import generate_floor
from hospital.data.scenario import FacilitySpec

if TYPE_CHECKING:
    from collections.abc import Iterable

# Elevator geometry. The shaft is a fixed footprint at the origin end of each floor's
# spine, so a floor's own geometry never has to make room for it.
_SHAFT_OFFSET_CM = 300
_SHAFT_PITCH_CM = 400
# A shaft edge's *distance* is the car's actual travel; its `seconds` are set from the
# spec, not derived from that distance. This is the one place the two are decoupled.
_FLOOR_HEIGHT_CM = 400


class FloorSpec(FrozenModel):
    """One floor: a name, and the geometry spec that fills it."""

    name: str = Field(min_length=1)
    facility: FacilitySpec


class HospitalSpec(FrozenModel):
    """A stack of floors and the elevators joining them.

    ``floors[0]`` is the ground floor and the only one with entrances: ambulances and
    walk-ins arrive at the emergency department, and everything above is reached through
    the shafts.
    """

    floors: tuple[FloorSpec, ...] = Field(min_length=1)
    elevator_shafts: int = Field(default=2, ge=1)
    # Time for the car to move one floor, and the fixed cost of a boarding/exit cycle.
    # `dwell` is charged once per shaft edge traversed, which is what makes a two-floor
    # trip cheaper than two one-floor trips through a lobby.
    seconds_per_floor: float = Field(default=12.0, gt=0.0, allow_inf_nan=False)
    dwell_seconds: float = Field(default=20.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _floor_names_are_unique(self) -> HospitalSpec:
        names = [floor.name for floor in self.floors]
        if len(names) != len(set(names)):
            raise ValueError("hospital.floors have duplicate names")
        return self


def _prefix(floor: int) -> str:
    """Floor 0 takes the empty prefix, so a one-floor hospital keeps the ER's own ids."""
    return "" if floor == 0 else f"f{floor:02d}_"


def _node_id(prefix: str, node: NodeId) -> NodeId:
    return NodeId(f"{prefix}{node.root}")


def _shaft_node(floor: int, shaft: int, y_cm: int) -> RouteNode:
    return RouteNode(
        id=NodeId(f"elev_{shaft:02d}_f{floor:02d}"),
        label=f"elevator {shaft} floor {floor}",
        x_cm=-_SHAFT_OFFSET_CM - shaft * _SHAFT_PITCH_CM,
        # Viz-only, and the one place a floor becomes a coordinate: stacking the shafts
        # vertically is what makes a rendered building look like a building.
        y_cm=y_cm + floor * _FLOOR_HEIGHT_CM,
    )


def _renamed(layout: FloorLayout, floor: int) -> FloorLayout:
    """One floor's layout with every id namespaced to it.

    Ids are unique only *within* a generated floor, so stacking two floors without this
    would silently merge their bays — two `bay_gen_00`s becoming one node with edges to
    both floors, and a patient able to walk between storeys without an elevator.
    """
    prefix = _prefix(floor)
    if not prefix:
        return layout

    graph = RouteGraph(
        nodes=tuple(
            node.model_copy(update={"id": _node_id(prefix, node.id)}) for node in layout.graph.nodes
        ),
        edges=tuple(
            edge.model_copy(update={"a": _node_id(prefix, edge.a), "b": _node_id(prefix, edge.b)})
            for edge in layout.graph.edges
        ),
    )
    return FloorLayout(
        graph=graph,
        zones=tuple(
            zone.model_copy(update={"id": ZoneId(f"{prefix}{zone.id.root}"), "floor": floor})
            for zone in layout.zones
        ),
        bays=tuple(
            bay.model_copy(
                update={
                    "id": BayId(f"{prefix}{bay.id.root}"),
                    "zone": ZoneId(f"{prefix}{bay.zone.root}"),
                    "node": _node_id(prefix, bay.node),
                    "serving_station": _node_id(prefix, bay.serving_station),
                }
            )
            for bay in layout.bays
        ),
        stations=tuple(_node_id(prefix, node) for node in layout.stations),
        entrances=tuple(_node_id(prefix, node) for node in layout.entrances),
        imaging_nodes=tuple(_node_id(prefix, node) for node in layout.imaging_nodes),
        lab_nodes=tuple(_node_id(prefix, node) for node in layout.lab_nodes),
    )


def _lobby(layout: FloorLayout) -> NodeId:
    """The node a floor's elevators open onto.

    The floor's first entrance where it has one, else its first station: on an upper floor
    with no doors to the street, the nurse station is the arrival end of the corridor.
    """
    if layout.entrances:
        return layout.entrances[0]
    if layout.stations:
        return layout.stations[0]
    raise LayoutError("a floor has neither an entrance nor a station to open onto")


def _shaft_edges(
    spec: HospitalSpec, floors: tuple[FloorLayout, ...], speed: WalkSpeed
) -> tuple[tuple[RouteNode, ...], tuple[RouteEdge, ...], tuple[NodeId, ...]]:
    """Shaft nodes per floor, the walk-in and vertical edges, and the boarding nodes."""
    lobbies = [_lobby(layout) for layout in floors]
    nodes: list[RouteNode] = []
    edges: list[RouteEdge] = []
    boarding: list[NodeId] = []

    per_floor = seconds(spec.seconds_per_floor)
    dwell = seconds(spec.dwell_seconds)
    for shaft in range(spec.elevator_shafts):
        column: list[RouteNode] = []
        for index, layout in enumerate(floors):
            lobby = next(n for n in layout.graph.nodes if n.id == lobbies[index])
            node = _shaft_node(index, shaft, lobby.y_cm)
            column.append(node)
            nodes.append(node)
            boarding.append(node.id)
            # Walking to the car is ordinary movement, so this edge derives its seconds
            # from its distance exactly like a corridor.
            distance = Distance(abs(lobby.x_cm - node.x_cm) + abs(lobby.y_cm - node.y_cm))
            if distance.root <= 0:
                raise LayoutError(f"elevator {shaft} on floor {index} coincides with its lobby")
            edges.append(
                RouteEdge(
                    a=node.id,
                    b=lobby.id,
                    distance=distance,
                    seconds=walk_duration(distance, speed),
                    bidirectional=True,
                )
            )
        for lower, upper in itertools.pairwise(column):
            # The one deliberately non-walking edge in the building: a short distance at
            # a cost that has nothing to do with walking it.
            edges.append(
                RouteEdge(
                    a=lower.id,
                    b=upper.id,
                    distance=Distance(_FLOOR_HEIGHT_CM),
                    seconds=Duration(per_floor.root + dwell.root),
                    bidirectional=True,
                )
            )
    return tuple(nodes), tuple(edges), tuple(boarding)


def _merged_graph(
    floors: Iterable[FloorLayout],
    extra_nodes: tuple[RouteNode, ...],
    extra_edges: tuple[RouteEdge, ...],
) -> RouteGraph:
    nodes = [node for layout in floors for node in layout.graph.nodes]
    edges = [edge for layout in floors for edge in layout.graph.edges]
    return RouteGraph(
        nodes=tuple(sorted([*nodes, *extra_nodes], key=lambda n: n.id.root)),
        edges=tuple(sorted([*edges, *extra_edges], key=lambda e: (e.a.root, e.b.root))),
    )


def generate_hospital(spec: HospitalSpec) -> FloorLayout:
    """Build the whole building as one :class:`~hospital.core.FloorLayout`.

    A single-floor spec returns exactly what :func:`generate_floor` would, ids included —
    floor 0 takes the empty prefix and a lone floor gets no shafts — so the committed
    goldens keep validating rather than needing a re-baseline.

    Raises :class:`~hospital.core.LayoutError` if the stack is disconnected, which is the
    failure this function exists to prevent: a ward nobody can reach is a silently
    unusable half of a hospital.
    """
    floors = tuple(
        _renamed(generate_floor(f.facility), index) for index, f in enumerate(spec.floors)
    )
    if len(floors) == 1:
        return floors[0]

    # Every floor is laid at its own spec's walk speed; the shafts use the ground floor's,
    # since that is the speed of the person walking to the car.
    speed = WalkSpeed(spec.floors[0].facility.walk_speed_cm_s)
    shaft_nodes, shaft_edges, boarding = _shaft_edges(spec, floors, speed)

    ground = floors[0]
    layout = FloorLayout(
        graph=_merged_graph(floors, shaft_nodes, shaft_edges),
        zones=tuple(zone for layout_ in floors for zone in layout_.zones),
        bays=tuple(bay for layout_ in floors for bay in layout_.bays),
        stations=tuple(node for layout_ in floors for node in layout_.stations),
        # Ground floor only: ambulances and walk-ins arrive at the ED, and an upper floor
        # with its own "entrance" would let arrivals teleport past the shafts.
        entrances=ground.entrances,
        imaging_nodes=tuple(node for layout_ in floors for node in layout_.imaging_nodes),
        lab_nodes=tuple(node for layout_ in floors for node in layout_.lab_nodes),
        elevators=tuple(sorted(boarding, key=lambda n: n.root)),
    )
    _assert_reachable(layout)
    return layout


def _assert_reachable(layout: FloorLayout) -> None:
    """Every node reachable from the walk-in entrance — checked, not assumed.

    The failure this exists to catch is a whole floor floating free: a ward nobody can
    reach is a silently unusable half of a hospital, and placement would keep offering
    its bays.
    """
    origin = layout.entrances[0]
    for node in layout.graph.nodes:
        try:
            layout.graph.dijkstra(origin, node.id)
        except LayoutError as exc:
            raise LayoutError(
                f"hospital graph is disconnected: {node.id.root} is unreachable from {origin.root}"
            ) from exc


__all__ = ["FloorSpec", "HospitalSpec", "generate_hospital"]
