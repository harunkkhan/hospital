"""``generate_floor`` — the deterministic ER floor geometry.

Turns a :class:`~hospital.data.scenario.FacilitySpec` into a concrete
:class:`~hospital.core.FloorLayout` wrapping a :class:`~hospital.core.RouteGraph`.

**Randomness-free** (doc 02 §2.2): this module draws no ``RandomStreams`` and is
pure integer-centimetre arithmetic — the same ``FacilitySpec`` always yields the
byte-identical ``FloorLayout``. Node/edge ordering is canonical (sorted by id)
so the serialized graph is byte-stable across machines. Every edge's ``seconds``
is *derived* from its Manhattan ``distance`` via
:func:`hospital.core.walk_duration` — never entered independently — and no two
adjacent nodes ever share coordinates, so no edge is zero-length. Connectivity
is verified (not assumed) by calling :meth:`~hospital.core.RouteGraph.dijkstra`
from the walk-in entrance to every other node; an unreachable node raises
:class:`~hospital.core.LayoutError` at construction, per the core registry —
this module builds the graph and calls ``dijkstra`` only to check it, never to
implement pathfinding of its own (anti-duplication rules 2/3).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from hospital.core import (
    Bay,
    BayId,
    Distance,
    FloorLayout,
    LayoutError,
    NodeId,
    RouteEdge,
    RouteGraph,
    RouteNode,
    WalkSpeed,
    Zone,
    ZoneId,
    ZoneType,
    walk_duration,
)
from hospital.data.scenario import FacilitySpec, ZoneQuota

_SQFT_TO_CM2 = 929.0304
_STATION_STUB_DIVISOR = 3  # station stub length = room_depth_cm // this (kept < room_depth_cm)
_CONNECTOR_DEPTH_MULT = 2  # imaging/lab connector sits this many room-depths off the spine
_IMAGING_LAB_ROW_MULT = 3  # imaging/lab node row sits this many room-depths off the spine


@dataclass(frozen=True)
class _Footprint:
    width_cm: int
    height_cm: int
    spine_y: int
    x0: int
    x1: int


@dataclass(frozen=True)
class _BayDraft:
    id: BayId
    zone_type: ZoneType
    node: RouteNode
    col: int
    isolation_capable: bool
    equipment: frozenset[str]


@dataclass(frozen=True)
class _ZoneBuild:
    zone: Zone
    bays: tuple[_BayDraft, ...]
    col_start: int
    col_count: int
    max_bays_per_station: int


@dataclass(frozen=True)
class _StationBuild:
    node: RouteNode
    anchor_col: int


@dataclass(frozen=True)
class _Specials:
    waiting_room: RouteNode
    entrance_walk_in: RouteNode
    entrance_ambulance: RouteNode
    triage: tuple[RouteNode, ...]
    connector: RouteNode | None
    imaging: tuple[RouteNode, ...]
    lab: tuple[RouteNode, ...]


def _footprint(facility: FacilitySpec) -> _Footprint:
    """Integer footprint ``(W, H, spine_y, x0, x1)`` — fixed round-half-to-even."""
    area_cm2 = facility.target_area_sqft * _SQFT_TO_CM2
    width = round(math.sqrt(area_cm2 * facility.aspect_ratio))
    height = round(area_cm2 / width)
    spine_y = height // 2
    x0 = facility.corridor_margin_cm
    x1 = width - facility.corridor_margin_cm
    if x1 <= x0:
        raise LayoutError("facility footprint is too small for the corridor margin")
    return _Footprint(width_cm=width, height_cm=height, spine_y=spine_y, x0=x0, x1=x1)


def _lay_spine(total_columns: int, fp: _Footprint) -> tuple[RouteNode, ...]:
    """Junction nodes ``corr_000..corr_{M-1}`` at a fixed, derived pitch."""
    pitch = (fp.x1 - fp.x0) // total_columns
    if pitch <= 0:
        raise LayoutError("derived corridor pitch is non-positive for this facility spec")
    return tuple(
        RouteNode(
            id=NodeId(f"corr_{i:03d}"), label="corridor", x_cm=fp.x0 + i * pitch, y_cm=fp.spine_y
        )
        for i in range(total_columns)
    )


def _place_zone_bays(
    zones: tuple[ZoneQuota, ...],
    column_counts: list[int],
    spine: tuple[RouteNode, ...],
    room_depth_cm: int,
) -> tuple[_ZoneBuild, ...]:
    """Walk ``zones`` west->east, alternating north/south bays one per column."""
    builds: list[_ZoneBuild] = []
    cursor = 0
    for i, (quota, block_size) in enumerate(zip(zones, column_counts, strict=True)):
        zone_id = ZoneId(f"{quota.zone_type.value}_{i:02d}")
        drafts: list[_BayDraft] = []
        bay_k = 0
        for j in range(block_size):
            junction = spine[cursor + j]
            for side_sign in (-1, 1):  # north first, then south
                bay_index = 2 * j + (0 if side_sign < 0 else 1)
                if bay_index >= quota.bays:
                    continue
                bay_id = BayId(f"bay_{zone_id.root}_{bay_k:02d}")
                node = RouteNode(
                    id=NodeId(bay_id.root),
                    label=quota.zone_type.value,
                    x_cm=junction.x_cm,
                    y_cm=junction.y_cm + side_sign * room_depth_cm,
                )
                drafts.append(
                    _BayDraft(
                        id=bay_id,
                        zone_type=quota.zone_type,
                        node=node,
                        col=cursor + j,
                        isolation_capable=bay_k < quota.isolation_bays,
                        equipment=quota.equipment,
                    )
                )
                bay_k += 1
        builds.append(
            _ZoneBuild(
                zone=Zone(id=zone_id, zone_type=quota.zone_type, capacity=quota.bays),
                bays=tuple(drafts),
                col_start=cursor,
                col_count=block_size,
                max_bays_per_station=quota.max_bays_per_station,
            )
        )
        cursor += block_size
    return tuple(builds)


def _place_stations(
    builds: tuple[_ZoneBuild, ...], spine: tuple[RouteNode, ...], room_depth_cm: int
) -> tuple[tuple[_StationBuild, ...], dict[BayId, NodeId]]:
    """One station per zone (split when ``bays > max_bays_per_station``), nearest-in-zone."""
    stub = max(1, room_depth_cm // _STATION_STUB_DIVISOR)
    stations: list[_StationBuild] = []
    serving: dict[BayId, NodeId] = {}
    for build in builds:
        if not build.bays:
            continue
        # Capped at col_count: a station is anchored to a distinct column, and asking
        # for more stations than columns would force two stations onto one column
        # (identical coordinates). With num_stations <= col_count this integer
        # bucketing is strictly increasing, so every station gets a distinct column.
        num_stations = min(
            build.col_count, max(1, math.ceil(len(build.bays) / build.max_bays_per_station))
        )
        cols = [
            build.col_start + (s * build.col_count) // num_stations for s in range(num_stations)
        ]
        zone_stations: list[_StationBuild] = []
        for s, col in enumerate(cols):
            junction = spine[col]
            station_node = RouteNode(
                id=NodeId(f"station_{build.zone.id.root}_{s:02d}"),
                label="station",
                x_cm=junction.x_cm,
                y_cm=junction.y_cm + stub,
            )
            zone_stations.append(_StationBuild(node=station_node, anchor_col=col))
        stations.extend(zone_stations)
        for draft in build.bays:
            nearest = min(range(len(cols)), key=lambda idx: abs(cols[idx] - draft.col))
            serving[draft.id] = zone_stations[nearest].node.id
    return tuple(stations), serving


def _place_specials(
    facility: FacilitySpec, spine: tuple[RouteNode, ...], fp: _Footprint
) -> _Specials:
    """Waiting room / entrances / triage rooms / imaging+lab off a north connector.

    All specials get their own coordinate *layer*, distinct from the spine
    (``y=spine_y``) and the bay layers (``y=spine_y +/- room_depth_cm``), so
    none can ever coincide with a bay or junction node.
    """
    room_depth = facility.room_depth_cm
    entrance_walk_in = RouteNode(
        id=NodeId("entrance_walk_in"), label="entrance", x_cm=0, y_cm=fp.spine_y
    )
    waiting_room = RouteNode(
        id=NodeId("waiting_room"), label="waiting", x_cm=0, y_cm=fp.spine_y - room_depth
    )
    entrance_ambulance = RouteNode(
        id=NodeId("entrance_ambulance"), label="entrance", x_cm=fp.width_cm, y_cm=fp.spine_y
    )
    # Triage rooms cluster at x=0 (the walk-in wall), stacked south of the entrance so
    # they never collide with the waiting room (which sits north of it).
    triage = tuple(
        RouteNode(
            id=NodeId(f"triage_{i:02d}"),
            label="triage",
            x_cm=0,
            y_cm=fp.spine_y + room_depth * (i + 1),
        )
        for i in range(facility.triage_rooms)
    )

    connector: RouteNode | None = None
    imaging: tuple[RouteNode, ...] = ()
    lab: tuple[RouteNode, ...] = ()
    if facility.imaging_suites > 0 or facility.lab_stations > 0:
        mid = spine[len(spine) // 2]
        connector = RouteNode(
            id=NodeId("connector_north"),
            label="connector",
            x_cm=mid.x_cm,
            y_cm=mid.y_cm - _CONNECTOR_DEPTH_MULT * room_depth,
        )
        row_y = mid.y_cm - _IMAGING_LAB_ROW_MULT * room_depth
        imaging = tuple(
            RouteNode(
                id=NodeId(f"imaging_{i:02d}"),
                label=ZoneType.IMAGING.value,
                x_cm=connector.x_cm + i * room_depth,
                y_cm=row_y,
            )
            for i in range(facility.imaging_suites)
        )
        lab = tuple(
            RouteNode(
                id=NodeId(f"lab_{j:02d}"),
                label=ZoneType.LAB.value,
                x_cm=connector.x_cm + (facility.imaging_suites + j) * room_depth,
                y_cm=row_y,
            )
            for j in range(facility.lab_stations)
        )
    return _Specials(
        waiting_room=waiting_room,
        entrance_walk_in=entrance_walk_in,
        entrance_ambulance=entrance_ambulance,
        triage=triage,
        connector=connector,
        imaging=imaging,
        lab=lab,
    )


def _edge(a: RouteNode, b: RouteNode, speed: WalkSpeed) -> RouteEdge:
    """A bidirectional Manhattan-distance edge; raises on a zero-length span."""
    dist = abs(a.x_cm - b.x_cm) + abs(a.y_cm - b.y_cm)
    if dist <= 0:
        raise LayoutError(f"zero-length edge between {a.id.root} and {b.id.root}")
    d = Distance(dist)
    return RouteEdge(
        a=a.id, b=b.id, distance=d, seconds=walk_duration(d, speed), bidirectional=True
    )


def _build_edges(
    spine: tuple[RouteNode, ...],
    builds: tuple[_ZoneBuild, ...],
    stations: tuple[_StationBuild, ...],
    specials: _Specials,
    speed: WalkSpeed,
) -> tuple[list[RouteNode], list[RouteEdge]]:
    nodes: list[RouteNode] = list(spine)
    edges: list[RouteEdge] = [_edge(a, b, speed) for a, b in itertools.pairwise(spine)]

    for build in builds:
        for draft in build.bays:
            nodes.append(draft.node)
            edges.append(_edge(draft.node, spine[draft.col], speed))

    for station in stations:
        nodes.append(station.node)
        edges.append(_edge(station.node, spine[station.anchor_col], speed))

    nodes.extend([specials.entrance_walk_in, specials.waiting_room, specials.entrance_ambulance])
    edges.append(_edge(specials.entrance_walk_in, spine[0], speed))
    edges.append(_edge(specials.waiting_room, specials.entrance_walk_in, speed))
    edges.append(_edge(specials.entrance_ambulance, spine[-1], speed))

    for triage_node in specials.triage:
        nodes.append(triage_node)
        edges.append(_edge(triage_node, specials.entrance_walk_in, speed))

    if specials.connector is not None:
        mid = spine[len(spine) // 2]
        nodes.append(specials.connector)
        edges.append(_edge(specials.connector, mid, speed))
        for node in (*specials.imaging, *specials.lab):
            nodes.append(node)
            edges.append(_edge(node, specials.connector, speed))

    return nodes, edges


def _assert_connected(graph: RouteGraph, probe: NodeId) -> None:
    """Every node must be reachable from ``probe`` (construction-time check, not a sim concern)."""
    for node in graph.nodes:
        try:
            graph.dijkstra(probe, node.id)
        except LayoutError as exc:
            raise LayoutError(
                f"floor graph is disconnected: {node.id.root} is unreachable from {probe.root}"
            ) from exc


def generate_floor(facility: FacilitySpec) -> FloorLayout:
    """Build the ER floor's :class:`~hospital.core.FloorLayout` (+ ``RouteGraph``).

    Pure integer geometry — draws no ``RandomStreams``. Raises
    :class:`~hospital.core.LayoutError` if ``facility`` yields a disconnected or
    degenerate (non-positive pitch/footprint) graph.
    """
    fp = _footprint(facility)
    column_counts = [math.ceil(q.bays / 2) for q in facility.zones]
    total_columns = sum(column_counts)
    if total_columns < 1:
        raise LayoutError("facility.zones allocate no bays; cannot lay a spine")

    spine = _lay_spine(total_columns, fp)
    speed = WalkSpeed(facility.walk_speed_cm_s)

    builds = _place_zone_bays(facility.zones, column_counts, spine, facility.room_depth_cm)
    stations, serving = _place_stations(builds, spine, facility.room_depth_cm)
    specials = _place_specials(facility, spine, fp)
    nodes, edges = _build_edges(spine, builds, stations, specials, speed)

    bays = tuple(
        Bay(
            id=draft.id,
            zone=build.zone.id,
            zone_type=draft.zone_type,
            node=draft.node.id,
            serving_station=serving[draft.id],
            isolation_capable=draft.isolation_capable,
            equipment=draft.equipment,
        )
        for build in builds
        for draft in build.bays
    )
    zones = (
        *(build.zone for build in builds),
        Zone(id=ZoneId("triage"), zone_type=ZoneType.TRIAGE, capacity=facility.triage_rooms),
        Zone(id=ZoneId("imaging"), zone_type=ZoneType.IMAGING, capacity=facility.imaging_suites),
        Zone(id=ZoneId("lab"), zone_type=ZoneType.LAB, capacity=facility.lab_stations),
    )

    graph = RouteGraph(
        nodes=tuple(sorted(nodes, key=lambda n: n.id.root)),
        edges=tuple(sorted(edges, key=lambda e: (e.a.root, e.b.root))),
    )
    layout = FloorLayout(
        graph=graph,
        zones=zones,
        bays=bays,
        stations=tuple(s.node.id for s in stations),
        entrances=(specials.entrance_walk_in.id, specials.entrance_ambulance.id),
        imaging_nodes=tuple(n.id for n in specials.imaging),
        lab_nodes=tuple(n.id for n in specials.lab),
    )
    _assert_connected(layout.graph, specials.entrance_walk_in.id)
    return layout


__all__ = ["generate_floor"]
