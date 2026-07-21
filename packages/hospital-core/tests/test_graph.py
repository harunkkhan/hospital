"""Graph: deterministic Dijkstra, path integrity, masking, closed nodes."""

from __future__ import annotations

import pytest
from _fixtures import diamond_graph, edge, node
from hypothesis import given
from hypothesis import strategies as st

from hospital.core import LayoutError, NodeId, RouteEdge, RouteGraph


def _edge_seconds_lookup(graph: RouteGraph) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for e in graph.edges:
        lookup[(e.a.root, e.b.root)] = e.seconds.root
        if e.bidirectional:
            lookup[(e.b.root, e.a.root)] = e.seconds.root
    return lookup


@st.composite
def connected_graph(draw: st.DrawFn) -> tuple[RouteGraph, NodeId, NodeId]:
    """A random *connected* graph (spanning tree + extra edges) plus a src/dst pair."""
    k = draw(st.integers(min_value=2, max_value=7))
    names = [f"n{i}" for i in range(k)]
    seen: set[tuple[str, str]] = set()
    edges: list[RouteEdge] = []

    def add(a: str, b: str, dist_cm: int) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        edges.append(edge(a, b, dist_cm))

    for i in range(1, k):
        j = draw(st.integers(min_value=0, max_value=i - 1))
        add(names[i], names[j], draw(st.integers(min_value=1, max_value=500)))
    for _ in range(draw(st.integers(min_value=0, max_value=k))):
        add(
            draw(st.sampled_from(names)),
            draw(st.sampled_from(names)),
            draw(st.integers(min_value=1, max_value=500)),
        )

    graph = RouteGraph(nodes=tuple(node(n) for n in names), edges=tuple(edges))
    return graph, NodeId(draw(st.sampled_from(names))), NodeId(draw(st.sampled_from(names)))


@given(connected_graph())
def test_path_total_equals_sum_of_traversed_edge_seconds(
    gd: tuple[RouteGraph, NodeId, NodeId],
) -> None:
    graph, src, dst = gd
    path = graph.dijkstra(src, dst)  # connected graph => always reachable
    lookup = _edge_seconds_lookup(graph)
    total = sum(lookup[(a.root, b.root)] for a, b in zip(path.nodes, path.nodes[1:], strict=False))
    assert path.total.root == total
    assert path.nodes[0] == src
    assert path.nodes[-1] == dst


@given(connected_graph())
def test_masking_an_edge_never_shortens(gd: tuple[RouteGraph, NodeId, NodeId]) -> None:
    graph, src, dst = gd
    path = graph.dijkstra(src, dst)
    if len(path.nodes) < 2:
        return
    blocked = frozenset({(path.nodes[0], path.nodes[1])})
    try:
        masked = graph.dijkstra(src, dst, blocked_edges=blocked)
    except LayoutError:
        return  # masking disconnected src/dst: "infinitely longer", still not shorter
    assert masked.total.root >= path.total.root


@given(connected_graph())
def test_closed_nodes_are_never_traversed(gd: tuple[RouteGraph, NodeId, NodeId]) -> None:
    graph, src, dst = gd
    if src == dst:
        return
    intermediates = graph.dijkstra(src, dst).nodes[1:-1]
    if not intermediates:
        return
    closed = frozenset({intermediates[0]})
    try:
        rerouted = graph.dijkstra(src, dst, closed_nodes=closed)
    except LayoutError:
        return
    assert intermediates[0] not in rerouted.nodes


def test_dijkstra_is_deterministic_with_total_order_tiebreak() -> None:
    graph = diamond_graph()
    src, dst = NodeId("src"), NodeId("dst")
    path = graph.dijkstra(src, dst)
    # Two equal-cost routes; tie-break (seconds, distance, node_id) picks "b" < "c".
    assert path.nodes == (NodeId("src"), NodeId("b"), NodeId("dst"))
    assert path.total.root == 20_000_000  # 2000 cm at 100 cm/s = 20 s
    for _ in range(5):
        again = graph.dijkstra(src, dst)
        assert again.nodes == path.nodes
        assert again.total == path.total


def test_masking_forces_the_longer_detour() -> None:
    graph = diamond_graph()
    src, dst = NodeId("src"), NodeId("dst")
    # Block both short branches (bidirectional entries close both orientations).
    blocked = frozenset({(NodeId("src"), NodeId("b")), (NodeId("src"), NodeId("c"))})
    detour = graph.dijkstra(src, dst, blocked_edges=blocked)
    assert detour.nodes == (NodeId("src"), NodeId("shortcut_mid"), NodeId("dst"))
    assert detour.total.root == 40_000_000  # 4000 cm > 2000 cm


def test_same_src_and_dst_is_zero() -> None:
    graph = diamond_graph()
    path = graph.dijkstra(NodeId("src"), NodeId("src"))
    assert path.nodes == (NodeId("src"),)
    assert path.total.root == 0


def test_closed_src_or_dst_raises() -> None:
    graph = diamond_graph()
    with pytest.raises(LayoutError):
        graph.dijkstra(NodeId("src"), NodeId("dst"), closed_nodes=frozenset({NodeId("src")}))
    with pytest.raises(LayoutError):
        graph.dijkstra(NodeId("src"), NodeId("dst"), closed_nodes=frozenset({NodeId("dst")}))


def test_unknown_node_raises() -> None:
    graph = diamond_graph()
    with pytest.raises(LayoutError):
        graph.dijkstra(NodeId("nope"), NodeId("dst"))


def test_unreachable_raises() -> None:
    # Two disconnected components.
    graph = RouteGraph(
        nodes=(node("a"), node("b"), node("x"), node("y")),
        edges=(edge("a", "b", 100), edge("x", "y", 100)),
    )
    with pytest.raises(LayoutError):
        graph.dijkstra(NodeId("a"), NodeId("x"))


def test_neighbors_sorted_by_node_id() -> None:
    graph = diamond_graph()
    names = [n.root for n, _ in graph.neighbors(NodeId("dst"))]
    assert names == sorted(names)
