"""oracle: distances equal core.dijkstra; memo is transparent; masks respected."""

from __future__ import annotations

import itertools

import pytest
from _solver_fixtures import tiny_graph, tiny_layout

from hospital.core import LayoutError, NodeId
from hospital.solver.oracle import EMPTY_MASK, GraphRoutingOracle, RouteMask


def _all_node_ids() -> list[NodeId]:
    return [n.id for n in tiny_graph().nodes]


def test_distance_equals_core_dijkstra_total() -> None:
    graph = tiny_graph()
    oracle = GraphRoutingOracle(graph)
    for src, dst in itertools.permutations(_all_node_ids(), 2):
        assert oracle.distance(src, dst) == graph.dijkstra(src, dst).total


def test_path_total_equals_distance_shared_memo() -> None:
    graph = tiny_graph()
    oracle = GraphRoutingOracle(graph)
    src, dst = NodeId("gstat"), NodeId("lab")
    assert oracle.path(src, dst).total == oracle.distance(src, dst)


def test_memo_hit_miss_accounting() -> None:
    graph = tiny_graph()
    oracle = GraphRoutingOracle(graph)
    src, dst = NodeId("gstat"), NodeId("b1")
    oracle.distance(src, dst)  # miss (computes path)
    first = oracle.stats()
    assert first.misses == 1 and first.hits == 0 and first.size == 1
    oracle.distance(src, dst)  # hit (distance reuses path memo)
    oracle.path(src, dst)  # hit
    second = oracle.stats()
    assert second.misses == 1 and second.hits == 2 and second.size == 1


def test_memo_returns_identical_object() -> None:
    oracle = GraphRoutingOracle(tiny_graph())
    src, dst = NodeId("gstat"), NodeId("b3")
    a = oracle.path(src, dst)
    b = oracle.path(src, dst)
    assert a is b  # cached object, byte-identical run to run


def test_eviction_when_cache_full() -> None:
    graph = tiny_graph()
    oracle = GraphRoutingOracle(graph, cache_size=2)
    pairs = list(itertools.permutations(_all_node_ids(), 2))[:5]
    for src, dst in pairs:
        oracle.distance(src, dst)
    stats = oracle.stats()
    assert stats.size == 2
    assert stats.evictions >= 1


def test_mask_change_recomputes_and_respects_block() -> None:
    graph = tiny_graph()
    oracle = GraphRoutingOracle(graph)
    src, dst = NodeId("gstat"), NodeId("lab")
    unmasked = oracle.distance(src, dst)
    # Block the only path to lab (via img): now unreachable -> LayoutError surfaces.
    mask = RouteMask(blocked_edges=frozenset({(NodeId("gstat"), NodeId("img"))}))
    with pytest.raises(LayoutError):
        oracle.distance(src, dst, mask=mask)
    # The distinct mask fingerprint produced a fresh lookup (a new miss).
    assert oracle.stats().misses == 2
    # The unmasked entry is untouched and still cached.
    assert oracle.distance(src, dst) == unmasked


def test_invalidate_purges_only_that_mask() -> None:
    oracle = GraphRoutingOracle(tiny_graph())
    src, dst = NodeId("gstat"), NodeId("b1")
    mask = RouteMask(closed_nodes=frozenset({NodeId("b4")}))
    oracle.distance(src, dst)  # EMPTY_MASK entry
    oracle.distance(src, dst, mask=mask)  # masked entry
    assert oracle.stats().size == 2
    oracle.invalidate(mask)
    assert oracle.stats().size == 1
    # The empty-mask entry survives.
    oracle.distance(src, dst)
    assert oracle.stats().hits >= 1


def test_fingerprint_stable_and_empty_mask_singleton() -> None:
    m1 = RouteMask(closed_nodes=frozenset({NodeId("a"), NodeId("b")}))
    m2 = RouteMask(closed_nodes=frozenset({NodeId("b"), NodeId("a")}))
    assert m1.fingerprint() == m2.fingerprint()
    assert EMPTY_MASK.fingerprint() == RouteMask().fingerprint()
    assert m1.fingerprint() != EMPTY_MASK.fingerprint()


def test_layout_graph_matches_oracle() -> None:
    layout = tiny_layout()
    oracle = GraphRoutingOracle(layout.graph)
    assert (
        oracle.distance(NodeId("gstat"), NodeId("img"))
        == layout.graph.dijkstra(NodeId("gstat"), NodeId("img")).total
    )
