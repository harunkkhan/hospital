"""``GraphRoutingOracle`` — the memoized front end to the one Dijkstra (doc 03 §4.2).

This is the *single* place ``solver`` touches pathfinding. It **wraps**
:meth:`hospital.core.graph.RouteGraph.dijkstra` behind an LRU memo keyed on
``(src, dst, mask.fingerprint())``; reimplementing Dijkstra here is forbidden
(doc 00 §5 rule 3). ``distance()`` is defined in terms of the memoized
``path()`` so both share one memo and never double-compute.

Determinism is *inherited*, not added: ``core.graph.dijkstra`` owns the
total-order tie-break ``(seconds, distance, node_id)``; the oracle only decides
*whether* a path is recomputed, never *which* path is returned.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Final

from hospital.core import Duration, FrozenModel, NodeId, RouteGraph, RoutePath


class RouteMask(FrozenModel):
    """Blocked edges / closed nodes for a live reroute; content-addressed key."""

    blocked_edges: frozenset[tuple[NodeId, NodeId]] = frozenset()
    closed_nodes: frozenset[NodeId] = frozenset()

    def fingerprint(self) -> str:
        """Stable ``blake2b`` over the *sorted* members — a reproducible memo key.

        A set's iteration order is not byte-stable across runs, so the mask is
        sorted-then-hashed; ``EMPTY_MASK`` collapses to one fingerprint and hits
        a hot cache, while each distinct live mask gets its own key space.
        """
        edges = sorted((a.root, b.root) for (a, b) in self.blocked_edges)
        nodes = sorted(n.root for n in self.closed_nodes)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(repr(edges).encode("utf-8"))
        digest.update(b"|")
        digest.update(repr(nodes).encode("utf-8"))
        return digest.hexdigest()


EMPTY_MASK: Final[RouteMask] = RouteMask()


class OracleStats(FrozenModel):
    """Observable memo accounting — tune ``cache_size`` to the floor, don't guess."""

    hits: int
    misses: int
    evictions: int
    size: int


class GraphRoutingOracle:
    """An LRU memo over ``core.graph.dijkstra`` (implements ``RoutingOracle``)."""

    def __init__(self, graph: RouteGraph, *, cache_size: int = 65_536) -> None:
        self._graph = graph
        self._cache_size = cache_size
        self._memo: OrderedDict[tuple[str, str, str], RoutePath] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def path(self, src: NodeId, dst: NodeId, *, mask: RouteMask = EMPTY_MASK) -> RoutePath:
        """Shortest path ``src -> dst`` — the primitive; the ONLY dijkstra call site."""
        key = (src.root, dst.root, mask.fingerprint())
        cached = self._memo.get(key)
        if cached is not None:
            self._hits += 1
            self._memo.move_to_end(key)
            return cached
        self._misses += 1
        route = self._graph.dijkstra(
            src, dst, blocked_edges=mask.blocked_edges, closed_nodes=mask.closed_nodes
        )
        self._memo[key] = route
        if len(self._memo) > self._cache_size:
            self._memo.popitem(last=False)
            self._evictions += 1
        return route

    def distance(self, src: NodeId, dst: NodeId, *, mask: RouteMask = EMPTY_MASK) -> Duration:
        """Total traversal time — shares ``path()``'s memo, so no double compute."""
        return self.path(src, dst, mask=mask).total

    def invalidate(self, mask: RouteMask) -> None:
        """Drop memo entries for ``mask`` (eager purge when the world mutates edges)."""
        fingerprint = mask.fingerprint()
        stale = [key for key in self._memo if key[2] == fingerprint]
        for key in stale:
            del self._memo[key]

    def stats(self) -> OracleStats:
        """A snapshot of hit/miss/eviction accounting and current memo size."""
        return OracleStats(
            hits=self._hits, misses=self._misses, evictions=self._evictions, size=len(self._memo)
        )


__all__ = ["EMPTY_MASK", "GraphRoutingOracle", "OracleStats", "RouteMask"]
