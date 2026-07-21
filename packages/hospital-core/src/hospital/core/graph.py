"""The single shortest-path implementation in the repo.

There is exactly **one** Dijkstra here; ``solver.oracle`` (with an LRU memo) and
``sim.physics.world`` (live rerouting around masks) both call it and never
reimplement it.

Determinism is a correctness requirement, not cosmetics (nuance 1.7): the total
order over equal-cost frontier nodes is the tie-break ``(seconds, distance,
node_id)``. Without it, Dijkstra's result would depend on heap/iteration order →
nondeterministic staff routes → nondeterministic ``staff_minutes_walked`` → CRN
comparison becomes meaningless.

Masks:

* ``blocked_edges`` — a set of ordered ``(a, b)`` pairs. Traversal ``u -> v`` is
  blocked if ``(u, v)`` is present, **and** — for a *bidirectional* edge — if
  ``(v, u)`` is present. So a single ``(a, b)`` entry closes a bidirectional
  corridor in *both* directions (you cannot leave one orientation quietly open);
  a one-way edge is only affected in its own direction.
* ``closed_nodes`` — never traversed, even as intermediates. If ``src`` or
  ``dst`` is itself closed, that is a caller error → :class:`LayoutError`.

No memoization lives here: a ``FrozenModel`` cannot hold a cache, and caching
would couple the graph to a call pattern. The oracle layers the LRU.

``x_cm``/``y_cm`` on a node are for viz/interpolation only — **never** used in
pathfinding, which keys entirely on edge ``seconds`` (with ``distance`` as the
secondary tie-break).
"""

from __future__ import annotations

import heapq

from hospital.core.errors import LayoutError
from hospital.core.ids import NodeId
from hospital.core.models import FrozenModel
from hospital.core.time import Duration
from hospital.core.units import Distance


class RouteNode(FrozenModel):
    """A graph vertex. ``x_cm``/``y_cm`` are viz-only, never used in routing."""

    id: NodeId
    label: str
    x_cm: int
    y_cm: int


class RouteEdge(FrozenModel):
    """A weighted edge. ``seconds`` is the routing cost; ``distance`` breaks ties.

    For an ordinary corridor ``seconds`` is derived from ``distance``/speed, but
    the two may be deliberately decoupled (e.g. an elevator: small distance,
    large ``seconds``) — routing always uses ``seconds``.
    """

    a: NodeId
    b: NodeId
    distance: Distance
    seconds: Duration
    bidirectional: bool = True


class RoutePath(FrozenModel):
    """A resolved path: the node sequence and its total traversal time."""

    nodes: tuple[NodeId, ...]
    total: Duration


class RouteGraph(FrozenModel):
    """An immutable weighted graph with a deterministic Dijkstra."""

    nodes: tuple[RouteNode, ...]
    edges: tuple[RouteEdge, ...]

    def _node_ids(self) -> frozenset[NodeId]:
        return frozenset(n.id for n in self.nodes)

    def _adjacency(self) -> dict[NodeId, list[tuple[NodeId, Duration, Distance, bool]]]:
        """Directed adjacency built fresh per call (no caching in the frozen model)."""
        adj: dict[NodeId, list[tuple[NodeId, Duration, Distance, bool]]] = {}
        for e in self.edges:
            adj.setdefault(e.a, []).append((e.b, e.seconds, e.distance, e.bidirectional))
            if e.bidirectional:
                adj.setdefault(e.b, []).append((e.a, e.seconds, e.distance, True))
        return adj

    def neighbors(self, n: NodeId) -> tuple[tuple[NodeId, Duration], ...]:
        """Direct neighbours of ``n`` with edge time, sorted by node id (deterministic)."""
        out = [(v, secs) for (v, secs, _dist, _bi) in self._adjacency().get(n, [])]
        out.sort(key=lambda pair: pair[0].root)
        return tuple(out)

    def dijkstra(
        self,
        src: NodeId,
        dst: NodeId,
        *,
        blocked_edges: frozenset[tuple[NodeId, NodeId]] = frozenset(),
        closed_nodes: frozenset[NodeId] = frozenset(),
    ) -> RoutePath:
        """Shortest path from ``src`` to ``dst`` by edge ``seconds``.

        Deterministic total order ``(cum_seconds, cum_distance, node_id)``.
        Raises :class:`LayoutError` if ``src``/``dst`` is unknown or closed, or
        if no path exists under the given masks.
        """
        node_ids = self._node_ids()
        if src not in node_ids:
            raise LayoutError(f"unknown src node: {src.root}")
        if dst not in node_ids:
            raise LayoutError(f"unknown dst node: {dst.root}")
        if src in closed_nodes:
            raise LayoutError(f"src node is closed: {src.root}")
        if dst in closed_nodes:
            raise LayoutError(f"dst node is closed: {dst.root}")

        if src == dst:
            return RoutePath(nodes=(src,), total=Duration(0))

        adj = self._adjacency()
        id_to_node: dict[str, NodeId] = {n.id.root: n.id for n in self.nodes}

        # best[node] = (cum_seconds, cum_distance) of the best path found so far.
        best: dict[NodeId, tuple[int, int]] = {src: (0, 0)}
        prev: dict[NodeId, NodeId] = {}
        settled: set[NodeId] = set()
        # Heap key is (cum_seconds, cum_distance, node_id_str): a strict total order.
        heap: list[tuple[int, int, str]] = [(0, 0, src.root)]

        while heap:
            cs, cd, u_id = heapq.heappop(heap)
            u = id_to_node[u_id]
            if u in settled:
                continue
            settled.add(u)
            if u == dst:
                break
            for v, secs, dist, bidi in adj.get(u, []):
                if v in closed_nodes or v in settled:
                    continue
                if (u, v) in blocked_edges or (bidi and (v, u) in blocked_edges):
                    continue
                ns = cs + secs.root
                nd = cd + dist.root
                if v not in best or (ns, nd) < best[v]:
                    best[v] = (ns, nd)
                    prev[v] = u
                    heapq.heappush(heap, (ns, nd, v.root))

        if dst not in best or dst not in settled:
            raise LayoutError(f"no path from {src.root} to {dst.root} under current masks")

        path: list[NodeId] = [dst]
        cur = dst
        while cur != src:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return RoutePath(nodes=tuple(path), total=Duration(best[dst][0]))


__all__ = ["RouteEdge", "RouteGraph", "RouteNode", "RoutePath"]
