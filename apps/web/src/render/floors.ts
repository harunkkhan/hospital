/**
 * Slicing a multi-floor hospital into one renderable storey.
 *
 * A building's floors share a footprint, so their `x_cm`/`y_cm` genuinely overlap — the
 * domain is not wrong about that, and offsetting them there would be a lie about the
 * geometry. Separation is a *rendering* decision, which is what these functions are:
 * pick a floor, keep what is on it, and let the one projection aspect-fit that.
 *
 * A single-floor scenario has exactly one floor and slices to itself, so every ED-only
 * run renders precisely as it did before floors existed.
 */

import type { Bay, FloorLayout, NodeId, RouteNode, Zone } from "../api/types";

/** Every storey present, ascending. `[0]` for a single-floor ED. */
export function floorsOf(layout: FloorLayout): readonly number[] {
  const seen = new Set<number>();
  for (const node of layout.graph.nodes) {
    seen.add(node.floor);
  }
  if (seen.size === 0) {
    seen.add(0);
  }
  return [...seen].sort((a, b) => a - b);
}

/**
 * The layout restricted to one storey.
 *
 * Edges are kept only when **both** endpoints are on the floor, which deliberately drops
 * the elevator shafts: a vertical edge drawn on a single-storey plan would be a line to
 * nowhere, since its far end is not on the canvas. The shaft's own boarding node stays —
 * it is on this floor and staff really do walk to it — so the route out of the storey is
 * still visible as a place, just not as a line.
 */
export function sliceToFloor(layout: FloorLayout, floor: number): FloorLayout {
  const nodes: RouteNode[] = layout.graph.nodes.filter((n) => n.floor === floor);
  const ids = new Set<NodeId>(nodes.map((n) => n.id));
  const zones: Zone[] = layout.zones.filter((z) => z.floor === floor);
  const zoneIds = new Set(zones.map((z) => z.id));
  const bays: Bay[] = layout.bays.filter((b) => zoneIds.has(b.zone));
  return {
    ...layout,
    graph: {
      nodes,
      edges: layout.graph.edges.filter((e) => ids.has(e.a) && ids.has(e.b)),
    },
    zones,
    bays,
    stations: layout.stations.filter((n) => ids.has(n)),
    entrances: layout.entrances.filter((n) => ids.has(n)),
    imaging_nodes: layout.imaging_nodes.filter((n) => ids.has(n)),
    lab_nodes: layout.lab_nodes.filter((n) => ids.has(n)),
    elevators: layout.elevators.filter((n) => ids.has(n)),
  };
}

/** A human label for a storey — the ED is the ground floor by construction. */
export function floorLabel(floor: number): string {
  return floor === 0 ? "Ground — ED" : `Floor ${floor}`;
}
