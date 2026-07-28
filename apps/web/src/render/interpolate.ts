/**
 * Edge-progress interpolation for streamed staff kinematics.
 *
 * The world is truth, the client fills the gap: between frames, a walking
 * staff dot dead-reckons along its REAL edge using the edge's authoritative
 * traversal time (`seconds`, µs), then SNAPS to the server position when the
 * next frame arrives (the caller resets `extraSimUs` to 0 on each frame).
 * Progress is clamped to [0, 1] — dead-reckoning never overshoots the node.
 */

import type { NodeId, RouteNode, RunState, StaffKinematic } from "../api/types";

export interface CmPoint {
  x_cm: number;
  y_cm: number;
}

/**
 * Sim-µs to dead-reckon past the last applied frame. ONLY a LIVE, playing
 * world extrapolates. A scrubbed view addresses a buffered historical frame —
 * that instant is fixed; advancing it by current wall-time would invent motion
 * that never happened (finding #8). A paused world holds still too.
 */
export function deadReckonSimUs(opts: {
  live: boolean;
  state: RunState | null;
  speed: number;
  wallElapsedMs: number;
}): number {
  if (!opts.live || opts.state !== "playing") {
    return 0;
  }
  return Math.max(0, opts.wallElapsedMs) * opts.speed * 1000;
}

export type NodeIndex = ReadonlyMap<NodeId, RouteNode>;

export function indexNodes(nodes: readonly RouteNode[]): NodeIndex {
  return new Map(nodes.map((n) => [n.id, n]));
}

/**
 * Dead-reckoned progress along the current edge: the streamed
 * `edge_progress` plus sim-time elapsed since the frame, over the edge's
 * total traversal time. Monotonic in `extraSimUs`, clamped to 1.
 */
export function extrapolateProgress(
  frameProgress: number,
  edgeTotalUs: number,
  extraSimUs: number,
): number {
  // Unknown traversal time: hold the server position (no dead-reckoning).
  const p = edgeTotalUs <= 0 ? frameProgress : frameProgress + extraSimUs / edgeTotalUs;
  return Math.max(0, Math.min(1, p));
}

/**
 * Current cm position of a kinematic: resting at a node, or lerped along its
 * edge polyline at `progress` (server progress + optional dead-reckoning).
 */
export function kinematicPosition(
  kin: StaffKinematic,
  nodes: NodeIndex,
  edgeTotalUs = 0,
  extraSimUs = 0,
): CmPoint | null {
  if (kin.edge !== null) {
    const a = nodes.get(kin.edge[0]);
    const b = nodes.get(kin.edge[1]);
    if (a === undefined || b === undefined) {
      return null;
    }
    const t = extrapolateProgress(kin.edge_progress, edgeTotalUs, extraSimUs);
    return {
      x_cm: a.x_cm + (b.x_cm - a.x_cm) * t,
      y_cm: a.y_cm + (b.y_cm - a.y_cm) * t,
    };
  }
  if (kin.at_node !== null) {
    const node = nodes.get(kin.at_node);
    return node === undefined ? null : { x_cm: node.x_cm, y_cm: node.y_cm };
  }
  return null;
}
