/**
 * The one cm -> px transform. Aspect-fits the floor's bounding box into the
 * viewport (shared by the SVG static layer and the Canvas dynamic layer so
 * they agree pixel-for-centimetre). Coordinates are viz-only — travel time
 * authority stays with edge `seconds`; nothing here feeds pathfinding.
 */

import type { RouteNode } from "../api/types";

export interface Projection {
  toX(xCm: number): number;
  toY(yCm: number): number;
  /** px per cm. */
  scale: number;
  width: number;
  height: number;
}

export function makeProjection(
  nodes: readonly RouteNode[],
  width: number,
  height: number,
  padding = 30,
): Projection {
  if (nodes.length === 0 || width <= 0 || height <= 0) {
    return { toX: () => 0, toY: () => 0, scale: 1, width, height };
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.x_cm);
    maxX = Math.max(maxX, node.x_cm);
    minY = Math.min(minY, node.y_cm);
    maxY = Math.max(maxY, node.y_cm);
  }
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const usableW = Math.max(width - 2 * padding, 1);
  const usableH = Math.max(height - 2 * padding, 1);
  const scale = Math.min(usableW / spanX, usableH / spanY);
  // center the fitted box
  const offsetX = padding + (usableW - spanX * scale) / 2;
  const offsetY = padding + (usableH - spanY * scale) / 2;
  return {
    toX: (xCm) => offsetX + (xCm - minX) * scale,
    toY: (yCm) => offsetY + (yCm - minY) * scale,
    scale,
    width,
    height,
  };
}
