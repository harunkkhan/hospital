import { describe, expect, test } from "bun:test";

import { makeMockLayout } from "../src/mock/fixtures";
import { makeProjection } from "../src/render/projection";

describe("makeProjection", () => {
  const nodes = makeMockLayout().graph.nodes;

  test("aspect-fits every node inside the padded viewport", () => {
    const proj = makeProjection(nodes, 800, 500, 30);
    for (const node of nodes) {
      const x = proj.toX(node.x_cm);
      const y = proj.toY(node.y_cm);
      expect(x).toBeGreaterThanOrEqual(30);
      expect(x).toBeLessThanOrEqual(800 - 30);
      expect(y).toBeGreaterThanOrEqual(30);
      expect(y).toBeLessThanOrEqual(500 - 30);
    }
  });

  test("uses one uniform scale for both axes (no distortion)", () => {
    const proj = makeProjection(nodes, 1000, 400);
    const dxCm = 1000;
    const dxPx = proj.toX(dxCm) - proj.toX(0);
    const dyPx = proj.toY(dxCm) - proj.toY(0);
    expect(dxPx).toBeCloseTo(dyPx, 6);
    expect(dxPx).toBeCloseTo(dxCm * proj.scale, 6);
  });

  test("degenerate inputs return a null projection instead of NaN", () => {
    const proj = makeProjection([], 800, 500);
    expect(proj.toX(123)).toBe(0);
    expect(Number.isNaN(makeProjection(nodes, 0, 0).toX(50))).toBe(false);
  });
});
