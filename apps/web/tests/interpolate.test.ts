import { describe, expect, test } from "bun:test";

import type { StaffKinematic } from "../src/api/types";
import {
  deadReckonSimUs,
  extrapolateProgress,
  indexNodes,
  kinematicPosition,
} from "../src/render/interpolate";

const NODES = indexNodes([
  { id: "a", label: "A", x_cm: 0, y_cm: 0 },
  { id: "b", label: "B", x_cm: 1000, y_cm: 0 },
]);

function walking(progress: number): StaffKinematic {
  return {
    staff: "nurse-1",
    role: "nurse",
    at_node: null,
    edge: ["a", "b"],
    edge_progress: progress,
    activity: "transport",
    current_task: null,
  };
}

describe("extrapolateProgress", () => {
  test("is monotonic in elapsed sim time and clamped to 1", () => {
    const edgeUs = 10_000_000; // 10s edge
    let previous = 0;
    for (let elapsed = 0; elapsed <= 15_000_000; elapsed += 1_000_000) {
      const p = extrapolateProgress(0.2, edgeUs, elapsed);
      expect(p).toBeGreaterThanOrEqual(previous);
      expect(p).toBeLessThanOrEqual(1);
      previous = p;
    }
    expect(extrapolateProgress(0.2, edgeUs, 15_000_000)).toBe(1);
  });

  test("zero elapsed lands exactly on the server position (snap on frame)", () => {
    expect(extrapolateProgress(0.37, 10_000_000, 0)).toBe(0.37);
  });
});

describe("deadReckonSimUs (finding #8: no extrapolation on buffered views)", () => {
  test("the live playing head dead-reckons by wall-time × speed", () => {
    expect(deadReckonSimUs({ live: true, state: "playing", speed: 60, wallElapsedMs: 250 })).toBe(
      250 * 60 * 1000,
    );
  });

  test("a scrubbed (non-live) view never extrapolates, even if it was playing", () => {
    expect(deadReckonSimUs({ live: false, state: "playing", speed: 60, wallElapsedMs: 5000 })).toBe(
      0,
    );
  });

  test("a paused live view holds still", () => {
    expect(deadReckonSimUs({ live: true, state: "paused", speed: 60, wallElapsedMs: 5000 })).toBe(0);
  });
});

describe("kinematicPosition", () => {
  test("lerps along the real edge", () => {
    const mid = kinematicPosition(walking(0.5), NODES);
    expect(mid).toEqual({ x_cm: 500, y_cm: 0 });
  });

  test("dead-reckons ahead of the frame using edge traversal time", () => {
    const pos = kinematicPosition(walking(0.5), NODES, 10_000_000, 2_000_000);
    expect(pos?.x_cm).toBeCloseTo(700, 6);
  });

  test("rests at a node when not on an edge", () => {
    const resting: StaffKinematic = { ...walking(0), edge: null, at_node: "b", edge_progress: 0 };
    expect(kinematicPosition(resting, NODES)).toEqual({ x_cm: 1000, y_cm: 0 });
  });

  test("unknown nodes yield null rather than a phantom position", () => {
    const ghost: StaffKinematic = { ...walking(0), edge: ["a", "zzz"] };
    expect(kinematicPosition(ghost, NODES)).toBeNull();
  });
});
