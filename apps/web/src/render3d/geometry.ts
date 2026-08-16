/**
 * Plan-space primitives shared by the derivation and the builders.
 *
 * Everything upstream of the renderer works in **centimetres**, the unit the layout
 * generator emits, with +x east and +y south. `SCALE` converts to the metres the scene
 * graph uses; nothing else in this directory should hard-code that factor.
 */

export const SCALE = 0.01;

export type Side = "n" | "s" | "e" | "w";

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function rect(x0: number, y0: number, x1: number, y1: number): Rect {
  return { x0, y0, x1, y1 };
}

export const rectWidth = (r: Rect): number => r.x1 - r.x0;
export const rectHeight = (r: Rect): number => r.y1 - r.y0;
export const centerX = (r: Rect): number => (r.x0 + r.x1) / 2;
export const centerY = (r: Rect): number => (r.y0 + r.y1) / 2;
export const rectCenter = (r: Rect): readonly [number, number] => [centerX(r), centerY(r)];

export function rectsOverlap(a: Rect, b: Rect, tolerance = 1): boolean {
  return (
    a.x0 < b.x1 - tolerance &&
    a.x1 > b.x0 + tolerance &&
    a.y0 < b.y1 - tolerance &&
    a.y1 > b.y0 + tolerance
  );
}

/**
 * Representational heights. The layout carries no z at all — it is a plan graph — so
 * every one of these is a stated liberty rather than data.
 */
export const HEIGHTS = {
  perimeter: 340,
  partition: 250,
  counter: 105,
  bed: 62,
  chair: 46,
  curtainTrack: 210,
} as const;

/**
 * A room's local frame: `u` runs along the door wall, `v` runs into the room from it.
 *
 * Treatment pods run north-south, so a bay's door faces east or west; the front-of-house
 * rooms open north or south. Placing every fitting in (u, v) means one description of a
 * bed serves all four orientations instead of four rotated copies that can drift apart.
 */
export interface RoomFrame {
  /** Along the door wall. */
  readonly W: number;
  /** Into the room from the door wall. */
  readonly D: number;
  box(u0: number, v0: number, u1: number, v1: number): Rect;
}

export function frameOf(r: Rect, doorSide: Side): RoomFrame {
  const w = rectWidth(r);
  const h = rectHeight(r);
  switch (doorSide) {
    case "s":
      return { W: w, D: h, box: (u0, v0, u1, v1) => rect(r.x0 + u0, r.y1 - v1, r.x0 + u1, r.y1 - v0) };
    case "n":
      return { W: w, D: h, box: (u0, v0, u1, v1) => rect(r.x0 + u0, r.y0 + v0, r.x0 + u1, r.y0 + v1) };
    case "w":
      return { W: h, D: w, box: (u0, v0, u1, v1) => rect(r.x0 + v0, r.y0 + u0, r.x0 + v1, r.y0 + u1) };
    case "e":
      return { W: h, D: w, box: (u0, v0, u1, v1) => rect(r.x1 - v1, r.y0 + u0, r.x1 - v0, r.y0 + u1) };
  }
}

/**
 * The wall runs of a rect, with a door gap punched in one side.
 *
 * `doorFrac` is the fraction of that side left open: a treatment bay opens most of its
 * corridor face (curtain or sliding glass), a store takes a single leaf.
 */
export function wallRuns(
  r: Rect,
  doorSide: Side | "none",
  doorFrac: number,
): readonly (readonly [number, number, number, number])[] {
  const sides: Record<Side, readonly [number, number, number, number, "x" | "y"]> = {
    n: [r.x0, r.y0, r.x1, r.y0, "x"],
    s: [r.x0, r.y1, r.x1, r.y1, "x"],
    w: [r.x0, r.y0, r.x0, r.y1, "y"],
    e: [r.x1, r.y0, r.x1, r.y1, "y"],
  };
  const runs: (readonly [number, number, number, number])[] = [];
  for (const key of ["n", "s", "w", "e"] as const) {
    const [ax, ay, bx, by, axis] = sides[key];
    if (key !== doorSide) {
      runs.push([ax, ay, bx, by]);
      continue;
    }
    const span = axis === "x" ? bx - ax : by - ay;
    const jamb = (Math.abs(span) * (1 - doorFrac)) / 2;
    if (axis === "x") {
      runs.push([ax, ay, ax + jamb, ay], [bx - jamb, by, bx, by]);
    } else {
      runs.push([ax, ay, ax, ay + jamb], [bx, by - jamb, bx, by]);
    }
  }
  return runs;
}

/**
 * A small deterministic stream. Support-room subdivision has to look irregular without
 * being random: the same layout must draw identically on every client and across reloads,
 * or two operators comparing screens would see different buildings.
 */
export function seeded(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
