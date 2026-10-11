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
 * Where the bed goes in a treatment bay, and which way round the patient lies.
 *
 * Shared rather than duplicated because TWO renderers need the same answer: `scene.ts` draws
 * the mattress and pillow, and `live.ts` lays the occupant on it. Computed independently they
 * would drift, and the failure mode is a patient floating beside their own bed.
 */
export const BED = {
  halfWidth: 45,
  maxLength: 205,
  /** Clear floor kept between the foot of the bed and the door wall. */
  clearance: 120,
  footInset: 45,
} as const;

export interface BedPlacement {
  readonly mattress: Rect;
  readonly pillow: Rect;
  /** Unit vector in plan pointing from the foot of the bed toward the pillow. */
  readonly toHead: readonly [number, number];
}

export function bedOf(room: Rect, doorSide: Side): BedPlacement {
  const f = frameOf(room, doorSide);
  const length = Math.min(BED.maxLength, f.D - BED.clearance);
  const u = f.W / 2;
  const head = f.D - BED.footInset;
  // The head of the bed is against the wall OPPOSITE the door, which is the far end of the
  // frame's own `v` axis — so the head-ward direction is that axis, read off the frame rather
  // than switched on the side. One less place for a door orientation to be got wrong.
  const near = f.box(u, 0, u, 0);
  const far = f.box(u, 1, u, 1);
  return {
    mattress: f.box(u - BED.halfWidth, head - length, u + BED.halfWidth, head),
    pillow: f.box(u - 40, head - 40, u + 40, head - 6),
    toHead: [centerX(far) - centerX(near), centerY(far) - centerY(near)],
  };
}

/**
 * The waiting hall's seats, in the order people take them.
 *
 * Same reason as `bedOf`: `scene.ts` draws the chairs and `live.ts` sits the queue on them.
 * A waiting room whose people hover between the seats reads as a rendering bug, which is
 * exactly what two independent grids would produce.
 */
export function seatsOf(hall: Rect): readonly (readonly [number, number])[] {
  const rows = Math.max(2, Math.min(6, Math.floor(rectHeight(hall) / 380)));
  const perRow = Math.max(3, Math.min(14, Math.floor(rectWidth(hall) / 150)));
  const x0 = hall.x0 + 340;
  const x1 = hall.x1 - 220;
  const y0 = hall.y0 + 300;
  const y1 = hall.y1 - 240;
  if (x1 <= x0 || y1 <= y0) {
    return [];
  }
  const seats: (readonly [number, number])[] = [];
  for (let row = 0; row < rows; row += 1) {
    const y = y0 + (row * (y1 - y0)) / (rows - 1);
    for (let i = 0; i < perRow; i += 1) {
      seats.push([x0 + (i * (x1 - x0)) / (perRow - 1), y]);
    }
  }
  return seats;
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
