/**
 * Turn a `FloorLayout` — a graph of POINTS — into architecture: ROOMS on a floor plate.
 *
 * The generator lays every bay along ONE corridor (`data/layout.py` walks its zones
 * west→east, hanging bays north and south off a single spine). Drawn literally that is a
 * 108 × 39 m ribbon: legible, but nothing like a hospital floor. This module re-plans the
 * same department onto a squarish plate organised the way emergency departments actually
 * are — a corridor GRID with treatment PODS hanging off it.
 *
 * **The pod decomposition is data, not taste.** `_place_stations` already assigns every bay
 * to a serving nurse station and splits a zone once it exceeds `max_bays_per_station`.
 * Grouping `layout.bays` by `serving_station` recovers those teams exactly — for the shipped
 * ER scenario, eight pods of 12/8/12/8/12/6/10/8 — and that uneven split is what makes the
 * plan read as a hospital rather than a comb. Nothing here hard-codes those numbers.
 *
 * **Everything else is computed too.** The wire `FloorLayout` carries no facility spec —
 * no target area, no aspect ratio, no room depth — so the plate cannot be taken from the
 * scenario the way a dumped layout allowed. Band depths, column positions, where the cross
 * corridor lands and how big the department ends up are all consequences of packing the
 * pods this layout actually contains. A scenario with different quotas re-plans itself.
 *
 * What stays faithful to the wire: bay count, zone membership, station assignment, isolation
 * flags, imaging / lab / triage counts, and both entrances. What is authored: where each pod
 * sits, and the back-of-house the clinical program never mentions. `anchors` records where
 * every node is DRAWN, so the live layer follows the geometry instead of the graph.
 *
 * Pure and deterministic: no `Math.random`, no clock. Two operators comparing screens must
 * see the same building.
 */

import type { Bay, BayId, FloorLayout, NodeId, RouteNode, ZoneId, ZoneType } from "../api/types";
import {
  centerX,
  centerY,
  rect,
  rectCenter,
  rectHeight,
  rectWidth,
  rectsOverlap,
  seeded,
  type Rect,
  type Side,
} from "./geometry";

// ---------------------------------------------------------------------------
// The dimension table. Centimetres, like the generator.
// ---------------------------------------------------------------------------

/**
 * Per-zone room sizes. `width` runs along the pod corridor, `depth` into the room.
 *
 * A resus room is nearly four times the area of a fast-track cubicle, and that spread is
 * most of what makes the plan read as a department. Zone types the ED never allocates
 * (wards, in M4) fall back to the general room rather than being dropped.
 */
const ZONE_ROOM: Partial<Record<ZoneType, { readonly width: number; readonly depth: number }>> = {
  fast_track: { width: 300, depth: 340 },
  general: { width: 360, depth: 420 },
  observation: { width: 400, depth: 420 },
  resus_trauma: { width: 440, depth: 700 },
};
const DEFAULT_ROOM = { width: 360, depth: 420 } as const;

const POD_CORRIDOR = 450; // clear width inside a treatment pod
const STATION_SIZE = { w: 330, d: 165 } as const;

const SPINE_DEPTH = 700; // the main east–west corridor
const CROSS_WIDTH = 700; // the north–south cross corridor
const PERIMETER_DEPTH = 400; // the corridors along the outer walls
const NORTH_STRIP = 740; // imaging / lab + north back-of-house
const SOUTH_STRIP = 620; // south back-of-house
const BAND_MIN_DEPTH = 1200; // a pod band never collapses below a walkable depth

const POD_GAP = 150; // minimum clear between two pods
const POD_GAP_JITTER = 190; // ...deliberately off a constant pitch (see `packBand`)
const DEPT_EDGE = 300; // clear inside the department wall before anything is placed
const PLATE_MARGIN = 900; // the plate the department sits on, all four sides

const IMAGING_WIDTH = 660; // wide enough for a bore ring and its table
const LAB_WIDTH = 360;
const TRIAGE_WIDTH = 300;
const TRIAGE_DEPTH = 340;
const WAITING_MIN_WIDTH = 1600;
const WAITING_MIN_DEPTH = 900;
const AMBULANCE_WIDTH = 640;

const SUPPORT_NAMES = [
  "Clean utility",
  "Soiled hold",
  "Medication",
  "Equipment",
  "Linen",
  "Staff",
  "Store",
  "Office",
  "Plant",
  "Cleaner",
] as const;

/** One deterministic stream for the whole derivation. */
const PLAN_SEED = 0x5eed1;

// ---------------------------------------------------------------------------
// The shapes this module produces
// ---------------------------------------------------------------------------

export type RoomKind =
  | "bay"
  | "triage"
  | "waiting"
  | "imaging"
  | "lab"
  | "ambulance"
  | "support";

interface RoomBase {
  /** Unique within the plan; a bay room's id IS its `BayId`. */
  readonly id: string;
  /** The graph node drawn here, when the room stands for one. Support rooms have none. */
  readonly node: NodeId | null;
  readonly rect: Rect;
  readonly doorSide: Side;
  readonly label: string;
}

export interface BayRoom extends RoomBase {
  readonly kind: "bay";
  readonly id: BayId;
  readonly node: NodeId;
  readonly zone: ZoneId;
  readonly zoneType: ZoneType;
  readonly isolation: boolean;
  readonly station: NodeId;
}

export interface PlainRoom extends RoomBase {
  readonly kind: Exclude<RoomKind, "bay">;
}

export type Room = BayRoom | PlainRoom;

/** A team station: the counter in a pod corridor at its spine mouth. */
export interface StationDesk {
  readonly id: NodeId;
  readonly zone: ZoneId;
  readonly zoneType: ZoneType;
  readonly rect: Rect;
}

export type BandId = "n" | "s";
/** Which corridor a pod hangs off: the spine, or the outer perimeter corridor. */
export type PodAnchor = "spine" | "outer";

export interface Pod {
  readonly station: NodeId;
  readonly zone: ZoneId;
  readonly zoneType: ZoneType;
  readonly count: number;
  readonly band: BandId;
  readonly anchor: PodAnchor;
  /** The whole pod: bays both sides plus the corridor between them. */
  readonly rect: Rect;
  /** Just the corridor. */
  readonly corridor: Rect;
  readonly bays: readonly BayRoom[];
}

export interface Bands {
  readonly northSupport: Rect;
  readonly northPerimeter: Rect;
  readonly north: Rect;
  readonly spine: Rect;
  readonly south: Rect;
  readonly southPerimeter: Rect;
  readonly southSupport: Rect;
}

export type Point = readonly [number, number];

export interface FloorArchitecture {
  /** The slab the department sits on — the department plus a margin. */
  readonly plate: Rect;
  /** The department envelope: everything inside the perimeter wall. */
  readonly dept: Rect;
  readonly bands: Bands;
  /** The main east–west corridor. */
  readonly spine: Rect;
  /** The north–south cross corridor. */
  readonly cross: Rect;
  readonly spineY: number;
  /** Every corridor surface, for the floor wash. */
  readonly circulation: readonly Rect[];
  readonly pods: readonly Pod[];
  readonly bays: readonly BayRoom[];
  readonly stations: readonly StationDesk[];
  readonly triage: readonly PlainRoom[];
  readonly wing: readonly PlainRoom[];
  readonly wingBlock: Rect;
  readonly support: readonly PlainRoom[];
  readonly waiting: PlainRoom | null;
  readonly ambulance: PlainRoom | null;
  /** Where the walk-in door is drawn on the west facade. */
  readonly walkIn: Point | null;
  /** Every room, bays included — one list for the batched builders. */
  readonly rooms: readonly Room[];
  /**
   * Where each graph node is DRAWN. Every node in `layout.graph` has an entry, so the live
   * layer can place anyone anywhere the sim puts them without hovering over a wall.
   */
  readonly anchors: ReadonlyMap<NodeId, Point>;
  readonly bayById: ReadonlyMap<BayId, BayRoom>;
}

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------

interface PodPlan {
  readonly station: NodeId;
  readonly zone: ZoneId;
  readonly zoneType: ZoneType;
  readonly bays: readonly Bay[];
  readonly room: { readonly width: number; readonly depth: number };
  /** Along the pod, north–south. */
  readonly length: number;
  /** Across the pod, east–west: two rows of rooms plus their corridor. */
  readonly width: number;
}

/** A pod, or the front-of-house block, waiting for an x position in a band. */
interface BandItem {
  readonly width: number;
  readonly pod: PodPlan | null;
}

interface PlacedItem extends BandItem {
  readonly x0: number;
}

interface PackedBand {
  readonly items: readonly PlacedItem[];
  /** The width consumed west and east of the cross corridor, gaps included. */
  readonly westWidth: number;
  readonly eastWidth: number;
  readonly splitAt: number;
}

/**
 * Measure a band as two groups either side of the cross corridor.
 *
 * The split lands at the item boundary nearest the band's own midpoint, so the cross
 * corridor cuts the department roughly in half however lopsided the pods are. It is clamped
 * off both ends of a multi-item band: the front of house is always the first item in the
 * south band and the resus pod always the last, and neither may end up on the wrong side of
 * the corridor.
 */
function splitBand(items: readonly BandItem[], gaps: readonly number[]): number {
  if (items.length < 2) {
    return items.length;
  }
  let total = 0;
  for (const item of items) {
    total += item.width;
  }
  let best = 1;
  let bestError = Number.POSITIVE_INFINITY;
  let running = 0;
  for (let i = 0; i < items.length; i += 1) {
    running += items[i]?.width ?? 0;
    if (i === items.length - 1) {
      break;
    }
    const error = Math.abs(running - total / 2);
    if (error < bestError) {
      bestError = error;
      best = i + 1;
    }
    running += gaps[i] ?? 0;
  }
  return best;
}

function groupWidth(
  items: readonly BandItem[],
  gaps: readonly number[],
  from: number,
  to: number,
): number {
  let width = 0;
  for (let i = from; i < to; i += 1) {
    width += items[i]?.width ?? 0;
    if (i > from) {
      width += gaps[i - 1] ?? 0;
    }
  }
  return width;
}

// ---------------------------------------------------------------------------
// Support-room subdivision
// ---------------------------------------------------------------------------

/**
 * Split a rect into rooms along `axis` with deliberately uneven widths. A row of identical
 * boxes is exactly the over-organised look this plan is trying to lose, so the weights come
 * off the seeded stream rather than from an even division.
 */
function subdivide(
  r: Rect,
  axis: "x" | "y",
  n: number,
  rnd: () => number,
  minSpan = 260,
): readonly Rect[] {
  const span = axis === "x" ? rectWidth(r) : rectHeight(r);
  if (n < 1 || span < minSpan) {
    return [];
  }
  const count = Math.max(1, Math.min(n, Math.floor(span / minSpan)));
  const weights: number[] = [];
  let total = 0;
  for (let i = 0; i < count; i += 1) {
    const w = 0.72 + rnd() * 0.56;
    weights.push(w);
    total += w;
  }
  const out: Rect[] = [];
  let cursor = axis === "x" ? r.x0 : r.y0;
  for (const weight of weights) {
    const size = (span * weight) / total;
    out.push(
      axis === "x" ? rect(cursor, r.y0, cursor + size, r.y1) : rect(r.x0, cursor, r.x1, cursor + size),
    );
    cursor += size;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The derivation
// ---------------------------------------------------------------------------

export function deriveFloor(layout: FloorLayout): FloorArchitecture {
  const rnd = seeded(PLAN_SEED);
  const nodes = layout.graph.nodes;

  // --- pods: one per nurse station -----------------------------------------------------
  const byStation = new Map<NodeId, Bay[]>();
  for (const bay of layout.bays) {
    const members = byStation.get(bay.serving_station);
    if (members === undefined) {
      byStation.set(bay.serving_station, [bay]);
    } else {
      members.push(bay);
    }
  }
  // `layout.stations` is in the generator's own west→east build order, which is the only
  // ordering on the wire that means anything; a station that somehow serves bays without
  // being listed still gets a pod, appended in id order so the plan stays deterministic.
  const stationOrder: NodeId[] = [];
  for (const id of layout.stations) {
    if (byStation.has(id)) {
      stationOrder.push(id);
    }
  }
  for (const id of [...byStation.keys()].sort()) {
    if (!layout.stations.includes(id)) {
      stationOrder.push(id);
    }
  }

  const plans: PodPlan[] = [];
  for (const station of stationOrder) {
    const members = byStation.get(station) ?? [];
    const first = members[0];
    if (first === undefined) {
      continue;
    }
    const room = ZONE_ROOM[first.zone_type] ?? DEFAULT_ROOM;
    // Two rows of rooms face each other across the pod corridor, so the pod is half as long
    // as its bay count — the arithmetic that makes an 8-bay pod visibly shorter than a 12.
    const perSide = Math.ceil(members.length / 2);
    plans.push({
      station,
      zone: first.zone,
      zoneType: first.zone_type,
      bays: members,
      room,
      length: perSide * room.width,
      width: 2 * room.depth + POD_CORRIDOR,
    });
  }

  // --- front of house ------------------------------------------------------------------
  // Triage rooms and the waiting hall are graph nodes, not bays, in the real generator; the
  // count of nodes labelled `triage` is the wire's only statement of how many there are.
  const triageNodes = nodes
    .filter((n) => n.label === "triage")
    .slice()
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const waitingNode = nodes.find((n) => n.label === "waiting") ?? null;
  const hasFoh = triageNodes.length > 0 || waitingNode !== null;
  const fohWidth = hasFoh
    ? Math.max(WAITING_MIN_WIDTH, triageNodes.length * TRIAGE_WIDTH)
    : 0;

  // --- band assignment -----------------------------------------------------------------
  // The resus pod takes the east end of the south band, hard against the ambulance entrance:
  // the one adjacency in an ED that is a clinical fact rather than a drafting preference.
  // Everything else is dealt to whichever band is currently narrower, widest pod first, so
  // the two bands come out near the same length without anyone choosing a slot table.
  const resusPlans = plans.filter((p) => p.zoneType === "resus_trauma");
  const freePlans = plans.filter((p) => p.zoneType !== "resus_trauma");
  const dealOrder = freePlans
    .map((plan, index) => ({ plan, index }))
    .sort((a, b) => b.plan.width - a.plan.width || a.index - b.index);

  const bandOf = new Map<NodeId, BandId>();
  let northLoad = 0;
  let southLoad = fohWidth;
  for (const plan of resusPlans) {
    bandOf.set(plan.station, "s");
    southLoad += plan.width;
  }
  for (const { plan } of dealOrder) {
    if (northLoad <= southLoad) {
      bandOf.set(plan.station, "n");
      northLoad += plan.width;
    } else {
      bandOf.set(plan.station, "s");
      southLoad += plan.width;
    }
  }

  // Within a band the pods keep the generator's own order, so a plan still reads west→east
  // the way the zone list does.
  const northItems: BandItem[] = [];
  const southItems: BandItem[] = [];
  if (hasFoh) {
    southItems.push({ width: fohWidth, pod: null });
  }
  for (const plan of plans) {
    if (plan.zoneType === "resus_trauma") {
      continue;
    }
    (bandOf.get(plan.station) === "n" ? northItems : southItems).push({
      width: plan.width,
      pod: plan,
    });
  }
  for (const plan of resusPlans) {
    southItems.push({ width: plan.width, pod: plan });
  }

  // Gaps are off a constant pitch on purpose: an even pitch reads as a comb even when the
  // pods themselves differ in length.
  const gapsFor = (items: readonly BandItem[]): number[] =>
    items.slice(1).map(() => POD_GAP + Math.round(rnd() * POD_GAP_JITTER));
  const northGaps = gapsFor(northItems);
  const southGaps = gapsFor(southItems);

  const northSplit = splitBand(northItems, northGaps);
  const southSplit = splitBand(southItems, southGaps);
  const westWidth = Math.max(
    groupWidth(northItems, northGaps, 0, northSplit),
    groupWidth(southItems, southGaps, 0, southSplit),
  );
  const eastWidth = Math.max(
    groupWidth(northItems, northGaps, northSplit, northItems.length),
    groupWidth(southItems, southGaps, southSplit, southItems.length),
  );

  // --- the department envelope, sized by what was just packed ---------------------------
  const deptX0 = PLATE_MARGIN;
  const westX0 = deptX0 + DEPT_EDGE;
  const crossX0 = westX0 + westWidth + (westWidth > 0 ? POD_GAP : 0);
  const eastX0 = crossX0 + CROSS_WIDTH + POD_GAP;
  const deptX1 = Math.max(eastX0 + eastWidth + DEPT_EDGE, crossX0 + CROSS_WIDTH + DEPT_EDGE);

  const packBand = (items: readonly BandItem[], gaps: readonly number[], splitAt: number): PackedBand => {
    // The west group starts at the west wall and the east group ENDS at the east wall, so
    // whichever band is shorter spends its slack against the cross corridor rather than at
    // the department's edge. That is what puts the resus pod hard against the ambulance
    // entrance, and it leaves the junction of the two corridors open the way a real one is.
    const bandEast = groupWidth(items, gaps, splitAt, items.length);
    const eastStart = Math.max(eastX0, deptX1 - DEPT_EDGE - bandEast);
    const placed: PlacedItem[] = [];
    let cursor = westX0;
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      if (item === undefined) {
        continue;
      }
      if (i === splitAt) {
        cursor = eastStart;
      } else if (i > 0) {
        cursor += gaps[i - 1] ?? POD_GAP;
      }
      placed.push({ ...item, x0: cursor });
      cursor += item.width;
    }
    return {
      items: placed,
      westWidth: groupWidth(items, gaps, 0, splitAt),
      eastWidth: bandEast,
      splitAt,
    };
  };
  const northBand = packBand(northItems, northGaps, northSplit);
  const southBand = packBand(southItems, southGaps, southSplit);

  const bandDepth = (band: PackedBand, floor: number): number => {
    let depth = floor;
    for (const item of band.items) {
      if (item.pod !== null) {
        depth = Math.max(depth, item.pod.length);
      }
    }
    return depth;
  };
  const northDepth = bandDepth(northBand, BAND_MIN_DEPTH);
  const southDepth = bandDepth(
    southBand,
    hasFoh ? Math.max(BAND_MIN_DEPTH, TRIAGE_DEPTH + WAITING_MIN_DEPTH) : BAND_MIN_DEPTH,
  );

  const deptY0 = PLATE_MARGIN;
  const northSupportBand = rect(deptX0, deptY0, deptX1, deptY0 + NORTH_STRIP);
  const northPerimeter = rect(deptX0, northSupportBand.y1, deptX1, northSupportBand.y1 + PERIMETER_DEPTH);
  const northPods = rect(deptX0, northPerimeter.y1, deptX1, northPerimeter.y1 + northDepth);
  const spineBand = rect(deptX0, northPods.y1, deptX1, northPods.y1 + SPINE_DEPTH);
  const southPods = rect(deptX0, spineBand.y1, deptX1, spineBand.y1 + southDepth);
  const southPerimeter = rect(deptX0, southPods.y1, deptX1, southPods.y1 + PERIMETER_DEPTH);
  const southSupportBand = rect(deptX0, southPerimeter.y1, deptX1, southPerimeter.y1 + SOUTH_STRIP);

  const dept = rect(deptX0, deptY0, deptX1, southSupportBand.y1);
  const plate = rect(0, 0, dept.x1 + PLATE_MARGIN, dept.y1 + PLATE_MARGIN);
  const bands: Bands = {
    northSupport: northSupportBand,
    northPerimeter,
    north: northPods,
    spine: spineBand,
    south: southPods,
    southPerimeter,
    southSupport: southSupportBand,
  };

  // The ambulance vestibule takes the east end of the spine, so the spine stops at its wall.
  const ambulanceEntrance = layout.entrances.length > 1 ? layout.entrances[layout.entrances.length - 1] : null;
  const ambulance: PlainRoom | null =
    ambulanceEntrance == null
      ? null
      : {
          kind: "ambulance",
          id: ambulanceEntrance,
          node: ambulanceEntrance,
          doorSide: "w",
          rect: rect(dept.x1 - AMBULANCE_WIDTH, spineBand.y0, dept.x1, spineBand.y1),
          label: "Ambulance",
        };
  const spine = rect(
    dept.x0,
    spineBand.y0,
    ambulance === null ? dept.x1 : ambulance.rect.x0,
    spineBand.y1,
  );
  const spineY = centerY(spine);
  // The cross corridor runs the full depth of the department below the north strip, which is
  // where the imaging suite sits — it heads the corridor rather than being cut in half by it.
  const cross = rect(crossX0, northPerimeter.y0, crossX0 + CROSS_WIDTH, dept.y1);

  // --- pods, bays and stations become rooms ---------------------------------------------
  const pods: Pod[] = [];
  const bays: BayRoom[] = [];
  const stations: StationDesk[] = [];

  const layoutBand = (band: PackedBand, which: BandId, extent: Rect): void => {
    let podIndex = 0;
    for (const item of band.items) {
      const plan = item.pod;
      if (plan === null) {
        continue;
      }
      // Pods alternate their anchor: some hang off the spine, some off the perimeter
      // corridor. A short pod therefore leaves its pocket against the outer wall in one slot
      // and against the spine in the next, and the band edges stay ragged. Both ends of every
      // pod still open onto circulation either way.
      const anchor: PodAnchor = podIndex % 2 === 0 ? "spine" : "outer";
      podIndex += 1;
      const north = which === "n";
      const onSpine = anchor === "spine";
      const [y0, y1] = north
        ? onSpine
          ? [extent.y1 - plan.length, extent.y1]
          : [extent.y0, extent.y0 + plan.length]
        : onSpine
          ? [extent.y0, extent.y0 + plan.length]
          : [extent.y1 - plan.length, extent.y1];
      const podRect = rect(item.x0, y0, item.x0 + plan.width, y1);
      const corridor = rect(
        podRect.x0 + plan.room.depth,
        podRect.y0,
        podRect.x1 - plan.room.depth,
        podRect.y1,
      );

      // Bays alternate west / east down the pod in the generator's own bay order, filling
      // from the spine outward — so bay 00 is the one nearest the department's spine.
      const podBays: BayRoom[] = [];
      plan.bays.forEach((bay, k) => {
        const west = k % 2 === 0;
        const rank = Math.floor(k / 2);
        const top = north
          ? podRect.y1 - (rank + 1) * plan.room.width
          : podRect.y0 + rank * plan.room.width;
        const r = west
          ? rect(podRect.x0, top, podRect.x0 + plan.room.depth, top + plan.room.width)
          : rect(podRect.x1 - plan.room.depth, top, podRect.x1, top + plan.room.width);
        podBays.push({
          kind: "bay",
          id: bay.id,
          node: bay.node,
          zone: bay.zone,
          zoneType: bay.zone_type,
          isolation: bay.isolation_capable,
          station: plan.station,
          // The door faces the pod corridor, so a west-side bay opens east and vice versa.
          doorSide: west ? "e" : "w",
          rect: r,
          label: bay.id,
        });
      });
      bays.push(...podBays);

      // The team station sits in the pod corridor at its spine mouth, where the generator's
      // station stub hangs off the junction.
      const stationY = north ? podRect.y1 - STATION_SIZE.d - 40 : podRect.y0 + 40;
      stations.push({
        id: plan.station,
        zone: plan.zone,
        zoneType: plan.zoneType,
        rect: rect(
          centerX(corridor) - STATION_SIZE.w / 2,
          stationY,
          centerX(corridor) + STATION_SIZE.w / 2,
          stationY + STATION_SIZE.d,
        ),
      });

      pods.push({
        station: plan.station,
        zone: plan.zone,
        zoneType: plan.zoneType,
        count: plan.bays.length,
        band: which,
        anchor,
        rect: podRect,
        corridor,
        bays: podBays,
      });
    }
  };
  layoutBand(northBand, "n", northPods);
  layoutBand(southBand, "s", southPods);

  // --- front of house: waiting hall over a triage row, west end of the south band --------
  const fohItem = hasFoh ? southBand.items.find((item) => item.pod === null) : undefined;
  const fohRect =
    fohItem === undefined ? null : rect(fohItem.x0, southPods.y0, fohItem.x0 + fohItem.width, southPods.y1);
  const triageTop = fohRect === null ? 0 : fohRect.y1 - (triageNodes.length > 0 ? TRIAGE_DEPTH : 0);
  const waiting: PlainRoom | null =
    fohRect === null || waitingNode === null
      ? null
      : {
          kind: "waiting",
          id: waitingNode.id,
          node: waitingNode.id,
          doorSide: "n",
          rect: rect(fohRect.x0, fohRect.y0, fohRect.x1, triageTop),
          label: "Waiting",
        };
  const triage: PlainRoom[] = [];
  if (fohRect !== null && triageNodes.length > 0) {
    const width = rectWidth(fohRect) / triageNodes.length;
    triageNodes.forEach((node, i) => {
      triage.push({
        kind: "triage",
        id: node.id,
        node: node.id,
        doorSide: "s",
        rect: rect(fohRect.x0 + i * width, triageTop, fohRect.x0 + (i + 1) * width, fohRect.y1),
        label: `T${i + 1}`,
      });
    });
  }

  // --- imaging + lab, heading the cross corridor -----------------------------------------
  const wing: PlainRoom[] = [];
  const wingWidth =
    layout.imaging_nodes.length * IMAGING_WIDTH + layout.lab_nodes.length * LAB_WIDTH;
  // Centred on the cross corridor it heads, then held inside the department when the suite is
  // wide enough to run off the end of the north strip.
  const wingX0 = Math.min(
    Math.max(centerX(cross) - wingWidth / 2, dept.x0 + DEPT_EDGE),
    Math.max(dept.x0 + DEPT_EDGE, dept.x1 - DEPT_EDGE - wingWidth),
  );
  const wingRect = rect(wingX0, northSupportBand.y0, wingX0 + wingWidth, northSupportBand.y1);
  let wingCursor = wingRect.x0;
  for (const id of layout.imaging_nodes) {
    wing.push({
      kind: "imaging",
      id,
      node: id,
      doorSide: "s",
      rect: rect(wingCursor, wingRect.y0, wingCursor + IMAGING_WIDTH, wingRect.y1),
      label: "CT",
    });
    wingCursor += IMAGING_WIDTH;
  }
  for (const id of layout.lab_nodes) {
    wing.push({
      kind: "lab",
      id,
      node: id,
      doorSide: "s",
      rect: rect(wingCursor, wingRect.y0, wingCursor + LAB_WIDTH, wingRect.y1),
      label: "Lab",
    });
    wingCursor += LAB_WIDTH;
  }

  const support = layOutSupportRooms({
    pods,
    wing,
    cross,
    fohRect,
    ambulance,
    dept,
    spineY,
    wingRect,
    northSupportBand,
    southSupportBand,
    northBand,
    southBand,
    northPods,
    southPods,
    rnd,
  });

  const { anchors, walkIn } = buildAnchors({
    layout,
    nodes,
    dept,
    spine,
    spineY,
    cross,
    northPerimeter,
    southPods,
    bays,
    wing,
    triage,
    stations,
    waiting,
    ambulance,
  });

  const rooms: Room[] = [...bays, ...wing, ...triage, ...support];
  if (waiting !== null) {
    rooms.push(waiting);
  }
  if (ambulance !== null) {
    rooms.push(ambulance);
  }

  return {
    plate,
    dept,
    bands,
    spine,
    cross,
    spineY,
    circulation: [spine, cross, northPerimeter, southPerimeter, ...pods.map((p) => p.corridor)],
    pods,
    bays,
    stations,
    triage,
    wing,
    wingBlock: wingRect,
    support,
    waiting,
    ambulance,
    walkIn,
    rooms,
    anchors,
    bayById: new Map(bays.map((b) => [b.id, b])),
  };
}

interface SupportInput {
  readonly pods: readonly Pod[];
  readonly wing: readonly PlainRoom[];
  readonly cross: Rect;
  readonly fohRect: Rect | null;
  readonly ambulance: PlainRoom | null;
  readonly dept: Rect;
  readonly spineY: number;
  readonly wingRect: Rect;
  readonly northSupportBand: Rect;
  readonly southSupportBand: Rect;
  readonly northBand: PackedBand;
  readonly southBand: PackedBand;
  readonly northPods: Rect;
  readonly southPods: Rect;
  readonly rnd: () => number;
}

/**
 * Back-of-house: every room the clinical program did not claim.
 *
 * Split out of the plan pipeline because it is a pure consumer of already-final geometry —
 * it reads the packed bands and writes an independent list, touching nothing upstream. It is
 * also the one stage that is wholly invented (see the note inside), so keeping it separate
 * keeps the boundary between derived and invented architecture visible.
 */
function layOutSupportRooms(input: SupportInput): PlainRoom[] {
  const {
    pods, wing, cross, fohRect, ambulance, dept, spineY, wingRect,
    northSupportBand, southSupportBand, northBand, southBand, northPods, southPods, rnd,
  } = input;
  // The layout carries no back-of-house at all, so all of this is invented — drawn to fill
  // the space the program leaves, and named from a fixed vocabulary.
  const claimed: Rect[] = [
    ...pods.map((p) => p.rect),
    ...wing.map((w) => w.rect),
    cross,
    ...(fohRect === null ? [] : [fohRect]),
    ...(ambulance === null ? [] : [ambulance.rect]),
  ];
  const support: PlainRoom[] = [];
  const addSupport = (
    r: Rect,
    axis: "x" | "y",
    n: number,
    options: { jitterDepth?: boolean } = {},
  ): void => {
    if (rectWidth(r) < 240 || rectHeight(r) < 240) {
      return;
    }
    if (claimed.some((c) => rectsOverlap(r, c))) {
      return;
    }
    const other = axis === "x" ? "y" : "x";
    for (const piece of subdivide(r, axis, n, rnd)) {
      // A back-of-house room 15 m across is a hall, not a store; an oversized piece splits
      // again on the other axis so the leftovers read as a warren rather than empty blocks.
      const w = rectWidth(piece);
      const h = rectHeight(piece);
      const parts =
        Math.max(w, h) > 900 && Math.min(w, h) > 380 ? subdivide(piece, other, 2, rnd) : [piece];
      for (const part of parts) {
        if (rectWidth(part) < 200 || rectHeight(part) < 200) {
          continue;
        }
        // Perimeter rows get uneven depth, so the north and south walls are not one flat
        // ribbon of identical boxes.
        let q = part;
        if (options.jitterDepth === true) {
          const inset = rnd() * 130;
          q =
            centerY(q) < spineY
              ? rect(q.x0, q.y0, q.x1, q.y1 - inset)
              : rect(q.x0, q.y0 + inset, q.x1, q.y1);
        }
        const index = support.length;
        support.push({
          kind: "support",
          id: `support_${index}`,
          node: null,
          doorSide:
            axis === "x"
              ? centerY(q) < spineY
                ? "s"
                : "n"
              : centerX(q) < centerX(dept)
                ? "e"
                : "w",
          rect: q,
          label: SUPPORT_NAMES[index % SUPPORT_NAMES.length] ?? "Store",
        });
      }
    }
  };

  // North wall, either side of the imaging suite.
  addSupport(rect(dept.x0, northSupportBand.y0, wingRect.x0 - 150, northSupportBand.y1), "x", 5, {
    jitterDepth: true,
  });
  addSupport(rect(wingRect.x1 + 150, northSupportBand.y0, dept.x1, northSupportBand.y1), "x", 6, {
    jitterDepth: true,
  });
  // South wall, either side of the cross corridor.
  addSupport(rect(dept.x0, southSupportBand.y0, cross.x0 - 150, southSupportBand.y1), "x", 6, {
    jitterDepth: true,
  });
  addSupport(rect(cross.x1 + 150, southSupportBand.y0, dept.x1, southSupportBand.y1), "x", 7, {
    jitterDepth: true,
  });
  // The gaps a band's packing leaves between and beyond its pods, and the pocket every pod
  // that does not span its band leaves at the end it does not reach.
  for (const [band, extent] of [
    [northBand, northPods],
    [southBand, southPods],
  ] as const) {
    const edges: number[] = [dept.x0];
    for (const item of band.items) {
      edges.push(item.x0 - 150, item.x0 + item.width + 150);
    }
    edges.push(dept.x1);
    for (let i = 0; i < edges.length - 1; i += 2) {
      const x0 = edges[i];
      const x1 = edges[i + 1];
      if (x0 === undefined || x1 === undefined) {
        continue;
      }
      // A gap reaching the cross corridor is two pockets, not one: filled whole it would be
      // rejected for overlapping circulation, and the department would carry a conspicuous
      // void either side of its busiest junction.
      const spans: readonly (readonly [number, number])[] = [
        [x0, Math.min(x1, cross.x0 - 150)],
        [Math.max(x0, cross.x1 + 150), x1],
      ];
      for (const [a, b] of spans) {
        if (b - a < 300) {
          continue;
        }
        addSupport(rect(a, extent.y0, b, extent.y1), "y", 4);
      }
    }
  }
  for (const pod of pods) {
    const extent = pod.band === "n" ? northPods : southPods;
    addSupport(rect(pod.rect.x0, extent.y0, pod.rect.x1, pod.rect.y0), "x", 2);
    addSupport(rect(pod.rect.x0, pod.rect.y1, pod.rect.x1, extent.y1), "x", 2);
  }
  return support;
}

interface AnchorInput {
  readonly layout: FloorLayout;
  readonly nodes: readonly RouteNode[];
  readonly dept: Rect;
  readonly spine: Rect;
  readonly spineY: number;
  readonly cross: Rect;
  readonly northPerimeter: Rect;
  readonly southPods: Rect;
  readonly bays: readonly BayRoom[];
  readonly wing: readonly PlainRoom[];
  readonly triage: readonly PlainRoom[];
  readonly stations: readonly StationDesk[];
  readonly waiting: PlainRoom | null;
  readonly ambulance: PlainRoom | null;
}

/**
 * Where every graph node is DRAWN.
 *
 * The live layer places people by node id, and the drawn plan is not the graph's own
 * geometry, so this map is the seam between the two. Split out of the pipeline for the same
 * reason as the support rooms: it consumes final geometry and produces one independent value.
 */
function buildAnchors(input: AnchorInput): { anchors: Map<NodeId, Point>; walkIn: Point | null } {
  const {
    layout, nodes, dept, spine, spineY, cross, northPerimeter, southPods,
    bays, wing, triage, stations, waiting, ambulance,
  } = input;
  const anchors = new Map<NodeId, Point>();
  seedFallbackAnchors(anchors, nodes, dept);
  for (const room of [...bays, ...wing, ...triage]) {
    if (room.node !== null) {
      anchors.set(room.node, rectCenter(room.rect));
    }
  }
  for (const desk of stations) {
    anchors.set(desk.id, rectCenter(desk.rect));
  }
  if (waiting !== null) {
    anchors.set(waiting.node ?? waiting.id, rectCenter(waiting.rect));
  }
  if (ambulance !== null) {
    anchors.set(ambulance.node ?? ambulance.id, rectCenter(ambulance.rect));
  }
  // The walk-in door is on the west facade at the waiting hall, opposite the ambulance door
  // on the east — which is both how EDs work and what breaks the plan's symmetry.
  const walkInEntrance = layout.entrances[0] ?? null;
  const walkIn: Point | null =
    walkInEntrance == null
      ? null
      : [dept.x0, waiting === null ? centerY(southPods) : centerY(waiting.rect)];
  if (walkInEntrance != null && walkIn !== null) {
    anchors.set(walkInEntrance, [walkIn[0] + 200, walkIn[1]]);
  }
  for (const node of nodes) {
    if (node.label === "connector") {
      anchors.set(node.id, [centerX(cross), centerY(northPerimeter)]);
    }
  }
  // Corridor junctions collapse onto the spine, keeping their original west→east order, so a
  // staff member walking the graph's corridor walks the drawn corridor.
  const junctions = nodes
    .filter((n) => n.label === "corridor")
    .slice()
    .sort((a, b) => a.x_cm - b.x_cm || (a.id < b.id ? -1 : 1));
  junctions.forEach((node, i) => {
    const t = junctions.length === 1 ? 0.5 : i / (junctions.length - 1);
    anchors.set(node.id, [spine.x0 + 300 + t * (rectWidth(spine) - 600), spineY]);
  });
  return { anchors, walkIn };
}

/**
 * Every node gets an anchor before any room claims one, by squeezing the layout's own
 * coordinates into the department.
 *
 * The specific placements below overwrite most of these. What is left is the safety net that
 * makes requirement "every node has an anchor" true by construction: a node this module has
 * never heard of (an elevator lobby, a ward corridor in M4) still lands inside the building
 * in roughly the relative position the graph gave it, rather than dropping the person
 * standing on it off the drawing.
 */
function seedFallbackAnchors(
  anchors: Map<NodeId, Point>,
  nodes: readonly RouteNode[],
  dept: Rect,
): void {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const node of nodes) {
    minX = Math.min(minX, node.x_cm);
    maxX = Math.max(maxX, node.x_cm);
    minY = Math.min(minY, node.y_cm);
    maxY = Math.max(maxY, node.y_cm);
  }
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const inset = 300;
  const usableW = Math.max(0, rectWidth(dept) - 2 * inset);
  const usableH = Math.max(0, rectHeight(dept) - 2 * inset);
  for (const node of nodes) {
    const u = spanX > 0 ? (node.x_cm - minX) / spanX : 0.5;
    const v = spanY > 0 ? (node.y_cm - minY) / spanY : 0.5;
    anchors.set(node.id, [dept.x0 + inset + u * usableW, dept.y0 + inset + v * usableH]);
  }
}
