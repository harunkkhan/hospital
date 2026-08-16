import { describe, expect, test } from "bun:test";

import type { Bay, FloorLayout, RouteEdge, RouteNode, ZoneType } from "../src/api/types";
import { deriveFloor } from "../src/render3d/derive";
import { rectHeight, rectWidth, rectsOverlap, type Rect } from "../src/render3d/geometry";
import { makeMockLayout } from "../src/mock/fixtures";

/**
 * The shipped ER floor, reproduced on the wire contract.
 *
 * Geometry mirrors `generate_floor(scenarios/er_floor.yaml)` exactly: a 38-column spine at
 * a 284 cm pitch, bays 420 cm north and south of it, and the specials on their own
 * coordinate layers. The per-station loads are the ones the generator actually produces —
 * `max_bays_per_station` is 12, so each 20-bay general zone splits 8/12 and the 16-bay
 * observation zone splits 6/10. The derivation is what is under test here, so the split
 * arrives as data rather than being recomputed.
 */
const ER_ZONES: readonly {
  zone: string;
  zoneType: ZoneType;
  bays: number;
  isolation: number;
  stationLoads: readonly number[];
}[] = [
  { zone: "fast_track_00", zoneType: "fast_track", bays: 12, isolation: 0, stationLoads: [12] },
  { zone: "general_01", zoneType: "general", bays: 20, isolation: 2, stationLoads: [8, 12] },
  { zone: "general_02", zoneType: "general", bays: 20, isolation: 2, stationLoads: [8, 12] },
  { zone: "observation_03", zoneType: "observation", bays: 16, isolation: 4, stationLoads: [6, 10] },
  { zone: "resus_trauma_04", zoneType: "resus_trauma", bays: 8, isolation: 8, stationLoads: [8] },
];

/** Every fixture node is on the ground floor: this reproduces the single-storey ED. */
const GROUND = 0;
const SPINE_Y = 3871;
const PITCH = 284;
const ROOM_DEPTH = 420;
const FLOOR_WIDTH = 12000;

function makeErFloorLayout(): FloorLayout {
  const nodes: RouteNode[] = [];
  const bays: Bay[] = [];
  const stations: string[] = [];

  let column = 0;
  for (const zone of ER_ZONES) {
    const columns = Math.ceil(zone.bays / 2);
    const loads = zone.stationLoads;
    loads.forEach((_, s) => {
      const id = `station_${zone.zone}_${String(s).padStart(2, "0")}`;
      stations.push(id);
      nodes.push({ floor: GROUND,
        id,
        label: "station",
        x_cm: 600 + (column + Math.floor((s * columns) / loads.length)) * PITCH,
        y_cm: SPINE_Y + (s === 0 ? 140 : -140),
      });
    });
    // Bays walk the zone's columns west->east, north bay then south bay, exactly as
    // `_place_zone_bays` does; each station takes its load in that order.
    let k = 0;
    let stationIndex = 0;
    let taken = 0;
    for (let j = 0; j < columns; j += 1) {
      for (const side of [-1, 1]) {
        if (2 * j + (side < 0 ? 0 : 1) >= zone.bays) {
          continue;
        }
        while (taken >= (loads[stationIndex] ?? 0) && stationIndex < loads.length - 1) {
          stationIndex += 1;
          taken = 0;
        }
        const id = `bay_${zone.zone}_${String(k).padStart(2, "0")}`;
        nodes.push({ floor: GROUND,
          id,
          label: zone.zoneType,
          x_cm: 600 + (column + j) * PITCH,
          y_cm: SPINE_Y + side * ROOM_DEPTH,
        });
        bays.push({
          id,
          zone: zone.zone,
          zone_type: zone.zoneType,
          node: id,
          serving_station: `station_${zone.zone}_${String(stationIndex).padStart(2, "0")}`,
          isolation_capable: k < zone.isolation,
          equipment: [],
        });
        k += 1;
        taken += 1;
      }
    }
    column += columns;
  }

  for (let i = 0; i < column; i += 1) {
    nodes.push({ floor: GROUND,
      id: `corr_${String(i).padStart(3, "0")}`,
      label: "corridor",
      x_cm: 600 + i * PITCH,
      y_cm: SPINE_Y,
    });
  }
  nodes.push({ floor: GROUND, id: "entrance_walk_in", label: "entrance", x_cm: 0, y_cm: SPINE_Y });
  nodes.push({ floor: GROUND, id: "waiting_room", label: "waiting", x_cm: 0, y_cm: SPINE_Y - ROOM_DEPTH });
  nodes.push({ floor: GROUND, id: "entrance_ambulance", label: "entrance", x_cm: FLOOR_WIDTH, y_cm: SPINE_Y });
  for (let i = 0; i < 6; i += 1) {
    nodes.push({ floor: GROUND,
      id: `triage_${String(i).padStart(2, "0")}`,
      label: "triage",
      x_cm: 0,
      y_cm: SPINE_Y + ROOM_DEPTH * (i + 1),
    });
  }
  const mid = 600 + Math.floor(column / 2) * PITCH;
  nodes.push({ floor: GROUND, id: "connector_north", label: "connector", x_cm: mid, y_cm: SPINE_Y - 2 * ROOM_DEPTH });
  const imaging: string[] = [];
  const lab: string[] = [];
  for (let i = 0; i < 3; i += 1) {
    const id = `imaging_${String(i).padStart(2, "0")}`;
    imaging.push(id);
    nodes.push({ floor: GROUND, id, label: "imaging", x_cm: mid + i * ROOM_DEPTH, y_cm: SPINE_Y - 3 * ROOM_DEPTH });
  }
  for (let j = 0; j < 2; j += 1) {
    const id = `lab_${String(j).padStart(2, "0")}`;
    lab.push(id);
    nodes.push({ floor: GROUND, id, label: "lab", x_cm: mid + (3 + j) * ROOM_DEPTH, y_cm: SPINE_Y - 3 * ROOM_DEPTH });
  }

  // The derivation never routes, so a spine-only edge list is enough to be a real graph.
  const edges: RouteEdge[] = [];
  for (let i = 1; i < column; i += 1) {
    edges.push({
      a: `corr_${String(i - 1).padStart(3, "0")}`,
      b: `corr_${String(i).padStart(3, "0")}`,
      distance: PITCH,
      seconds: Math.round((PITCH / 120) * 1_000_000),
      bidirectional: true,
    });
  }

  return {
    graph: { nodes, edges },
    zones: ER_ZONES.map((z) => ({
      id: z.zone,
      zone_type: z.zoneType,
      capacity: z.bays,
      floor: 0,
    })),
    bays,
    stations,
    entrances: ["entrance_walk_in", "entrance_ambulance"],
    imaging_nodes: imaging,
    lab_nodes: lab,
    elevators: [],
  };
}

const contains = (r: Rect, p: readonly [number, number]): boolean =>
  p[0] >= r.x0 && p[0] <= r.x1 && p[1] >= r.y0 && p[1] <= r.y1;

describe("deriveFloor", () => {
  const layout = makeErFloorLayout();
  const arch = deriveFloor(layout);

  test("makes one pod per nurse station, with that station's bays", () => {
    const stationsWithBays = new Set(layout.bays.map((b) => b.serving_station));
    expect(arch.pods.length).toBe(stationsWithBays.size);
    for (const pod of arch.pods) {
      const expected = layout.bays.filter((b) => b.serving_station === pod.station);
      expect(pod.count).toBe(expected.length);
      expect(pod.bays.map((b) => b.id).sort()).toEqual(expected.map((b) => b.id).sort());
    }
    // The generator's own uneven split — the source of the plan's irregularity. Pods come
    // back band by band rather than in station order, so compare the sizes as a multiset.
    expect(arch.pods.map((p) => p.count).sort((a, b) => b - a)).toEqual([
      12, 12, 12, 10, 8, 8, 8, 6,
    ]);
    expect(arch.bays.length).toBe(layout.bays.length);
  });

  test("sizes rooms by zone: a resus room is not a fast-track cubicle", () => {
    const dims = (zoneType: ZoneType): readonly [number, number] => {
      const bay = arch.bays.find((b) => b.zoneType === zoneType);
      if (bay === undefined) {
        throw new Error(`no ${zoneType} bay derived`);
      }
      // A bay's door faces the pod corridor east or west, so its depth is the x span.
      return [rectHeight(bay.rect), rectWidth(bay.rect)];
    };
    expect(dims("fast_track")).toEqual([300, 340]);
    expect(dims("general")).toEqual([360, 420]);
    expect(dims("observation")).toEqual([400, 420]);
    expect(dims("resus_trauma")).toEqual([440, 700]);
  });

  test("no two pods overlap, and every pod stays inside the department", () => {
    for (let i = 0; i < arch.pods.length; i += 1) {
      for (let j = i + 1; j < arch.pods.length; j += 1) {
        const a = arch.pods[i];
        const b = arch.pods[j];
        if (a === undefined || b === undefined) {
          throw new Error("pod index out of range");
        }
        expect([a.station, b.station, rectsOverlap(a.rect, b.rect)]).toEqual([
          a.station,
          b.station,
          false,
        ]);
      }
    }
    for (const pod of arch.pods) {
      expect(pod.rect.x0).toBeGreaterThanOrEqual(arch.dept.x0);
      expect(pod.rect.x1).toBeLessThanOrEqual(arch.dept.x1);
      expect(pod.rect.y0).toBeGreaterThanOrEqual(arch.dept.y0);
      expect(pod.rect.y1).toBeLessThanOrEqual(arch.dept.y1);
      // Neither may a pod be built across the cross corridor.
      expect(rectsOverlap(pod.rect, arch.cross)).toBe(false);
    }
  });

  test("gives every graph node an anchor", () => {
    for (const node of layout.graph.nodes) {
      expect(arch.anchors.has(node.id)).toBe(true);
    }
    for (const [, point] of arch.anchors) {
      expect(contains(arch.dept, point)).toBe(true);
    }
  });

  test("anchors land inside the room they stand for, never on a wall", () => {
    for (const bay of arch.bays) {
      const point = arch.anchors.get(bay.node);
      expect(point === undefined ? bay.id : contains(bay.rect, point)).toBe(true);
    }
    for (const desk of arch.stations) {
      const point = arch.anchors.get(desk.id);
      expect(point === undefined ? desk.id : contains(desk.rect, point)).toBe(true);
    }
    for (const room of [...arch.triage, ...arch.wing]) {
      const node = room.node;
      const point = node === null ? undefined : arch.anchors.get(node);
      expect(point === undefined ? room.id : contains(room.rect, point)).toBe(true);
    }
  });

  test("corridor junctions spread along the spine in their original order", () => {
    const junctions = layout.graph.nodes
      .filter((n) => n.label === "corridor")
      .slice()
      .sort((a, b) => a.x_cm - b.x_cm);
    let previous = Number.NEGATIVE_INFINITY;
    for (const node of junctions) {
      const point = arch.anchors.get(node.id);
      if (point === undefined) {
        throw new Error(`junction ${node.id} has no anchor`);
      }
      expect(point[0]).toBeGreaterThan(previous);
      expect(contains(arch.spine, point)).toBe(true);
      previous = point[0];
    }
  });

  test("re-plans the ribbon into a squarish plate", () => {
    const xs = layout.graph.nodes.map((n) => n.x_cm);
    const ys = layout.graph.nodes.map((n) => n.y_cm);
    const rawAspect =
      (Math.max(...xs) - Math.min(...xs)) / (Math.max(...ys) - Math.min(...ys));
    const aspect = rectWidth(arch.dept) / rectHeight(arch.dept);
    expect(rawAspect).toBeGreaterThan(2.5); // the generator's ribbon
    expect(aspect).toBeGreaterThan(0.6);
    expect(aspect).toBeLessThan(1.8);
  });

  test("puts the resus pod at the ambulance end and the front of house at the other", () => {
    const resus = arch.pods.find((p) => p.zoneType === "resus_trauma");
    if (resus === undefined || arch.ambulance === null || arch.waiting === null) {
      throw new Error("the ER fixture must derive a resus pod, an ambulance bay and a hall");
    }
    for (const pod of arch.pods) {
      expect(pod.rect.x1).toBeLessThanOrEqual(resus.rect.x1);
    }
    expect(resus.band).toBe("s");
    expect(arch.ambulance.rect.x1).toBe(arch.dept.x1);
    expect(arch.waiting.rect.x0).toBeLessThan(resus.rect.x0);
    expect(arch.triage.length).toBe(6);
  });

  test("is deterministic — two clients draw the same building", () => {
    const again = deriveFloor(layout);
    expect(JSON.stringify(again.rooms)).toBe(JSON.stringify(arch.rooms));
    expect(JSON.stringify([...again.anchors])).toBe(JSON.stringify([...arch.anchors]));
  });

  test("plans the hand-authored mock floor too (no triage nodes, no waiting hall)", () => {
    const mock = makeMockLayout();
    const small = deriveFloor(mock);
    expect(small.pods.length).toBe(new Set(mock.bays.map((b) => b.serving_station)).size);
    expect(small.bays.length).toBe(mock.bays.length);
    expect(small.triage.length).toBe(0);
    expect(small.waiting).toBe(null);
    for (const node of mock.graph.nodes) {
      expect(small.anchors.has(node.id)).toBe(true);
    }
  });
});
