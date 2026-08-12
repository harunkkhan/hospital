import { describe, expect, it } from "bun:test";

import type { FloorLayout } from "../src/api/types";
import { floorLabel, floorsOf, sliceToFloor } from "../src/render/floors";

/** Two storeys whose bays deliberately SHARE coordinates — the situation being solved. */
function building(): FloorLayout {
  return {
    graph: {
      nodes: [
        { id: "ed_bay", label: "ed bay", x_cm: 600, y_cm: 100, floor: 0 },
        { id: "ed_stat", label: "ed station", x_cm: 0, y_cm: 100, floor: 0 },
        { id: "elev_f00", label: "elevator g", x_cm: -300, y_cm: 100, floor: 0 },
        // Same x/y as the ED bay: floors of a real building share a footprint.
        { id: "f01_bay", label: "icu bay", x_cm: 600, y_cm: 100, floor: 1 },
        { id: "f01_stat", label: "icu station", x_cm: 0, y_cm: 100, floor: 1 },
        { id: "elev_f01", label: "elevator 1", x_cm: -300, y_cm: 500, floor: 1 },
      ],
      edges: [
        { a: "ed_stat", b: "ed_bay", distance: 600, seconds: 5, bidirectional: true },
        { a: "ed_stat", b: "elev_f00", distance: 300, seconds: 3, bidirectional: true },
        { a: "f01_stat", b: "f01_bay", distance: 600, seconds: 5, bidirectional: true },
        // The shaft: the one edge whose endpoints are on different storeys.
        { a: "elev_f00", b: "elev_f01", distance: 400, seconds: 32, bidirectional: true },
      ],
    },
    zones: [
      { id: "z_gen", zone_type: "general", capacity: 1, floor: 0 },
      { id: "z_icu", zone_type: "icu", capacity: 1, floor: 1 },
    ],
    bays: [
      {
        id: "bay_gen",
        zone: "z_gen",
        zone_type: "general",
        node: "ed_bay",
        serving_station: "ed_stat",
        isolation_capable: false,
        equipment: [],
      },
      {
        id: "bay_icu",
        zone: "z_icu",
        zone_type: "icu",
        node: "f01_bay",
        serving_station: "f01_stat",
        isolation_capable: true,
        equipment: [],
      },
    ],
    stations: ["ed_stat", "f01_stat"],
    entrances: ["ed_stat"],
    imaging_nodes: [],
    lab_nodes: [],
    elevators: ["elev_f00", "elev_f01"],
  };
}

function singleFloor(): FloorLayout {
  const b = building();
  const nodes = b.graph.nodes.filter((n) => n.floor === 0);
  const ids = new Set(nodes.map((n) => n.id));
  return {
    ...b,
    graph: {
      nodes,
      // Both endpoints on the floor — a dangling edge would be a malformed layout, not a
      // single-floor one, and would make the "unchanged" assertion below meaningless.
      edges: b.graph.edges.filter((e) => ids.has(e.a) && ids.has(e.b)),
    },
    zones: b.zones.filter((z) => z.floor === 0),
    bays: b.bays.filter((x) => x.zone === "z_gen"),
    stations: ["ed_stat"],
    elevators: [],
  };
}

describe("floorsOf", () => {
  it("lists every storey, ascending", () => {
    expect(floorsOf(building())).toEqual([0, 1]);
  });

  it("reports a single-floor ED as exactly one storey", () => {
    // This is what keeps the picker hidden and the existing console untouched.
    expect(floorsOf(singleFloor())).toEqual([0]);
  });
});

describe("sliceToFloor", () => {
  it("keeps only the chosen storey's zones, bays, and nodes", () => {
    const ground = sliceToFloor(building(), 0);
    expect(ground.bays.map((b) => b.id)).toEqual(["bay_gen"]);
    expect(ground.zones.map((z) => z.id)).toEqual(["z_gen"]);
    expect(ground.graph.nodes.every((n) => n.floor === 0)).toBe(true);
    expect(ground.stations).toEqual(["ed_stat"]);

    const upstairs = sliceToFloor(building(), 1);
    expect(upstairs.bays.map((b) => b.id)).toEqual(["bay_icu"]);
    expect(upstairs.stations).toEqual(["f01_stat"]);
  });

  it("separates bays that share coordinates", () => {
    // The whole reason this module exists: both bays sit at (600, 100), so rendering the
    // unsliced graph would draw the ICU on top of the ED.
    const all = building();
    const ed = all.bays.find((b) => b.id === "bay_gen");
    const icu = all.bays.find((b) => b.id === "bay_icu");
    const node = (id: string) => all.graph.nodes.find((n) => n.id === id);
    expect(node(ed!.node)!.x_cm).toBe(node(icu!.node)!.x_cm);
    expect(node(ed!.node)!.y_cm).toBe(node(icu!.node)!.y_cm);
    // ...and after slicing, each storey shows exactly one of them.
    expect(sliceToFloor(all, 0).bays).toHaveLength(1);
    expect(sliceToFloor(all, 1).bays).toHaveLength(1);
  });

  it("drops the shaft edge but keeps its boarding node", () => {
    // A vertical edge on a single-storey plan is a line to nowhere — its far end is not on
    // the canvas. The boarding node stays: staff really do walk to it.
    const ground = sliceToFloor(building(), 0);
    expect(ground.graph.edges.some((e) => e.a === "elev_f00" && e.b === "elev_f01")).toBe(false);
    expect(ground.graph.nodes.some((n) => n.id === "elev_f00")).toBe(true);
    expect(ground.elevators).toEqual(["elev_f00"]);
  });

  it("leaves a single-floor layout materially unchanged", () => {
    const one = singleFloor();
    const sliced = sliceToFloor(one, 0);
    expect(sliced.bays).toEqual(one.bays);
    expect(sliced.zones).toEqual(one.zones);
    expect(sliced.graph.nodes).toEqual(one.graph.nodes);
    expect(sliced.graph.edges).toEqual(one.graph.edges);
  });
});

describe("floorLabel", () => {
  it("names the ground floor for what it is", () => {
    expect(floorLabel(0)).toBe("Ground — ED");
    expect(floorLabel(2)).toBe("Floor 2");
  });
});
