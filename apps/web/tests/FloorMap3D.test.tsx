/**
 * The 3D floor's storey seam.
 *
 * There is no WebGL context under happy-dom, which is deliberate cover rather than a
 * limitation: `FloorMap3D` catches the failed context and stays mounted so the operator can
 * fall back to the 2D map, and that leaves its chrome — including the storey picker —
 * rendered and assertable without a GPU.
 */

import { describe, expect, test } from "bun:test";
import { render, screen } from "@testing-library/react";

import type { Bay, FloorLayout, RouteNode, Zone } from "../src/api/types";
import { FloorMap3D } from "../src/render3d/FloorMap3D";
import { initialWorld } from "../src/state/streamReducer";

function node(id: string, label: string, x: number, y: number, floor: number): RouteNode {
  return { id, label, x_cm: x, y_cm: y, floor };
}

function bay(id: string, zone: string, station: string): Bay {
  return {
    id,
    zone,
    zone_type: "general",
    node: id,
    serving_station: station,
    isolation_capable: false,
    equipment: [],
  };
}

/** Two storeys sharing a footprint — the case that made the slice necessary. */
function building(): FloorLayout {
  const nodes: RouteNode[] = [];
  const bays: Bay[] = [];
  const zones: Zone[] = [
    { id: "z-ed", zone_type: "general", capacity: 2, floor: 0 },
    { id: "z-ward", zone_type: "med_surg", capacity: 2, floor: 1 },
  ];
  for (const [floor, zone, station] of [
    [0, "z-ed", "st-ed"],
    [1, "z-ward", "st-ward"],
  ] as const) {
    nodes.push(node(station, "station", 600, 400, floor));
    nodes.push(node(`corr-${floor}`, "corridor", 600, 300, floor));
    for (let i = 0; i < 2; i++) {
      const id = `bay-${floor}-${i}`;
      // Identical coordinates on both storeys: floors of a real building share a footprint.
      nodes.push(node(id, "general", 400 + i * 300, 700, floor));
      bays.push(bay(id, zone, station));
    }
  }
  return {
    graph: { nodes, edges: [] },
    zones,
    bays,
    stations: ["st-ed", "st-ward"],
    entrances: [],
    imaging_nodes: [],
    lab_nodes: [],
    elevators: [],
  };
}

function singleFloor(): FloorLayout {
  const full = building();
  return {
    ...full,
    graph: { nodes: full.graph.nodes.filter((n) => n.floor === 0), edges: [] },
    zones: full.zones.filter((z) => z.floor === 0),
    bays: full.bays.filter((b) => b.id.startsWith("bay-0")),
    stations: ["st-ed"],
  };
}


function mount(layout: FloorLayout): void {
  render(
    <FloorMap3D layout={layout} world={initialWorld()} selected={null} onSelect={() => {}} live={false} />,
  );
}

describe("FloorMap3D storeys", () => {
  test("offers a storey picker when the building has more than one", () => {
    mount(building());
    expect(screen.getByRole("button", { name: "Ground — ED" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Floor 1" })).toBeDefined();
  });

  test("an ED-only scenario is not asked which floor it is on", () => {
    mount(singleFloor());
    expect(screen.queryByRole("button", { name: "Ground — ED" })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Floor /u })).toBeNull();
  });

  test("says so, and stays mounted, when there is no WebGL context", () => {
    // Losing the canvas must not cost the operator the session — the 2D map is one click away.
    mount(building());
    expect(screen.getByText(/no WebGL context/u)).toBeDefined();
  });
});
