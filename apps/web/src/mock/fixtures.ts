/**
 * Deterministic fixtures for the mock stream mode: a small hand-authored ER
 * floor (13 bays, 4 zones, a corridor spine) plus a staff roster. Everything
 * is plain wire-contract data — the mock engine walks the same RouteGraph the
 * FloorMap renders, so staff visibly traverse real edges.
 */

import type { Bay, FloorLayout, RouteEdge, RouteNode, StaffRole, Zone } from "../api/types";

/** Walk speed used to derive edge traversal time: 140 cm/s (~1.4 m/s). */
const WALK_CM_PER_S = 140;

interface StaffFixture {
  id: string;
  role: StaffRole;
  home: string;
}

export const STAFF_ROSTER: readonly StaffFixture[] = [
  { id: "phys-1", role: "physician", home: "station-general" },
  { id: "phys-2", role: "physician", home: "station-fast" },
  { id: "nurse-1", role: "nurse", home: "station-triage" },
  { id: "nurse-2", role: "nurse", home: "station-general" },
  { id: "nurse-3", role: "nurse", home: "station-resus" },
  { id: "tech-1", role: "tech", home: "node-imaging" },
  { id: "porter-1", role: "porter", home: "station-general" },
  { id: "hk-1", role: "housekeeping", home: "station-fast" },
];

export const COMPLAINTS: readonly string[] = [
  "chest pain",
  "abdominal pain",
  "laceration",
  "shortness of breath",
  "fever",
  "fracture suspected",
  "headache",
  "back pain",
];

function node(id: string, label: string, x: number, y: number, floor = 0): RouteNode {
  return { id, label, x_cm: x, y_cm: y, floor };
}

function edge(a: string, b: string, coords: Map<string, RouteNode>): RouteEdge {
  const na = coords.get(a);
  const nb = coords.get(b);
  if (na === undefined || nb === undefined) {
    throw new Error(`fixture edge references unknown node ${a} or ${b}`);
  }
  const distance = Math.round(Math.hypot(na.x_cm - nb.x_cm, na.y_cm - nb.y_cm));
  return {
    a,
    b,
    distance,
    seconds: Math.round((distance / WALK_CM_PER_S) * 1_000_000),
    bidirectional: true,
  };
}

export function makeMockLayout(): FloorLayout {
  const nodes: RouteNode[] = [
    node("entr-main", "Main entrance", 300, 3600),
    node("entr-ambo", "Ambulance bay", 300, 1200),
    node("c-1", "Corridor 1", 1500, 3000),
    node("c-2", "Corridor 2", 3000, 3000),
    node("c-3", "Corridor 3", 4800, 3000),
    node("c-4", "Corridor 4", 6600, 3000),
    node("c-5", "Corridor 5", 8100, 3000),
    node("station-triage", "Triage station", 1500, 1500),
    node("bay-t1", "Triage 1", 900, 1800),
    node("bay-t2", "Triage 2", 2100, 1800),
    node("station-resus", "Resus station", 3000, 1200),
    node("bay-r1", "Resus 1", 2500, 1800),
    node("bay-r2", "Resus 2", 3500, 1800),
    node("station-general", "General station", 4800, 2100),
    node("bay-g1", "General 1", 4200, 1500),
    node("bay-g2", "General 2", 4800, 1200),
    node("bay-g3", "General 3", 5400, 1500),
    node("bay-g4", "General 4", 4200, 4200),
    node("bay-g5", "General 5", 4800, 4500),
    node("bay-g6", "General 6", 5400, 4200),
    node("station-fast", "Fast track station", 8100, 1500),
    node("bay-f1", "Fast track 1", 7500, 1800),
    node("bay-f2", "Fast track 2", 8100, 1000),
    node("bay-f3", "Fast track 3", 8700, 1800),
    node("node-imaging", "Imaging", 6600, 4200),
    node("node-lab", "Lab", 8100, 4200),
  ];
  const byId = new Map(nodes.map((n) => [n.id, n]));

  const edges: RouteEdge[] = [
    edge("entr-main", "c-1", byId),
    edge("entr-ambo", "station-triage", byId),
    edge("c-1", "c-2", byId),
    edge("c-2", "c-3", byId),
    edge("c-3", "c-4", byId),
    edge("c-4", "c-5", byId),
    edge("c-1", "station-triage", byId),
    edge("station-triage", "bay-t1", byId),
    edge("station-triage", "bay-t2", byId),
    edge("c-2", "station-resus", byId),
    edge("station-resus", "bay-r1", byId),
    edge("station-resus", "bay-r2", byId),
    edge("c-3", "station-general", byId),
    edge("station-general", "bay-g1", byId),
    edge("station-general", "bay-g2", byId),
    edge("station-general", "bay-g3", byId),
    edge("c-3", "bay-g4", byId),
    edge("bay-g4", "bay-g5", byId),
    edge("bay-g5", "bay-g6", byId),
    edge("c-5", "station-fast", byId),
    edge("station-fast", "bay-f1", byId),
    edge("station-fast", "bay-f2", byId),
    edge("station-fast", "bay-f3", byId),
    edge("c-4", "node-imaging", byId),
    edge("c-5", "node-lab", byId),
  ];

  const zones: Zone[] = [
    { id: "zone-triage", zone_type: "triage", capacity: 2, floor: 0 },
    { id: "zone-resus", zone_type: "resus_trauma", capacity: 2, floor: 0 },
    { id: "zone-general", zone_type: "general", capacity: 6, floor: 0 },
    { id: "zone-fast", zone_type: "fast_track", capacity: 3, floor: 0 },
  ];

  const bay = (
    id: string,
    zone: string,
    zoneType: Zone["zone_type"],
    station: string,
    isolation = false,
    equipment: readonly string[] = [],
  ): Bay => ({
    id,
    zone,
    zone_type: zoneType,
    node: id,
    serving_station: station,
    isolation_capable: isolation,
    equipment,
  });

  const bays: Bay[] = [
    bay("bay-t1", "zone-triage", "triage", "station-triage"),
    bay("bay-t2", "zone-triage", "triage", "station-triage"),
    bay("bay-r1", "zone-resus", "resus_trauma", "station-resus", false, ["monitor", "defib"]),
    bay("bay-r2", "zone-resus", "resus_trauma", "station-resus", true, ["monitor", "defib"]),
    bay("bay-g1", "zone-general", "general", "station-general"),
    bay("bay-g2", "zone-general", "general", "station-general"),
    bay("bay-g3", "zone-general", "general", "station-general"),
    bay("bay-g4", "zone-general", "general", "station-general"),
    bay("bay-g5", "zone-general", "general", "station-general"),
    bay("bay-g6", "zone-general", "general", "station-general", true),
    bay("bay-f1", "zone-fast", "fast_track", "station-fast"),
    bay("bay-f2", "zone-fast", "fast_track", "station-fast"),
    bay("bay-f3", "zone-fast", "fast_track", "station-fast"),
  ];

  return {
    graph: { nodes, edges },
    zones,
    bays,
    stations: ["station-triage", "station-resus", "station-general", "station-fast"],
    entrances: ["entr-main", "entr-ambo"],
    imaging_nodes: ["node-imaging"],
    lab_nodes: ["node-lab"],
    // Single-floor mock: nowhere to go, so no shafts.
    elevators: [],
  };
}

/** ESI whitelist mirroring the acuity/zone rules the real validator enforces. */
export function allowedZoneTypes(esi: number): readonly Zone["zone_type"][] {
  switch (esi) {
    case 1:
      return ["resus_trauma"];
    case 2:
      return ["resus_trauma", "general"];
    case 3:
      return ["general"];
    default:
      return ["general", "fast_track"];
  }
}
