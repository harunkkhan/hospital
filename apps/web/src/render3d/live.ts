/**
 * The live layer: patients and staff on the drawn floor.
 *
 * **Drawn distances are not graph distances.** The route graph puts every bay on one long
 * spine; this view puts them in pods on a corridor grid. Walk times still come from the
 * graph — nothing here touches routing — but a person's POSITION has to be expressed in the
 * building the operator is looking at, or staff walk through walls and patients hover over
 * back-of-house.
 *
 * The trick that makes that free: `derive.ts` already exports where every node is DRAWN, so
 * this module builds a `NodeIndex` whose coordinates are the drawn anchors and hands it
 * straight to `render/interpolate.ts`. Dead reckoning, edge progress and its clamping are
 * then EXACTLY the code the 2D map runs — one implementation, two renderers. Re-deriving
 * that maths here would be a second definition of "where is this person right now", and the
 * two would drift.
 *
 * Both populations are instanced: one draw call for every patient, one for every staff
 * member, updated in place each frame. Shape carries meaning as it does in 2D — patients are
 * rounded, staff are squared — so acuity colour is never the only channel.
 */

import * as THREE from "three";

import type { FloorLayout, RouteNode } from "../api/types";
import { deadReckonSimUs, indexNodes, kinematicPosition } from "../render/interpolate";
import type { WorldView } from "../state/streamReducer";
import type { FloorArchitecture } from "./derive";
import { SCALE } from "./geometry";
import type { StylePack } from "./styles";

const S = SCALE;

/** Representational, like every height here: the layout has no z. */
const PERSON_HEIGHT = 1.08;
const PERSON_CENTRE = 0.64;

/** Chips fan out when several people share one node, so a busy bay is visibly busy. */
const FAN_RADIUS_CM = 62;
const FAN_STEP_CM = 26;
const FAN_PER_RING = 8;

export interface LiveLayer {
  readonly root: THREE.Group;
  /**
   * Reposition everyone from the given world. Called once per animation frame — it writes
   * instance matrices and colours, and allocates only when the population outgrows the
   * buffers it already has.
   */
  update(world: WorldView, live: boolean, frameWallMs: number): void;
  dispose(): void;
}

interface Population {
  mesh: THREE.InstancedMesh;
  capacity: number;
}

export function buildLiveLayer(
  layout: FloorLayout,
  arch: FloorArchitecture,
  style: StylePack,
): LiveLayer {
  const root = new THREE.Group();
  root.position.set(0, 0, 0);

  // Every node, re-coordinated to where it is drawn. `interpolate.ts` reads nothing else off
  // a RouteNode, so this substitution is total: it cannot half-apply.
  const drawnNodes: RouteNode[] = layout.graph.nodes.map((node) => {
    const anchor = arch.anchors.get(node.id);
    return anchor === undefined ? node : { ...node, x_cm: anchor[0], y_cm: anchor[1] };
  });
  const nodeIndex = indexNodes(drawnNodes);

  // Authoritative traversal time per directed edge, for dead reckoning (µs, despite `seconds`).
  const edgeUs = new Map<string, number>();
  for (const edge of layout.graph.edges) {
    edgeUs.set(`${edge.a}>${edge.b}`, edge.seconds);
    edgeUs.set(`${edge.b}>${edge.a}`, edge.seconds);
  }

  const patientGeometry = new THREE.CapsuleGeometry(0.21, PERSON_HEIGHT - 0.42, 4, 10);
  const staffGeometry = new THREE.BoxGeometry(0.4, PERSON_HEIGHT, 0.4);
  const material = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 });

  const makePopulation = (geometry: THREE.BufferGeometry, capacity: number): Population => {
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.count = 0;
    mesh.renderOrder = 20;
    mesh.frustumCulled = false;
    root.add(mesh);
    return { mesh, capacity };
  };
  let patients = makePopulation(patientGeometry, 128);
  let staff = makePopulation(staffGeometry, 64);

  const grow = (population: Population, needed: number, geometry: THREE.BufferGeometry): Population => {
    if (needed <= population.capacity) {
      return population;
    }
    root.remove(population.mesh);
    population.mesh.dispose();
    // Double until it fits: a run that admits one more patient must not reallocate the whole
    // buffer on the next frame as well.
    let capacity = Math.max(1, population.capacity);
    while (capacity < needed) {
      capacity *= 2;
    }
    return makePopulation(geometry, capacity);
  };

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3(1, 1, 1);
  const rotation = new THREE.Quaternion();
  const color = new THREE.Color();
  const staffColor = new THREE.Color(style.staff);

  const place = (population: Population, index: number, xCm: number, yCm: number): void => {
    position.set(xCm * S, PERSON_CENTRE, yCm * S);
    matrix.compose(position, rotation, scale);
    population.mesh.setMatrixAt(index, matrix);
  };

  const update = (world: WorldView, live: boolean, frameWallMs: number): void => {
    // Dead-reckon ONLY on the live, playing head. A scrubbed view addresses a buffered
    // instant; advancing it by wall time would invent motion that never happened.
    const extraSimUs = deadReckonSimUs({
      live,
      state: world.state,
      speed: world.speed,
      wallElapsedMs: performance.now() - frameWallMs,
    });

    const chips = Object.values(world.patients);
    patients = grow(patients, chips.length, patientGeometry);
    const perNode = new Map<string, number>();
    let drawn = 0;
    for (const chip of chips) {
      if (chip.at_node === null) {
        continue;
      }
      const node = nodeIndex.get(chip.at_node);
      if (node === undefined) {
        continue;
      }
      const seen = perNode.get(chip.at_node) ?? 0;
      perNode.set(chip.at_node, seen + 1);
      const angle = (seen * Math.PI * 2) / FAN_PER_RING;
      const spread = seen === 0 ? 0 : FAN_RADIUS_CM + FAN_STEP_CM * Math.floor(seen / FAN_PER_RING);
      place(
        patients,
        drawn,
        node.x_cm + Math.cos(angle) * spread,
        node.y_cm + Math.sin(angle) * spread,
      );
      color.set(style.esi[chip.esi]);
      patients.mesh.setColorAt(drawn, color);
      drawn += 1;
    }
    patients.mesh.count = drawn;
    patients.mesh.instanceMatrix.needsUpdate = true;
    if (patients.mesh.instanceColor !== null) {
      patients.mesh.instanceColor.needsUpdate = true;
    }

    const kinematics = Object.values(world.staff);
    staff = grow(staff, kinematics.length, staffGeometry);
    let walking = 0;
    for (const kin of kinematics) {
      const totalUs = kin.edge === null ? 0 : (edgeUs.get(`${kin.edge[0]}>${kin.edge[1]}`) ?? 0);
      const point = kinematicPosition(kin, nodeIndex, totalUs, extraSimUs);
      if (point === null) {
        continue;
      }
      place(staff, walking, point.x_cm, point.y_cm);
      staff.mesh.setColorAt(walking, staffColor);
      walking += 1;
    }
    staff.mesh.count = walking;
    staff.mesh.instanceMatrix.needsUpdate = true;
    if (staff.mesh.instanceColor !== null) {
      staff.mesh.instanceColor.needsUpdate = true;
    }
  };

  return {
    root,
    update,
    dispose() {
      patients.mesh.dispose();
      staff.mesh.dispose();
      patientGeometry.dispose();
      staffGeometry.dispose();
      material.dispose();
    },
  };
}
