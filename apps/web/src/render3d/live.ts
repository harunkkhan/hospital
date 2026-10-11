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
 * **Posture is derived, not decorative.** A patient in a treatment bay is in the bed, not
 * standing beside it; a patient in the waiting hall is in a chair, not hovering over the
 * seat rows. `geometry.ts` owns both placements and `scene.ts` draws the same furniture from
 * them, so the body and the thing it rests on cannot disagree. Everyone else stands.
 *
 * Shape still carries meaning independently of colour, as it does in 2D: a patient's torso is
 * a smooth capsule, a staff member's is a six-sided prism that reads as faceted at any zoom.
 * Posture is a second non-colour channel on top — the recumbent bodies are the patients.
 *
 * Four instanced draws in total (a torso and a head per population), rewritten in place each
 * frame; the buffers only ever grow.
 */

import * as THREE from "three";

import type { FloorLayout, NodeId, RouteNode } from "../api/types";
import { deadReckonSimUs, indexNodes, kinematicPosition } from "../render/interpolate";
import type { WorldView } from "../state/streamReducer";
import type { FloorArchitecture } from "./derive";
import { HEIGHTS, SCALE, bedOf, centerX, centerY, seatsOf } from "./geometry";
import type { StylePack } from "./styles";

const S = SCALE;

/**
 * A body, in metres. Representational like every height here — the layout has no z.
 *
 * The base capsule is one metre tall and centred on the origin, so a posture is expressed as
 * a scale along its own axis plus a rotation. That keeps all three postures in ONE instanced
 * mesh instead of three.
 */
const BODY = {
  radius: 0.19,
  /** Cylinder length; total base height is this plus two radii, i.e. 1.0 m. */
  length: 0.62,
  headRadius: 0.115,
  standing: 1.34,
  seated: 0.74,
  lying: 1.55,
} as const;

const UP = new THREE.Vector3(0, 1, 0);

/** Chips fan out when several people share one node, so a busy queue is visibly busy. */
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
  torso: THREE.InstancedMesh;
  head: THREE.InstancedMesh;
  capacity: number;
}

/** Where a body rests when it is not simply standing on a node. */
interface Berth {
  readonly x: number;
  readonly y: number;
  /** Unit vector in plan from the foot of the bed toward the pillow. */
  readonly dx: number;
  readonly dy: number;
}

export function buildLiveLayer(
  layout: FloorLayout,
  arch: FloorArchitecture,
  style: StylePack,
): LiveLayer {
  const root = new THREE.Group();

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

  // The bed in each treatment bay, from the same placement `scene.ts` drew.
  const berthByNode = new Map<NodeId, Berth>();
  for (const bay of arch.bays) {
    const placement = bedOf(bay.rect, bay.doorSide);
    const length = Math.hypot(placement.toHead[0], placement.toHead[1]) || 1;
    berthByNode.set(bay.node, {
      x: centerX(placement.mattress),
      y: centerY(placement.mattress),
      dx: placement.toHead[0] / length,
      dy: placement.toHead[1] / length,
    });
  }

  // The waiting hall's chairs, in the order the queue takes them.
  const seats = arch.waiting === null ? [] : seatsOf(arch.waiting.rect);
  const seatedNode = arch.waiting?.node ?? null;

  const torsoGeometry = new THREE.CapsuleGeometry(BODY.radius, BODY.length, 4, 12);
  // Six radial segments read as a faceted prism rather than a cylinder, which is the shape
  // channel that keeps staff distinguishable from patients without relying on colour.
  const staffTorsoGeometry = new THREE.CylinderGeometry(
    BODY.radius * 0.92,
    BODY.radius * 1.06,
    1,
    6,
  );
  const headGeometry = new THREE.SphereGeometry(BODY.headRadius, 10, 8);
  // Lit, like the fit-out they stand among: an unshaded body is a flat lozenge from every
  // angle, which is most of why the old capsules read as chips rather than people.
  const material = new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.97 });

  const makePopulation = (torsoGeom: THREE.BufferGeometry, capacity: number): Population => {
    const torso = new THREE.InstancedMesh(torsoGeom, material, capacity);
    const head = new THREE.InstancedMesh(headGeometry, material, capacity);
    for (const mesh of [torso, head]) {
      mesh.count = 0;
      mesh.renderOrder = 20;
      mesh.frustumCulled = false;
      mesh.castShadow = true;
      root.add(mesh);
    }
    return { torso, head, capacity };
  };
  let patients = makePopulation(torsoGeometry, 128);
  let staff = makePopulation(staffTorsoGeometry, 64);

  const grow = (
    population: Population,
    needed: number,
    torsoGeom: THREE.BufferGeometry,
  ): Population => {
    if (needed <= population.capacity) {
      return population;
    }
    for (const mesh of [population.torso, population.head]) {
      root.remove(mesh);
      mesh.dispose();
    }
    // Double until it fits: a run that admits one more patient must not reallocate the whole
    // buffer on the next frame as well.
    let capacity = Math.max(1, population.capacity);
    while (capacity < needed) {
      capacity *= 2;
    }
    return makePopulation(torsoGeom, capacity);
  };

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3(1, 1, 1);
  const rotation = new THREE.Quaternion();
  const axis = new THREE.Vector3();
  const color = new THREE.Color();
  const staffColor = new THREE.Color(style.staff);
  const identity = new THREE.Quaternion();

  /** An upright body: torso scaled along its own axis, head sitting on top of it. */
  const standUpright = (
    population: Population,
    index: number,
    xCm: number,
    yCm: number,
    height: number,
    footY: number,
  ): void => {
    position.set(xCm * S, footY + height / 2, yCm * S);
    scale.set(1, height, 1);
    matrix.compose(position, identity, scale);
    population.torso.setMatrixAt(index, matrix);

    position.set(xCm * S, footY + height + BODY.headRadius * 0.55, yCm * S);
    scale.set(1, 1, 1);
    matrix.compose(position, identity, scale);
    population.head.setMatrixAt(index, matrix);
  };

  /** A body lying on a bed, its long axis along the mattress and its head at the pillow. */
  const lieDown = (population: Population, index: number, berth: Berth): void => {
    const restY = HEIGHTS.bed * S + BODY.radius * 0.8;
    axis.set(berth.dx, 0, berth.dy).normalize();
    rotation.setFromUnitVectors(UP, axis);

    position.set(berth.x * S, restY, berth.y * S);
    scale.set(0.92, BODY.lying, 0.92);
    matrix.compose(position, rotation, scale);
    population.torso.setMatrixAt(index, matrix);

    // The head rides just beyond the shoulder end, so the silhouette reads head-on-pillow.
    const reach = BODY.lying / 2 - BODY.radius * 0.5;
    position.set(berth.x * S + axis.x * reach, restY, berth.y * S + axis.z * reach);
    scale.set(1, 1, 1);
    matrix.compose(position, identity, scale);
    population.head.setMatrixAt(index, matrix);
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
    patients = grow(patients, chips.length, torsoGeometry);
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

      const berth = berthByNode.get(chip.at_node);
      if (berth !== undefined && seen === 0) {
        // The bay's occupant is in its bed. A second body at the same node is someone else's
        // problem — a visitor, or a hand-over — so it stands beside the bed rather than
        // sharing the mattress.
        lieDown(patients, drawn, berth);
      } else {
        const seat = chip.at_node === seatedNode ? seats[seen % Math.max(1, seats.length)] : undefined;
        const angle = (seen * Math.PI * 2) / FAN_PER_RING;
        const spread =
          seen === 0 ? 0 : FAN_RADIUS_CM + FAN_STEP_CM * Math.floor(seen / FAN_PER_RING);
        const x = seat === undefined ? node.x_cm + Math.cos(angle) * spread : seat[0];
        const y = seat === undefined ? node.y_cm + Math.sin(angle) * spread : seat[1];
        const height = seat === undefined ? BODY.standing : BODY.seated;
        const footY = seat === undefined ? 0 : HEIGHTS.chair * S;
        standUpright(patients, drawn, x, y, height, footY);
      }

      color.set(style.esi[chip.esi]);
      patients.torso.setColorAt(drawn, color);
      patients.head.setColorAt(drawn, color);
      drawn += 1;
    }
    patients.torso.count = drawn;
    patients.head.count = drawn;
    markDirty(patients);

    const kinematics = Object.values(world.staff);
    staff = grow(staff, kinematics.length, staffTorsoGeometry);
    let walking = 0;
    for (const kin of kinematics) {
      const totalUs = kin.edge === null ? 0 : (edgeUs.get(`${kin.edge[0]}>${kin.edge[1]}`) ?? 0);
      const point = kinematicPosition(kin, nodeIndex, totalUs, extraSimUs);
      if (point === null) {
        continue;
      }
      standUpright(staff, walking, point.x_cm, point.y_cm, BODY.standing, 0);
      staff.torso.setColorAt(walking, staffColor);
      staff.head.setColorAt(walking, staffColor);
      walking += 1;
    }
    staff.torso.count = walking;
    staff.head.count = walking;
    markDirty(staff);
  };

  return {
    root,
    update,
    dispose() {
      for (const population of [patients, staff]) {
        population.torso.dispose();
        population.head.dispose();
      }
      torsoGeometry.dispose();
      staffTorsoGeometry.dispose();
      headGeometry.dispose();
      material.dispose();
    },
  };
}

function markDirty(population: Population): void {
  for (const mesh of [population.torso, population.head]) {
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor !== null) {
      mesh.instanceColor.needsUpdate = true;
    }
  }
}
