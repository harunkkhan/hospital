/**
 * The three.js builders: derived architecture + a style pack in, a scene graph out.
 *
 * Plan coordinates are centimetres with +x east and +y south; the scene maps `x_cm` to
 * `scene.x` and `y_cm` to `scene.z`, with `scene.y` as height. The root group is translated
 * so the department's centre sits on the origin — the camera then orbits the origin and the
 * fit maths never has to know where the plate happens to be.
 *
 * **Everything is batched.** 76 bays of walls and fit-out plus ~40 support rooms would be
 * hundreds of meshes drawn one at a time; here line work merges into one `LineSegments` per
 * material, floor washes into one merged quad buffer per colour, and solids into one
 * `InstancedMesh` per material — a few dozen draw calls for the whole floor. The console
 * runs for hours, so every geometry, material and texture allocated here is tracked and
 * released by `dispose()`.
 *
 * The bay wash is an `InstancedMesh` on purpose: live status is then a per-instance colour
 * write on geometry that never changes, so a bay going from occupied to cleaning costs no
 * rebuild — and the same mesh is what the pointer raycasts against for selection.
 */

import * as THREE from "three";

import type { BayId, BayStatus } from "../api/types";
import type { FloorArchitecture } from "./derive";
import {
  BED,
  HEIGHTS,
  SCALE,
  bedOf,
  centerX,
  centerY,
  frameOf,
  rect,
  rectHeight,
  rectWidth,
  seatsOf,
  wallRuns,
  type Rect,
  type Side,
} from "./geometry";
import { LABEL_FONT, type StylePack, type TintKey } from "./styles";

const S = SCALE;

/**
 * Floor washes stack with millimetre offsets so coplanar surfaces never z-fight. Rooms and
 * circulation share a layer because they never overlap in plan.
 */
const LAYER = {
  plate: 0,
  dept: 0.004,
  circulation: 0.008,
  room: 0.008,
  zone: 0.013,
  status: 0.018,
  marking: 0.024,
} as const;

/** Representational heights the layout cannot supply, beyond the shared ones. */
const GANTRY_HEIGHT = 190;
const BOOM_HEIGHT = 232;

export interface SceneOptions {
  /** The camera's vertical field of view, degrees — labels are sized against it. */
  readonly vfovDeg: number;
}

export interface FloorScene {
  readonly root: THREE.Group;
  /** The bay status wash: one instance per bay, and the pointer's pick target. */
  readonly bayMesh: THREE.InstancedMesh;
  /** Instance index -> bay, so a raycast hit names a bay. */
  readonly bayOrder: readonly BayId[];
  /** Repaint the wash from the live world. Cheap; safe to call every frame. */
  paintBays(statusOf: (bay: BayId) => BayStatus): void;
  /** Cage the selected bay, or clear the selection with `null`. */
  setSelectedBay(bay: BayId | null): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

/** Line work accumulated into one buffer, emitted as a single `LineSegments`. */
class LineBatch {
  private readonly points: number[] = [];

  segment(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): void {
    this.points.push(x0, y0, z0, x1, y1, z1);
  }

  /** A rect's outline at height `h`, in scene units. */
  ring(r: Rect, h: number): void {
    const a = r.x0 * S;
    const b = r.x1 * S;
    const c = r.y0 * S;
    const d = r.y1 * S;
    this.segment(a, h, c, b, h, c);
    this.segment(b, h, c, b, h, d);
    this.segment(b, h, d, a, h, d);
    this.segment(a, h, d, a, h, c);
  }

  emit(parent: THREE.Object3D, material: THREE.Material, track: Tracker): void {
    if (this.points.length === 0) {
      return;
    }
    const geometry = track(new THREE.BufferGeometry());
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(this.points, 3));
    const lines = new THREE.LineSegments(geometry, material);
    lines.renderOrder = 14;
    parent.add(lines);
  }
}

/** Horizontal quads merged into one buffer — the floor washes. */
class QuadBatch {
  private readonly points: number[] = [];

  add(r: Rect, h: number): void {
    const a = r.x0 * S;
    const b = r.x1 * S;
    const c = r.y0 * S;
    const d = r.y1 * S;
    this.points.push(a, h, c, a, h, d, b, h, d, a, h, c, b, h, d, b, h, c);
  }

  emit(parent: THREE.Object3D, material: THREE.Material, order: number, track: Tracker): void {
    if (this.points.length === 0) {
      return;
    }
    const geometry = track(new THREE.BufferGeometry());
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(this.points, 3));
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = order;
    parent.add(mesh);
  }
}

/** Axis-aligned boxes drawn as one `InstancedMesh`. */
class BoxBatch {
  private readonly boxes: (readonly [number, number, number, number, number, number])[] = [];

  /** `r` in plan cm, `bottom`/`top` in cm above the floor. */
  box(r: Rect, bottom: number, top: number): void {
    this.boxes.push([
      centerX(r) * S,
      ((bottom + top) / 2) * S,
      centerY(r) * S,
      Math.abs(rectWidth(r)) * S,
      Math.abs(top - bottom) * S,
      Math.abs(rectHeight(r)) * S,
    ]);
  }

  emit(
    parent: THREE.Object3D,
    material: THREE.Material,
    track: Tracker,
    castShadow = false,
  ): void {
    if (this.boxes.length === 0) {
      return;
    }
    const geometry = track(new THREE.BoxGeometry(1, 1, 1));
    const mesh = new THREE.InstancedMesh(geometry, material, this.boxes.length);
    mesh.castShadow = castShadow;
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    const rotation = new THREE.Quaternion();
    this.boxes.forEach(([x, y, z, sx, sy, sz], i) => {
      position.set(x, y, z);
      scale.set(Math.max(sx, 1e-4), Math.max(sy, 1e-4), Math.max(sz, 1e-4));
      matrix.compose(position, rotation, scale);
      mesh.setMatrixAt(i, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    // Solids sit above every floor wash: the transparent, depth-write-free slabs would
    // otherwise paint straight over them.
    mesh.renderOrder = 10;
    parent.add(mesh);
  }
}

interface Disposable {
  dispose(): void;
}
type Tracker = <T extends Disposable>(item: T) => T;

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

export function buildFloorScene(
  arch: FloorArchitecture,
  style: StylePack,
  options: SceneOptions,
): FloorScene {
  const disposables: Disposable[] = [];
  const track: Tracker = (item) => {
    disposables.push(item);
    return item;
  };

  const root = new THREE.Group();
  root.position.set(-centerX(arch.dept) * S, 0, -centerY(arch.dept) * S);

  // Two lights and a shadow catcher — the whole lighting rig.
  //
  // The intensities are chosen so an up-facing white surface still reads at the pack's own
  // value: this is meant to give the solids form, not to relight the drawing. The key comes
  // from the north-west because that is where the plan's own labels and leaders sit, so the
  // shadows fall away from the text rather than under it.
  const sky = track(new THREE.HemisphereLight(0xffffff, 0xbfc6cf, 1.6));
  root.add(sky);
  const key = track(new THREE.DirectionalLight(0xffffff, 1.15));
  const spanX = rectWidth(arch.dept) * S;
  const spanZ = rectHeight(arch.dept) * S;
  const reach = Math.max(spanX, spanZ);
  key.position.set(centerX(arch.dept) * S - spanX * 0.5, reach * 0.9, centerY(arch.dept) * S - spanZ * 0.6);
  key.target.position.set(centerX(arch.dept) * S, 0, centerY(arch.dept) * S);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  const half = reach * 0.62;
  Object.assign(key.shadow.camera, {
    left: -half,
    right: half,
    top: half,
    bottom: -half,
    near: 1,
    far: reach * 2.4,
  });
  key.shadow.bias = -0.0009;
  key.shadow.normalBias = 0.02;
  root.add(key);
  root.add(key.target);

  // Shadows land on a ShadowMaterial plane rather than on the floor washes themselves. The
  // washes are stacked, translucent and depth-write-free — a receiver has to be opaque to the
  // depth buffer, and making them so would change the palette. This catches the shadow and is
  // otherwise invisible.
  const catcher = new THREE.Mesh(
    track(new THREE.PlaneGeometry(spanX, spanZ)),
    track(new THREE.ShadowMaterial({ opacity: 0.17 })),
  );
  catcher.rotation.x = -Math.PI / 2;
  catcher.position.set(centerX(arch.dept) * S, 0.03, centerY(arch.dept) * S);
  catcher.receiveShadow = true;
  catcher.renderOrder = 6;
  root.add(catcher);

  const fill = (color: string, opacity: number): THREE.Material =>
    track(
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
  // Volumes are LIT; the flat floor washes are not.
  //
  // A hairline drawing whose only solids are unshaded reads as a diagram — every box the same
  // value from every angle, which is exactly the blandness. Lambert costs one light pass and
  // gives the fit-out a top-versus-side falloff, so a bed reads as a bed. The stacked,
  // depth-write-free floor washes stay unlit on purpose: shading them would tint the palette
  // the pack has already chosen, and they are horizontal, so there is nothing to shade.
  const solid = (color: string, opacity: number): THREE.Material =>
    track(
      new THREE.MeshLambertMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        side: THREE.DoubleSide,
      }),
    );
  const stroke = (color: string, opacity: number): THREE.Material =>
    track(new THREE.LineBasicMaterial({ color, transparent: true, opacity }));

  // =========================================================================
  // 1. Plate and department envelope
  // =========================================================================
  const plate = new QuadBatch();
  plate.add(arch.plate, LAYER.plate);
  plate.emit(root, fill(style.plateFill, style.slabOp), 0, track);

  const dept = new QuadBatch();
  dept.add(arch.dept, LAYER.dept);
  dept.emit(root, fill(style.deptFill, style.slabOp), 1, track);

  // A metre grid over the plate: it gives the drawing a scale without a single label.
  const grid = new LineBatch();
  for (let x = 0; x <= arch.plate.x1; x += 1000) {
    grid.segment(x * S, 0.003, 0, x * S, 0.003, arch.plate.y1 * S);
  }
  for (let y = 0; y <= arch.plate.y1; y += 1000) {
    grid.segment(0, 0.003, y * S, arch.plate.x1 * S, 0.003, y * S);
  }
  grid.emit(root, stroke(style.ink, style.gridOp), track);

  const plateEdge = new LineBatch();
  plateEdge.ring(arch.plate, 0.002);
  plateEdge.emit(root, stroke(style.inkSoft, 0.55), track);

  // =========================================================================
  // 2. Circulation — the corridor grid the whole plan hangs off
  // =========================================================================
  const circulation = new QuadBatch();
  for (const corridor of arch.circulation) {
    circulation.add(corridor, LAYER.circulation);
  }
  circulation.emit(root, fill(style.corridorFill, 0.95), 2, track);

  // A dashed centre line down the main spine, the way a real department marks its route.
  const marking = new LineBatch();
  const spineZ = arch.spineY * S;
  for (let x = arch.spine.x0; x < arch.spine.x1; x += 240) {
    marking.segment(x * S, LAYER.marking, spineZ, Math.min(x + 130, arch.spine.x1) * S, LAYER.marking, spineZ);
  }

  // =========================================================================
  // 3. Rooms — floors, zone tint, walls
  // =========================================================================
  const roomFloors = new QuadBatch();
  const tints = new Map<string, QuadBatch>();
  const tintOf = (key: TintKey): string | undefined => style.zoneTint[key];
  const addTint = (color: string, r: Rect, h: number): void => {
    let batch = tints.get(color);
    if (batch === undefined) {
      batch = new QuadBatch();
      tints.set(color, batch);
    }
    batch.add(r, h);
  };

  const walls = new LineBatch();
  const drawWalls = (r: Rect, doorSide: Side | "none", doorFrac: number, height: number): void => {
    for (const [ax, ay, bx, by] of wallRuns(r, doorSide, doorFrac)) {
      if (Math.abs(bx - ax) < 1 && Math.abs(by - ay) < 1) {
        continue;
      }
      // A wall drawn as its four corner lines: base, head, and the two jambs. Cheaper than a
      // solid by an order of magnitude, and it is what makes the pack read as a drawing.
      walls.segment(ax * S, 0.01, ay * S, bx * S, 0.01, by * S);
      walls.segment(ax * S, height * S, ay * S, bx * S, height * S, by * S);
      walls.segment(ax * S, 0.01, ay * S, ax * S, height * S, ay * S);
      walls.segment(bx * S, 0.01, by * S, bx * S, height * S, by * S);
    }
  };

  const envelope = new LineBatch();
  for (const [ax, ay, bx, by] of wallRuns(arch.dept, "none", 0)) {
    envelope.segment(ax * S, 0.01, ay * S, bx * S, 0.01, by * S);
    envelope.segment(ax * S, HEIGHTS.perimeter * S, ay * S, bx * S, HEIGHTS.perimeter * S, by * S);
    envelope.segment(ax * S, 0.01, ay * S, ax * S, HEIGHTS.perimeter * S, ay * S);
  }
  envelope.emit(root, stroke(style.inkStrong, Math.min(1, style.wall.edgeOp + 0.2)), track);

  for (const room of arch.rooms) {
    roomFloors.add(room.rect, LAYER.room);
    // A bay floor carries live status instead of a zone hue: two washes over one surface
    // turn every bay the same muddy grey. Bay zone identity lives on the pod corridor.
    if (room.kind !== "bay") {
      const tint = tintOf(room.kind);
      if (tint !== undefined) {
        addTint(tint, room.rect, LAYER.zone);
      }
    }
    drawWalls(room.rect, room.doorSide, room.kind === "bay" ? 0.6 : 0.3, HEIGHTS.partition);
  }
  roomFloors.emit(root, fill(style.roomFill, style.room.fillOp), 3, track);

  // Pod corridors are banded in their team's zone colour — real departments do exactly this,
  // and it frees the bay floor to carry status alone.
  for (const pod of arch.pods) {
    const tint = tintOf(pod.zoneType);
    if (tint !== undefined) {
      addTint(tint, pod.corridor, LAYER.marking);
    }
  }
  for (const [color, batch] of tints) {
    batch.emit(root, fill(color, style.zoneTintOp), 4, track);
  }

  // A curtain track across every bay opening; isolation bays are glazed full height instead.
  const track3 = new LineBatch();
  for (const bay of arch.bays) {
    const r = bay.rect;
    const vertical = bay.doorSide === "e" || bay.doorSide === "w";
    const dx = (bay.doorSide === "e" ? r.x1 : r.x0) * S;
    const dy = (bay.doorSide === "s" ? r.y1 : r.y0) * S;
    const ax = vertical ? dx : r.x0 * S;
    const az = vertical ? r.y0 * S : dy;
    const bx = vertical ? dx : r.x1 * S;
    const bz = vertical ? r.y1 * S : dy;
    track3.segment(ax, HEIGHTS.curtainTrack * S, az, bx, HEIGHTS.curtainTrack * S, bz);
    if (bay.isolation) {
      const h = HEIGHTS.partition * S;
      track3.segment(ax, h, az, bx, h, bz);
      track3.segment((ax + bx) / 2, 0.01, (az + bz) / 2, (ax + bx) / 2, h, (az + bz) / 2);
    }
  }
  track3.emit(root, stroke(style.ink, style.wall.edgeOp * 0.6), track);
  walls.emit(root, stroke(style.ink, style.wall.edgeOp), track);

  // =========================================================================
  // 4. Fit-out — the layer that makes it a hospital and not a warehouse
  // =========================================================================
  const furniture = new BoxBatch();
  const furnitureLight = new BoxBatch();
  const furnitureEdge = new LineBatch();

  // A bed in every bay, head to the wall opposite the door, with a headwall services panel,
  // a bedside monitor pole and its trolley. All placed in the room's own (u, v) frame, so
  // one description serves a bay opening east, west, north or south.
  for (const bay of arch.bays) {
    const f = frameOf(bay.rect, bay.doorSide);
    const u = f.W / 2;
    const head = f.D - 45;
    // `bedOf` is shared with the live layer, which lays the occupant on this exact mattress.
    const placement = bedOf(bay.rect, bay.doorSide);
    const bedLength = Math.min(BED.maxLength, f.D - BED.clearance);
    furnitureLight.box(placement.mattress, HEIGHTS.bed - 18, HEIGHTS.bed);
    furnitureEdge.ring(placement.mattress, HEIGHTS.bed * S);
    furnitureLight.box(placement.pillow, HEIGHTS.bed, HEIGHTS.bed + 12);
    furniture.box(f.box(u - 70, f.D - 22, u + 70, f.D - 8), 90, 150);
    furniture.box(f.box(u + 52, head - 22, u + 72, head - 2), 0, 130);
    furniture.box(f.box(u + 38, head - 20, u + 86, head - 4), 130, 165);
    if (bay.zoneType === "resus_trauma") {
      // An overhead equipment boom, which is what a resus room has and a cubicle does not.
      const mid = head - bedLength / 2;
      furniture.box(f.box(u - 130, mid - 8, u + 130, mid + 8), BOOM_HEIGHT, BOOM_HEIGHT + 14);
      furniture.box(f.box(u - 8, mid - 90, u + 8, mid + 90), BOOM_HEIGHT, BOOM_HEIGHT + 14);
    }
  }

  // Team stations in the pod corridors, with monitors along the counter and two stools.
  for (const desk of arch.stations) {
    const r = desk.rect;
    furniture.box(r, 0, HEIGHTS.counter);
    furnitureEdge.ring(r, HEIGHTS.counter * S);
    for (let i = 0; i < 3; i += 1) {
      const mx = r.x0 + ((i + 0.5) * rectWidth(r)) / 3;
      furnitureLight.box(
        rect(mx - 24, r.y1 - 24, mx + 24, r.y1 - 14),
        HEIGHTS.counter,
        HEIGHTS.counter + 40,
      );
    }
    for (let i = 0; i < 2; i += 1) {
      const sx = r.x0 + ((i + 0.5) * rectWidth(r)) / 2;
      furniture.box(rect(sx - 15, r.y1 + 34, sx + 15, r.y1 + 64), 0, 50);
    }
  }

  // Imaging: a bore ring and its table. One torus does more work than any other object here,
  // so all of them are instanced off a single geometry.
  const bores: THREE.Matrix4[] = [];
  const boreQuaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
  for (const room of arch.wing) {
    const f = frameOf(room.rect, room.doorSide);
    const u = f.W / 2;
    if (room.kind === "imaging") {
      const gantry = f.box(u - 95, f.D - 300, u + 95, f.D - 110);
      bores.push(
        new THREE.Matrix4().compose(
          new THREE.Vector3(centerX(gantry) * S, GANTRY_HEIGHT * S * 0.55, centerY(gantry) * S),
          boreQuaternion,
          new THREE.Vector3(1, 1, 1),
        ),
      );
      furnitureLight.box(f.box(u - 38, 120, u + 38, f.D - 150), 62, 80);
      furniture.box(f.box(u - 24, 120, u + 24, 220), 0, 62);
      furniture.box(f.box(30, 40, 170, 110), 0, 95);
    } else {
      furniture.box(f.box(20, 90, 90, f.D - 60), 0, 92);
      furniture.box(f.box(f.W - 90, 90, f.W - 20, f.D - 60), 0, 92);
      for (let i = 0; i < 3; i += 1) {
        furniture.box(f.box(u - 52, 160 + i * 180, u + 52, 260 + i * 180), 0, 126);
      }
    }
  }
  if (bores.length > 0) {
    const geometry = track(new THREE.TorusGeometry(0.95, 0.3, 12, 26));
    const mesh = new THREE.InstancedMesh(geometry, fill(style.ink, style.furnitureOp * 1.3), bores.length);
    bores.forEach((matrix, i) => mesh.setMatrixAt(i, matrix));
    mesh.instanceMatrix.needsUpdate = true;
    mesh.renderOrder = 10;
    root.add(mesh);
  }

  // Triage rooms: desk, chair, exam couch.
  for (const room of arch.triage) {
    const f = frameOf(room.rect, room.doorSide);
    furniture.box(f.box(35, f.D - 190, 185, f.D - 120), 0, 74);
    furniture.box(f.box(80, f.D - 110, 130, f.D - 60), 0, 48);
    furnitureLight.box(f.box(f.W - 130, 80, f.W - 40, 280), 0, HEIGHTS.bed);
  }

  // The waiting hall: rows of seating and a reception desk, sized to the hall it is given.
  if (arch.waiting !== null) {
    const r = arch.waiting.rect;
    // `seatsOf` is shared with the live layer, which sits the queue on these exact chairs.
    for (const [x, y] of seatsOf(r)) {
      furniture.box(rect(x - 22, y - 22, x + 22, y + 22), 0, HEIGHTS.chair);
      furniture.box(rect(x - 22, y + 16, x + 22, y + 24), HEIGHTS.chair, HEIGHTS.chair + 42);
    }
    furniture.box(rect(r.x0 + 70, r.y0 + 90, r.x0 + 300, r.y0 + 210), 0, 108);
  }

  // Support rooms get a single bench each — enough to read as fitted out at any zoom.
  for (const room of arch.support) {
    const f = frameOf(room.rect, room.doorSide);
    if (f.W < 200 || f.D < 160) {
      continue;
    }
    furniture.box(f.box(24, f.D - 80, f.W - 24, f.D - 24), 0, 88);
  }

  furniture.emit(root, solid(style.ink, style.furnitureOp), track, true);
  furnitureLight.emit(root, solid(style.ink, style.furnitureOp * 0.7), track, true);
  furnitureEdge.emit(root, stroke(style.ink, style.furnitureEdgeOp), track);

  // The ambulance apron, marked with chevrons the way a real one is.
  if (arch.ambulance !== null) {
    const r = arch.ambulance.rect;
    const chevrons = new LineBatch();
    for (let i = 0; i < 5; i += 1) {
      const y = r.y0 + 70 + i * 130;
      if (y + 90 > r.y1) {
        break;
      }
      chevrons.segment(r.x0 * S + 0.3, LAYER.marking, y * S, centerX(r) * S, LAYER.marking, (y + 90) * S);
      chevrons.segment(centerX(r) * S, LAYER.marking, (y + 90) * S, r.x1 * S - 0.3, LAYER.marking, y * S);
    }
    chevrons.emit(root, stroke(style.status.closed, 0.5), track);
  }

  // =========================================================================
  // 5. The bay wash — live status, and the pointer's pick target
  // =========================================================================
  const bayOrder: BayId[] = [];
  const bayGeometry = track(new THREE.PlaneGeometry(1, 1));
  const bayMaterial = track(
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: style.statusOp,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  const bayMesh = new THREE.InstancedMesh(bayGeometry, bayMaterial, Math.max(1, arch.bays.length));
  bayMesh.renderOrder = 5;
  bayMesh.count = arch.bays.length;
  {
    const matrix = new THREE.Matrix4();
    const flat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    arch.bays.forEach((bay, i) => {
      bayOrder.push(bay.id);
      matrix.compose(
        new THREE.Vector3(centerX(bay.rect) * S, LAYER.status, centerY(bay.rect) * S),
        flat,
        new THREE.Vector3(rectWidth(bay.rect) * S, rectHeight(bay.rect) * S, 1),
      );
      bayMesh.setMatrixAt(i, matrix);
    });
    bayMesh.instanceMatrix.needsUpdate = true;
  }
  root.add(bayMesh);

  const statusColors = new Map<BayStatus, THREE.Color>();
  for (const key of ["free", "occupied", "cleaning", "closed"] as const) {
    statusColors.set(key, new THREE.Color(style.status[key]));
  }
  const fallbackColor = new THREE.Color(style.status.free);
  const paintBays = (statusOf: (bay: BayId) => BayStatus): void => {
    bayOrder.forEach((id, i) => {
      bayMesh.setColorAt(i, statusColors.get(statusOf(id)) ?? fallbackColor);
    });
    if (bayMesh.instanceColor !== null) {
      bayMesh.instanceColor.needsUpdate = true;
    }
  };
  paintBays(() => "free");

  // The selection is a cage around the whole room rather than a brighter floor: status
  // already owns the floor colour, and an operator has to be able to see what is selected
  // from any of the four camera presets, including from inside the corridor. Depth testing is
  // off so the cage is never buried by the bed standing in it.
  const selectionBox = new THREE.BoxGeometry(1, 1, 1);
  const selectionGeometry = track(new THREE.EdgesGeometry(selectionBox));
  selectionBox.dispose();
  const selection = new THREE.LineSegments(
    selectionGeometry,
    track(
      new THREE.LineBasicMaterial({
        color: style.selection,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      }),
    ),
  );
  selection.renderOrder = 45;
  selection.visible = false;
  root.add(selection);

  const bayRects = new Map<BayId, Rect>(arch.bays.map((bay) => [bay.id, bay.rect]));
  const setSelectedBay = (bay: BayId | null): void => {
    const r = bay === null ? undefined : bayRects.get(bay);
    if (r === undefined) {
      selection.visible = false;
      return;
    }
    selection.visible = true;
    selection.position.set(centerX(r) * S, (HEIGHTS.partition * S) / 2, centerY(r) * S);
    selection.scale.set(rectWidth(r) * S, HEIGHTS.partition * S, rectHeight(r) * S);
  };

  // =========================================================================
  // 6. Labels — signage, sized in SCREEN space
  // =========================================================================
  const leaders = new LineBatch();
  const textures = new Map<string, { texture: THREE.Texture; aspect: number }>();

  /**
   * Sprites, not floor decals: a decal reads mirrored the moment the camera swings past 90
   * degrees, and this view orbits freely.
   *
   * They are also sized in SCREEN space rather than world space. A world-sized label is
   * correct in plan but grows without limit as the camera walks into a corridor — one pod
   * name ends up filling the frame. With `sizeAttenuation` off three multiplies the sprite
   * scale by view depth, so on-screen height is `scale / (2 * tan(vfov / 2))`; solving for
   * scale makes `sizeFrac` a constant fraction of viewport height at every distance.
   */
  const vfovTan = Math.tan(((options.vfovDeg * Math.PI) / 180) / 2);
  const label = (
    text: string,
    x: number,
    z: number,
    height: number,
    sizeFrac: number,
    tracking = 3,
  ): void => {
    const key = `${text}|${tracking}`;
    let entry = textures.get(key);
    if (entry === undefined) {
      entry = makeTextTexture(text, style.label, tracking, track);
      textures.set(key, entry);
    }
    const material = track(
      new THREE.SpriteMaterial({
        map: entry.texture,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        sizeAttenuation: false,
      }),
    );
    const sprite = new THREE.Sprite(material);
    const sy = sizeFrac * 2 * vfovTan;
    sprite.scale.set(sy * entry.aspect, sy, 1);
    sprite.position.set(x * S, height, z * S);
    sprite.renderOrder = 40;
    root.add(sprite);
  };

  // Pod names sit in the main spine at each pod's mouth, north pods above the centre line and
  // south pods below. Over the pod itself the text would land on beds and status colour; the
  // spine is the one surface that stays empty, and the result reads as a corridor directory.
  for (const pod of arch.pods) {
    const z = arch.spineY + (pod.band === "n" ? -215 : 215);
    label(`${pod.zoneType.replace(/_/g, " ")} · ${pod.count}`, centerX(pod.corridor), z, 4.2, 0.021);
    leaders.segment(
      centerX(pod.corridor) * S,
      0.02,
      (pod.band === "n" ? arch.spine.y0 : arch.spine.y1) * S,
      centerX(pod.corridor) * S,
      0.02,
      z * S,
    );
  }

  const named: (readonly [string, number, number])[] = [];
  if (arch.waiting !== null) {
    named.push(["Waiting", centerX(arch.waiting.rect), centerY(arch.waiting.rect)]);
  }
  if (arch.ambulance !== null) {
    named.push(["Ambulance", centerX(arch.ambulance.rect), arch.ambulance.rect.y0 - 250]);
  }
  const firstTriage = arch.triage[0];
  const lastTriage = arch.triage[arch.triage.length - 1];
  if (firstTriage !== undefined && lastTriage !== undefined) {
    named.push([
      "Triage",
      (firstTriage.rect.x0 + lastTriage.rect.x1) / 2,
      firstTriage.rect.y1 + 200,
    ]);
  }
  if (arch.wing.length > 0) {
    named.push(["Imaging · Lab", centerX(arch.wingBlock), arch.wingBlock.y1 + 200]);
  }
  for (const [text, x, z] of named) {
    label(text, x, z, 3.2, 0.024);
  }

  // Drawing furniture — a scale bar and a north point — in the west margin, the one place the
  // department leaves free.
  const marginX = arch.dept.x0 / 2;
  const barY0 = centerY(arch.dept) - 500;
  const barY1 = barY0 + 1000;
  leaders.segment(marginX * S, 0.02, barY0 * S, marginX * S, 0.02, barY1 * S);
  for (let i = 0; i <= 4; i += 1) {
    const y = barY0 + (i * (barY1 - barY0)) / 4;
    leaders.segment((marginX - 55) * S, 0.02, y * S, (marginX + 55) * S, 0.02, y * S);
  }
  label("10 m", marginX, barY1 + 320, 0.4, 0.016, 2);

  const northY = arch.dept.y0 + 900;
  leaders.segment(marginX * S, 0.02, (northY + 330) * S, marginX * S, 0.02, (northY - 330) * S);
  leaders.segment((marginX - 125) * S, 0.02, (northY - 50) * S, marginX * S, 0.02, (northY - 330) * S);
  leaders.segment((marginX + 125) * S, 0.02, (northY - 50) * S, marginX * S, 0.02, (northY - 330) * S);
  label("N", marginX, northY + 620, 0.4, 0.015, 1);

  leaders.emit(root, stroke(style.ink, 0.34), track);
  marking.emit(root, stroke(style.ink, 0.28), track);

  return {
    root,
    bayMesh,
    bayOrder,
    paintBays,
    setSelectedBay,
    dispose() {
      for (const item of disposables) {
        item.dispose();
      }
      bayGeometry.dispose();
      bayMaterial.dispose();
      bayMesh.dispose();
    },
  };
}

/**
 * A label rendered to a canvas texture in Times New Roman, matching the view's own chrome.
 *
 * The canvas is sized to the MEASURED text so cap height stays constant however long the
 * label is — a fixed canvas would either clip a long pod name or shrink it against its
 * neighbours — and the sprite is given the resulting aspect so nothing is stretched.
 */
function makeTextTexture(
  text: string,
  color: string,
  tracking: number,
  track: Tracker,
): { texture: THREE.Texture; aspect: number } {
  const height = 128;
  const font = `400 66px ${LABEL_FONT}`;
  const measure = document.createElement("canvas").getContext("2d");
  let width = height;
  if (measure !== null) {
    measure.font = font;
    measure.letterSpacing = `${tracking}px`;
    width = Math.max(height, Math.ceil(measure.measureText(text).width) + 52);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (ctx !== null) {
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.letterSpacing = `${tracking}px`;
    ctx.fillStyle = color;
    ctx.fillText(text, width / 2, height / 2);
  }
  const texture = track(new THREE.CanvasTexture(canvas));
  texture.anisotropy = 8;
  return { texture, aspect: width / height };
}
