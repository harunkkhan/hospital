/**
 * Camera framing for the 3D floor — pure maths, no three.js.
 *
 * The renderer only ever asks this module two things: which way is the camera looking, and
 * how far back does it have to stand. Keeping both here means the framing can be tested
 * without a WebGL context, which is the only way an "is the whole department actually in
 * shot?" assertion can exist at all.
 *
 * Convention matches the scene: +x east, +z south, +y up, and the department centred on the
 * origin with its floor at y = 0. Azimuth is measured from +z toward +x, elevation up from
 * the floor plane — so (elevation 90°) is straight down and (elevation 0°) is eye level.
 */

export type Vec3 = readonly [number, number, number];

/**
 * Vertical field of view, degrees. The fit maths and the screen-space label sizing must
 * agree on this number or labels drift as the camera moves, so it lives here once.
 */
export const VFOV_DEG = 34;

export type PresetId = "isometric" | "plan" | "corridor" | "approach";

export interface CameraPreset {
  readonly id: PresetId;
  readonly label: string;
  readonly elevationDeg: number;
  readonly azimuthDeg: number;
  /**
   * Fraction of the fitted distance actually used. Above 1 leaves air around the department;
   * well below 1 deliberately puts the camera INSIDE it, which is the whole point of the
   * corridor view.
   */
  readonly margin: number;
  /** Height of the orbit target above the floor, metres. */
  readonly targetHeight: number;
}

/**
 * The department is near-square once re-planned, so a true isometric frames it. The
 * near-frontal angle the generator's 108 m ribbon would have needed reads as an elevation
 * and hides the corridor grid entirely.
 */
const ISOMETRIC: CameraPreset = {
  id: "isometric",
  label: "Isometric",
  elevationDeg: 34,
  azimuthDeg: 31,
  margin: 0.99,
  targetHeight: 1,
};

export const CAMERA_PRESETS: readonly CameraPreset[] = [
  ISOMETRIC,
  // Just off vertical: at exactly 90° the camera basis is undefined, and the last two degrees
  // are what keep the wall heights legible as walls.
  { id: "plan", label: "Plan", elevationDeg: 88, azimuthDeg: 0, margin: 1.03, targetHeight: 1 },
  { id: "corridor", label: "Corridor", elevationDeg: 9, azimuthDeg: 86, margin: 0.36, targetHeight: 1.4 },
  { id: "approach", label: "Approach", elevationDeg: 23, azimuthDeg: 149, margin: 0.95, targetHeight: 1 },
];

export const DEFAULT_PRESET: PresetId = ISOMETRIC.id;

export function presetById(id: PresetId): CameraPreset {
  return CAMERA_PRESETS.find((preset) => preset.id === id) ?? ISOMETRIC;
}

/** The department's bounding half-extents in metres, floor at y = 0. */
export interface FloorBox {
  readonly halfWidth: number;
  readonly halfDepth: number;
  readonly height: number;
}

/** Orbit state: where the camera is looking and from how far. */
export interface Orbit {
  readonly azimuth: number;
  readonly elevation: number;
  readonly distance: number;
  readonly target: Vec3;
}

// Elevation stops just short of vertical: at exactly 90° the view direction is parallel to
// the world up vector and the camera basis is undefined.
export const MIN_ELEVATION = 0.03;
export const MAX_ELEVATION = 1.55;
export const MIN_DISTANCE = 3;
export const MAX_DISTANCE = 1400;

export const clampElevation = (value: number): number =>
  Math.min(MAX_ELEVATION, Math.max(MIN_ELEVATION, value));
export const clampDistance = (value: number): number =>
  Math.min(MAX_DISTANCE, Math.max(MIN_DISTANCE, value));

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  return length === 0 ? [0, 0, 1] : [v[0] / length, v[1] / length, v[2] / length];
}

/** Unit vector FROM the target TOWARD the camera. */
export function directionFrom(elevationDeg: number, azimuthDeg: number): Vec3 {
  const phi = (elevationDeg * Math.PI) / 180;
  const theta = (azimuthDeg * Math.PI) / 180;
  return [Math.cos(phi) * Math.sin(theta), Math.sin(phi), Math.cos(phi) * Math.cos(theta)];
}

/** The camera basis looking back down `direction` at the target. */
export function cameraBasis(direction: Vec3): { right: Vec3; up: Vec3; forward: Vec3 } {
  const forward: Vec3 = [-direction[0], -direction[1], -direction[2]];
  // Looking straight down, forward is parallel to world up and the cross product vanishes;
  // fall back to a fixed reference so a plan view still has a defined horizon.
  const reference: Vec3 = Math.abs(forward[1]) > 0.9999 ? [0, 0, -1] : [0, 1, 0];
  const right = normalize(cross(forward, reference));
  return { right, up: cross(right, forward), forward };
}

/**
 * The smallest distance along `direction` that keeps every corner of the box inside both the
 * vertical and the aspect-corrected horizontal frustum.
 *
 * For a corner `q` measured from the orbit target, the camera sees it at depth
 * `distance - q·direction`, offset `q·right` across and `q·up` up. Requiring
 * `|q·right| <= tanH * depth` and `|q·up| <= tanV * depth` and solving each for distance
 * gives the two candidate bounds below; the answer is the largest over all eight corners.
 * Doing it per corner rather than on a bounding sphere is what stops a wide, shallow
 * department from being framed as if it were a cube.
 */
export function fitDistance(
  box: FloorBox,
  direction: Vec3,
  aspect: number,
  vfovDeg: number,
  targetHeight: number,
  margin = 1,
): number {
  const { right, up } = cameraBasis(direction);
  const tanV = Math.tan(((vfovDeg * Math.PI) / 180) / 2);
  const tanH = tanV * Math.max(aspect, 1e-3);
  let distance = 0;
  for (const x of [-box.halfWidth, box.halfWidth]) {
    for (const y of [0 - targetHeight, box.height - targetHeight]) {
      for (const z of [-box.halfDepth, box.halfDepth]) {
        const q: Vec3 = [x, y, z];
        const along = dot(q, direction);
        distance = Math.max(
          distance,
          along + Math.abs(dot(q, right)) / tanH,
          along + Math.abs(dot(q, up)) / tanV,
        );
      }
    }
  }
  return distance * margin;
}

/** Where the camera sits for an orbit state. */
export function orbitEye(orbit: Orbit): Vec3 {
  const ce = Math.cos(orbit.elevation);
  const se = Math.sin(orbit.elevation);
  return [
    orbit.target[0] + orbit.distance * ce * Math.sin(orbit.azimuth),
    orbit.target[1] + orbit.distance * se,
    orbit.target[2] + orbit.distance * ce * Math.cos(orbit.azimuth),
  ];
}

/** The orbit state a preset asks for, fitted to this box and viewport. */
export function orbitForPreset(
  preset: CameraPreset,
  box: FloorBox,
  aspect: number,
  vfovDeg: number,
): Orbit {
  const direction = directionFrom(preset.elevationDeg, preset.azimuthDeg);
  return {
    azimuth: Math.atan2(direction[0], direction[2]),
    elevation: clampElevation(Math.asin(direction[1])),
    distance: clampDistance(
      fitDistance(box, direction, aspect, vfovDeg, preset.targetHeight, preset.margin),
    ),
    target: [0, preset.targetHeight, 0],
  };
}

/**
 * Drag-to-pan, in the ground plane.
 *
 * Panning moves the TARGET, not the camera, so the orbit keeps working afterwards. The
 * screen-to-world rate scales with distance: at 200 m out a pixel is worth far more ground
 * than at 5 m, and a fixed rate makes the far view unusable and the near view uncontrollable.
 */
export function panTarget(orbit: Orbit, dxPixels: number, dyPixels: number): Vec3 {
  const rate = orbit.distance * 0.0016;
  const rightX = Math.cos(orbit.azimuth);
  const rightZ = -Math.sin(orbit.azimuth);
  return [
    orbit.target[0] - dxPixels * rightX * rate - dyPixels * Math.sin(orbit.azimuth) * rate,
    orbit.target[1],
    orbit.target[2] - dxPixels * rightZ * rate - dyPixels * Math.cos(orbit.azimuth) * rate,
  ];
}

/** One wheel notch. `deltaY` is only ever read for its sign — trackpads lie about magnitude. */
export function zoomDistance(distance: number, deltaY: number): number {
  return clampDistance(distance * (1 + Math.sign(deltaY) * 0.09));
}
