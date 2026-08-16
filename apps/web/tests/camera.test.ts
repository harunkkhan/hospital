import { describe, expect, test } from "bun:test";

import {
  CAMERA_PRESETS,
  DEFAULT_PRESET,
  cameraBasis,
  clampDistance,
  clampElevation,
  directionFrom,
  fitDistance,
  orbitEye,
  orbitForPreset,
  panTarget,
  presetById,
  zoomDistance,
  type FloorBox,
  type Vec3,
} from "../src/render3d/camera";

const VFOV = 34;
/** Roughly the shipped ER department: 92 x 72 m, one storey. */
const BOX: FloorBox = { halfWidth: 46, halfDepth: 36, height: 3.4 };

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Every corner of the box, measured from the orbit target. */
function corners(box: FloorBox, targetHeight: number): readonly Vec3[] {
  const out: Vec3[] = [];
  for (const x of [-box.halfWidth, box.halfWidth]) {
    for (const y of [-targetHeight, box.height - targetHeight]) {
      for (const z of [-box.halfDepth, box.halfDepth]) {
        out.push([x, y, z]);
      }
    }
  }
  return out;
}

describe("directionFrom", () => {
  test("elevation is up and azimuth swings from +z toward +x", () => {
    const overhead = directionFrom(90, 0);
    expect(overhead[1]).toBeCloseTo(1, 6);
    const east = directionFrom(0, 90);
    expect(east[0]).toBeCloseTo(1, 6);
    expect(east[1]).toBeCloseTo(0, 6);
    const south = directionFrom(0, 0);
    expect(south[2]).toBeCloseTo(1, 6);
  });

  test("is always a unit vector", () => {
    for (const preset of CAMERA_PRESETS) {
      const d = directionFrom(preset.elevationDeg, preset.azimuthDeg);
      expect(Math.hypot(d[0], d[1], d[2])).toBeCloseTo(1, 9);
    }
  });
});

describe("cameraBasis", () => {
  test("right, up and forward stay orthonormal, straight down included", () => {
    for (const elevation of [0, 12, 45, 88, 90]) {
      const { right, up, forward } = cameraBasis(directionFrom(elevation, 25));
      expect(Math.hypot(right[0], right[1], right[2])).toBeCloseTo(1, 9);
      expect(Math.hypot(up[0], up[1], up[2])).toBeCloseTo(1, 9);
      expect(dot(right, up)).toBeCloseTo(0, 9);
      expect(dot(right, forward)).toBeCloseTo(0, 9);
      expect(dot(up, forward)).toBeCloseTo(0, 9);
    }
  });
});

describe("fitDistance", () => {
  test("frames every corner of the department inside the frustum", () => {
    const tanV = Math.tan(((VFOV * Math.PI) / 180) / 2);
    for (const aspect of [0.6, 1, 16 / 9, 3]) {
      for (const preset of CAMERA_PRESETS) {
        if (preset.margin < 1) {
          continue; // the corridor view deliberately stands inside the box
        }
        const direction = directionFrom(preset.elevationDeg, preset.azimuthDeg);
        const { right, up } = cameraBasis(direction);
        const distance = fitDistance(BOX, direction, aspect, VFOV, preset.targetHeight, preset.margin);
        const tanH = tanV * aspect;
        for (const q of corners(BOX, preset.targetHeight)) {
          const depth = distance - dot(q, direction);
          expect(depth).toBeGreaterThan(0);
          // Allow the margin's own slack, but nothing may fall outside the frustum.
          expect(Math.abs(dot(q, right))).toBeLessThanOrEqual(tanH * depth + 1e-6);
          expect(Math.abs(dot(q, up))).toBeLessThanOrEqual(tanV * depth + 1e-6);
        }
      }
    }
  });

  test("a bigger department needs a longer lens", () => {
    const direction = directionFrom(34, 31);
    const near = fitDistance(BOX, direction, 1.6, VFOV, 1);
    const far = fitDistance(
      { halfWidth: 92, halfDepth: 72, height: 3.4 },
      direction,
      1.6,
      VFOV,
      1,
    );
    expect(far).toBeGreaterThan(near);
    expect(far / near).toBeCloseTo(2, 1);
  });

  test("a wider viewport does not have to stand as far back", () => {
    const direction = directionFrom(88, 0);
    const square = fitDistance(BOX, direction, 1, VFOV, 1);
    const wide = fitDistance(BOX, direction, 16 / 9, VFOV, 1);
    expect(wide).toBeLessThan(square);
  });

  test("margin scales the fit linearly", () => {
    const direction = directionFrom(34, 31);
    const base = fitDistance(BOX, direction, 1.6, VFOV, 1);
    expect(fitDistance(BOX, direction, 1.6, VFOV, 1, 0.5)).toBeCloseTo(base * 0.5, 6);
  });

  test("a narrower field of view backs the camera off", () => {
    const direction = directionFrom(34, 31);
    expect(fitDistance(BOX, direction, 1.6, 20, 1)).toBeGreaterThan(
      fitDistance(BOX, direction, 1.6, 50, 1),
    );
  });
});

describe("orbitForPreset", () => {
  test("the default is the isometric view, and it looks down at the floor", () => {
    expect(DEFAULT_PRESET).toBe("isometric");
    const orbit = orbitForPreset(presetById(DEFAULT_PRESET), BOX, 1.6, VFOV);
    const eye = orbitEye(orbit);
    expect(eye[1]).toBeGreaterThan(0);
    expect(orbit.distance).toBeGreaterThan(BOX.halfWidth);
    expect(Math.hypot(eye[0] - orbit.target[0], eye[1] - orbit.target[1], eye[2] - orbit.target[2])).toBeCloseTo(
      orbit.distance,
      6,
    );
  });

  test("the plan view is nearly overhead; the corridor view is nearly level and inside", () => {
    const plan = orbitForPreset(presetById("plan"), BOX, 1.6, VFOV);
    const eye = orbitEye(plan);
    expect(Math.hypot(eye[0], eye[2])).toBeLessThan(plan.distance * 0.05);

    const corridor = orbitForPreset(presetById("corridor"), BOX, 1.6, VFOV);
    expect(corridor.elevation).toBeLessThan(0.3);
    expect(corridor.distance).toBeLessThan(
      orbitForPreset(presetById("isometric"), BOX, 1.6, VFOV).distance,
    );
    expect(corridor.target[1]).toBeGreaterThan(plan.target[1]);
  });

  test("every preset stays within the orbit limits", () => {
    for (const preset of CAMERA_PRESETS) {
      const orbit = orbitForPreset(preset, BOX, 1.6, VFOV);
      expect(orbit.elevation).toBe(clampElevation(orbit.elevation));
      expect(orbit.distance).toBe(clampDistance(orbit.distance));
    }
  });

  test("an unknown preset id falls back to the isometric view", () => {
    expect(presetById("nope" as never).id).toBe("isometric");
  });
});

describe("controls", () => {
  const orbit = orbitForPreset(presetById("isometric"), BOX, 1.6, VFOV);

  test("panning slides the target in the ground plane, never up or down", () => {
    const moved = panTarget(orbit, 40, -25);
    expect(moved[1]).toBe(orbit.target[1]);
    expect(Math.hypot(moved[0] - orbit.target[0], moved[2] - orbit.target[2])).toBeGreaterThan(0);
  });

  test("panning scales with distance — a pixel is worth more ground from further out", () => {
    const near = panTarget({ ...orbit, distance: 10 }, 100, 0);
    const far = panTarget({ ...orbit, distance: 200 }, 100, 0);
    expect(Math.hypot(far[0], far[2])).toBeGreaterThan(Math.hypot(near[0], near[2]));
  });

  test("the wheel reads only the sign, and zoom is clamped both ways", () => {
    expect(zoomDistance(100, 1)).toBeGreaterThan(100);
    expect(zoomDistance(100, 1)).toBe(zoomDistance(100, 4000));
    expect(zoomDistance(100, -1)).toBeLessThan(100);
    expect(zoomDistance(1, -1)).toBeGreaterThanOrEqual(3);
    expect(zoomDistance(1400, 1)).toBeLessThanOrEqual(1400);
  });

  test("elevation never reaches vertical, where the camera basis collapses", () => {
    expect(clampElevation(Math.PI / 2)).toBeLessThan(Math.PI / 2);
    expect(clampElevation(-4)).toBeGreaterThan(0);
  });
});
