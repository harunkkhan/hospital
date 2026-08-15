/**
 * The 3D floor: a WebGL canvas over the derived architecture, with orbit / pan / zoom and
 * the four camera presets.
 *
 * three.js state does NOT live in React state. The renderer, camera and orbit are held in a
 * ref and mutated in place: a drag that re-rendered the component sixty times a second would
 * cost far more than the frame it is trying to draw, and React has no business owning a
 * scene graph. What React does own is the two choices an operator makes — style pack and
 * camera preset — and the effects that translate those into a rebuild.
 *
 * The scene is rebuilt only when the layout or the style pack changes. Everything else
 * (camera moves, and later the live layer) writes into geometry that is already on the GPU.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { FloorLayout } from "../api/types";
import type { SelectedEntity } from "../state/runStore";
import type { WorldView } from "../state/streamReducer";
import {
  CAMERA_PRESETS,
  DEFAULT_PRESET,
  VFOV_DEG,
  clampElevation,
  orbitEye,
  orbitForPreset,
  panTarget,
  presetById,
  zoomDistance,
  type FloorBox,
  type Orbit,
  type PresetId,
  type Vec3,
} from "./camera";
import { deriveFloor } from "./derive";
import { HEIGHTS, SCALE, rectHeight, rectWidth } from "./geometry";
import { buildLiveLayer, type LiveLayer } from "./live";
import { buildFloorScene, type FloorScene } from "./scene";
import { DEFAULT_STYLE_ID, STYLE_PACKS, stylePackById } from "./styles";

/** The department is one storey; the camera fit only needs its height in metres. */
const STOREY_M = HEIGHTS.perimeter * SCALE;

interface FloorMap3DProps {
  layout: FloorLayout;
  world: WorldView;
  selected: SelectedEntity | null;
  onSelect: (selected: SelectedEntity | null) => void;
  /** True only for the live head; a scrubbed buffered view must not extrapolate. */
  live: boolean;
}

/** A pointer that travelled further than this between down and up was a drag, not a click. */
const CLICK_SLOP_PX = 4;

interface Stage {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  orbit: Orbit;
  floor: FloorScene | null;
  people: LiveLayer | null;
  box: FloorBox;
}

export function FloorMap3D({ layout, world, selected, onSelect, live }: FloorMap3DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<Stage | null>(null);
  const [styleId, setStyleId] = useState<string>(DEFAULT_STYLE_ID);
  const [presetId, setPresetId] = useState<PresetId>(DEFAULT_PRESET);
  const [failure, setFailure] = useState<string | null>(null);
  // The rebuild effect has to re-fit the camera without taking a dependency on the preset:
  // changing preset is a camera move, and rebuilding the whole floor for one would be absurd.
  // Written during render rather than in an effect so it is never a commit behind.
  const presetIdRef = useRef<PresetId>(presetId);
  presetIdRef.current = presetId;

  const arch = useMemo(() => deriveFloor(layout), [layout]);
  const style = useMemo(() => stylePackById(styleId), [styleId]);

  // The latest world, its wall-clock arrival time and whether it is the live head — read from
  // the animation loop without dragging React into every frame.
  const frameRef = useRef<{ world: WorldView; wallMs: number; live: boolean }>({
    world,
    wallMs: performance.now(),
    live,
  });
  useEffect(() => {
    frameRef.current = { world, wallMs: performance.now(), live };
  }, [world, live]);

  const aspectOf = useCallback((): number => {
    const canvas = canvasRef.current;
    if (canvas === null || canvas.clientHeight === 0) {
      return 1;
    }
    return canvas.clientWidth / canvas.clientHeight;
  }, []);

  const applyPreset = useCallback(
    (id: PresetId): void => {
      const stage = stageRef.current;
      if (stage === null) {
        return;
      }
      stage.orbit = orbitForPreset(presetById(id), stage.box, aspectOf(), VFOV_DEG);
    },
    [aspectOf],
  );

  // --- the renderer, created once and kept until unmount ---------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (canvas === null || container === null) {
      return;
    }
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    } catch {
      // No WebGL context (a locked-down browser, a headless run). Say so and stay mounted:
      // the operator can switch back to the 2D map without losing the session.
      setFailure("this browser has no WebGL context");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const stage: Stage = {
      renderer,
      scene: new THREE.Scene(),
      camera: new THREE.PerspectiveCamera(VFOV_DEG, 1, 0.5, 4000),
      orbit: { azimuth: 0, elevation: 0.6, distance: 120, target: [0, 1, 0] },
      floor: null,
      people: null,
      box: { halfWidth: 1, halfDepth: 1, height: STOREY_M },
    };
    stageRef.current = stage;

    const resize = (): void => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) {
        return;
      }
      renderer.setSize(width, height, false);
      stage.camera.aspect = width / height;
      stage.camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let raf = 0;
    const frame = (): void => {
      // People move between frames; the building does not. Only the live layer is touched at
      // display rate — the floor is geometry already on the GPU.
      const current = frameRef.current;
      stage.people?.update(current.world, current.live, current.wallMs);
      const eye: Vec3 = orbitEye(stage.orbit);
      stage.camera.position.set(eye[0], eye[1], eye[2]);
      stage.camera.lookAt(stage.orbit.target[0], stage.orbit.target[1], stage.orbit.target[2]);
      stage.renderer.render(stage.scene, stage.camera);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      stage.people?.dispose();
      stage.floor?.dispose();
      renderer.dispose();
      stageRef.current = null;
    };
  }, []);

  // --- the floor itself, rebuilt only when the plan or the style changes -----------------
  useEffect(() => {
    const stage = stageRef.current;
    if (stage === null) {
      return;
    }
    if (stage.floor !== null) {
      stage.scene.remove(stage.floor.root);
    }
    stage.people?.dispose();
    stage.floor?.dispose();

    const floor = buildFloorScene(arch, style, { vfovDeg: VFOV_DEG });
    const people = buildLiveLayer(layout, arch, style);
    // The live layer hangs off the floor's group so it inherits the same plan-to-scene
    // translation: one offset, defined once, and people cannot drift from the building.
    floor.root.add(people.root);
    stage.floor = floor;
    stage.people = people;
    stage.scene.add(floor.root);
    stage.scene.background = new THREE.Color(style.bg);
    floor.paintBays((bay) => frameRef.current.world.bays[bay]?.status ?? "free");
    stage.box = {
      halfWidth: (rectWidth(arch.dept) * SCALE) / 2,
      halfDepth: (rectHeight(arch.dept) * SCALE) / 2,
      height: STOREY_M,
    };
    // Re-fit to whichever preset is showing: a new plan or a new pack must not leave the
    // camera framing the previous building.
    applyPreset(presetIdRef.current);
  }, [arch, style, layout, applyPreset]);

  useEffect(() => {
    applyPreset(presetId);
  }, [presetId, applyPreset]);

  // Bay status is a per-instance colour write on a mesh that already exists, so a bay turning
  // over costs one buffer upload rather than a rebuild. Frames arrive a few times a second;
  // repainting on frame arrival rather than per animation frame keeps it that way.
  useEffect(() => {
    stageRef.current?.floor?.paintBays((bay) => world.bays[bay]?.status ?? "free");
  }, [world.bays, style]);

  // Declared after the rebuild effect so a style swap re-cages the selected bay on the new
  // floor in the same commit, without the selection having to survive inside the builder.
  useEffect(() => {
    stageRef.current?.floor?.setSelectedBay(
      selected !== null && selected.type === "bay" ? selected.id : null,
    );
  }, [selected, style, arch]);

  // --- pointer controls ------------------------------------------------------------------
  const dragRef = useRef<{ x: number; y: number; pan: boolean; travel: number } | null>(null);
  const pickerRef = useRef(new THREE.Raycaster());

  const onPointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement>): void => {
    event.currentTarget.setPointerCapture(event.pointerId);
    // Shift-drag or right-drag pans; plain drag orbits.
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      pan: event.shiftKey || event.button === 2,
      travel: 0,
    };
  }, []);

  const onPointerMove = useCallback((event: React.PointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current;
    const stage = stageRef.current;
    if (drag === null || stage === null) {
      return;
    }
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    drag.travel += Math.abs(dx) + Math.abs(dy);
    stage.orbit = drag.pan
      ? { ...stage.orbit, target: panTarget(stage.orbit, dx, dy) }
      : {
          ...stage.orbit,
          azimuth: stage.orbit.azimuth - dx * 0.005,
          elevation: clampElevation(stage.orbit.elevation + dy * 0.005),
        };
  }, []);

  /**
   * Selection parity with the 2D map: a bay click feeds `OverridePanel` the same
   * `{ type: "bay", id }`, clicking the selected bay again clears it, and so does clicking
   * empty floor. The pick target is the status wash — one `InstancedMesh` for all 76 bays,
   * so `instanceId` names the bay directly and there is nothing per-bay to raycast against.
   */
  const onPointerUp = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>): void => {
      const drag = dragRef.current;
      dragRef.current = null;
      const stage = stageRef.current;
      if (drag === null || stage === null || stage.floor === null) {
        return;
      }
      if (drag.travel > CLICK_SLOP_PX || drag.pan) {
        return; // the operator was moving the camera, not choosing a bay
      }
      const bounds = event.currentTarget.getBoundingClientRect();
      if (bounds.width === 0 || bounds.height === 0) {
        return;
      }
      pickerRef.current.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
          -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
        ),
        stage.camera,
      );
      const hit = pickerRef.current.intersectObject(stage.floor.bayMesh, false)[0];
      const id = hit?.instanceId === undefined ? undefined : stage.floor.bayOrder[hit.instanceId];
      if (id === undefined) {
        onSelect(null);
        return;
      }
      const already = selected !== null && selected.type === "bay" && selected.id === id;
      onSelect(already ? null : { type: "bay", id });
    },
    [onSelect, selected],
  );

  const onPointerCancel = useCallback((): void => {
    dragRef.current = null;
  }, []);

  // The wheel listener has to be non-passive to suppress page scroll, which React's own
  // onWheel cannot promise — so it is attached by hand.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const onWheel = (event: WheelEvent): void => {
      event.preventDefault();
      const stage = stageRef.current;
      if (stage !== null) {
        stage.orbit = { ...stage.orbit, distance: zoomDistance(stage.orbit.distance, event.deltaY) };
      }
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div ref={containerRef} className="floor3d" style={{ position: "absolute", inset: 0 }}>
      <canvas
        ref={canvasRef}
        className="floor3d-canvas"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onPointerMove={onPointerMove}
        onContextMenu={(event) => event.preventDefault()}
      />
      {failure !== null && <div className="floor3d-failure">{failure}</div>}
      <div className="floor3d-chrome">
        <div className="floor3d-group">
          {CAMERA_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={preset.id === presetId ? "on" : undefined}
              onClick={() => setPresetId(preset.id)}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <div className="floor3d-group">
          {STYLE_PACKS.map((pack) => (
            <button
              key={pack.id}
              type="button"
              className={pack.id === styleId ? "on" : undefined}
              onClick={() => setStyleId(pack.id)}
            >
              {pack.name}
            </button>
          ))}
        </div>
      </div>
      <p className="floor3d-tagline">{style.tagline}</p>
    </div>
  );
}
