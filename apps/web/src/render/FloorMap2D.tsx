/**
 * The live floor: an SVG static layer (corridor edges, stations, zone
 * labels, clickable bay rects colored by status) under a Canvas dynamic
 * layer (patient chips by acuity, staff dots walking real edges).
 *
 * Canvas — not SVG DOM — for the moving layer: dozens of staff and chips
 * repainting at display rate would thrash the DOM. The render loop runs at
 * requestAnimationFrame rate, DECOUPLED from frame arrival: between frames
 * staff dead-reckon along their edge (interpolate.ts) and snap when the next
 * server-authoritative frame lands.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import type { FloorLayout, NodeId } from "../api/types";
import type { SelectedEntity } from "../state/runStore";
import type { WorldView } from "../state/streamReducer";
import {
  BAY_STATUS_COLORS,
  CHIP_RING,
  EDGE_COLOR,
  ESI_COLORS,
  LABEL_COLOR,
  NODE_COLOR,
  SELECTION_COLOR,
  STAFF_DOT_COLOR,
  STAFF_DOT_RING,
} from "./colors";
import { floorLabel, floorsOf, sliceToFloor } from "./floors";
import { deadReckonSimUs, indexNodes, kinematicPosition } from "./interpolate";
import { makeProjection } from "./projection";

const BAY_W_CM = 260;
const BAY_H_CM = 200;

interface FloorMapProps {
  layout: FloorLayout;
  world: WorldView;
  selected: SelectedEntity | null;
  onSelect: (selected: SelectedEntity | null) => void;
  /** True only for the live head; a scrubbed buffered view must not extrapolate. */
  live: boolean;
}

interface Size {
  width: number;
  height: number;
}

export function FloorMap2D({ layout, world, selected, onSelect, live }: FloorMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState<Size>({ width: 800, height: 520 });
  const [floor, setFloor] = useState(0);

  // A building's floors share a footprint, so their coordinates overlap and projecting the
  // whole graph would stack every ward on the ED. Everything below renders ONE storey; a
  // single-floor scenario slices to itself, so an ED-only run is unchanged.
  const floors = useMemo(() => floorsOf(layout), [layout]);
  const visible = useMemo(() => sliceToFloor(layout, floor), [layout, floor]);

  useEffect(() => {
    const el = containerRef.current;
    if (el === null) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry !== undefined) {
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const projection = useMemo(
    () => makeProjection(visible.graph.nodes, size.width, size.height),
    [visible, size],
  );
  const nodeIndex = useMemo(() => indexNodes(visible.graph.nodes), [visible]);
  const edgeUs = useMemo(() => {
    const map = new Map<string, number>();
    for (const e of visible.graph.edges) {
      map.set(`${e.a}>${e.b}`, e.seconds);
      map.set(`${e.b}>${e.a}`, e.seconds);
    }
    return map;
  }, [visible]);

  // Latest world + its wall arrival time + whether it is the live head,
  // readable from the rAF loop without retriggering React renders.
  const frameRef = useRef<{ world: WorldView; wallMs: number; live: boolean }>({
    world,
    wallMs: performance.now(),
    live,
  });
  useEffect(() => {
    frameRef.current = { world, wallMs: performance.now(), live };
  }, [world, live]);

  // The dynamic layer: patients + staff at display rate.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d") ?? null;
    if (canvas === null || ctx === null) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);

    let raf = 0;
    const draw = (): void => {
      const { world: w, wallMs, live: isLive } = frameRef.current;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.width, size.height);

      // Dead-reckon only on the live, playing head — a scrubbed or paused view
      // is a frozen instant (finding #8).
      const extraSimUs = deadReckonSimUs({
        live: isLive,
        state: w.state,
        speed: w.speed,
        wallElapsedMs: performance.now() - wallMs,
      });

      // patient chips, fanned out when several share a node
      const perNode = new Map<NodeId, number>();
      for (const chip of Object.values(w.patients)) {
        if (chip.at_node === null) {
          continue;
        }
        const node = nodeIndex.get(chip.at_node);
        if (node === undefined) {
          continue;
        }
        const n = perNode.get(chip.at_node) ?? 0;
        perNode.set(chip.at_node, n + 1);
        const angle = (n * Math.PI * 2) / 8;
        const spread = n === 0 ? 0 : 11 + 4 * Math.floor(n / 8);
        const x = projection.toX(node.x_cm) + Math.cos(angle) * spread;
        const y = projection.toY(node.y_cm) + Math.sin(angle) * spread;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = ESI_COLORS[chip.esi];
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = CHIP_RING;
        ctx.stroke();
      }

      // staff dots walking real edges
      for (const kin of Object.values(w.staff)) {
        const totalUs =
          kin.edge !== null ? (edgeUs.get(`${kin.edge[0]}>${kin.edge[1]}`) ?? 0) : 0;
        const pos = kinematicPosition(kin, nodeIndex, totalUs, extraSimUs);
        if (pos === null) {
          continue;
        }
        const x = projection.toX(pos.x_cm);
        const y = projection.toY(pos.y_cm);
        ctx.beginPath();
        ctx.rect(x - 4.5, y - 4.5, 9, 9);
        ctx.fillStyle = STAFF_DOT_COLOR;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = STAFF_DOT_RING;
        ctx.stroke();
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [size, projection, nodeIndex, edgeUs]);

  const zoneLabel = (zoneId: string): { x: number; y: number; text: string } | null => {
    const members = visible.bays.filter((b) => b.zone === zoneId);
    if (members.length === 0) {
      return null;
    }
    let sx = 0;
    let sy = 0;
    for (const bay of members) {
      const node = nodeIndex.get(bay.node);
      if (node === undefined) {
        return null;
      }
      sx += node.x_cm;
      sy += node.y_cm;
    }
    const zone = visible.zones.find((z) => z.id === zoneId);
    return {
      x: projection.toX(sx / members.length),
      y: projection.toY(sy / members.length) - (BAY_H_CM * projection.scale) / 2 - 16,
      text: zone === undefined ? zoneId : zone.zone_type.replace("_", " "),
    };
  };

  const bayW = BAY_W_CM * projection.scale;
  const bayH = BAY_H_CM * projection.scale;

  return (
    <div ref={containerRef} style={{ position: "absolute", inset: 0 }}>
      {/* Storey picker. Absent for a single-floor ED, so the console a reader already
          knows is untouched until there is genuinely somewhere else to look. */}
      {floors.length > 1 && (
        <div
          role="group"
          aria-label="Floor"
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            zIndex: 1,
            display: "flex",
            gap: 4,
          }}
        >
          {floors.map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFloor(f)}
              aria-pressed={f === floor}
              style={{
                padding: "2px 8px",
                fontSize: 12,
                cursor: "pointer",
                fontWeight: f === floor ? 700 : 400,
              }}
            >
              {floorLabel(f)}
            </button>
          ))}
        </div>
      )}
      <svg width={size.width} height={size.height} role="img" aria-label="ER floor map">
        {/* corridor edges */}
        {visible.graph.edges.map((e) => {
          const a = nodeIndex.get(e.a);
          const b = nodeIndex.get(e.b);
          if (a === undefined || b === undefined) {
            return null;
          }
          return (
            <line
              key={`${e.a}>${e.b}`}
              x1={projection.toX(a.x_cm)}
              y1={projection.toY(a.y_cm)}
              x2={projection.toX(b.x_cm)}
              y2={projection.toY(b.y_cm)}
              stroke={EDGE_COLOR}
              strokeWidth={3}
              strokeLinecap="round"
            />
          );
        })}
        {/* stations / entrances / service nodes */}
        {[...visible.stations, ...visible.entrances, ...visible.imaging_nodes, ...visible.lab_nodes].map(
          (id) => {
            const node = nodeIndex.get(id);
            if (node === undefined) {
              return null;
            }
            const x = projection.toX(node.x_cm);
            const y = projection.toY(node.y_cm);
            return (
              <g key={id}>
                <circle cx={x} cy={y} r={4} fill={NODE_COLOR} />
                <text x={x} y={y - 8} textAnchor="middle" fontSize={10} fill={LABEL_COLOR}>
                  {node.label}
                </text>
              </g>
            );
          },
        )}
        {/* zone labels */}
        {visible.zones.map((zone) => {
          const label = zoneLabel(zone.id);
          if (label === null) {
            return null;
          }
          return (
            <text
              key={zone.id}
              x={label.x}
              y={label.y}
              textAnchor="middle"
              fontSize={11}
              fontWeight={600}
              fill={LABEL_COLOR}
              style={{ textTransform: "uppercase", letterSpacing: "0.06em" }}
            >
              {label.text}
            </text>
          );
        })}
        {/* bays, colored by live status; click to select for overrides */}
        {visible.bays.map((bay) => {
          const node = nodeIndex.get(bay.node);
          if (node === undefined) {
            return null;
          }
          const status = world.bays[bay.id]?.status ?? "free";
          const color = BAY_STATUS_COLORS[status];
          const x = projection.toX(node.x_cm) - bayW / 2;
          const y = projection.toY(node.y_cm) - bayH / 2;
          const isSelected = selected?.type === "bay" && selected.id === bay.id;
          return (
            <g
              key={bay.id}
              onClick={() => onSelect(isSelected ? null : { type: "bay", id: bay.id })}
              style={{ cursor: "pointer" }}
            >
              <rect
                x={x}
                y={y}
                width={bayW}
                height={bayH}
                rx={4}
                fill={color}
                fillOpacity={0.26}
                stroke={isSelected ? SELECTION_COLOR : color}
                strokeWidth={isSelected ? 2.5 : 1.5}
                strokeDasharray={isSelected ? "5 3" : undefined}
              />
              <text
                x={x + bayW / 2}
                y={y + bayH + 11}
                textAnchor="middle"
                fontSize={9}
                fill={LABEL_COLOR}
              >
                {bay.id.replace("bay-", "")}
              </text>
            </g>
          );
        })}
      </svg>
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: size.width,
          height: size.height,
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
