/**
 * play / pause / step / speed / scrub. Controls mutate SESSION state only —
 * speed is pacing, never sampling (16x is the identical run, paced faster),
 * so the speed picker never implies a "rougher" simulation. `step` is only
 * meaningful from paused and is disabled while playing. Scrub is LOCAL and
 * buffer-bounded: it addresses the retained frame ring, not sim history.
 */

import { useState } from "react";

import type { ControlCommand, RunState, StepGranularity } from "../api/types";
import { formatSimTime } from "./format";

const SPEED_OPTIONS = [1, 4, 16, 60, 240, 960] as const;

export interface PlaybackControlsProps {
  state: RunState | null;
  speed: number;
  simTime: number;
  horizon: number;
  onCommand: (cmd: ControlCommand) => void;
  /** Local scrub over the frame buffer: null index == live. */
  bufferLength: number;
  scrubIndex: number | null;
  onScrub: (index: number | null) => void;
}

export function PlaybackControls({
  state,
  speed,
  simTime,
  horizon,
  onCommand,
  bufferLength,
  scrubIndex,
  onScrub,
}: PlaybackControlsProps) {
  const [granularity, setGranularity] = useState<StepGranularity>("decision");
  const playing = state === "playing";
  const finished = state === "finished";
  const scrubbing = scrubIndex !== null;
  const progress = horizon > 0 ? Math.min(1, simTime / horizon) : 0;

  return (
    <div className="panel playback">
      <span className="clock" title="sim time (decoupled from wall-clock)">
        {formatSimTime(simTime)}
        <span className="muted small"> / {Math.round(progress * 100)}%</span>
      </span>

      {playing ? (
        <button onClick={() => onCommand({ action: "pause" })} disabled={finished}>
          Pause
        </button>
      ) : (
        <button className="primary" onClick={() => onCommand({ action: "play" })} disabled={finished}>
          Play
        </button>
      )}

      <button
        onClick={() => onCommand({ action: "step", granularity, count: 1 })}
        disabled={playing || finished}
        title={playing ? "Pause to step" : `Advance one ${granularity}`}
      >
        Step
      </button>
      <label className="field">
        <select
          aria-label="step granularity"
          value={granularity}
          onChange={(e) => setGranularity(e.target.value as StepGranularity)}
          disabled={playing || finished}
        >
          <option value="decision">decision</option>
          <option value="event">event</option>
          <option value="tick">tick</option>
        </select>
      </label>

      <label className="field">
        speed
        <select
          aria-label="playback speed"
          value={String(speed)}
          onChange={(e) => onCommand({ action: "speed", multiplier: Number(e.target.value) })}
        >
          {SPEED_OPTIONS.map((s) => (
            <option key={s} value={String(s)}>
              {s}x
            </option>
          ))}
          {!SPEED_OPTIONS.includes(speed as (typeof SPEED_OPTIONS)[number]) && (
            <option value={String(speed)}>{speed}x</option>
          )}
        </select>
      </label>

      <label className="field" style={{ flex: 1 }}>
        scrub
        <input
          type="range"
          aria-label="scrub buffered frames"
          min={0}
          max={Math.max(bufferLength - 1, 0)}
          value={scrubIndex ?? Math.max(bufferLength - 1, 0)}
          disabled={bufferLength < 2}
          onChange={(e) => {
            const idx = Number(e.target.value);
            onScrub(idx >= bufferLength - 1 ? null : idx);
          }}
        />
      </label>
      {scrubbing ? (
        <button onClick={() => onScrub(null)}>Live</button>
      ) : (
        <span className="badge live">live</span>
      )}
      {finished && <span className="badge">finished</span>}
    </div>
  );
}
