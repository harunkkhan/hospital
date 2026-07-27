/**
 * Stream hook: opens the run's frame stream, folds frames through the pure
 * reducer, buffers reduced moments for local scrub, and reconnects (which
 * yields a fresh snapshot) whenever a seq gap desyncs the view.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { ConsoleApi, StreamStatus } from "../api/client";
import type { RunId, StreamFrame } from "../api/types";
import { FrameBuffer } from "../state/frameBuffer";
import { applyFrame, initialWorld, type WorldView } from "../state/streamReducer";

export interface StreamView {
  world: WorldView;
  status: StreamStatus;
  buffer: FrameBuffer;
  /** Force a reconnect, which yields a fresh authoritative snapshot. */
  resync: () => void;
}

export function useStream(api: ConsoleApi, run: RunId | null): StreamView {
  const [world, setWorld] = useState<WorldView>(initialWorld);
  const [status, setStatus] = useState<StreamStatus>("closed");
  const [epoch, setEpoch] = useState(0);
  const worldRef = useRef(world);
  const bufferRef = useRef<FrameBuffer | null>(null);
  bufferRef.current ??= new FrameBuffer();

  // Re-subscribing (bumping epoch) reruns the effect below, which closes and
  // reopens the stream — the reconnect protocol re-snapshots from scratch.
  const resync = useCallback(() => setEpoch((e) => e + 1), []);

  useEffect(() => {
    if (run === null) {
      setWorld(initialWorld());
      worldRef.current = initialWorld();
      setStatus("closed");
      return;
    }

    // A fresh subscription always starts from a snapshot; drop stale state.
    const fresh = initialWorld();
    worldRef.current = fresh;
    setWorld(fresh);
    bufferRef.current?.clear();

    let desyncScheduled = false;
    const handle = api.openStream(run, {
      onFrame: (frame: StreamFrame) => {
        const next = applyFrame(worldRef.current, frame);
        if (next === worldRef.current) {
          return;
        }
        worldRef.current = next;
        if (next.desynced) {
          // Gap detected: resume == re-snapshot (doc 07 nuances §7.3).
          if (!desyncScheduled) {
            desyncScheduled = true;
            setEpoch((e) => e + 1);
          }
          return;
        }
        bufferRef.current?.push({ frame, world: next });
        setWorld(next);
      },
      onStatus: setStatus,
    });

    return () => {
      handle.close();
    };
  }, [api, run, epoch]);

  return { world, status, buffer: bufferRef.current, resync };
}
