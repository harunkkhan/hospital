/**
 * Owns the run lifecycle: the initial launch and every re-run/replacement.
 *
 * Two failure modes this guards against:
 * - StrictMode (and any rapid double-launch) mounting the create effect twice
 *   and leaking the loser — a monotonic generation makes every superseded
 *   start self-delete, so exactly ONE run survives (finding #4);
 * - deleting the current run BEFORE its replacement is ready — a mid-flight
 *   failure would strand every control on a run that no longer exists. Instead
 *   we create → swap → delete, and on failure keep the old run and bin only the
 *   half-created replacement (finding #10).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { ConsoleApi } from "../api/client";
import type { FloorLayout, RunHandle, RunRequest } from "../api/types";

export interface RunManager {
  handle: RunHandle | null;
  layout: FloorLayout | null;
  error: string | null;
  /** Launch a fresh run, replacing the current one via create → swap → delete. */
  start: (req: RunRequest) => void;
}

export function useRunManager(api: ConsoleApi, initial: RunRequest): RunManager {
  const [handle, setHandle] = useState<RunHandle | null>(null);
  const [layout, setLayout] = useState<FloorLayout | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The run the controls currently address — read inside the async start, so a ref.
  const currentRef = useRef<RunHandle | null>(null);
  // Only the newest start may commit; older attempts (incl. StrictMode's second
  // mount) find a mismatch and delete the run they created instead of leaking it.
  const genRef = useRef(0);

  const start = useCallback(
    (req: RunRequest) => {
      const gen = (genRef.current += 1);
      void (async () => {
        let created: RunHandle | null = null;
        try {
          created = await api.createRun(req);
          const nextLayout = await api.getLayout(created.run);
          if (genRef.current !== gen) {
            // A newer start (or unmount) superseded us — drop the orphan run.
            await api.deleteRun(created.run).catch(() => undefined);
            return;
          }
          const previous = currentRef.current;
          currentRef.current = created;
          setLayout(nextLayout);
          setHandle(created);
          setError(null);
          // Swap first, delete the old run last — controls never point at a
          // deleted run.
          if (previous !== null && previous.run !== created.run) {
            await api.deleteRun(previous.run).catch(() => undefined);
          }
        } catch (err) {
          // createRun succeeded but layout failed ⇒ orphan run; clean it up.
          if (created !== null) {
            await api.deleteRun(created.run).catch(() => undefined);
          }
          // The current run (if any) is untouched — the console keeps working.
          if (genRef.current === gen) {
            setError(err instanceof Error ? err.message : String(err));
          }
        }
      })();
    },
    [api],
  );

  useEffect(() => {
    start(initial);
    // Retire any in-flight launch on teardown so it self-deletes rather than
    // committing after unmount (this is also what neutralizes StrictMode's
    // duplicate mount).
    return () => {
      genRef.current += 1;
    };
  }, [start, initial]);

  return { handle, layout, error, start };
}
