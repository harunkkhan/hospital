/**
 * Poll an async fetcher on a light interval (REST pull — metrics and
 * indicators are pulled, never folded client-side from the events tail).
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface Polled<T> {
  data: T | null;
  error: string | null;
  refresh: () => void;
}

export function usePolled<T>(fetcher: (() => Promise<T>) | null, intervalMs: number): Polled<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Every load() gets a strictly-increasing request id. A resolved request may
  // update state only if its id is not older than the latest already applied —
  // so when two polls overlap, the SLOWER-but-OLDER response can never clobber
  // the newer one, regardless of which promise settles first.
  const nextIdRef = useRef(0);
  const appliedRef = useRef(0);

  const load = useCallback(() => {
    if (fetcher === null) {
      return;
    }
    const id = nextIdRef.current;
    nextIdRef.current += 1;
    fetcher().then(
      (value) => {
        if (id >= appliedRef.current) {
          appliedRef.current = id;
          setData(value);
          setError(null);
        }
      },
      (err: unknown) => {
        if (id >= appliedRef.current) {
          appliedRef.current = id;
          setError(err instanceof Error ? err.message : String(err));
        }
      },
    );
  }, [fetcher]);

  useEffect(() => {
    // Fetcher/interval change: retire every in-flight request (their ids are
    // all below the current counter) and clear the view for the new source.
    appliedRef.current = nextIdRef.current;
    setData(null);
    setError(null);
    if (fetcher === null) {
      return;
    }
    load();
    const timer = setInterval(load, intervalMs);
    return () => clearInterval(timer);
  }, [fetcher, intervalMs, load]);

  return { data, error, refresh: load };
}
