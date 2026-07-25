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
  const epochRef = useRef(0);

  const load = useCallback(() => {
    if (fetcher === null) {
      return;
    }
    const epoch = epochRef.current;
    fetcher().then(
      (value) => {
        if (epochRef.current === epoch) {
          setData(value);
          setError(null);
        }
      },
      (err: unknown) => {
        if (epochRef.current === epoch) {
          setError(err instanceof Error ? err.message : String(err));
        }
      },
    );
  }, [fetcher]);

  useEffect(() => {
    epochRef.current += 1;
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
