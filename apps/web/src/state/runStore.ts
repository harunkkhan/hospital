/**
 * Minimal external store (React 19 `useSyncExternalStore`-thin, no deps).
 * Holds cross-panel session state: the run handle, the last authoritative
 * SessionState from /control, and the map selection feeding OverridePanel.
 */

import { useSyncExternalStore } from "react";

import type { BayId, PatientId, RunHandle, SessionState, StaffId } from "../api/types";

export interface Store<T> {
  get(): T;
  set(next: T | ((prev: T) => T)): void;
  subscribe(listener: () => void): () => void;
}

export function createStore<T>(initial: T): Store<T> {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => state,
    set(next) {
      state = typeof next === "function" ? (next as (prev: T) => T)(state) : next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

export type SelectedEntity =
  | { type: "bay"; id: BayId }
  | { type: "patient"; id: PatientId }
  | { type: "staff"; id: StaffId };

export interface RunUiState {
  handle: RunHandle | null;
  session: SessionState | null;
  selected: SelectedEntity | null;
}

export const runStore = createStore<RunUiState>({
  handle: null,
  session: null,
  selected: null,
});
