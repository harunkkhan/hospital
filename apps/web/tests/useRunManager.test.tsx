import { describe, expect, test } from "bun:test";
import { StrictMode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";

import type { ConsoleApi } from "../src/api/client";
import type { FloorLayout, RunRequest } from "../src/api/types";
import { useRunManager } from "../src/hooks/useRunManager";

const REQ: RunRequest = { scenario: { id: "er_floor" }, seed: 1, arm: "optimized", start: "paused" };

const EMPTY_LAYOUT: FloorLayout = {
  graph: { nodes: [], edges: [] },
  zones: [],
  bays: [],
  stations: [],
  entrances: [],
  imaging_nodes: [],
  lab_nodes: [],
};

function makeFakeApi(opts: { failLayoutFor?: (runId: string) => boolean } = {}): {
  api: ConsoleApi;
  created: string[];
  deleted: string[];
} {
  const created: string[] = [];
  const deleted: string[] = [];
  let counter = 0;
  const unused = () => Promise.reject(new Error("unused in these tests"));
  const api: ConsoleApi = {
    mode: "mock",
    async createRun(req) {
      counter += 1;
      const run = `run-${counter}`;
      created.push(run);
      return {
        run,
        arm: req.arm,
        seed: req.seed,
        horizon: 0,
        state: "paused",
        sim_time: 0,
        stream_url: "",
        shadow: null,
      };
    },
    async getLayout(run) {
      if (opts.failLayoutFor?.(run)) {
        throw new Error(`layout failed for ${run}`);
      }
      return EMPTY_LAYOUT;
    },
    async deleteRun(run) {
      deleted.push(run);
    },
    getRun: unused,
    control: unused,
    override: unused,
    getMetrics: unused,
    getCompare: unused,
    getBottleneck: unused,
    async listScenarios() {
      return [];
    },
    async createScenario() {
      return { id: "scn" };
    },
    openStream() {
      return { close() {} };
    },
  };
  return { api, created, deleted };
}

const liveRuns = (created: string[], deleted: string[]): string[] =>
  created.filter((id) => !deleted.includes(id));

describe("useRunManager (finding #4: StrictMode double-create)", () => {
  test("a StrictMode double mount leaves exactly one live run", async () => {
    const { api, created, deleted } = makeFakeApi();
    const { result } = renderHook(() => useRunManager(api, REQ), { wrapper: StrictMode });

    await waitFor(() => expect(result.current.handle).not.toBeNull());
    // Any superseded create self-deletes — settle to exactly one live run.
    await waitFor(() => expect(liveRuns(created, deleted)).toHaveLength(1));
    expect(result.current.handle?.run).toBe(liveRuns(created, deleted)[0]);
    // Every run the double-mount created except the survivor was cleaned up —
    // nothing leaked (created here is 2 under StrictMode's duplicate mount).
    expect(deleted.length).toBe(created.length - 1);
  });
});

describe("useRunManager (finding #10: create → swap → delete)", () => {
  test("replacing a run swaps before deleting the old one", async () => {
    const { api, deleted } = makeFakeApi();
    const { result } = renderHook(() => useRunManager(api, REQ));

    await waitFor(() => expect(result.current.handle).not.toBeNull());
    const first = result.current.handle!.run;
    expect(deleted).not.toContain(first);

    act(() => {
      result.current.start(REQ);
    });
    await waitFor(() => expect(result.current.handle?.run).not.toBe(first));
    const second = result.current.handle!.run;

    await waitFor(() => expect(deleted).toContain(first)); // old deleted after swap
    expect(deleted).not.toContain(second); // replacement kept
    expect(result.current.error).toBeNull();
  });

  test("a failed replacement keeps the old run and bins the partial create", async () => {
    const { api, created, deleted } = makeFakeApi({ failLayoutFor: (id) => id === "run-2" });
    const { result } = renderHook(() => useRunManager(api, REQ));

    await waitFor(() => expect(result.current.handle?.run).toBe("run-1"));

    act(() => {
      result.current.start(REQ);
    });
    await waitFor(() => expect(result.current.error).not.toBeNull());

    expect(result.current.handle?.run).toBe("run-1"); // controls stay on the old run
    expect(deleted).not.toContain("run-1");
    expect(created).toContain("run-2");
    expect(deleted).toContain("run-2"); // orphaned replacement cleaned up
  });
});
