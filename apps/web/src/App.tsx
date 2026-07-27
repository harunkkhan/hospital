/**
 * Layout shell: FloorMap center, side panels, bottom PlaybackControls.
 *
 * API mode: mock by default (fully demoable without the backend); append
 * `?live` to the URL to talk to the FastAPI backend through the Vite proxy.
 * The map renders the SCRUBBED world when the operator drags back through
 * the local frame buffer; every panel that acts (override, control) is wired
 * to the LIVE world only.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { createHttpApi, type ConsoleApi } from "./api/client";
import type { OverrideOutcome, OverrideRequest, RunRequest } from "./api/types";
import { BottleneckPanel } from "./components/BottleneckPanel";
import { CompareView } from "./components/CompareView";
import { formatSimTime } from "./components/format";
import { KPIPanel } from "./components/KPIPanel";
import { Legend } from "./components/Legend";
import { OverridePanel } from "./components/OverridePanel";
import { PlaybackControls } from "./components/PlaybackControls";
import { ScenarioControls } from "./components/ScenarioControls";
import { usePolled } from "./hooks/usePolled";
import { useRunManager } from "./hooks/useRunManager";
import { useStream } from "./hooks/useStream";
import { createMockApi } from "./mock/mockApi";
import { FloorMap2D } from "./render/FloorMap2D";
import type { SelectedEntity } from "./state/runStore";

const DEFAULT_RUN: RunRequest = {
  scenario: { id: "er_floor" },
  seed: 42,
  arm: "optimized",
  compare_to: "baseline",
  start: "paused",
};

function makeApi(): ConsoleApi {
  const live = new URLSearchParams(window.location.search).has("live");
  return live ? createHttpApi() : createMockApi();
}

export function App() {
  const api = useMemo(makeApi, []);
  const { handle, layout, error: bootError, start } = useRunManager(api, DEFAULT_RUN);
  const [selected, setSelected] = useState<SelectedEntity | null>(null);
  const [scrubIndex, setScrubIndex] = useState<number | null>(null);

  const run = handle?.run ?? null;

  // A run switch invalidates the map selection and the local scrub position.
  useEffect(() => {
    setSelected(null);
    setScrubIndex(null);
  }, [run]);

  const { world: liveWorld, status, buffer, resync } = useStream(api, run);
  const scrubbedWorld =
    scrubIndex !== null ? (buffer.at(scrubIndex)?.world ?? liveWorld) : liveWorld;

  const metricsFetcher = useMemo(
    () => (run === null ? null : () => api.getMetrics(run)),
    [api, run],
  );
  const bottleneckFetcher = useMemo(
    () => (run === null ? null : () => api.getBottleneck(run)),
    [api, run],
  );
  const compareFetcher = useMemo(
    () => (run === null || handle?.shadow == null ? null : () => api.getCompare(run)),
    [api, run, handle],
  );
  const scenariosFetcher = useMemo(() => () => api.listScenarios(), [api]);

  const metrics = usePolled(metricsFetcher, 4000);
  const bottleneck = usePolled(bottleneckFetcher, 8000);
  const compare = usePolled(compareFetcher, 20000);
  const scenarios = usePolled(scenariosFetcher, 60000);

  const submitOverride = useCallback(
    (req: OverrideRequest): Promise<OverrideOutcome> => {
      if (run === null) {
        return Promise.reject(new Error("no active run"));
      }
      return api.override(run, req);
    },
    [api, run],
  );

  const statusBadge =
    status === "open" ? "live" : status === "closed" ? "err" : "warn";

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>ER Operator Console</h1>
        <span className="badge">{api.mode} mode</span>
        <span className={`badge ${statusBadge}`}>stream {status}</span>
        {handle !== null && (
          <span className="muted small mono">
            {handle.run} · arm {handle.arm} · seed {handle.seed}
          </span>
        )}
        {scrubIndex !== null && (
          <span className="badge warn">viewing buffered frame {formatSimTime(scrubbedWorld.simTime)}</span>
        )}
        <span className="spacer" />
        {bootError !== null && <span className="badge err">{bootError}</span>}
      </header>

      <main className="map-pane">
        {layout !== null ? (
          <FloorMap2D
            layout={layout}
            world={scrubbedWorld}
            selected={selected}
            onSelect={setSelected}
            live={scrubIndex === null}
          />
        ) : (
          <div style={{ padding: 20 }} className="muted">
            {bootError ?? "starting run…"}
          </div>
        )}
      </main>

      <aside className="side-pane">
        <Legend />
        <OverridePanel
          layout={layout}
          world={liveWorld}
          selected={selected}
          onSubmit={submitOverride}
          runId={run}
          onResync={resync}
        />
        <KPIPanel metrics={liveWorld.kpiPreview ?? metrics.data} />
        <BottleneckPanel report={bottleneck.data} />
        <CompareView
          compare={compare.data}
          error={handle !== null && handle.shadow == null ? "no shadow arm (compare_to unset)" : compare.error}
          onRefresh={compare.refresh}
        />
        <ScenarioControls
          scenarios={scenarios.data}
          currentSeed={handle?.seed ?? DEFAULT_RUN.seed}
          onRerun={start}
          onSaveScenario={(req) => api.createScenario(req)}
        />
      </aside>

      <div className="controls-pane">
        <PlaybackControls
          state={liveWorld.state ?? handle?.state ?? null}
          speed={liveWorld.speed}
          simTime={scrubbedWorld.simTime}
          horizon={handle?.horizon ?? 0}
          onCommand={(cmd) => {
            if (run !== null) {
              void api.control(run, cmd);
            }
          }}
          bufferLength={buffer.length}
          scrubIndex={scrubIndex}
          onScrub={setScrubIndex}
        />
      </div>
    </div>
  );
}
