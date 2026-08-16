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
import { ScenarioLab } from "./components/ScenarioLab";
import { usePolled } from "./hooks/usePolled";
import { useRunManager } from "./hooks/useRunManager";
import { useStream } from "./hooks/useStream";
import { createMockApi } from "./mock/mockApi";
import { FloorMap2D } from "./render/FloorMap2D";
import { FloorMap3D } from "./render3d/FloorMap3D";
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
  // The 3D floor is the default view; the flat map stays one click away for anyone whose
  // browser has no WebGL, and as the reference for what the route graph literally says.
  const [floorView, setFloorView] = useState<"2d" | "3d">("3d");

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
  // Stable, so the lab's per-base catalogue effect fires on the base, not on us.
  const loadCatalogue = useCallback((id: string) => api.getSliders(id), [api]);

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
        <button className="badge" onClick={() => setFloorView(floorView === "3d" ? "2d" : "3d")}>
          {floorView === "3d" ? "3D floor" : "2D map"}
        </button>
        <span className="spacer" />
        {bootError !== null && <span className="badge err">{bootError}</span>}
      </header>

      <main className="map-pane">
        {layout !== null ? (
          floorView === "3d" ? (
            <FloorMap3D
              layout={layout}
              world={scrubbedWorld}
              selected={selected}
              onSelect={setSelected}
              live={scrubIndex === null}
            />
          ) : (
            <FloorMap2D
              layout={layout}
              world={scrubbedWorld}
              selected={selected}
              onSelect={setSelected}
              live={scrubIndex === null}
            />
          )
        ) : (
          <div style={{ padding: 20 }} className="muted">
            {bootError ?? "starting run…"}
          </div>
        )}
        {/* The legend describes the floor, so it sits on it rather than competing for rail
            space with the numbers. */}
        <div className="map-legend">
          <Legend />
        </div>
      </main>

      {/* Monitoring on the left, controls on the right: reading what the department is doing
          and changing what it is asked to do are different jobs, and interleaving them meant
          scrolling past a slider to reach a KPI. */}
      <aside className="rail rail-left" aria-label="Live metrics">
        <KPIPanel metrics={liveWorld.kpiPreview ?? metrics.data} />
        <BottleneckPanel report={bottleneck.data} />
        <CompareView
          compare={compare.data}
          error={handle !== null && handle.shadow == null ? "no shadow arm (compare_to unset)" : compare.error}
          onRefresh={compare.refresh}
        />
        <OverridePanel
          layout={layout}
          world={liveWorld}
          selected={selected}
          onSubmit={submitOverride}
          runId={run}
          onResync={resync}
        />
      </aside>

      <aside className="rail rail-right" aria-label="Scenario controls">
        <ScenarioLab
          scenarios={scenarios.data}
          currentSeed={handle?.seed ?? DEFAULT_RUN.seed}
          loadCatalogue={loadCatalogue}
          onRerun={start}
          onSaveScenario={(req) => api.createScenario(req)}
          runId={run}
          metrics={liveWorld.kpiPreview ?? metrics.data}
          simTime={liveWorld.simTime}
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
