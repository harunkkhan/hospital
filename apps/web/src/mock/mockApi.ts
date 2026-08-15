/**
 * Mock implementation of ConsoleApi: a per-run MockEngine plus a wall-clock
 * driver interval. Pacing mirrors the real driver contract — `speed` scales
 * only how much sim-time each wall tick consumes; the engine's realized event
 * sequence is fixed by the seed (see engine.ts). The whole console is
 * demoable and testable against this with the FastAPI backend switched off.
 */

import type { ConsoleApi, StreamCallbacks, StreamHandle } from "../api/client";
import type {
  CompareResponse,
  ControlCommand,
  KpiContrast,
  RunHandle,
  RunId,
  RunRequest,
  ScenarioSummary,
  SessionState,
  SliderCatalogue,
  SliderGroup,
  SliderSpec,
  StaffRole,
  StepGranularity,
} from "../api/types";
import { MockEngine, mulberry32, WEEK_US, type ScenarioConfig } from "./engine";
import { makeMockLayout, STAFF_ROSTER } from "./fixtures";

const TICK_MS = 100;
const HEARTBEAT_TICKS = 20;
const S = 1_000_000;

interface MockRun {
  engine: MockEngine;
  subscribers: Set<StreamCallbacks>;
  timer: ReturnType<typeof setInterval> | null;
  ticksSinceEmit: number;
  compareTo: "baseline" | "optimized" | null;
}

function contrastSpec(
  key: string,
  base: number,
  improve: number,
  noise: number,
): { key: string; base: number; improve: number; noise: number } {
  return { key, base, improve, noise };
}

/**
 * Deterministic synthetic compare payload. Includes honest regressions
 * (optimized worse on turnaround) and non-significant contrasts (CI spans 0)
 * so CompareView's honest-display paths are exercised in the demo.
 */
export function synthesizeCompare(
  seed: number,
  baselineRun: RunId,
  optimizedRun: RunId,
): CompareResponse {
  const rng = mulberry32(seed + 7919);
  const specs = [
    contrastSpec("completions_per_week", 968, -0.034, 0.012),
    contrastSpec("wip_end_of_week", 26, 0.16, 0.06),
    contrastSpec("door_to_triage_s_mean", 540, 0.04, 0.05),
    contrastSpec("door_to_provider_s_mean", 1920, 0.22, 0.05),
    contrastSpec("door_to_provider_s_p90", 4260, 0.26, 0.07),
    contrastSpec("los_s_mean_by_esi_1", 14300, 0.02, 0.04),
    contrastSpec("los_s_mean_by_esi_3", 9840, 0.12, 0.04),
    contrastSpec("boarding_time_s_mean", 1380, 0.31, 0.08),
    contrastSpec("turnaround_time_s_mean", 312, -0.07, 0.02),
    contrastSpec("staff_minutes_walked", 5180, 0.18, 0.05),
    contrastSpec("bay_utilization", 0.63, -0.05, 0.03),
    contrastSpec("provider_util", 0.71, -0.03, 0.05),
  ];
  const contrasts: KpiContrast[] = specs.map(({ key, base, improve, noise }) => {
    const jitter = (rng() - 0.5) * 0.2 * improve;
    const delta = base * (improve + jitter);
    const optimized = base - delta;
    const halfWidth = Math.abs(base) * noise * (0.8 + rng() * 0.4) * 1.96;
    const ci_lo = delta - halfWidth;
    const ci_hi = delta + halfWidth;
    return {
      key,
      baseline: base,
      optimized,
      delta,
      ci_lo,
      ci_hi,
      significant: !(ci_lo <= 0 && 0 <= ci_hi),
    };
  });
  return { baseline_run: baselineRun, optimized_run: optimizedRun, replications: 16, contrasts };
}

/**
 * The mock backend's slider catalogue — the same wire shape the API publishes,
 * describing THIS floor rather than the reference ED.
 *
 * Two deliberate differences from the live catalogue, both honesty rather than
 * laziness. (1) Values and bay ranges are read off the hand-authored fixture
 * (13 bays, 8 staff), so they describe what the mock engine will actually run;
 * a bay knob therefore caps at the fixture's own count, because there is no
 * geometry to open a fourteenth bay into. (2) Knobs this engine does not model —
 * observation bays (no such zone here), imaging suites and lab stations (no
 * imaging/lab service physics) — are OMITTED rather than published as sliders
 * that move nothing. A control that silently does nothing teaches the operator
 * something false about the model; an absent one only teaches them about the
 * mock. The live catalogue is the full vocabulary.
 */
function mockKnobs(overrides: Readonly<Record<string, number>>): SliderSpec[] {
  const layout = makeMockLayout();
  const baysOfType = (zoneType: string): number =>
    layout.bays.filter((b) => b.zone_type === zoneType).length;
  const staffOfRole = (role: StaffRole): number =>
    STAFF_ROSTER.filter((s) => s.role === role).length;

  const spec = (
    key: string,
    label: string,
    group: SliderGroup,
    unit: string,
    base: number,
    min: number,
    max: number,
    step: number,
  ): SliderSpec => ({
    key,
    label,
    group,
    min,
    max,
    step,
    unit,
    // Read against the scenario, exactly as the server does: a saved variant
    // reports where IT sits, not where the canned base does.
    value: overrides[key] ?? base,
  });

  return [
    // The multiplier is relative, so it reads 1.0 against its own base — the
    // override is already baked into the rate the base describes.
    { ...spec("workload.arrival_rate_multiplier", "Arrival rate", "demand", "x base", 1, 0.25, 3, 0.05), value: 1 },
    spec("workload.ambulance_share", "Ambulance arrivals", "demand", "share", 0.25, 0, 1, 0.01),
    spec("workload.isolation_share", "Isolation required", "demand", "share", 0.05, 0, 1, 0.01),
    spec("staffing.physician_count", "Physicians", "staffing", "on duty", staffOfRole("physician"), 0, 12, 1),
    spec("staffing.nurse_count", "Nurses", "staffing", "on duty", staffOfRole("nurse"), 0, 12, 1),
    spec("staffing.tech_count", "Techs", "staffing", "on duty", staffOfRole("tech"), 0, 12, 1),
    spec("facility.general_bays", "General bays", "capacity", "bays", baysOfType("general"), 0, baysOfType("general"), 1),
    spec("facility.resus_bays", "Resus / trauma bays", "capacity", "bays", baysOfType("resus_trauma"), 0, baysOfType("resus_trauma"), 1),
    spec("facility.fast_track_bays", "Fast-track bays", "capacity", "bays", baysOfType("fast_track"), 0, baysOfType("fast_track"), 1),
    spec("facility.triage_rooms", "Triage rooms", "capacity", "rooms", baysOfType("triage"), 0, baysOfType("triage"), 1),
    spec("staffing.porter_count", "Porters", "capacity", "on duty", staffOfRole("porter"), 0, 12, 1),
    spec("staffing.housekeeping_count", "Housekeeping", "capacity", "on duty", staffOfRole("housekeeping"), 0, 12, 1),
  ];
}

const CANNED_SCENARIOS: ScenarioSummary[] = [
  { id: "er_floor", name: "Reference ER week", horizon: WEEK_US, note: "100k sqft, 1-week workload" },
  { id: "surge_monday", name: "Monday surge", horizon: WEEK_US, note: "arrival surge overlay" },
  { id: "short_staffed", name: "Short-staffed nights", horizon: WEEK_US, note: "night staffing -2" },
];

export function createMockApi(): ConsoleApi {
  const runs = new Map<RunId, MockRun>();
  const scenarios: ScenarioSummary[] = [...CANNED_SCENARIOS];
  // Overrides for every SAVED scenario, so re-running one by id replays its
  // parameters instead of silently reverting to the canned defaults.
  const savedOverrides = new Map<string, Readonly<Record<string, number>>>();
  let runCounter = 0;
  let scenarioCounter = 0;

  /** Resolve a run request's scenario ref/inline into concrete engine inputs. */
  const resolveScenario = (req: RunRequest): ScenarioConfig => {
    const scenario = req.scenario;
    if ("base" in scenario) {
      return { base: scenario.base, overrides: scenario.overrides };
    }
    return { base: scenario.id, overrides: savedOverrides.get(scenario.id) ?? {} };
  };

  const handleOf = (run: MockRun, shadow: RunId | null = null): RunHandle => ({
    run: run.engine.runId,
    arm: run.engine.arm,
    seed: run.engine.seed,
    horizon: run.engine.horizonUs,
    state: run.engine.state,
    sim_time: run.engine.simTimeUs,
    stream_url: `/runs/${run.engine.runId}/stream`,
    shadow,
  });

  const sessionOf = (run: MockRun): SessionState => ({
    run: run.engine.runId,
    state: run.engine.state,
    sim_time: run.engine.simTimeUs,
    speed: run.engine.speed,
    horizon: run.engine.horizonUs,
  });

  const broadcast = (run: MockRun, kind: "snapshot" | "delta"): void => {
    if (run.subscribers.size === 0) {
      return;
    }
    const frame = run.engine.buildFrame(kind);
    for (const sub of run.subscribers) {
      sub.onFrame(frame);
    }
    run.ticksSinceEmit = 0;
  };

  const ensureDriver = (run: MockRun): void => {
    run.timer ??= setInterval(() => {
      if (run.engine.state === "playing") {
        run.engine.advance(run.engine.speed * TICK_MS * 1000);
        broadcast(run, "delta"); // carries the "finished" state too, if reached
        return;
      }
      // paused/finished: heartbeat so the client can tell pause from a stall
      run.ticksSinceEmit += 1;
      if (run.ticksSinceEmit >= HEARTBEAT_TICKS) {
        broadcast(run, "delta");
      }
    }, TICK_MS);
  };

  const mustRun = (id: RunId): MockRun => {
    const run = runs.get(id);
    if (run === undefined) {
      throw new Error(`mock: unknown run ${id}`);
    }
    return run;
  };

  const step = (run: MockRun, granularity: StepGranularity, count: number): void => {
    const engine = run.engine;
    for (let i = 0; i < count; i += 1) {
      if (granularity === "tick") {
        engine.advance(S);
      } else if (granularity === "decision") {
        engine.advance(60 * S);
      } else {
        // "event": advance 1s at a time until something happened (bounded)
        for (let guard = 0; guard < 600; guard += 1) {
          engine.advance(S);
          if (engine.hasPendingEvents()) {
            break;
          }
        }
      }
    }
  };

  return {
    mode: "mock",

    async createRun(req: RunRequest) {
      runCounter += 1;
      const id = `run-${String(runCounter).padStart(2, "0")}-${req.arm}`;
      const engine = new MockEngine(id, req.seed, req.arm, resolveScenario(req));
      const run: MockRun = {
        engine,
        subscribers: new Set(),
        timer: null,
        ticksSinceEmit: 0,
        compareTo: req.compare_to ?? null,
      };
      runs.set(id, run);
      engine.state = req.start === "playing" ? "playing" : "paused";
      ensureDriver(run);
      const shadow = req.compare_to != null ? `${id}-shadow` : null;
      return handleOf(run, shadow);
    },

    async getRun(id) {
      return handleOf(mustRun(id));
    },

    async deleteRun(id) {
      const run = runs.get(id);
      if (run !== undefined) {
        if (run.timer !== null) {
          clearInterval(run.timer);
        }
        for (const sub of run.subscribers) {
          sub.onStatus?.("closed");
        }
        runs.delete(id);
      }
    },

    async getLayout(id) {
      return mustRun(id).engine.layout;
    },

    async control(id, cmd: ControlCommand) {
      const run = mustRun(id);
      const engine = run.engine;
      if (engine.state !== "finished") {
        switch (cmd.action) {
          case "play":
            engine.state = "playing";
            break;
          case "pause":
            engine.state = "paused";
            break;
          case "speed":
            engine.speed = cmd.multiplier ?? engine.speed;
            break;
          case "step":
            engine.state = "paused";
            step(run, cmd.granularity ?? "decision", cmd.count ?? 1);
            break;
        }
      }
      broadcast(run, "delta");
      return sessionOf(run);
    },

    async override(id, req) {
      const run = mustRun(id);
      const outcome = run.engine.applyOverride(req.action, req.pin ?? true);
      if (outcome.status === "applied") {
        broadcast(run, "delta"); // reflect the accepted action immediately
      }
      return outcome;
    },

    async getMetrics(id) {
      return mustRun(id).engine.metrics();
    },

    async getCompare(id) {
      const run = mustRun(id);
      const shadow = `${id}-shadow`;
      const [baseline, optimized] =
        run.engine.arm === "optimized" ? [shadow, id] : [id, shadow];
      return synthesizeCompare(run.engine.seed, baseline, optimized);
    },

    async getBottleneck(id) {
      return mustRun(id).engine.bottleneck();
    },

    async listScenarios() {
      return scenarios;
    },

    async getSliders(scenario): Promise<SliderCatalogue> {
      if (!scenarios.some((s) => s.id === scenario)) {
        throw new Error(`mock: unknown scenario ${scenario}`);
      }
      return { scenario, knobs: mockKnobs(savedOverrides.get(scenario) ?? {}) };
    },

    async createScenario(req) {
      scenarioCounter += 1;
      const id = `scn-${String(scenarioCounter).padStart(2, "0")}`;
      // Retain the overrides so a later createRun({ scenario: { id } }) replays
      // them — otherwise a saved scenario would be a hollow name.
      savedOverrides.set(id, req.overrides);
      scenarios.push({
        id,
        name: `${req.base} (custom ${scenarioCounter})`,
        horizon: WEEK_US,
        note: `overrides: ${Object.keys(req.overrides).join(", ") || "none"}`,
      });
      return { id };
    },

    openStream(id, callbacks): StreamHandle {
      const run = mustRun(id);
      run.subscribers.add(callbacks);
      callbacks.onStatus?.("open");
      // New subscriptions reset everyone to a snapshot — seq is per run, and a
      // snapshot is always a safe reset for existing subscribers too.
      broadcast(run, "snapshot");
      return {
        close() {
          run.subscribers.delete(callbacks);
          callbacks.onStatus?.("closed");
        },
      };
    },
  };
}
