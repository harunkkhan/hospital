/**
 * Typed REST client over the wire contract in ./types.ts.
 *
 * `ConsoleApi` is the one interface the UI codes against; it has two
 * implementations — this HTTP client and the deterministic mock in
 * src/mock/mockApi.ts — so every component is demoable and testable without
 * the FastAPI backend running.
 */

import { openSseStream } from "./stream";
import type {
  BottleneckReport,
  CompareResponse,
  ControlCommand,
  FloorLayout,
  KpiVector,
  OverrideOutcome,
  OverrideRejected,
  OverrideRequest,
  RunHandle,
  RunId,
  RunRequest,
  ScenarioCreated,
  ScenarioCreateRequest,
  ScenarioSummary,
  SessionState,
  SliderCatalogue,
  StreamFrame,
} from "./types";

export type StreamStatus = "connecting" | "open" | "reconnecting" | "closed";

export interface StreamCallbacks {
  onFrame: (frame: StreamFrame) => void;
  onStatus?: (status: StreamStatus) => void;
}

export interface StreamHandle {
  close: () => void;
}

export interface ConsoleApi {
  readonly mode: "live" | "mock";
  createRun(req: RunRequest): Promise<RunHandle>;
  getRun(run: RunId): Promise<RunHandle>;
  deleteRun(run: RunId): Promise<void>;
  /** Static geometry, fetched ONCE per run — frames carry only the mutable delta. */
  getLayout(run: RunId): Promise<FloorLayout>;
  control(run: RunId, cmd: ControlCommand): Promise<SessionState>;
  /** 200 → applied; 422 → rejected with verbatim Violations. Anything else throws. */
  override(run: RunId, req: OverrideRequest): Promise<OverrideOutcome>;
  getMetrics(run: RunId): Promise<KpiVector>;
  getCompare(run: RunId): Promise<CompareResponse>;
  getBottleneck(run: RunId): Promise<BottleneckReport>;
  listScenarios(): Promise<readonly ScenarioSummary[]>;
  createScenario(req: ScenarioCreateRequest): Promise<ScenarioCreated>;
  /**
   * The parameter vocabulary, read against one base — names, ranges, and where
   * that base currently sits on each knob. Per-scenario because the value half
   * is a fact about the base; a global catalogue could only guess it.
   */
  getSliders(scenario: string): Promise<SliderCatalogue>;
  openStream(run: RunId, callbacks: StreamCallbacks): StreamHandle;
}

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string, url: string) {
    super(`API ${status} on ${url}: ${body.slice(0, 300)}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "content-type": "application/json", accept: "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new ApiError(res.status, await res.text(), url);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export function createHttpApi(baseUrl = "/api"): ConsoleApi {
  const u = (path: string): string => `${baseUrl}${path}`;

  return {
    mode: "live",

    createRun(req) {
      return requestJson<RunHandle>(u("/runs"), { method: "POST", body: JSON.stringify(req) });
    },

    getRun(run) {
      return requestJson<RunHandle>(u(`/runs/${run}`));
    },

    async deleteRun(run) {
      await requestJson<void>(u(`/runs/${run}`), { method: "DELETE" });
    },

    getLayout(run) {
      return requestJson<FloorLayout>(u(`/runs/${run}/layout`));
    },

    control(run, cmd) {
      return requestJson<SessionState>(u(`/runs/${run}/control`), {
        method: "POST",
        body: JSON.stringify(cmd),
      });
    },

    async override(run, req) {
      const url = u(`/runs/${run}/override`);
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: JSON.stringify(req),
      });
      if (res.status === 422) {
        // The rejection body IS the contract: verbatim Violations, nothing applied.
        return (await res.json()) as OverrideRejected;
      }
      if (!res.ok) {
        throw new ApiError(res.status, await res.text(), url);
      }
      return (await res.json()) as OverrideOutcome;
    },

    getMetrics(run) {
      return requestJson<KpiVector>(u(`/runs/${run}/metrics`));
    },

    getCompare(run) {
      return requestJson<CompareResponse>(u(`/runs/${run}/compare`));
    },

    getBottleneck(run) {
      return requestJson<BottleneckReport>(u(`/runs/${run}/bottleneck`));
    },

    listScenarios() {
      return requestJson<readonly ScenarioSummary[]>(u("/scenarios"));
    },

    createScenario(req) {
      return requestJson<ScenarioCreated>(u("/scenarios"), {
        method: "POST",
        body: JSON.stringify(req),
      });
    },

    getSliders(scenario) {
      return requestJson<SliderCatalogue>(u(`/scenarios/${scenario}/sliders`));
    },

    openStream(run, callbacks) {
      return openSseStream(u(`/runs/${run}/stream`), callbacks);
    },
  };
}
