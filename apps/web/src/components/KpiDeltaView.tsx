/**
 * What the last slider move actually did: the KPI delta between the run the
 * operator came from and the run they are in now.
 *
 * This is NOT `/compare`, and the difference matters enough to be on the label.
 * `/compare` contrasts the baseline and optimized ARMS of one run under common
 * random numbers, and its numbers carry a paired bootstrap's CIs. This contrasts
 * TWO RUNS the operator launched, each a single realized week: it is a point
 * delta with no confidence bound at all, and no amount of formatting can turn it
 * into one. Two runs at the same seed differ only by the parameters that moved
 * (that is what holding the seed buys); at different seeds the delta is
 * parameter and weather together, and the caption says which it is.
 *
 * The second honesty rule is the cut. Live KPIs are folded over elapsed sim time
 * (`_live_window` in the API), so every rate- and duration-normalized figure is
 * relative to how far its run has got. Comparing a five-day reading against a
 * five-minute one is not a parameter effect, so both cuts are shown and a wide
 * gap is called out rather than quietly differenced.
 *
 * Sign convention matches CompareView: `delta = previous − current`, so the
 * *current* run plays the role optimized plays there and `contrastVerdict` reads
 * "better"/"worse" for it without a second direction-of-good table.
 */

import type { KpiVector } from "../api/types";
import { contrastVerdict, EM_DASH, formatKpiValue, formatSigned, formatSimTime, kpiLabel } from "./format";

/** One run's live reading, and the cut it was folded at. */
export interface RunReading {
  run: string;
  metrics: KpiVector;
  simTime: number;
  seed: number;
}

const DELTA_KEYS: readonly string[] = [
  "completions_per_week",
  "wip_end_of_week",
  "door_to_provider_s_mean",
  "boarding_time_s_mean",
  "turnaround_time_s_mean",
  "bay_utilization",
  "provider_util",
  "nurse_util",
];

/** Cuts further apart than this make a live-fold comparison misleading. */
const CUT_TOLERANCE_US = 60 * 60 * 1_000_000;

export interface KpiDeltaViewProps {
  previous: RunReading | null;
  current: RunReading | null;
  /** Pre-formatted "Nurses 3 → 6" lines for the knobs that moved. */
  changes: readonly string[];
}

function valueOf(reading: RunReading, key: string): number | null | undefined {
  return reading.metrics.values[key];
}

export function KpiDeltaView({ previous, current, changes }: KpiDeltaViewProps) {
  if (previous === null || current === null || previous.run === current.run) {
    return (
      <section className="knob-group" aria-label="Run delta">
        <h3>Since the last run</h3>
        <div className="small muted">
          {previous === null
            ? "Move a knob and Run — the run you are in now becomes what the next one is measured against."
            : "waiting for the new run's first fold…"}
        </div>
      </section>
    );
  }

  const sameSeed = previous.seed === current.seed;
  const cutGap = Math.abs(current.simTime - previous.simTime);

  return (
    <section className="knob-group" aria-label="Run delta">
      <h3>Since the last run</h3>
      <div className="small muted">
        {changes.length === 0 ? "no knobs moved — a re-run of the same parameters" : changes.join(" · ")}
      </div>
      <div className="small muted">
        {previous.run} @ {formatSimTime(previous.simTime)} → {current.run} @{" "}
        {formatSimTime(current.simTime)}
      </div>
      <div className="small" role="note">
        {sameSeed
          ? `Single-seed point delta (seed ${current.seed} held) — no confidence bound. Baseline-vs-optimized CIs live in the compare panel.`
          : `Seed changed (${previous.seed} → ${current.seed}), so this delta is parameter AND weather. Hold the seed to isolate the knob.`}
      </div>
      {cutGap > CUT_TOLERANCE_US && (
        <div className="small" role="alert">
          The two runs were read {formatSimTime(cutGap)} apart. Live KPIs are folded over
          elapsed sim time, so advance the new run to a comparable cut before reading this.
        </div>
      )}
      <div
        className="tile-grid"
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", marginTop: 6 }}
      >
        {DELTA_KEYS.map((key) => {
          const before = valueOf(previous, key);
          const after = valueOf(current, key);
          const delta =
            before === null || before === undefined || after === null || after === undefined
              ? null
              : before - after;
          const verdict = contrastVerdict({ key, delta });
          return (
            <div key={key} className="contrast-tile">
              <div className="key" title={key}>
                {kpiLabel(key)}
              </div>
              <div className={`delta ${verdict === "neutral" ? "" : verdict}`.trim()}>
                {formatSigned(key, delta)}
                {verdict !== "neutral" && (
                  <span style={{ fontSize: 11, marginLeft: 5 }}>{verdict}</span>
                )}
              </div>
              <div className="arms">
                {formatKpiValue(key, before)} {EM_DASH}&gt; {formatKpiValue(key, after)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="small muted" style={{ marginTop: 6 }}>
        Δ = previous − current, same sign convention as the arm compare: a key where
        less is better reads &quot;better&quot; on a positive Δ.
      </div>
    </section>
  );
}
