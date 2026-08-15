/**
 * The scenario lab: push demand and supply around, re-run, see what it did.
 *
 * Replaces the old type-a-dotted-key override table. Three things make it a lab
 * rather than a form:
 *
 * - **The knobs are DATA.** Names, groups, ranges, units and each knob's current
 *   value come from `GET /scenarios/{id}/sliders`, read against the selected
 *   base. The panel hard-codes no parameter list, so it cannot drift from what
 *   the data layer accepts, and it opens on the scenario's real values instead
 *   of an invented default.
 * - **Only what moved is submitted.** A knob left at its base value is not sent.
 *   Re-stating a value is not always a no-op server-side (a headcount override
 *   also flattens per-shift variation), so "unchanged" must mean "absent".
 * - **A re-run REPLACES the session** (doc 07 nuances 7.12): workload and layout
 *   are generation-time inputs drawn at build under `RandomStreams(seed)`, so a
 *   slider cannot hot-edit a live run — only a new run realizes it.
 *
 * Supply here is CAPACITY, not consumables: this simulator has no inventory of
 * gloves, saline or kits to model. What it does have is bays, rooms, suites and
 * the labour that turns a bay over — which is what actually gates a patient —
 * and the panel says so rather than offering a knob the engine cannot honor.
 *
 * The seed / arm / CRN-shadow controls are load-bearing and survive verbatim:
 * holding the seed across a slider sweep is what makes two runs comparable
 * (common random numbers), and changing seed and slider together confounds the
 * parameter with the weather.
 */

import { useCallback, useEffect, useState } from "react";

import type {
  Arm,
  KpiVector,
  RunRequest,
  ScenarioCreated,
  ScenarioCreateRequest,
  ScenarioSummary,
  SliderCatalogue,
  SliderGroup,
  SliderSpec,
} from "../api/types";
import { KpiDeltaView, type RunReading } from "./KpiDeltaView";

const GROUP_ORDER: readonly SliderGroup[] = ["demand", "staffing", "capacity"];

const GROUP_COPY: Readonly<Record<SliderGroup, { title: string; note: string }>> = {
  demand: { title: "Demand", note: "how much work arrives, and in what mix" },
  staffing: { title: "Staffing", note: "the clinicians who see patients" },
  capacity: {
    title: "Capacity",
    note:
      "where patients are seen, and the labour that turns a bay over. " +
      "This model has capacity, not consumables — nothing here tracks gloves, " +
      "saline or kits, so housekeeping and porters are the supply knobs that " +
      "actually free beds.",
  },
};

/** Slider positions differing by less than this are the same position. */
const EPSILON = 1e-9;

export function isDirty(knob: SliderSpec, value: number): boolean {
  return Math.abs(value - knob.value) > EPSILON;
}

export function formatKnobValue(knob: SliderSpec, value: number): string {
  return knob.step >= 1 ? String(Math.round(value)) : value.toFixed(2);
}

function formatDelta(knob: SliderSpec, value: number): string {
  const delta = value - knob.value;
  const sign = delta > 0 ? "+" : "−";
  return `${sign}${formatKnobValue(knob, Math.abs(delta))}`;
}

function baseValues(catalogue: SliderCatalogue): Record<string, number> {
  return Object.fromEntries(catalogue.knobs.map((knob) => [knob.key, knob.value]));
}

export interface ScenarioLabProps {
  scenarios: readonly ScenarioSummary[] | null;
  currentSeed: number;
  /** Reads the knob catalogue for a base scenario. Must be referentially stable. */
  loadCatalogue: (base: string) => Promise<SliderCatalogue>;
  onRerun: (req: RunRequest) => void;
  onSaveScenario: (req: ScenarioCreateRequest) => Promise<ScenarioCreated>;
  /** The live run's id and reading — snapshotted on Run to become the "before". */
  runId: string | null;
  metrics: KpiVector | null;
  simTime: number;
}

export function ScenarioLab({
  scenarios,
  currentSeed,
  loadCatalogue,
  onRerun,
  onSaveScenario,
  runId,
  metrics,
  simTime,
}: ScenarioLabProps) {
  const [base, setBase] = useState("");
  const [seed, setSeed] = useState(currentSeed);
  const [arm, setArm] = useState<Arm>("optimized");
  const [withShadow, setWithShadow] = useState(true);
  const [catalogue, setCatalogue] = useState<SliderCatalogue | null>(null);
  const [catalogueError, setCatalogueError] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, number>>({});
  const [savedId, setSavedId] = useState<string | null>(null);
  // The run being left, captured at the moment Run is pressed. Held here rather
  // than refetched because it is gone the instant the session is replaced — the
  // API keeps no history, and this is the only place that reading still exists.
  const [previous, setPrevious] = useState<RunReading | null>(null);
  const [changes, setChanges] = useState<readonly string[]>([]);

  useEffect(() => {
    if (base === "" && scenarios !== null && scenarios.length > 0) {
      setBase(scenarios[0]?.id ?? "");
    }
  }, [scenarios, base]);

  // The catalogue is per-base, so a base switch invalidates both it and every
  // slider position: the values belong to the scenario they were read against.
  useEffect(() => {
    if (base === "") {
      return;
    }
    let cancelled = false;
    setCatalogue(null);
    setCatalogueError(null);
    void loadCatalogue(base)
      .then((next) => {
        if (!cancelled) {
          setCatalogue(next);
          setValues(baseValues(next));
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setCatalogueError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [base, loadCatalogue]);

  const knobs = catalogue?.knobs ?? [];
  const moved = knobs.filter((knob) => isDirty(knob, values[knob.key] ?? knob.value));
  // Only the knobs that actually moved: an unchanged slider must be ABSENT, not
  // re-stated, because re-stating one is not always a server-side no-op.
  const overrides = Object.fromEntries(
    moved.map((knob) => [knob.key, values[knob.key] ?? knob.value]),
  );

  const reset = useCallback(() => {
    if (catalogue !== null) {
      setValues(baseValues(catalogue));
    }
  }, [catalogue]);

  const run = (): void => {
    if (base === "") {
      return;
    }
    // Snapshot what we are leaving BEFORE the session is replaced, together with
    // the knob moves that separate the two runs — the delta is only readable as
    // "this changed, and here is what it did".
    setPrevious(
      runId === null || metrics === null
        ? null
        : { run: runId, metrics, simTime, seed: currentSeed },
    );
    setChanges(
      moved.map(
        (knob) =>
          `${knob.label} ${formatKnobValue(knob, knob.value)} → ` +
          `${formatKnobValue(knob, values[knob.key] ?? knob.value)}`,
      ),
    );
    onRerun({
      scenario: moved.length > 0 ? { base, overrides } : { id: base },
      seed,
      arm,
      compare_to: withShadow ? (arm === "optimized" ? "baseline" : "optimized") : null,
      start: "paused",
    });
  };

  const save = async (): Promise<void> => {
    if (base === "") {
      return;
    }
    setSavedId((await onSaveScenario({ base, overrides })).id);
  };

  return (
    <div className="panel" aria-label="Scenario lab">
      <h2>Scenario lab</h2>

      <div className="lab-head">
        <label className="field">
          base
          <select aria-label="base scenario" value={base} onChange={(e) => setBase(e.target.value)}>
            {(scenarios ?? []).map((s) => (
              <option key={s.id} value={s.id} title={s.note}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          seed
          <input
            aria-label="seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </label>
        <label className="field">
          arm
          <select aria-label="arm" value={arm} onChange={(e) => setArm(e.target.value as Arm)}>
            <option value="optimized">optimized</option>
            <option value="baseline">baseline</option>
          </select>
        </label>
        <label className="field">
          <input
            type="checkbox"
            aria-label="CRN shadow arm"
            checked={withShadow}
            onChange={(e) => setWithShadow(e.target.checked)}
          />
          CRN shadow arm
        </label>
      </div>

      {catalogueError !== null && (
        <div className="small" role="alert">
          parameter catalogue unavailable: {catalogueError}
        </div>
      )}
      {catalogue === null && catalogueError === null && (
        <div className="small muted">reading the parameter catalogue…</div>
      )}

      {GROUP_ORDER.map((group) => {
        const inGroup = knobs.filter((knob) => knob.group === group);
        if (inGroup.length === 0) {
          return null;
        }
        return (
          <section key={group} aria-label={`${group} knobs`} className="knob-group">
            <h3>{GROUP_COPY[group].title}</h3>
            <p className="small muted">{GROUP_COPY[group].note}</p>
            {inGroup.map((knob) => {
              const value = values[knob.key] ?? knob.value;
              const dirty = isDirty(knob, value);
              return (
                <div key={knob.key} className="knob-row">
                  <div className="knob-label" title={knob.key}>
                    {knob.label}
                  </div>
                  <div className="knob-value mono">
                    {formatKnobValue(knob, value)} <span className="muted">{knob.unit}</span>
                  </div>
                  <div className="knob-flag small">
                    {dirty ? (
                      <span className="badge warn">
                        {formatDelta(knob, value)} vs base {formatKnobValue(knob, knob.value)}
                      </span>
                    ) : (
                      <span className="muted">= base</span>
                    )}
                  </div>
                  <input
                    type="range"
                    aria-label={knob.key}
                    // The server's range is an affordance, and the base value is a
                    // fact — so the track is widened to admit a base that sits
                    // outside it rather than snapping the scenario to the range.
                    min={Math.min(knob.min, knob.value)}
                    max={Math.max(knob.max, knob.value)}
                    step={knob.step}
                    value={value}
                    onChange={(e) =>
                      setValues({ ...values, [knob.key]: Number(e.target.value) })
                    }
                  />
                </div>
              );
            })}
          </section>
        );
      })}

      <div className="lab-actions">
        <button className="primary" onClick={run} disabled={base === ""}>
          Run
        </button>
        <button onClick={reset} disabled={moved.length === 0}>
          Reset to base
        </button>
        <button onClick={() => void save()} disabled={base === "" || moved.length === 0}>
          Save scenario
        </button>
        <span className="small muted" role="status">
          {moved.length === 0
            ? "at base"
            : `${moved.length} knob${moved.length === 1 ? "" : "s"} moved`}
        </span>
      </div>

      <KpiDeltaView
        previous={previous}
        current={
          runId === null || metrics === null
            ? null
            : { run: runId, metrics, simTime, seed: currentSeed }
        }
        changes={changes}
      />

      {savedId !== null && <div className="small muted">saved as {savedId}</div>}
      <div className="small muted">
        A run replaces the live session — parameters are generation-time inputs, so
        nothing here edits the run in flight. The seed is held so a slider sweep is
        read under common random numbers rather than against different weather.
      </div>
    </div>
  );
}
