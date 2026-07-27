/**
 * Scenario parameter overrides that RE-RUN (never hot-edit a live run —
 * parameters are generation-time inputs). Seed is an explicit control and is
 * held by default so a slider sweep isolates the parameter under CRN;
 * changing seed and slider together confounds parameter with weather.
 */

import { useEffect, useState } from "react";

import type { Arm, RunRequest, ScenarioCreated, ScenarioCreateRequest, ScenarioSummary } from "../api/types";

interface OverrideRow {
  key: string;
  value: number;
}

const SUGGESTED_KEYS = [
  "workload.arrival_rate_multiplier",
  "workload.ambulance_share",
  "staffing.nurse_count",
  "staffing.physician_count",
  "facility.fast_track_bays",
];

export interface ScenarioControlsProps {
  scenarios: readonly ScenarioSummary[] | null;
  currentSeed: number;
  onRerun: (req: RunRequest) => void;
  onSaveScenario: (req: ScenarioCreateRequest) => Promise<ScenarioCreated>;
}

export function ScenarioControls({
  scenarios,
  currentSeed,
  onRerun,
  onSaveScenario,
}: ScenarioControlsProps) {
  const [base, setBase] = useState("");
  const [seed, setSeed] = useState(currentSeed);
  const [arm, setArm] = useState<Arm>("optimized");
  const [withShadow, setWithShadow] = useState(true);
  const [rows, setRows] = useState<OverrideRow[]>([]);
  const [savedId, setSavedId] = useState<string | null>(null);

  useEffect(() => {
    if (base === "" && scenarios !== null && scenarios.length > 0) {
      setBase(scenarios[0]?.id ?? "");
    }
  }, [scenarios, base]);

  const overrides = Object.fromEntries(
    rows.filter((r) => r.key.trim() !== "").map((r) => [r.key.trim(), r.value]),
  );

  const rerun = (): void => {
    if (base === "") {
      return;
    }
    onRerun({
      scenario: rows.length > 0 ? { base, overrides } : { id: base },
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
    const created = await onSaveScenario({ base, overrides });
    setSavedId(created.id);
  };

  return (
    <div className="panel" aria-label="Scenario controls">
      <h2>Scenario</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
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
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
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
              checked={withShadow}
              onChange={(e) => setWithShadow(e.target.checked)}
            />
            CRN shadow arm
          </label>
        </div>

        {rows.map((row, i) => (
          <div key={i} style={{ display: "flex", gap: 6 }}>
            <input
              aria-label={`override key ${i + 1}`}
              list="scenario-override-keys"
              placeholder="parameter key"
              value={row.key}
              style={{ flex: 1, minWidth: 0 }}
              onChange={(e) =>
                setRows(rows.map((r, j) => (j === i ? { ...r, key: e.target.value } : r)))
              }
            />
            <input
              aria-label={`override value ${i + 1}`}
              type="number"
              step="0.1"
              value={row.value}
              style={{ width: 76 }}
              onChange={(e) =>
                setRows(rows.map((r, j) => (j === i ? { ...r, value: Number(e.target.value) } : r)))
              }
            />
            <button onClick={() => setRows(rows.filter((_, j) => j !== i))}>×</button>
          </div>
        ))}
        <datalist id="scenario-override-keys">
          {SUGGESTED_KEYS.map((k) => (
            <option key={k} value={k} />
          ))}
        </datalist>

        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={() => setRows([...rows, { key: "", value: 1 }])}>Add override</button>
          <button onClick={() => void save()} disabled={base === "" || rows.length === 0}>
            Save scenario
          </button>
          <button className="primary" onClick={rerun} disabled={base === ""}>
            Re-run
          </button>
        </div>
        {savedId !== null && (
          <div className="small muted" role="status">
            saved as {savedId}
          </div>
        )}
        <div className="small muted">
          Re-run replaces the live session. Seed is held so parameter deltas are CRN-comparable.
        </div>
      </div>
    </div>
  );
}
