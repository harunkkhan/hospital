/**
 * Operator overrides: pick an entity → choose an action → POST → render the
 * verdict. Two hard rules from the doc:
 * - a 422's Violation[] is rendered VERBATIM (kind + detail + entity), never
 *   softened into a generic "invalid";
 * - feedback is pending-then-authoritative, not optimistic: the map only
 *   recolors from streamed frames, so a stale-frame rejection ("the bay you
 *   saw free is now occupied") can never leave a phantom state behind.
 */

import { useEffect, useMemo, useState } from "react";

import type {
  FloorLayout,
  OperatorAction,
  OperatorActionKind,
  OverrideOutcome,
  OverrideRequest,
} from "../api/types";
import type { SelectedEntity } from "../state/runStore";
import type { WorldView } from "../state/streamReducer";
import { formatSimTime } from "./format";

const ACTION_LABELS: Readonly<Record<OperatorActionKind, string>> = {
  reassign: "Reassign patient → bay",
  bump_priority: "Bump patient priority",
  reroute: "Reroute staff → task",
  expedite_clean: "Expedite cleanup",
  expedite_discharge: "Expedite discharge",
  close_bay: "Hold / close bay",
  block_edge: "Block corridor edge",
};

export interface OverridePanelProps {
  layout: FloorLayout | null;
  world: WorldView;
  selected: SelectedEntity | null;
  onSubmit: (req: OverrideRequest) => Promise<OverrideOutcome>;
}

export function OverridePanel({ layout, world, selected, onSubmit }: OverridePanelProps) {
  const [kind, setKind] = useState<OperatorActionKind>("reassign");
  const [patient, setPatient] = useState("");
  const [bay, setBay] = useState("");
  const [staff, setStaff] = useState("");
  const [task, setTask] = useState("");
  const [edge, setEdge] = useState("");
  const [priority, setPriority] = useState(1);
  const [pin, setPin] = useState(true);
  const [pending, setPending] = useState(false);
  const [outcome, setOutcome] = useState<OverrideOutcome | null>(null);

  // A map selection prefills the matching field.
  useEffect(() => {
    if (selected?.type === "bay") {
      setBay(selected.id);
    } else if (selected?.type === "patient") {
      setPatient(selected.id);
    } else if (selected?.type === "staff") {
      setStaff(selected.id);
    }
  }, [selected]);

  const patients = useMemo(
    () => Object.values(world.patients).sort((a, b) => a.patient.localeCompare(b.patient)),
    [world.patients],
  );
  const staffList = useMemo(
    () => Object.values(world.staff).sort((a, b) => a.staff.localeCompare(b.staff)),
    [world.staff],
  );
  /** Pending-work targets derived from the live world (mock task vocabulary). */
  const tasks = useMemo(() => {
    const out: { id: string; label: string }[] = [];
    for (const bayFrame of Object.values(world.bays)) {
      if (bayFrame.status === "cleaning") {
        out.push({ id: `cleaning:${bayFrame.bay}`, label: `clean ${bayFrame.bay}` });
      }
      if (bayFrame.status === "occupied") {
        out.push({ id: `nurse_visit:${bayFrame.bay}`, label: `nurse visit ${bayFrame.bay}` });
        out.push({ id: `provider_visit:${bayFrame.bay}`, label: `provider visit ${bayFrame.bay}` });
      }
    }
    return out;
  }, [world.bays]);

  const buildAction = (): OperatorAction | null => {
    switch (kind) {
      case "reassign":
        return patient !== "" && bay !== "" ? { kind, patient, bay } : null;
      case "bump_priority":
        return patient !== "" ? { kind, patient, priority } : null;
      case "reroute":
        return staff !== "" && task !== "" ? { kind, staff, task } : null;
      case "expedite_clean":
        return bay !== "" ? { kind, bay } : null;
      case "expedite_discharge":
        return patient !== "" ? { kind, patient } : null;
      case "close_bay":
        return bay !== "" ? { kind, bay } : null;
      case "block_edge": {
        const [a, b] = edge.split("|");
        return a !== undefined && b !== undefined && a !== "" && b !== ""
          ? { kind, edge: [a, b] }
          : null;
      }
    }
  };

  const action = buildAction();

  const submit = async (): Promise<void> => {
    if (action === null || pending) {
      return;
    }
    setPending(true);
    setOutcome(null);
    try {
      setOutcome(await onSubmit({ action, pin }));
    } catch (err) {
      setOutcome({
        status: "rejected",
        violations: [
          {
            kind: "unknown_entity",
            detail: err instanceof Error ? err.message : String(err),
            entity: "request",
          },
        ],
      });
    } finally {
      setPending(false);
    }
  };

  const needsPatient = kind === "reassign" || kind === "bump_priority" || kind === "expedite_discharge";
  const needsBay = kind === "reassign" || kind === "close_bay" || kind === "expedite_clean";

  return (
    <div className="panel" aria-label="Override panel">
      <h2>Operator override</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <label className="field">
          action
          <select
            aria-label="override action"
            value={kind}
            onChange={(e) => {
              setKind(e.target.value as OperatorActionKind);
              setOutcome(null);
            }}
          >
            {Object.entries(ACTION_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>

        {needsPatient && (
          <label className="field">
            patient
            <select aria-label="patient" value={patient} onChange={(e) => setPatient(e.target.value)}>
              <option value="">choose…</option>
              {patients.map((p) => (
                <option key={p.patient} value={p.patient}>
                  {p.patient} · ESI-{p.esi} · {p.stage}
                </option>
              ))}
            </select>
          </label>
        )}

        {needsBay && (
          <label className="field">
            bay
            <select aria-label="bay" value={bay} onChange={(e) => setBay(e.target.value)}>
              <option value="">choose…</option>
              {(layout?.bays ?? []).map((b) => {
                const status = world.bays[b.id]?.status ?? "free";
                return (
                  <option key={b.id} value={b.id}>
                    {b.id} · {b.zone_type} · {status}
                  </option>
                );
              })}
            </select>
          </label>
        )}

        {kind === "bump_priority" && (
          <label className="field">
            priority
            <input
              aria-label="priority"
              type="number"
              min={1}
              max={9}
              value={priority}
              onChange={(e) => setPriority(Number(e.target.value))}
              style={{ width: 64 }}
            />
          </label>
        )}

        {kind === "reroute" && (
          <>
            <label className="field">
              staff
              <select aria-label="staff" value={staff} onChange={(e) => setStaff(e.target.value)}>
                <option value="">choose…</option>
                {staffList.map((s) => (
                  <option key={s.staff} value={s.staff}>
                    {s.staff} · {s.role} · {s.activity}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              task
              <select aria-label="task" value={task} onChange={(e) => setTask(e.target.value)}>
                <option value="">choose…</option>
                {tasks.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}

        {kind === "block_edge" && (
          <label className="field">
            edge
            <select aria-label="edge" value={edge} onChange={(e) => setEdge(e.target.value)}>
              <option value="">choose…</option>
              {(layout?.graph.edges ?? []).map((e) => (
                <option key={`${e.a}|${e.b}`} value={`${e.a}|${e.b}`}>
                  {e.a} ↔ {e.b}
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="field">
          <input type="checkbox" checked={pin} onChange={(e) => setPin(e.target.checked)} />
          pin (hold against solver re-solve)
        </label>

        <div>
          <button className="primary" onClick={() => void submit()} disabled={action === null || pending}>
            {pending ? "Validating…" : "Submit override"}
          </button>
        </div>
      </div>

      {outcome?.status === "applied" && (
        <div className="applied-note" role="status">
          <span className="status">applied</span> at {formatSimTime(outcome.applied_at)} · plan
          now carries {outcome.plan.items.length} item{outcome.plan.items.length === 1 ? "" : "s"}
          {pin && " · pinned"}
        </div>
      )}
      {outcome?.status === "rejected" && (
        <div role="alert">
          {outcome.violations.map((v, i) => (
            <div key={i} className="violation">
              <span className="kind">{v.kind}</span> — {v.detail}{" "}
              <span className="entity">({v.entity})</span>
            </div>
          ))}
          <div className="small muted" style={{ marginTop: 4 }}>
            Nothing was applied — the world is unchanged.
          </div>
        </div>
      )}
    </div>
  );
}
