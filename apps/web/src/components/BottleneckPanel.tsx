/**
 * Binding-constraint + staff work-concentration indicators, rendered
 * VERBATIM from the analysis projection (the panel presents; it computes
 * nothing — Gini is an analysis concept, not a UI metric).
 */

import type { BottleneckReport } from "../api/types";
import { EM_DASH, isAbsent } from "./format";

export interface BottleneckPanelProps {
  report: BottleneckReport | null;
}

export function BottleneckPanel({ report }: BottleneckPanelProps) {
  return (
    <div className="panel" aria-label="Bottleneck panel">
      <h2>Bottleneck</h2>
      {report === null ? (
        <div className="muted small">waiting for analysis…</div>
      ) : (
        <>
          <div style={{ marginBottom: 6 }}>
            binding constraint: <strong>{report.binding}</strong>
          </div>
          {report.resources.map((r) => {
            // A share is absent (null/NaN) until some patient-time has been
            // observed. `null * 100` is 0, so rendering it unguarded would draw an
            // empty bar and "0%" — reading as "this resource holds nobody up"
            // when the truth is "nothing measured here yet".
            const share = r.share_of_cycle;
            const pct = isAbsent(share) ? null : share * 100;
            return (
              <div
                key={r.resource}
                className={`share-row${r.resource === report.binding ? " binding" : ""}`}
              >
                <span>{r.resource}</span>
                <span className="bar-track">
                  <span className="bar-fill" style={{ width: `${Math.round(pct ?? 0)}%` }} />
                </span>
                <span className="pct">{pct === null ? EM_DASH : `${pct.toFixed(0)}%`}</span>
              </div>
            );
          })}
          <div className="small muted" style={{ marginTop: 6 }}>
            work concentration (Gini): overall {report.gini_overall.toFixed(2)}
            {Object.entries(report.gini_by_role).map(([role, g]) => (
              <span key={role}>
                {" "}
                · {role} {g.toFixed(2)}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
