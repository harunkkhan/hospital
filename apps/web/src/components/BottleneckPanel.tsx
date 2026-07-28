/**
 * Binding-constraint + staff work-concentration indicators, rendered
 * VERBATIM from the analysis projection (the panel presents; it computes
 * nothing — Gini is an analysis concept, not a UI metric).
 */

import type { BottleneckReport } from "../api/types";

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
          {report.resources.map((r) => (
            <div
              key={r.resource}
              className={`share-row${r.resource === report.binding ? " binding" : ""}`}
            >
              <span>{r.resource}</span>
              <span className="bar-track">
                <span
                  className="bar-fill"
                  style={{ width: `${Math.round(r.share_of_cycle * 100)}%` }}
                />
              </span>
              <span className="pct">{(r.share_of_cycle * 100).toFixed(0)}%</span>
            </div>
          ))}
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
