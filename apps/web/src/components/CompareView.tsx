/**
 * Baseline-vs-optimized delta tiles. Honesty rules (doc 07 nuances §7.11):
 * - CIs and significance come from the API's paired bootstrap — the view
 *   computes no stats;
 * - replications == 1 renders an explicit "n=1 · point delta, no CI" state,
 *   never a fake band;
 * - negative deltas (optimized worse) are shown as-is and labeled "worse";
 *   direction-of-good is per key, and neutral keys get no verdict color.
 * Verdict color never carries meaning alone — the label says better/worse.
 */

import type { CompareResponse } from "../api/types";
import {
  contrastVerdict,
  formatCi,
  formatKpiValue,
  formatSigned,
  kpiLabel,
  significanceLabel,
} from "./format";

export interface CompareViewProps {
  compare: CompareResponse | null;
  error?: string | null;
  onRefresh: () => void;
}

export function CompareView({ compare, error = null, onRefresh }: CompareViewProps) {
  return (
    <div className="panel" aria-label="Compare view">
      <h2>
        Baseline vs optimized{" "}
        <button style={{ float: "right", fontSize: 11, padding: "1px 8px" }} onClick={onRefresh}>
          refresh
        </button>
      </h2>
      {error !== null && <div className="small muted">compare unavailable: {error}</div>}
      {compare === null && error === null && <div className="small muted">waiting…</div>}
      {compare !== null && (
        <>
          <div className="small muted" style={{ marginBottom: 6 }}>
            {compare.baseline_run} vs {compare.optimized_run} ·{" "}
            {compare.replications === 1
              ? "single paired seed (n=1) — point deltas only"
              : `${compare.replications} paired replications`}
          </div>
          <div className="tile-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))" }}>
            {compare.contrasts.map((c) => {
              const verdict = contrastVerdict(c);
              return (
                <div key={c.key} className="contrast-tile">
                  <div className="key" title={c.key}>
                    {kpiLabel(c.key)}
                  </div>
                  <div className={`delta ${verdict === "neutral" ? "" : verdict}`.trim()}>
                    {formatSigned(c.key, c.delta)}
                    {verdict !== "neutral" && (
                      <span style={{ fontSize: 11, marginLeft: 5 }}>{verdict}</span>
                    )}
                  </div>
                  <div className="arms">
                    b {formatKpiValue(c.key, c.baseline)} → o {formatKpiValue(c.key, c.optimized)}
                  </div>
                  <div className="ci">
                    {formatCi(c, compare.replications)} ·{" "}
                    {significanceLabel(c, compare.replications)}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            Δ = baseline − optimized. CIs from the paired bootstrap in hospital.analysis.
          </div>
        </>
      )}
    </div>
  );
}
