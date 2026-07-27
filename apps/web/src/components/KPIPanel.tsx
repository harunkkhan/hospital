/**
 * Live KpiVector tiles (GET /runs/{id}/metrics, pulled on a light interval).
 * Keys are treated as ⊆ KPI_KEYS, present-or-absent: an empty ESI stratum
 * mid-run renders "—", never 0 or NaN. No KPI math happens here — the API's
 * fold is the only folder.
 */

import type { KpiVector } from "../api/types";
import { EM_DASH, formatKpiValue, kpiLabel } from "./format";

const HEADLINE_KEYS: readonly string[] = [
  "completions_per_week",
  "wip_end_of_week",
  "door_to_triage_s_mean",
  "door_to_provider_s_mean",
  "door_to_provider_s_p90",
  "boarding_time_s_mean",
  "turnaround_time_s_mean",
  "bay_utilization",
  "provider_util",
  "nurse_util",
  "staff_minutes_walked",
];

const STAFF_FRAC_KEYS: readonly string[] = [
  "staff_frac_walk",
  "staff_frac_direct_care",
  "staff_frac_cleaning",
  "staff_frac_documentation",
  "staff_frac_idle",
];

export interface KPIPanelProps {
  metrics: KpiVector | null;
}

function valueOf(metrics: KpiVector | null, key: string): number | null | undefined {
  return metrics?.values[key];
}

export function KPIPanel({ metrics }: KPIPanelProps) {
  return (
    <div className="panel" aria-label="KPI panel">
      <h2>Live KPIs {metrics === null && <span className="muted">(waiting)</span>}</h2>
      <div className="tile-grid">
        {HEADLINE_KEYS.map((key) => (
          <div key={key} className="stat-tile">
            <div className="label" title={key}>
              {kpiLabel(key)}
            </div>
            <div className="value">{formatKpiValue(key, valueOf(metrics, key))}</div>
          </div>
        ))}
      </div>

      <h2 style={{ marginTop: 10 }}>Length of stay by acuity</h2>
      <table className="kv">
        <tbody>
          {([1, 2, 3, 4, 5] as const).map((esi) => {
            const meanKey = `los_s_mean_by_esi_${esi}`;
            const p90Key = `los_s_p90_by_esi_${esi}`;
            return (
              <tr key={esi}>
                <td>ESI-{esi}</td>
                <td>
                  mean {formatKpiValue(meanKey, valueOf(metrics, meanKey))}
                  <span className="muted"> · p90 </span>
                  {formatKpiValue(p90Key, valueOf(metrics, p90Key))}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h2 style={{ marginTop: 10 }}>Staff time split</h2>
      <table className="kv">
        <tbody>
          {STAFF_FRAC_KEYS.map((key) => (
            <tr key={key}>
              <td>{kpiLabel(key)}</td>
              <td>{metrics === null ? EM_DASH : formatKpiValue(key, valueOf(metrics, key))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
