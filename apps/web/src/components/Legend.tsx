/** Status + acuity legends, fed by the one ramp definition in colors.ts. */

import {
  BAY_STATUS_COLORS,
  BAY_STATUS_LABELS,
  ESI_COLORS,
  STAFF_DOT_COLOR,
} from "../render/colors";
import type { BayStatus, EsiAcuity } from "../api/types";

const STATUSES: readonly BayStatus[] = ["free", "occupied", "cleaning", "closed"];
const ACUITIES: readonly EsiAcuity[] = [1, 2, 3, 4, 5];

export function Legend() {
  return (
    <div className="panel">
      <h2>Legend</h2>
      <div className="legend-row" style={{ marginBottom: 6 }}>
        {STATUSES.map((status) => (
          <span key={status}>
            <span className="swatch" style={{ background: BAY_STATUS_COLORS[status] }} />
            Bay {BAY_STATUS_LABELS[status].toLowerCase()}
          </span>
        ))}
      </div>
      <div className="legend-row">
        {ACUITIES.map((esi) => (
          <span key={esi}>
            <span
              className="swatch"
              style={{ background: ESI_COLORS[esi], borderRadius: "50%" }}
            />
            ESI-{esi}
            {esi === 1 ? " (most critical)" : ""}
          </span>
        ))}
        <span>
          <span
            className="swatch"
            style={{ background: STAFF_DOT_COLOR, borderRadius: 1 }}
          />
          Staff
        </span>
      </div>
    </div>
  );
}
