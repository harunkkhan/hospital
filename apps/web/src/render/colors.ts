/**
 * Single source of truth for both ramps (consumed by the FloorMap layers and
 * the Legend — one definition, per doc 07 §7.4).
 *
 * Colors come from the validated dataviz reference palette (dark-surface
 * steps; the console is dark-first). Status colors are stateful by design:
 * good/warning/critical map onto free/cleaning/closed; occupied is the
 * series blue (normal operation, not an alert).
 *
 * The acuity sign trap (core.enums): EsiAcuity 1 = MOST critical, so visual
 * weight lands on LOW numbers — ESI-1 is the critical red and the ramp cools
 * toward ESI-5. Color never carries meaning alone: bays are rects, patients
 * circles, staff squares, and the Legend labels every hue.
 */

import type { BayStatus, EsiAcuity } from "../api/types";

export const BAY_STATUS_COLORS: Readonly<Record<BayStatus, string>> = {
  free: "#0ca30c", // status good
  occupied: "#3987e5", // series-1 blue (dark step)
  cleaning: "#fab219", // status warning
  closed: "#d03b3b", // status critical
};

export const BAY_STATUS_LABELS: Readonly<Record<BayStatus, string>> = {
  free: "Free",
  occupied: "Occupied",
  cleaning: "Cleaning",
  closed: "Closed",
};

/** ESI 1 (most critical) -> hottest; ESI 5 -> muted. */
export const ESI_COLORS: Readonly<Record<EsiAcuity, string>> = {
  1: "#d03b3b", // critical
  2: "#ec835a", // serious
  3: "#fab219", // warning
  4: "#0ca30c", // good
  5: "#898781", // muted
};

export const STAFF_DOT_COLOR = "#ffffff";
export const STAFF_DOT_RING = "#0d0d0d";
export const CHIP_RING = "#0d0d0d";
export const EDGE_COLOR = "#383835";
export const BLOCKED_EDGE_COLOR = "#d03b3b";
export const NODE_COLOR = "#52514e";
export const LABEL_COLOR = "#898781";
export const SELECTION_COLOR = "#ffffff";
