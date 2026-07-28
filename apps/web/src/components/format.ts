/**
 * Pure formatting/classification helpers for the KPI tiles and compare
 * tiles. All display honesty rules live here so they are unit-testable:
 * - absent/NaN values render as an em dash, never 0 (empty ESI strata);
 * - a contrast's verdict follows a per-key direction-of-good map — negative
 *   deltas are shown as-is and classified "worse" when they are worse;
 * - replications == 1 means "point delta, no CI" — no fake bands.
 */

import type { KpiContrast } from "../api/types";

export const EM_DASH = "—";

// ---------------------------------------------------------------- sim time

/** µs since run start -> "D3 14:22:05". */
export function formatSimTime(us: number): string {
  const totalS = Math.max(0, Math.floor(us / 1_000_000));
  const day = Math.floor(totalS / 86_400) + 1;
  const h = Math.floor((totalS % 86_400) / 3600);
  const m = Math.floor((totalS % 3600) / 60);
  const s = totalS % 60;
  const pad = (n: number): string => String(n).padStart(2, "0");
  return `D${day} ${pad(h)}:${pad(m)}:${pad(s)}`;
}

// ---------------------------------------------------------------- KPI tiles

export type KpiUnit = "count" | "seconds" | "minutes" | "fraction";

const KEY_UNITS: readonly [RegExp, KpiUnit][] = [
  [/^completions|^wip/, "count"],
  [/_s_mean|_s_p90|^los_s_/, "seconds"],
  [/^staff_minutes/, "minutes"],
  [/util$|^staff_frac_|^bay_utilization$/, "fraction"],
];

export function kpiUnit(key: string): KpiUnit {
  for (const [pattern, unit] of KEY_UNITS) {
    if (pattern.test(key)) {
      return unit;
    }
  }
  return "count";
}

export function kpiLabel(key: string): string {
  return key
    .replace(/^los_s_(mean|p90)_by_esi_(\d)$/, "LOS $1 · ESI-$2")
    .replace(/_s_(mean|p90)$/, " $1 (s)")
    .replace(/^staff_frac_/, "time ")
    .replace(/_/g, " ");
}

function isAbsent(v: number | null | undefined): v is null | undefined {
  return v === null || v === undefined || Number.isNaN(v);
}

export function formatSeconds(v: number): string {
  if (v >= 3600) {
    return `${(v / 3600).toFixed(1)}h`;
  }
  if (v >= 60) {
    return `${(v / 60).toFixed(1)}m`;
  }
  return `${Math.round(v)}s`;
}

/** Empty strata (null/NaN/missing) render as an em dash, never 0. */
export function formatKpiValue(key: string, v: number | null | undefined): string {
  if (isAbsent(v)) {
    return EM_DASH;
  }
  switch (kpiUnit(key)) {
    case "seconds":
      return formatSeconds(v);
    case "minutes":
      return `${Math.round(v).toLocaleString()}m`;
    case "fraction":
      return `${(v * 100).toFixed(1)}%`;
    case "count":
      return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1);
  }
}

// ---------------------------------------------------------------- contrasts

export type Direction = "up" | "down" | "neutral";

/**
 * Which way is GOOD for each KPI. Utilizations and staff-time fractions are
 * deliberately neutral — more utilization is not unambiguously better, and
 * coloring them would editorialize.
 */
export function goodDirection(key: string): Direction {
  if (key === "completions_per_week") {
    return "up";
  }
  if (
    key === "wip_end_of_week" ||
    key === "staff_minutes_walked" ||
    key.startsWith("door_to_") ||
    key.startsWith("los_s_") ||
    key === "turnaround_time_s_mean" ||
    key === "boarding_time_s_mean" ||
    key === "staff_frac_walk"
  ) {
    return "down";
  }
  return "neutral";
}

export type Verdict = "better" | "worse" | "neutral";

/**
 * Honest classification of a contrast. delta = baseline - optimized, so for
 * a "down is good" key, a POSITIVE delta means optimized improved. Neutral
 * keys and zero deltas never get a color.
 */
export function contrastVerdict(c: Pick<KpiContrast, "key" | "delta">): Verdict {
  const dir = goodDirection(c.key);
  if (dir === "neutral" || c.delta === 0) {
    return "neutral";
  }
  const optimizedImproved = dir === "down" ? c.delta > 0 : c.delta < 0;
  return optimizedImproved ? "better" : "worse";
}

function formatByUnit(key: string, v: number): string {
  return formatKpiValue(key, v);
}

export function formatSigned(key: string, v: number): string {
  const sign = v > 0 ? "+" : v < 0 ? "−" : "±";
  return `${sign}${formatByUnit(key, Math.abs(v))}`;
}

/** "Δ b−o +4.2m · 95% CI [+1.1m, +7.4m]" or the honest n=1 state. */
export function formatCi(c: Pick<KpiContrast, "key" | "ci_lo" | "ci_hi">, replications: number): string {
  if (replications <= 1) {
    return "n=1 · point delta, no CI";
  }
  return `95% CI [${formatSigned(c.key, c.ci_lo)}, ${formatSigned(c.key, c.ci_hi)}]`;
}

export function significanceLabel(c: Pick<KpiContrast, "significant">, replications: number): string {
  if (replications <= 1) {
    return "unpowered";
  }
  return c.significant ? "significant" : "not significant";
}
