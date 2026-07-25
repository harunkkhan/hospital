import { describe, expect, test } from "bun:test";

import {
  contrastVerdict,
  EM_DASH,
  formatCi,
  formatKpiValue,
  formatSigned,
  formatSimTime,
  goodDirection,
  significanceLabel,
} from "../src/components/format";

describe("formatKpiValue (empty strata honesty)", () => {
  test("null, undefined, and NaN render as an em dash — never 0", () => {
    expect(formatKpiValue("los_s_mean_by_esi_1", null)).toBe(EM_DASH);
    expect(formatKpiValue("los_s_mean_by_esi_1", undefined)).toBe(EM_DASH);
    expect(formatKpiValue("los_s_mean_by_esi_1", Number.NaN)).toBe(EM_DASH);
  });

  test("units follow the key", () => {
    expect(formatKpiValue("door_to_provider_s_mean", 90)).toBe("1.5m");
    expect(formatKpiValue("los_s_mean_by_esi_3", 7200)).toBe("2.0h");
    expect(formatKpiValue("bay_utilization", 0.625)).toBe("62.5%");
    expect(formatKpiValue("completions_per_week", 968)).toBe("968");
    expect(formatKpiValue("staff_minutes_walked", 5181)).toBe("5,181m");
  });
});

describe("contrastVerdict (delta = baseline - optimized)", () => {
  test("down-good key, positive delta -> optimized improved", () => {
    expect(contrastVerdict({ key: "door_to_provider_s_mean", delta: 300 })).toBe("better");
  });

  test("down-good key, negative delta -> honest 'worse'", () => {
    expect(contrastVerdict({ key: "turnaround_time_s_mean", delta: -22 })).toBe("worse");
  });

  test("up-good key, negative delta -> optimized improved", () => {
    expect(contrastVerdict({ key: "completions_per_week", delta: -34 })).toBe("better");
  });

  test("up-good key, positive delta -> worse", () => {
    expect(contrastVerdict({ key: "completions_per_week", delta: 12 })).toBe("worse");
  });

  test("neutral keys and zero deltas never get a verdict", () => {
    expect(contrastVerdict({ key: "provider_util", delta: 0.2 })).toBe("neutral");
    expect(contrastVerdict({ key: "door_to_provider_s_mean", delta: 0 })).toBe("neutral");
  });

  test("direction map covers the contract families", () => {
    expect(goodDirection("los_s_p90_by_esi_2")).toBe("down");
    expect(goodDirection("staff_minutes_walked")).toBe("down");
    expect(goodDirection("completions_per_week")).toBe("up");
    expect(goodDirection("nurse_util")).toBe("neutral");
  });
});

describe("CI formatting honesty", () => {
  const c = { key: "door_to_provider_s_mean", ci_lo: 60, ci_hi: 420 };

  test("replications > 1 renders the bootstrap interval", () => {
    expect(formatCi(c, 16)).toBe("95% CI [+1.0m, +7.0m]");
  });

  test("replications == 1 renders the explicit point-delta state, no fake band", () => {
    expect(formatCi(c, 1)).toBe("n=1 · point delta, no CI");
    expect(significanceLabel({ significant: true }, 1)).toBe("unpowered");
  });

  test("significance labels mirror the API flag", () => {
    expect(significanceLabel({ significant: true }, 16)).toBe("significant");
    expect(significanceLabel({ significant: false }, 16)).toBe("not significant");
  });
});

describe("formatSigned / formatSimTime", () => {
  test("signs are explicit, including the true minus sign", () => {
    expect(formatSigned("completions_per_week", 34)).toBe("+34");
    expect(formatSigned("completions_per_week", -34)).toBe("−34");
  });

  test("sim time renders day + clock", () => {
    expect(formatSimTime(0)).toBe("D1 00:00:00");
    expect(formatSimTime((2 * 86_400 + 3 * 3600 + 4 * 60 + 5) * 1_000_000)).toBe("D3 03:04:05");
  });
});
