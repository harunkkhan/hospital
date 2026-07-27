import { describe, expect, test } from "bun:test";
import { render, screen } from "@testing-library/react";

import type { CompareResponse } from "../src/api/types";
import { CompareView } from "../src/components/CompareView";

const BOOTSTRAPPED: CompareResponse = {
  baseline_run: "run-base",
  optimized_run: "run-opt",
  replications: 16,
  contrasts: [
    {
      key: "door_to_provider_s_mean",
      baseline: 1920,
      optimized: 1500,
      delta: 420,
      ci_lo: 120,
      ci_hi: 720,
      significant: true,
    },
    {
      key: "turnaround_time_s_mean",
      baseline: 312,
      optimized: 334,
      delta: -22,
      ci_lo: -31,
      ci_hi: -12,
      significant: true,
    },
    {
      key: "door_to_triage_s_mean",
      baseline: 540,
      optimized: 528,
      delta: 12,
      ci_lo: -40,
      ci_hi: 66,
      significant: false,
    },
  ],
};

describe("CompareView with bootstrap CIs", () => {
  test("renders deltas with verdicts, CIs, and significance flags", () => {
    render(<CompareView compare={BOOTSTRAPPED} onRefresh={() => undefined} />);

    // improvements on down-good keys (verdict is independent of significance)
    expect(screen.getByText("+7.0m").textContent).toBeDefined();
    expect(screen.getAllByText("better").length).toBe(2);

    // honest regression: negative delta labeled worse, shown as-is
    expect(screen.getByText("−22s")).toBeDefined();
    expect(screen.getAllByText("worse").length).toBe(1);

    // significance mirrors the API flag
    expect(screen.getAllByText(/·\s*significant/).length).toBeGreaterThan(0);
    expect(screen.getByText(/not significant/)).toBeDefined();

    expect(screen.getByText(/16 paired replications/)).toBeDefined();
  });
});

describe("CompareView single-seed honesty", () => {
  test("replications == 1 shows the point-delta state and no CI band", () => {
    const pointDelta: CompareResponse = {
      ...BOOTSTRAPPED,
      replications: 1,
      contrasts: [BOOTSTRAPPED.contrasts[0] as CompareResponse["contrasts"][number]],
    };
    render(<CompareView compare={pointDelta} onRefresh={() => undefined} />);
    expect(screen.getByText(/single paired seed \(n=1\)/)).toBeDefined();
    expect(screen.getByText(/n=1 · point delta, no CI/)).toBeDefined();
    expect(screen.getByText(/unpowered/)).toBeDefined();
    expect(screen.queryByText(/95% CI/)).toBeNull();
  });
});
