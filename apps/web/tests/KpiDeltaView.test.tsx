import { describe, expect, test } from "bun:test";
import { render, screen } from "@testing-library/react";

import { KpiDeltaView, type RunReading } from "../src/components/KpiDeltaView";

const HOUR = 3600 * 1_000_000;

function reading(
  run: string,
  values: Record<string, number | null>,
  simTime = 24 * HOUR,
  seed = 42,
): RunReading {
  return { run, metrics: { values }, simTime, seed };
}

function view(previous: RunReading | null, current: RunReading | null, changes: string[] = []) {
  render(<KpiDeltaView previous={previous} current={current} changes={changes} />);
  return screen.getByLabelText("Run delta");
}

describe("KpiDeltaView", () => {
  test("renders a signed, verdict-labelled delta per key", () => {
    const panel = view(
      reading("run-1", { completions_per_week: 100, boarding_time_s_mean: 1200 }),
      reading("run-2", { completions_per_week: 130, boarding_time_s_mean: 900 }),
      ["Nurses 3 → 6"],
    );
    // completions: more is better, and it went UP -> better (delta = prev - cur = -30)
    expect(panel.textContent).toContain("−30");
    // boarding: less is better, and it FELL by 300s -> better
    expect(panel.textContent).toContain("+5.0m");
    expect(panel.textContent).toContain("better");
    expect(panel.textContent).toContain("Nurses 3 → 6");
  });

  test("names itself a single-seed point delta and points CIs elsewhere", () => {
    const panel = view(reading("run-1", { completions_per_week: 100 }), reading("run-2", {}));
    expect(panel.textContent).toContain("Single-seed point delta");
    expect(panel.textContent).toContain("seed 42 held");
    expect(panel.textContent).toContain("no confidence bound");
    expect(panel.textContent).toContain("compare panel");
  });

  test("a changed seed is called out as parameter AND weather", () => {
    const panel = view(
      reading("run-1", { completions_per_week: 100 }, 24 * HOUR, 42),
      reading("run-2", { completions_per_week: 130 }, 24 * HOUR, 43),
    );
    expect(panel.textContent).toContain("Seed changed (42 → 43)");
    expect(panel.textContent).toContain("parameter AND weather");
    expect(panel.textContent).not.toContain("Single-seed point delta");
  });

  test("mismatched cuts are flagged, because a live fold normalizes by elapsed time", () => {
    view(
      reading("run-1", { completions_per_week: 900 }, 96 * HOUR),
      reading("run-2", { completions_per_week: 3 }, HOUR / 4),
    );
    expect(screen.getByRole("alert").textContent).toContain("Live KPIs are folded over");
  });

  test("comparable cuts raise no alarm", () => {
    view(
      reading("run-1", { completions_per_week: 900 }, 24 * HOUR),
      reading("run-2", { completions_per_week: 950 }, 24 * HOUR + 60_000_000),
    );
    expect(screen.queryByRole("alert")).toBeNull();
  });

  test("an absent figure on either side is an em dash, never a delta of zero", () => {
    const panel = view(
      reading("run-1", { completions_per_week: 100, provider_util: null }),
      reading("run-2", { completions_per_week: 130 }),
    );
    // provider_util is null on one side and missing on the other -> no delta at all.
    expect(panel.textContent).toContain("—");
    expect(panel.textContent).not.toContain("±");
  });

  test("without a previous run it explains what to do instead of showing zeros", () => {
    const panel = view(null, reading("run-1", { completions_per_week: 100 }));
    expect(panel.textContent).toContain("Move a knob and Run");
  });

  test("a run compared with itself waits rather than differencing", () => {
    const panel = view(
      reading("run-1", { completions_per_week: 100 }),
      reading("run-1", { completions_per_week: 100 }),
    );
    expect(panel.textContent).toContain("waiting for the new run");
  });
});
