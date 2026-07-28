import { describe, expect, test } from "bun:test";
import { render, screen } from "@testing-library/react";

import type { BottleneckReport } from "../src/api/types";
import { BottleneckPanel } from "../src/components/BottleneckPanel";
import { EM_DASH } from "../src/components/format";

function report(overrides: Partial<BottleneckReport> = {}): BottleneckReport {
  return {
    binding: "provider",
    resources: [
      {
        resource: "provider",
        total_wait_s: 4200,
        n_requests: 12,
        mean_wait_s: 350,
        share_of_cycle: 0.42,
      },
      // Nothing has queued for the lab yet: `analysis` reports NaN for both the
      // mean and the share, which cross the wire as null.
      { resource: "lab", total_wait_s: 0, n_requests: 0, mean_wait_s: null, share_of_cycle: null },
    ],
    total_cycle_s: 10_000,
    gini_by_role: { physician: 0.21, nurse: 0.34 },
    gini_overall: 0.28,
    ...overrides,
  };
}

describe("BottleneckPanel", () => {
  test("renders the binding constraint and the full ranked table", () => {
    render(<BottleneckPanel report={report()} />);
    // Named twice on purpose: once as the binding constraint, once in the table.
    expect(screen.getAllByText("provider")).toHaveLength(2);
    expect(screen.getByText("42%")).toBeDefined();
    // The ranking is not trimmed to the winner -- a co-binding partner stays visible.
    expect(screen.getByText("lab")).toBeDefined();
  });

  test("an unmeasured share renders as an em dash, never 0%", () => {
    // `null * 100` is 0, so an unguarded render says "this resource holds nobody
    // up" when the truth is "nothing measured here yet".
    render(<BottleneckPanel report={report()} />);
    const labRow = screen.getByText("lab").closest("div");
    expect(labRow?.textContent).toContain(EM_DASH);
    expect(labRow?.textContent).not.toContain("0%");
  });

  test("no report at all is an explicit waiting state", () => {
    render(<BottleneckPanel report={null} />);
    expect(screen.getByText(/waiting for analysis/)).toBeDefined();
  });
});
