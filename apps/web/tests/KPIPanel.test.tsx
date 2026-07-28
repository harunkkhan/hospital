import { describe, expect, test } from "bun:test";
import { render, screen } from "@testing-library/react";

import type { KpiVector } from "../src/api/types";
import { KPI_KEYS } from "../src/api/types";
import { KPIPanel } from "../src/components/KPIPanel";

function vector(overrides: Record<string, number | null>): KpiVector {
  const values: Record<string, number | null> = {};
  for (const key of KPI_KEYS) {
    values[key] = 0.1;
  }
  return { values: { ...values, ...overrides } };
}

describe("KPIPanel", () => {
  test("renders values and em-dashes empty strata (never 0 for missing)", () => {
    render(
      <KPIPanel
        metrics={vector({
          completions_per_week: 968,
          los_s_mean_by_esi_1: null, // no ESI-1 completions yet
          los_s_p90_by_esi_1: null,
          door_to_provider_s_mean: 1920,
        })}
      />,
    );
    expect(screen.getByText("968")).toBeDefined();
    expect(screen.getByText("32.0m")).toBeDefined(); // 1920s
    const esi1Row = screen.getByText("ESI-1").closest("tr");
    expect(esi1Row?.textContent).toContain("—");
  });

  test("shows a waiting state before the first metrics arrive", () => {
    render(<KPIPanel metrics={null} />);
    expect(screen.getByText("(waiting)")).toBeDefined();
  });
});
