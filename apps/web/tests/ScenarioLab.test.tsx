import { describe, expect, test } from "bun:test";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import type {
  KpiVector,
  RunRequest,
  ScenarioSummary,
  SliderCatalogue,
} from "../src/api/types";
import { ScenarioLab } from "../src/components/ScenarioLab";

const SCENARIOS: ScenarioSummary[] = [
  { id: "er_floor", name: "Reference ER week", horizon: 1, note: "seeded" },
  { id: "surge", name: "Monday surge", horizon: 1, note: "surge overlay" },
];

function catalogueFor(scenario: string): SliderCatalogue {
  const nurses = scenario === "surge" ? 9 : 3;
  return {
    scenario,
    knobs: [
      {
        key: "workload.arrival_rate_multiplier",
        label: "Arrival rate",
        group: "demand",
        min: 0.25,
        max: 3,
        step: 0.05,
        unit: "x base",
        value: 1,
      },
      {
        key: "staffing.nurse_count",
        label: "Nurses",
        group: "staffing",
        min: 0,
        max: 12,
        step: 1,
        unit: "on duty",
        value: nurses,
      },
      {
        key: "facility.general_bays",
        label: "General bays",
        group: "capacity",
        min: 0,
        max: 6,
        step: 1,
        unit: "bays",
        value: 6,
      },
      {
        key: "staffing.housekeeping_count",
        label: "Housekeeping",
        group: "capacity",
        min: 0,
        max: 12,
        step: 1,
        unit: "on duty",
        value: 1,
      },
    ],
  };
}

interface Reading {
  runId: string | null;
  metrics: KpiVector | null;
  simTime: number;
}

const NO_RUN: Reading = { runId: null, metrics: null, simTime: 0 };

function kpis(values: Record<string, number>): KpiVector {
  return { values };
}

function setup(reading: Reading = NO_RUN): {
  runs: RunRequest[];
  loads: string[];
  show: (next: Reading) => void;
} {
  const runs: RunRequest[] = [];
  const loads: string[] = [];
  // One stable identity: the catalogue effect keys on the base, and a fresh
  // function per render would reload (and reset) the panel on every rerender.
  const loadCatalogue = (base: string): Promise<SliderCatalogue> => {
    loads.push(base);
    return Promise.resolve(catalogueFor(base));
  };
  const view = (next: Reading) => (
    <ScenarioLab
      scenarios={SCENARIOS}
      currentSeed={42}
      loadCatalogue={loadCatalogue}
      onRerun={(req) => runs.push(req)}
      onSaveScenario={() => Promise.resolve({ id: "scn-01" })}
      runId={next.runId}
      metrics={next.metrics}
      simTime={next.simTime}
    />
  );
  const { rerender } = render(view(reading));
  return { runs, loads, show: (next) => rerender(view(next)) };
}

async function nurseSlider(): Promise<HTMLInputElement> {
  return (await screen.findByLabelText("staffing.nurse_count")) as HTMLInputElement;
}

describe("ScenarioLab renders from the catalogue", () => {
  test("draws every knob the server published, grouped, with its base value", async () => {
    setup();
    const nurses = await nurseSlider();
    expect(nurses.value).toBe("3");
    expect(nurses.min).toBe("0");
    expect(nurses.max).toBe("12");
    expect(nurses.step).toBe("1");

    // Labels and units are the server's, not the panel's.
    expect(screen.getByText("Nurses")).toBeDefined();
    expect(screen.getByText("General bays")).toBeDefined();
    expect(screen.getAllByText("on duty").length).toBe(2);

    // Grouped demand / staffing / capacity...
    expect(screen.getByLabelText("demand knobs")).toBeDefined();
    expect(screen.getByLabelText("staffing knobs")).toBeDefined();
    const capacity = screen.getByLabelText("capacity knobs");
    // ...with the turnaround labour under capacity, and the honest note about
    // what "supply" means in a model with no consumables.
    expect(capacity.textContent).toContain("Housekeeping");
    expect(capacity.textContent).toContain("capacity, not consumables");

    // Nothing moved yet.
    expect(screen.getAllByText("= base").length).toBe(4);
    expect(screen.getByRole("status").textContent).toBe("at base");
  });

  test("switching base re-reads the catalogue for the new scenario", async () => {
    const { loads } = setup();
    expect((await nurseSlider()).value).toBe("3");

    fireEvent.change(screen.getByLabelText("base scenario"), { target: { value: "surge" } });

    await waitFor(async () => expect((await nurseSlider()).value).toBe("9"));
    expect(loads).toEqual(["er_floor", "surge"]);
  });

  test("a catalogue that cannot be read is reported, not faked", async () => {
    render(
      <ScenarioLab
        scenarios={SCENARIOS}
        currentSeed={42}
        loadCatalogue={() => Promise.reject(new Error("no backend"))}
        onRerun={() => undefined}
        onSaveScenario={() => Promise.resolve({ id: "scn-01" })}
        runId={null}
        metrics={null}
        simTime={0}
      />,
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("no backend");
    expect(screen.queryByLabelText("staffing.nurse_count")).toBeNull();
  });
});

describe("ScenarioLab dirty state", () => {
  test("a moved slider marks itself against the base it moved from", async () => {
    setup();
    fireEvent.change(await nurseSlider(), { target: { value: "6" } });

    const flag = screen.getByText(/vs base 3/);
    expect(flag.textContent).toContain("+3");
    // The knobs that did not move keep saying so.
    expect(screen.getAllByText("= base").length).toBe(3);
    expect(screen.getByRole("status").textContent).toBe("1 knob moved");

    // ...and the sign is honest in the other direction too.
    fireEvent.change(await nurseSlider(), { target: { value: "1" } });
    expect(screen.getByText(/vs base 3/).textContent).toContain("−2");
  });
});

describe("ScenarioLab submits", () => {
  test("Run sends ONLY the knobs that moved, holding the seed and the CRN shadow", async () => {
    const { runs } = setup();
    fireEvent.change(await nurseSlider(), { target: { value: "6" } });
    fireEvent.change(screen.getByLabelText("facility.general_bays"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(runs).toHaveLength(1);
    expect(runs[0]).toEqual({
      scenario: {
        base: "er_floor",
        // The untouched arrival-rate and housekeeping knobs are ABSENT, not
        // re-stated at their base values.
        overrides: { "staffing.nurse_count": 6, "facility.general_bays": 2 },
      },
      seed: 42,
      arm: "optimized",
      compare_to: "baseline",
      start: "paused",
    });
  });

  test("Run with nothing moved launches the stored scenario, not an empty overlay", async () => {
    const { runs } = setup();
    await nurseSlider();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(runs[0]?.scenario).toEqual({ id: "er_floor" });
  });

  test("the seed and CRN-shadow controls still reach the request", async () => {
    const { runs } = setup();
    await nurseSlider();
    fireEvent.change(screen.getByLabelText("seed"), { target: { value: "7" } });
    fireEvent.click(screen.getByLabelText("CRN shadow arm"));
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    expect(runs[0]?.seed).toBe(7);
    expect(runs[0]?.compare_to).toBeNull();
  });
});

describe("ScenarioLab reset", () => {
  test("Reset to base restores every slider and clears the dirty marks", async () => {
    setup();
    const nurses = await nurseSlider();
    const reset = screen.getByRole("button", { name: "Reset to base" }) as HTMLButtonElement;
    expect(reset.disabled).toBe(true);

    fireEvent.change(nurses, { target: { value: "6" } });
    fireEvent.change(screen.getByLabelText("facility.general_bays"), { target: { value: "2" } });
    expect(reset.disabled).toBe(false);

    fireEvent.click(reset);

    expect((await nurseSlider()).value).toBe("3");
    expect((screen.getByLabelText("facility.general_bays") as HTMLInputElement).value).toBe("6");
    expect(screen.getAllByText("= base").length).toBe(4);
    expect(screen.getByRole("status").textContent).toBe("at base");
    expect(reset.disabled).toBe(true);
  });
});

const HOUR = 3600 * 1_000_000;

describe("ScenarioLab shows what the move did", () => {
  test("after a re-run it contrasts the run left behind with the one now live", async () => {
    const { show } = setup({
      runId: "run-1",
      metrics: kpis({ completions_per_week: 100, boarding_time_s_mean: 1200 }),
      simTime: 24 * HOUR,
    });
    fireEvent.change(await nurseSlider(), { target: { value: "6" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    // Before the replacement lands there is nothing to contrast — and the panel
    // says so rather than differencing a run against itself.
    expect(screen.getByLabelText("Run delta").textContent).toContain("waiting for the new run");

    show({
      runId: "run-2",
      metrics: kpis({ completions_per_week: 130, boarding_time_s_mean: 900 }),
      simTime: 25 * HOUR,
    });

    const delta = screen.getByLabelText("Run delta");
    // What moved, and what it did to the numbers.
    expect(delta.textContent).toContain("Nurses 3 → 6");
    expect(delta.textContent).toContain("run-1");
    expect(delta.textContent).toContain("run-2");
    expect(delta.textContent).toContain("100 —> 130"); // completions, before -> after
    expect(delta.textContent).toContain("+5.0m"); // boarding fell by 300s
    expect(delta.textContent).toContain("better");
    // ...labelled for exactly what it is.
    expect(delta.textContent).toContain("Single-seed point delta");
    expect(delta.textContent).toContain("no confidence bound");
  });

  test("with no live reading to snapshot, the delta stays absent rather than invented", async () => {
    const { show } = setup();
    fireEvent.change(await nurseSlider(), { target: { value: "6" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    show({ runId: "run-2", metrics: kpis({ completions_per_week: 130 }), simTime: HOUR });

    expect(screen.getByLabelText("Run delta").textContent).toContain("Move a knob and Run");
  });
});
