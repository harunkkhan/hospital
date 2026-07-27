import { describe, expect, test } from "bun:test";
import { fireEvent, render, screen } from "@testing-library/react";

import type { OverrideOutcome, OverrideRequest } from "../src/api/types";
import { OverridePanel } from "../src/components/OverridePanel";
import { makeMockLayout } from "../src/mock/fixtures";
import { initialWorld, type WorldView } from "../src/state/streamReducer";

const LAYOUT = makeMockLayout();

function world(): WorldView {
  return {
    ...initialWorld(),
    patients: {
      "p-0001": { patient: "p-0001", esi: 3, at_node: "station-triage", stage: "waiting_bay", waited: 0 },
    },
    staff: {
      "hk-1": {
        staff: "hk-1",
        role: "housekeeping",
        at_node: "station-fast",
        edge: null,
        edge_progress: 0,
        activity: "idle",
        current_task: null,
      },
    },
    bays: {
      "bay-g1": { bay: "bay-g1", status: "free", occupant: null, cleaning_eta: null },
      "bay-r1": { bay: "bay-r1", status: "occupied", occupant: "p-0002", cleaning_eta: null },
    },
  };
}

function setupReassign(outcome: OverrideOutcome): { requests: OverrideRequest[] } {
  const requests: OverrideRequest[] = [];
  render(
    <OverridePanel
      layout={LAYOUT}
      world={world()}
      selected={null}
      onSubmit={(req) => {
        requests.push(req);
        return Promise.resolve(outcome);
      }}
    />,
  );
  fireEvent.change(screen.getByLabelText("override action"), { target: { value: "reassign" } });
  fireEvent.change(screen.getByLabelText("patient"), { target: { value: "p-0001" } });
  fireEvent.change(screen.getByLabelText("bay"), { target: { value: "bay-g1" } });
  fireEvent.click(screen.getByRole("button", { name: /submit override/i }));
  return { requests };
}

describe("OverridePanel accept path", () => {
  test("shows the applied verdict, plan size, pin state, and sends pin=true by default", async () => {
    const { requests } = setupReassign({
      status: "applied",
      plan: { items: [{ stable_id: "assign:p-0001", kind: "assign_bay", patient: "p-0001", bay: "bay-g1" }] },
      applied_at: 3_600_000_000,
    });

    const note = await screen.findByRole("status");
    expect(note.textContent).toContain("applied");
    expect(note.textContent).toContain("D1 01:00:00");
    expect(note.textContent).toContain("1 item");
    expect(note.textContent).toContain("pinned");

    expect(requests).toHaveLength(1);
    expect(requests[0]?.pin).toBe(true);
    expect(requests[0]?.action).toEqual({ kind: "reassign", patient: "p-0001", bay: "bay-g1" });
  });
});

describe("OverridePanel reject path", () => {
  test("renders every Violation verbatim: kind + detail + entity", async () => {
    setupReassign({
      status: "rejected",
      violations: [
        { kind: "bay_incompatible", detail: "bay not free (status occupied)", entity: "bay-r1" },
        { kind: "double_booked", detail: "2 patients on one bay", entity: "bay-r1" },
      ],
    });

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("bay_incompatible");
    expect(alert.textContent).toContain("bay not free (status occupied)");
    expect(alert.textContent).toContain("double_booked");
    expect(alert.textContent).toContain("2 patients on one bay");
    expect(alert.textContent).toContain("(bay-r1)");
    // atomicity is surfaced to the operator
    expect(alert.textContent).toContain("Nothing was applied");
    expect(screen.queryByRole("status")).toBeNull();
  });

  test("submit stays disabled until the action is complete", () => {
    render(
      <OverridePanel layout={LAYOUT} world={world()} selected={null} onSubmit={() => Promise.reject(new Error("nope"))} />,
    );
    const submit = screen.getByRole("button", { name: /submit override/i });
    expect((submit as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("OverridePanel reroute (finding #2: real task ids from the frame)", () => {
  test("lists the frame's pending tasks and echoes the opaque id verbatim", () => {
    const requests: OverrideRequest[] = [];
    const w: WorldView = {
      ...world(),
      pendingTasks: [{ id: "task_000042", kind: "cleaning", at: "bay-g1" }],
    };
    render(
      <OverridePanel
        layout={LAYOUT}
        world={w}
        selected={null}
        onSubmit={(req) => {
          requests.push(req);
          return Promise.resolve({ status: "applied", plan: { items: [] }, applied_at: 0 });
        }}
      />,
    );
    fireEvent.change(screen.getByLabelText("override action"), { target: { value: "reroute" } });
    fireEvent.change(screen.getByLabelText("staff"), { target: { value: "hk-1" } });
    const taskSelect = screen.getByLabelText("task") as HTMLSelectElement;
    expect(taskSelect.disabled).toBe(false);
    fireEvent.change(taskSelect, { target: { value: "task_000042" } });
    fireEvent.click(screen.getByRole("button", { name: /submit override/i }));

    expect(requests[0]?.action).toEqual({ kind: "reroute", staff: "hk-1", task: "task_000042" });
  });

  test("reroute is disabled when the frame carries no pending tasks", () => {
    render(
      <OverridePanel
        layout={LAYOUT}
        world={world()}
        selected={null}
        onSubmit={() => Promise.reject(new Error("unused"))}
      />,
    );
    fireEvent.change(screen.getByLabelText("override action"), { target: { value: "reroute" } });
    fireEvent.change(screen.getByLabelText("staff"), { target: { value: "hk-1" } });
    expect((screen.getByLabelText("task") as HTMLSelectElement).disabled).toBe(true);
    expect(
      (screen.getByRole("button", { name: /submit override/i }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

describe("OverridePanel map selection", () => {
  test("a selected bay prefills the bay field", () => {
    render(
      <OverridePanel
        layout={LAYOUT}
        world={world()}
        selected={{ type: "bay", id: "bay-g1" }}
        onSubmit={() => Promise.reject(new Error("unused"))}
      />,
    );
    expect((screen.getByLabelText("bay") as HTMLSelectElement).value).toBe("bay-g1");
  });
});
