import { describe, expect, test } from "bun:test";

import type { EventEnvelope, StreamFrame } from "../src/api/types";
import {
  applyFrame,
  EVENT_RING_CAPACITY,
  initialWorld,
  type WorldView,
} from "../src/state/streamReducer";

function frame(overrides: Partial<StreamFrame>): StreamFrame {
  return {
    run: "run-01",
    sim_time: 1_000_000,
    seq: 0,
    kind: "snapshot",
    state: "paused",
    speed: 60,
    staff: [],
    bays: [],
    queues: [],
    patients: [],
    events: [],
    kpi_preview: null,
    ...overrides,
  };
}

function envelope(sequence: number, patient = "p-0001"): EventEnvelope {
  return {
    event: { kind: "bay_requested", occurred_at: 1_000_000, patient },
    sequence,
    caused_by: null,
  };
}

const CHIP = {
  patient: "p-0001",
  esi: 3,
  at_node: "station-triage",
  stage: "waiting_bay",
  waited: 0,
} as const;

const BAY = { bay: "bay-g1", status: "occupied", occupant: "p-0001", cleaning_eta: null } as const;

describe("applyFrame snapshot", () => {
  test("replaces the whole view and clears desync", () => {
    const desynced: WorldView = { ...initialWorld(), desynced: true, seq: 10 };
    const next = applyFrame(
      desynced,
      frame({ seq: 42, kind: "snapshot", patients: [CHIP], bays: [BAY], sim_time: 5_000_000 }),
    );
    expect(next.desynced).toBe(false);
    expect(next.seq).toBe(42);
    expect(next.simTime).toBe(5_000_000);
    expect(next.patients["p-0001"]).toEqual(CHIP);
    expect(next.bays["bay-g1"]).toEqual(BAY);
  });
});

describe("applyFrame delta", () => {
  const base = applyFrame(initialWorld(), frame({ seq: 0, patients: [CHIP], bays: [BAY] }));

  test("upserts carried entities and keeps the rest", () => {
    const moved = { ...CHIP, patient: "p-0002", at_node: "bay-g2" };
    const next = applyFrame(base, frame({ seq: 1, kind: "delta", patients: [moved] }));
    expect(next.patients["p-0001"]).toEqual(CHIP);
    expect(next.patients["p-0002"]).toEqual(moved);
    expect(next.bays["bay-g1"]).toEqual(BAY); // empty facet -> kept
    expect(next.seq).toBe(1);
  });

  test("retires a patient on discharge_completed", () => {
    const discharge: EventEnvelope = {
      event: { kind: "discharge_completed", occurred_at: 2_000_000, patient: "p-0001" },
      sequence: 5,
      caused_by: null,
    };
    const next = applyFrame(base, frame({ seq: 1, kind: "delta", events: [discharge] }));
    expect(next.patients["p-0001"]).toBeUndefined();
  });

  test("a seq gap marks desync and applies nothing", () => {
    const next = applyFrame(base, frame({ seq: 5, kind: "delta", patients: [] }));
    expect(next.desynced).toBe(true);
    expect(next.seq).toBe(base.seq);
    expect(next.simTime).toBe(base.simTime);
  });

  test("a delta before any snapshot marks desync", () => {
    const next = applyFrame(initialWorld(), frame({ seq: 3, kind: "delta" }));
    expect(next.desynced).toBe(true);
  });

  test("stale/duplicate deltas are ignored", () => {
    const next = applyFrame(base, frame({ seq: 0, kind: "delta", sim_time: 99 }));
    expect(next).toBe(base);
  });

  test("event ring is appended in order and bounded", () => {
    let world = base;
    const perFrame = 10;
    const frames = Math.ceil(EVENT_RING_CAPACITY / perFrame) + 5;
    for (let i = 0; i < frames; i += 1) {
      const events = Array.from({ length: perFrame }, (_, j) => envelope(i * perFrame + j, "p-x"));
      world = applyFrame(world, frame({ seq: i + 1, kind: "delta", events }));
    }
    expect(world.events.length).toBe(EVENT_RING_CAPACITY);
    const sequences = world.events.map((e) => e.sequence);
    expect(sequences).toEqual([...sequences].sort((a, b) => a - b));
    expect(sequences[sequences.length - 1]).toBe(frames * perFrame - 1);
  });

  test("pending_tasks: snapshot sets, omitted delta keeps, present delta replaces", () => {
    const task = { id: "task_000042", kind: "cleaning", at: "bay-g1" } as const;
    const snap = applyFrame(
      initialWorld(),
      frame({ seq: 0, kind: "snapshot", pending_tasks: [task] }),
    );
    expect(snap.pendingTasks).toEqual([task]);

    const kept = applyFrame(snap, frame({ seq: 1, kind: "delta" }));
    expect(kept.pendingTasks).toEqual([task]); // omitted ⇒ unchanged

    const cleared = applyFrame(kept, frame({ seq: 2, kind: "delta", pending_tasks: [] }));
    expect(cleared.pendingTasks).toEqual([]); // present (empty) ⇒ authoritative
  });

  test("kpi_preview is kept from the last frame that carried one", () => {
    const withKpi = applyFrame(
      base,
      frame({ seq: 1, kind: "delta", kpi_preview: { values: { completions_per_week: 3 } } }),
    );
    const next = applyFrame(withKpi, frame({ seq: 2, kind: "delta" }));
    expect(next.kpiPreview?.values["completions_per_week"]).toBe(3);
  });
});
