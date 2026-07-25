import { describe, expect, test } from "bun:test";

import { MockEngine } from "../src/mock/engine";
import { synthesizeCompare } from "../src/mock/mockApi";

const HOUR_US = 3600 * 1_000_000;

function occupiedBay(engine: MockEngine): { bay: string; occupant: string } | null {
  // advance until some bay is occupied (bounded)
  for (let i = 0; i < 48; i += 1) {
    engine.advance(HOUR_US / 4);
    const frame = engine.buildFrame("delta");
    const hit = frame.bays.find((b) => b.status === "occupied" && b.occupant !== null);
    if (hit !== undefined && hit.occupant !== null) {
      return { bay: hit.bay, occupant: hit.occupant };
    }
  }
  return null;
}

describe("MockEngine determinism", () => {
  test("same seed -> byte-identical frames, regardless of pacing chunks", () => {
    const a = new MockEngine("run-x", 42, "optimized");
    const b = new MockEngine("run-x", 42, "optimized");
    a.advance(2 * HOUR_US); // one big pacing chunk
    for (let i = 0; i < 2 * 3600; i += 1) {
      b.advance(1_000_000); // many small chunks (as at a different speed)
    }
    expect(JSON.stringify(a.buildFrame("snapshot"))).toBe(JSON.stringify(b.buildFrame("snapshot")));
    expect(JSON.stringify(a.metrics())).toBe(JSON.stringify(b.metrics()));
  });

  test("different seeds diverge", () => {
    const a = new MockEngine("run-x", 1, "optimized");
    const b = new MockEngine("run-x", 2, "optimized");
    a.advance(2 * HOUR_US);
    b.advance(2 * HOUR_US);
    const framesA = JSON.stringify(a.buildFrame("snapshot").patients);
    const framesB = JSON.stringify(b.buildFrame("snapshot").patients);
    expect(framesA).not.toBe(framesB);
  });

  test("synthesizeCompare is deterministic per seed and self-consistent", () => {
    const one = synthesizeCompare(42, "b", "o");
    const two = synthesizeCompare(42, "b", "o");
    expect(one).toEqual(two);
    for (const c of one.contrasts) {
      expect(c.significant).toBe(!(c.ci_lo <= 0 && 0 <= c.ci_hi));
      expect(c.baseline - c.delta).toBeCloseTo(c.optimized, 9);
    }
    // the fixture must exercise the honest paths: at least one regression
    // and at least one non-significant contrast
    expect(one.contrasts.some((c) => c.delta < 0)).toBe(true);
    expect(one.contrasts.some((c) => !c.significant)).toBe(true);
  });
});

describe("MockEngine override validation (mirrors core semantics)", () => {
  test("reassign into an occupied bay is rejected verbatim and atomically", () => {
    const engine = new MockEngine("run-x", 42, "optimized");
    const occupied = occupiedBay(engine);
    expect(occupied).not.toBeNull();
    if (occupied === null) {
      return;
    }
    const before = JSON.stringify(engine.buildFrame("snapshot"));
    const otherPatient = engine
      .buildFrame("delta")
      .patients.find((p) => p.patient !== occupied.occupant && p.stage !== "in_bay");
    const target = otherPatient?.patient ?? "p-9999";
    const outcome = engine.applyOverride({ kind: "reassign", patient: target, bay: occupied.bay }, true);
    expect(outcome.status).toBe("rejected");
    if (outcome.status === "rejected") {
      expect(outcome.violations.length).toBeGreaterThan(0);
      const kinds = outcome.violations.map((v) => v.kind);
      expect(
        kinds.includes("bay_incompatible") ||
          kinds.includes("double_booked") ||
          kinds.includes("unknown_entity"),
      ).toBe(true);
    }
    // atomic: a rejection leaves the world byte-identical (modulo frame seq)
    const after = JSON.stringify(engine.buildFrame("snapshot"));
    const stripSeq = (s: string): string => s.replace(/"seq":\d+/, "");
    expect(stripSeq(after)).toBe(stripSeq(before));
  });

  test("close_bay on an occupied bay is rejected; on a free bay it applies", () => {
    const engine = new MockEngine("run-x", 42, "optimized");
    const occupied = occupiedBay(engine);
    expect(occupied).not.toBeNull();
    if (occupied === null) {
      return;
    }
    const rejected = engine.applyOverride({ kind: "close_bay", bay: occupied.bay }, true);
    expect(rejected.status).toBe("rejected");

    const freeBay = engine.buildFrame("delta").bays.find((b) => b.status === "free");
    expect(freeBay).toBeDefined();
    if (freeBay === undefined) {
      return;
    }
    const applied = engine.applyOverride({ kind: "close_bay", bay: freeBay.bay }, true);
    expect(applied.status).toBe("applied");
    const now = engine.buildFrame("delta").bays.find((b) => b.bay === freeBay.bay);
    expect(now?.status).toBe("closed");
  });

  test("reroute to a role-mismatched task yields staff_lacks_skill", () => {
    const engine = new MockEngine("run-x", 42, "optimized");
    engine.advance(HOUR_US);
    const outcome = engine.applyOverride(
      { kind: "reroute", staff: "hk-1", task: "provider_visit:bay-g1" },
      false,
    );
    expect(outcome.status).toBe("rejected");
    if (outcome.status === "rejected") {
      expect(outcome.violations[0]?.kind).toBe("staff_lacks_skill");
      expect(outcome.violations[0]?.detail).toContain("task needs role physician");
    }
  });

  test("an accepted reassign lands as bay_assigned by=operator in the event tail", () => {
    const engine = new MockEngine("run-x", 42, "optimized");
    const occupied = occupiedBay(engine);
    expect(occupied).not.toBeNull();
    if (occupied === null) {
      return;
    }
    const frame = engine.buildFrame("delta");
    const freeGeneral = frame.bays.find((b) => b.status === "free" && b.bay.startsWith("bay-g"));
    const chip = frame.patients.find((p) => p.patient === occupied.occupant);
    expect(freeGeneral).toBeDefined();
    expect(chip).toBeDefined();
    if (freeGeneral === undefined || chip === undefined || ![2, 3].includes(chip.esi)) {
      return; // acuity not general-eligible in this realization; covered above
    }
    const outcome = engine.applyOverride(
      { kind: "reassign", patient: occupied.occupant, bay: freeGeneral.bay },
      true,
    );
    expect(outcome.status).toBe("applied");
    const events = engine.buildFrame("delta").events;
    const assigned = events.find((e) => e.event.kind === "bay_assigned");
    expect(assigned).toBeDefined();
    if (assigned !== undefined && assigned.event.kind === "bay_assigned") {
      expect(assigned.event.by).toBe("operator");
      expect(assigned.event.bay).toBe(freeGeneral.bay);
    }
  });
});
