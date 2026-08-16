/**
 * Deterministic mock ER engine backing the mock stream mode.
 *
 * Fidelity target: the WIRE CONTRACT, not the physics. Patients arrive, get
 * triaged, wait for a bay, receive service, discharge; bays cycle through
 * free/occupied/cleaning; staff walk real RouteGraph edges. All randomness
 * comes from one seeded PRNG consumed in a fixed order inside fixed
 * 1-sim-second quanta, so a given seed always yields byte-identical frames —
 * pacing (speed) never touches sampling, mirroring the real driver invariant.
 *
 * Scenario knobs are REALIZED here, not recorded: arrival rate and ambulance
 * share shape the arrival stream, isolation share contends for the two
 * isolation-capable bays, the staffing counts build the roster (mirroring
 * data.realize_staff), and the bay-capacity counts take bays out of service. The
 * mock catalogue (mockApi.ts) publishes exactly the knobs this list covers, and
 * caps each at what this floor can express — a slider the demo cannot honor is a
 * placebo, and a placebo is worse than an absent control.
 *
 * Where the roster BITES is worth stating plainly, because it is the one place
 * this mock is thinner than the engine it stands in for. Triage is staff-gated:
 * a patient waits for a free nurse, so the nurse count moves throughput and
 * door-to-triage. Provider visits and bay cleaning are not gated — those roles
 * move utilization, walking and what is visible on the floor, but a bay still
 * turns over on a timer rather than on a housekeeper arriving. The real engine
 * (sim.physics) gates all of it; this is a wire-contract stand-in, and the live
 * console is where the full mechanism runs.
 */

import type {
  BayFrame,
  BayId,
  BayStatus,
  BottleneckReport,
  EsiAcuity,
  EventEnvelope,
  Event,
  FloorLayout,
  KpiVector,
  NodeId,
  OperatorAction,
  OverrideOutcome,
  PatientChip,
  PatientId,
  PendingTask,
  Plan,
  PlanItem,
  QueueFrame,
  ResourceWait,
  RunId,
  RunState,
  SimTime,
  StaffId,
  StaffKinematic,
  StaffRole,
  StreamFrame,
  Violation,
} from "../api/types";
import { KPI_KEYS } from "../api/types";
import { allowedZoneTypes, COMPLAINTS, makeMockLayout, STAFF_ROSTER } from "./fixtures";

export const WEEK_US = 7 * 24 * 3600 * 1_000_000;
const STEP_US = 1_000_000; // the fixed sampling quantum: 1 sim-second
const S = 1_000_000; // µs per second

/** Deterministic 32-bit PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface MockPatient {
  id: PatientId;
  esi: EsiAcuity;
  complaint: string;
  arrivalUs: SimTime;
  stage: "waiting_triage" | "triage" | "waiting_bay" | "in_bay" | "departed";
  stageSinceUs: SimTime;
  atNode: NodeId;
  bay: BayId | null;
  remainingUs: number;
  pinned: boolean;
  /** Needs an isolation-capable bay — the mock floor has only two. */
  isolation: boolean;
  /**
   * Queue-ordering priority ONLY — higher jumps the FIFO. Kept distinct from
   * stageSinceUs (stage-entry time) so a bump reorders the queue WITHOUT
   * rewriting the timestamps that drive displayed wait and boarding metrics.
   */
  priority: number;
}

interface MockBay {
  id: BayId;
  status: BayStatus;
  occupant: PatientId | null;
  cleaningEta: SimTime | null;
  remainingUs: number;
}

interface StaffTask {
  kind: "triage" | "provider_visit" | "nurse_visit" | "cleaning" | "documentation";
  node: NodeId;
  durationUs: number;
}

interface MockStaff {
  id: StaffId;
  role: StaffRole;
  home: NodeId;
  at: NodeId;
  /** Remaining nodes to visit, current edge target first. */
  path: NodeId[];
  edgeFrom: NodeId | null;
  edgeElapsedUs: number;
  edgeTotalUs: number;
  task: StaffTask | null;
  serving: boolean;
  idleCooldownUs: number;
  busyUs: number;
}

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) {
    return null;
  }
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[idx] ?? null;
}

function mean(xs: number[]): number | null {
  if (xs.length === 0) {
    return null;
  }
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function gini(values: number[]): number {
  const xs = values.filter((v) => v >= 0);
  const total = xs.reduce((a, b) => a + b, 0);
  if (xs.length < 2 || total <= 0) {
    return 0;
  }
  const sorted = [...xs].sort((a, b) => a - b);
  const n = sorted.length;
  let rankWeighted = 0;
  for (const [i, v] of sorted.entries()) {
    rankWeighted += (i + 1) * v;
  }
  const g = (2 * rankWeighted) / (n * total) - (n + 1) / n;
  return Math.max(0, Math.min(1, g));
}

/** Scenario inputs that shape a mock run (base id + numeric parameter overrides). */
export interface ScenarioConfig {
  base: string;
  overrides: Readonly<Record<string, number>>;
}

/**
 * Which slider closes bays of which zone type. The mock floor is hand-authored
 * geometry (fixtures.ts), so a capacity knob can only take bays OUT of service —
 * there is nowhere to put a fourteenth one. That is why the mock catalogue caps
 * each of these at the fixture's own count: the range the panel draws is the
 * range this engine can actually realize, so no slider position is a placebo.
 */
const BAY_CAPACITY_KEYS: Readonly<Record<string, string>> = {
  general: "facility.general_bays",
  fast_track: "facility.fast_track_bays",
  resus_trauma: "facility.resus_bays",
  triage: "facility.triage_rooms",
};

const ROLE_ORDER: readonly StaffRole[] = [
  "physician",
  "nurse",
  "tech",
  "porter",
  "housekeeping",
];

/**
 * The roster a staffing slider realizes — the mock's stand-in for
 * `data.realize_staff`: a count per role, homed round-robin over the stations.
 *
 * The fixture roster supplies the first members of each role, so an un-overridden
 * run is byte-identical to before this knob existed (same ids, same order, same
 * homes); beyond it, members are minted with the same id prefix.
 */
function scaleRoster(
  overrides: Readonly<Record<string, number>>,
  stations: readonly NodeId[],
): { id: StaffId; role: StaffRole; home: NodeId }[] {
  const out: { id: StaffId; role: StaffRole; home: NodeId }[] = [];
  for (const role of ROLE_ORDER) {
    const fixtures = STAFF_ROSTER.filter((f) => f.role === role);
    const requested = overrides[`staffing.${role}_count`];
    const count = Math.max(0, Math.round(requested ?? fixtures.length));
    const prefix = fixtures[0]?.id.split("-")[0] ?? role;
    for (let k = 0; k < count; k += 1) {
      const fixture = fixtures[k];
      out.push(
        fixture ?? {
          id: `${prefix}-${k + 1}`,
          role,
          home: stations[k % stations.length] ?? "c-1",
        },
      );
    }
  }
  return out;
}

export class MockEngine {
  readonly runId: RunId;
  readonly arm: "baseline" | "optimized";
  readonly seed: number;
  readonly layout: FloorLayout;
  readonly horizonUs = WEEK_US;

  /** The resolved scenario this run was generated from (never discarded). */
  readonly scenario: ScenarioConfig;

  simTimeUs = 0;
  state: RunState = "created";
  speed = 60;
  seq = -1;

  /** Interarrival divisor: >1 packs arrivals tighter, <1 spreads them out. */
  private readonly arrivalMultiplier: number;
  /** P(ambulance) for each arrival. */
  private readonly ambulanceShare: number;
  /** P(needs an isolation-capable bay) for each arrival. */
  private readonly isolationShare: number;

  private rng: () => number;
  private accUs = 0;
  private nextArrivalUs: number;
  private patientCounter = 0;
  private eventSeq = 0;
  private pendingEvents: EventEnvelope[] = [];

  private patients = new Map<PatientId, MockPatient>();
  private bays = new Map<BayId, MockBay>();
  private staff = new Map<StaffId, MockStaff>();
  private blockedEdges = new Set<string>();
  private pins = new Set<string>();

  // metric accumulators (all µs unless suffixed)
  private completions = 0;
  private doorToTriageS: number[] = [];
  private doorToProviderS: number[] = [];
  private losSByEsi: Record<number, number[]> = { 1: [], 2: [], 3: [], 4: [], 5: [] };
  private turnaroundS: number[] = [];
  private boardingS: number[] = [];
  private waitSums = { triage: 0, provider: 0, nurse: 0, imaging: 0, lab: 0, housekeeping: 0 };
  private waitCounts = { triage: 0, provider: 0, nurse: 0, imaging: 0, lab: 0, housekeeping: 0 };
  private fracUs = { walk: 0, direct_care: 0, cleaning: 0, documentation: 0, idle: 0 };
  private occupiedBayUs = 0;

  private adjacency = new Map<NodeId, NodeId[]>();
  private edgeUs = new Map<string, number>();

  constructor(
    runId: RunId,
    seed: number,
    arm: "baseline" | "optimized",
    scenario: ScenarioConfig = { base: "er_floor", overrides: {} },
  ) {
    this.runId = runId;
    this.seed = seed;
    this.arm = arm;
    this.scenario = scenario;
    const overrides = scenario.overrides;
    this.arrivalMultiplier = Math.max(0.1, overrides["workload.arrival_rate_multiplier"] ?? 1);
    const share = overrides["workload.ambulance_share"];
    this.ambulanceShare = share === undefined ? 0.25 : Math.max(0, Math.min(1, share));
    const isolation = overrides["workload.isolation_share"];
    this.isolationShare = isolation === undefined ? 0.05 : Math.max(0, Math.min(1, isolation));
    this.layout = makeMockLayout();
    this.rng = mulberry32(seed);
    this.nextArrivalUs = Math.round((30 * S + this.rng() * 120 * S) / this.arrivalMultiplier);

    // Bays beyond a zone type's requested capacity start CLOSED and stay closed:
    // a capacity slider takes real bays out of service rather than annotating a
    // number nothing reads (placement, cleaning and utilization all key off status).
    const openedPerZone = new Map<string, number>();
    for (const bay of this.layout.bays) {
      const capacityKey = BAY_CAPACITY_KEYS[bay.zone_type];
      const cap = capacityKey === undefined ? undefined : overrides[capacityKey];
      const opened = openedPerZone.get(bay.zone_type) ?? 0;
      const closed = cap !== undefined && opened >= Math.max(0, Math.round(cap));
      openedPerZone.set(bay.zone_type, opened + (closed ? 0 : 1));
      this.bays.set(bay.id, {
        id: bay.id,
        status: closed ? "closed" : "free",
        occupant: null,
        cleaningEta: null,
        remainingUs: 0,
      });
    }
    for (const fixture of scaleRoster(overrides, this.layout.stations)) {
      this.staff.set(fixture.id, {
        id: fixture.id,
        role: fixture.role,
        home: fixture.home,
        at: fixture.home,
        path: [],
        edgeFrom: null,
        edgeElapsedUs: 0,
        edgeTotalUs: 0,
        task: null,
        serving: false,
        idleCooldownUs: Math.round(30 * S + this.rng() * 60 * S),
        busyUs: 0,
      });
    }
    for (const e of this.layout.graph.edges) {
      const fwd = this.adjacency.get(e.a) ?? [];
      fwd.push(e.b);
      this.adjacency.set(e.a, fwd);
      this.edgeUs.set(`${e.a}>${e.b}`, e.seconds);
      if (e.bidirectional) {
        const back = this.adjacency.get(e.b) ?? [];
        back.push(e.a);
        this.adjacency.set(e.b, back);
        this.edgeUs.set(`${e.b}>${e.a}`, e.seconds);
      }
    }
  }

  // ------------------------------------------------------------------ events

  private emit(event: Event): void {
    this.pendingEvents.push({ event, sequence: this.eventSeq, caused_by: null });
    this.eventSeq += 1;
  }

  hasPendingEvents(): boolean {
    return this.pendingEvents.length > 0;
  }

  // ------------------------------------------------------------------ routing

  private isBlocked(a: NodeId, b: NodeId): boolean {
    return this.blockedEdges.has(`${a}>${b}`) || this.blockedEdges.has(`${b}>${a}`);
  }

  /** Deterministic BFS (adjacency in insertion order) avoiding blocked edges. */
  private findPath(from: NodeId, to: NodeId): NodeId[] | null {
    if (from === to) {
      return [];
    }
    const prev = new Map<NodeId, NodeId>();
    const queue: NodeId[] = [from];
    const seen = new Set<NodeId>([from]);
    while (queue.length > 0) {
      const cur = queue.shift() as NodeId;
      for (const next of this.adjacency.get(cur) ?? []) {
        if (seen.has(next) || this.isBlocked(cur, next)) {
          continue;
        }
        seen.add(next);
        prev.set(next, cur);
        if (next === to) {
          const path: NodeId[] = [to];
          let node: NodeId = to;
          while (node !== from) {
            node = prev.get(node) as NodeId;
            path.unshift(node);
          }
          return path.slice(1);
        }
        queue.push(next);
      }
    }
    return null;
  }

  private dispatchStaff(staff: MockStaff, task: StaffTask): void {
    const path = this.findPath(staff.at, task.node);
    if (path === null) {
      return; // unreachable (blocked) — task silently undone in the mock
    }
    staff.task = task;
    staff.serving = false;
    staff.path = path;
    staff.edgeFrom = null;
    staff.edgeElapsedUs = 0;
    staff.edgeTotalUs = 0;
  }

  private idleStaffOfRole(role: StaffRole): MockStaff | null {
    for (const member of this.staff.values()) {
      if (member.role === role && member.task === null) {
        return member;
      }
    }
    return null;
  }

  // ------------------------------------------------------------------ driver

  /** Advance sim time by `dtUs`, consumed in fixed 1s quanta (pacing-safe). */
  advance(dtUs: number): void {
    this.accUs += dtUs;
    while (this.accUs >= STEP_US && this.simTimeUs < this.horizonUs) {
      this.accUs -= STEP_US;
      this.stepOnce();
    }
    if (this.simTimeUs >= this.horizonUs) {
      this.state = "finished";
    }
  }

  private stepOnce(): void {
    this.simTimeUs += STEP_US;
    this.stepArrivals();
    this.stepTriage();
    this.stepPlacement();
    this.stepService();
    this.stepCleaning();
    this.stepStaff();
    this.stepAccounting();
  }

  private drawEsi(): EsiAcuity {
    const u = this.rng();
    if (u < 0.05) return 1;
    if (u < 0.25) return 2;
    if (u < 0.62) return 3;
    if (u < 0.88) return 4;
    return 5;
  }

  private stepArrivals(): void {
    while (this.simTimeUs >= this.nextArrivalUs) {
      this.patientCounter += 1;
      const id = `p-${String(this.patientCounter).padStart(4, "0")}`;
      const esi = this.drawEsi();
      const mode = this.rng() < this.ambulanceShare ? "ambulance" : "walk_in";
      const isolation = this.rng() < this.isolationShare;
      const complaint = COMPLAINTS[Math.floor(this.rng() * COMPLAINTS.length)] ?? "unwell";
      const atNode = mode === "ambulance" ? "entr-ambo" : "entr-main";
      this.patients.set(id, {
        id,
        esi,
        complaint,
        arrivalUs: this.simTimeUs,
        stage: "waiting_triage",
        stageSinceUs: this.simTimeUs,
        atNode,
        bay: null,
        remainingUs: 0,
        pinned: false,
        isolation,
        priority: 0,
      });
      this.emit({ kind: "patient_arrived", occurred_at: this.simTimeUs, patient: id, mode });
      // exponential-ish interarrival, mean ~10 min, tightened by the workload
      // arrival-rate override so scenario knobs actually move the load.
      this.nextArrivalUs += Math.round((-Math.log(1 - this.rng()) * 600 * S) / this.arrivalMultiplier);
    }
  }

  private waitingByStage(stage: "waiting_triage" | "waiting_bay"): MockPatient[] {
    return [...this.patients.values()]
      .filter((p) => p.stage === stage)
      .sort(
        (a, b) =>
          b.priority - a.priority ||
          a.stageSinceUs - b.stageSinceUs ||
          a.id.localeCompare(b.id),
      );
  }

  private bayFixture(id: BayId) {
    return this.layout.bays.find((b) => b.id === id);
  }

  private stepTriage(): void {
    // completions first
    for (const patient of this.patients.values()) {
      if (patient.stage !== "triage") {
        continue;
      }
      patient.remainingUs -= STEP_US;
      if (patient.remainingUs <= 0) {
        this.emit({
          kind: "triage_completed",
          occurred_at: this.simTimeUs,
          patient: patient.id,
          esi: patient.esi,
        });
        const bay = patient.bay === null ? null : this.bays.get(patient.bay);
        if (bay !== undefined && bay !== null) {
          bay.status = "free";
          bay.occupant = null;
        }
        patient.bay = null;
        patient.stage = "waiting_bay";
        patient.stageSinceUs = this.simTimeUs;
        patient.atNode = "station-triage";
        this.emit({ kind: "bay_requested", occurred_at: this.simTimeUs, patient: patient.id });
      }
    }
    // assignments
    for (const patient of this.waitingByStage("waiting_triage")) {
      const bay = [...this.bays.values()].find(
        (b) => b.status === "free" && this.bayFixture(b.id)?.zone_type === "triage",
      );
      if (bay === undefined) {
        break;
      }
      // Triage is STAFF-GATED: no free nurse, no triage, and the queue grows.
      // Without this the nurse slider would move utilization and walking while
      // leaving throughput untouched — a control that looks live and is not.
      const nurse = this.idleStaffOfRole("nurse");
      if (nurse === null) {
        break;
      }
      bay.status = "occupied";
      bay.occupant = patient.id;
      patient.stage = "triage";
      patient.bay = bay.id;
      patient.atNode = bay.id;
      patient.remainingUs = Math.round(90 * S + this.rng() * 180 * S);
      const waited = this.simTimeUs - patient.arrivalUs;
      this.doorToTriageS.push(waited / S);
      this.waitSums.triage += waited;
      this.waitCounts.triage += 1;
      patient.stageSinceUs = this.simTimeUs;
      this.emit({
        kind: "triage_started",
        occurred_at: this.simTimeUs,
        patient: patient.id,
        staff: nurse.id,
      });
      this.dispatchStaff(nurse, { kind: "triage", node: bay.id, durationUs: patient.remainingUs });
    }
  }

  /** Free bay compatible with the patient's acuity (and isolation need), or null.
   *
   * Isolation is a hard placement constraint, not a preference: the floor has two
   * isolation-capable bays, so raising the isolation share squeezes placement
   * against a much smaller pool and boarding time climbs — the contention the
   * demand knob exists to show. Isolation-capable bays are NOT reserved, so a
   * routine patient may still be placed in one; that is the real trade-off, and
   * hiding it would make the knob look free.
   */
  private findFreeBay(esi: EsiAcuity, isolation: boolean): MockBay | null {
    const allowed = allowedZoneTypes(esi);
    for (const fixture of this.layout.bays) {
      if (fixture.zone_type === "triage" || !allowed.includes(fixture.zone_type)) {
        continue;
      }
      if (isolation && !fixture.isolation_capable) {
        continue;
      }
      const bay = this.bays.get(fixture.id);
      if (bay !== undefined && bay.status === "free") {
        return bay;
      }
    }
    return null;
  }

  private placePatient(patient: MockPatient, bay: MockBay, by: "solver" | "operator" | "baseline"): void {
    bay.status = "occupied";
    bay.occupant = patient.id;
    const boarding = this.simTimeUs - patient.stageSinceUs;
    if (patient.stage === "waiting_bay") {
      this.boardingS.push(boarding / S);
      this.waitSums.provider += boarding;
      this.waitCounts.provider += 1;
      this.doorToProviderS.push((this.simTimeUs - patient.arrivalUs) / S);
    }
    patient.stage = "in_bay";
    patient.stageSinceUs = this.simTimeUs;
    patient.bay = bay.id;
    patient.atNode = bay.id;
    patient.remainingUs = Math.round((7 - patient.esi) * 1200 * S + this.rng() * 1800 * S);
    this.emit({
      kind: "bay_assigned",
      occurred_at: this.simTimeUs,
      patient: patient.id,
      bay: bay.id,
      by,
    });
    const physician = this.idleStaffOfRole("physician");
    if (physician !== null) {
      this.emit({
        kind: "provider_visit_started",
        occurred_at: this.simTimeUs,
        patient: patient.id,
        staff: physician.id,
      });
      this.dispatchStaff(physician, {
        kind: "provider_visit",
        node: bay.id,
        durationUs: Math.round(420 * S + this.rng() * 300 * S),
      });
    }
  }

  private stepPlacement(): void {
    for (const patient of this.waitingByStage("waiting_bay")) {
      const by = this.arm === "baseline" ? "baseline" : "solver";
      const bay = this.findFreeBay(patient.esi, patient.isolation);
      if (bay === null) {
        continue;
      }
      this.placePatient(patient, bay, by);
    }
  }

  private stepService(): void {
    for (const patient of this.patients.values()) {
      if (patient.stage !== "in_bay") {
        continue;
      }
      patient.remainingUs -= STEP_US;
      if (patient.remainingUs > 0) {
        continue;
      }
      const losS = (this.simTimeUs - patient.arrivalUs) / S;
      this.losSByEsi[patient.esi]?.push(losS);
      this.completions += 1;
      this.emit({
        kind: "disposition_decided",
        occurred_at: this.simTimeUs,
        patient: patient.id,
        disposition: "discharge",
      });
      this.emit({ kind: "discharge_completed", occurred_at: this.simTimeUs, patient: patient.id });
      const bay = patient.bay === null ? null : this.bays.get(patient.bay);
      patient.stage = "departed";
      this.pins.delete(patient.id);
      this.patients.delete(patient.id);
      if (bay !== undefined && bay !== null) {
        bay.status = "cleaning";
        bay.occupant = null;
        bay.remainingUs = Math.round(240 * S + this.rng() * 240 * S);
        bay.cleaningEta = this.simTimeUs + bay.remainingUs;
        this.turnaroundS.push(bay.remainingUs / S);
        this.waitSums.housekeeping += bay.remainingUs;
        this.waitCounts.housekeeping += 1;
        const hk = this.idleStaffOfRole("housekeeping");
        this.emit({
          kind: "bay_cleaning_started",
          occurred_at: this.simTimeUs,
          bay: bay.id,
          staff: hk?.id ?? "hk-1",
        });
        if (hk !== null) {
          this.dispatchStaff(hk, { kind: "cleaning", node: bay.id, durationUs: bay.remainingUs });
        }
      }
    }
  }

  private stepCleaning(): void {
    for (const bay of this.bays.values()) {
      if (bay.status !== "cleaning") {
        continue;
      }
      bay.remainingUs -= STEP_US;
      if (bay.remainingUs <= 0) {
        bay.status = "free";
        bay.cleaningEta = null;
        this.emit({
          kind: "bay_cleaning_completed",
          occurred_at: this.simTimeUs,
          bay: bay.id,
          staff: "hk-1",
        });
      }
    }
  }

  private stepStaff(): void {
    for (const member of this.staff.values()) {
      if (member.path.length > 0) {
        // walking an edge
        if (member.edgeFrom === null) {
          const next = member.path[0] as NodeId;
          member.edgeFrom = member.at;
          member.edgeTotalUs = this.edgeUs.get(`${member.at}>${next}`) ?? 30 * S;
          member.edgeElapsedUs = 0;
          this.emit({
            kind: "staff_moved",
            occurred_at: this.simTimeUs,
            staff: member.id,
            edge: [member.at, next],
            seconds: member.edgeTotalUs,
          });
        }
        member.edgeElapsedUs += STEP_US;
        if (member.edgeElapsedUs >= member.edgeTotalUs) {
          member.at = member.path.shift() as NodeId;
          member.edgeFrom = null;
          member.edgeElapsedUs = 0;
          if (member.path.length === 0 && member.task !== null) {
            member.serving = true;
          }
        }
        continue;
      }
      if (member.task !== null && member.serving) {
        member.task.durationUs -= STEP_US;
        if (member.task.durationUs <= 0) {
          const finished = member.task;
          member.task = null;
          member.serving = false;
          if (finished.kind === "provider_visit") {
            this.emit({
              kind: "provider_visit_completed",
              occurred_at: this.simTimeUs,
              patient: this.bays.get(finished.node)?.occupant ?? "p-0000",
              staff: member.id,
            });
            member.task = {
              kind: "documentation",
              node: member.at,
              durationUs: Math.round(120 * S + this.rng() * 60 * S),
            };
            member.serving = true;
          } else if (member.at !== member.home) {
            this.dispatchStaffHome(member);
          }
        }
        continue;
      }
      // idle: occasional deterministic stroll for ambient motion
      member.idleCooldownUs -= STEP_US;
      if (member.idleCooldownUs <= 0) {
        member.idleCooldownUs = Math.round(60 * S + this.rng() * 120 * S);
        const stations = this.layout.stations;
        const target = stations[Math.floor(this.rng() * stations.length)] ?? member.home;
        if (target !== member.at) {
          const path = this.findPath(member.at, target);
          if (path !== null) {
            member.path = path;
            member.edgeFrom = null;
          }
        }
      }
    }
  }

  private dispatchStaffHome(member: MockStaff): void {
    const path = this.findPath(member.at, member.home);
    if (path !== null) {
      member.path = path;
      member.edgeFrom = null;
    }
  }

  private stepAccounting(): void {
    for (const member of this.staff.values()) {
      if (member.path.length > 0 || member.edgeFrom !== null) {
        this.fracUs.walk += STEP_US;
        member.busyUs += STEP_US;
      } else if (member.task !== null && member.serving) {
        member.busyUs += STEP_US;
        if (member.task.kind === "cleaning") {
          this.fracUs.cleaning += STEP_US;
        } else if (member.task.kind === "documentation") {
          this.fracUs.documentation += STEP_US;
        } else {
          this.fracUs.direct_care += STEP_US;
        }
      } else {
        this.fracUs.idle += STEP_US;
      }
    }
    for (const bay of this.bays.values()) {
      if (bay.status === "occupied") {
        this.occupiedBayUs += STEP_US;
      }
    }
  }

  // ------------------------------------------------------------------ frames

  private staffKinematic(member: MockStaff): StaffKinematic {
    const walking = member.path.length > 0 && member.edgeFrom !== null;
    const activity =
      member.task !== null && member.serving
        ? member.task.kind === "provider_visit"
          ? "provider_visit"
          : member.task.kind === "nurse_visit"
            ? "nurse_visit"
            : member.task.kind === "triage"
              ? "triage"
              : member.task.kind === "documentation"
                ? "documentation"
                : "cleaning"
        : walking
          ? "transport"
          : "idle";
    return {
      staff: member.id,
      role: member.role,
      at_node: walking ? null : member.at,
      edge: walking ? [member.edgeFrom as NodeId, member.path[0] as NodeId] : null,
      edge_progress:
        walking && member.edgeTotalUs > 0
          ? Math.min(1, member.edgeElapsedUs / member.edgeTotalUs)
          : 0,
      activity,
      current_task: member.task === null ? null : `${member.task.kind}:${member.task.node}`,
    };
  }

  /**
   * Reroutable tasks the operator can target. The mock's task ids encode
   * kind+node (`nurse_visit:bay-g1`) so applyOverride can decode them, but the
   * console treats them as opaque — it echoes whatever id the frame carries.
   */
  private pendingTasks(): PendingTask[] {
    const out: PendingTask[] = [];
    for (const bay of this.bays.values()) {
      if (bay.status === "cleaning") {
        out.push({ id: `cleaning:${bay.id}`, kind: "cleaning", at: bay.id });
      } else if (bay.status === "occupied") {
        out.push({ id: `nurse_visit:${bay.id}`, kind: "nurse_visit", at: bay.id });
        out.push({ id: `provider_visit:${bay.id}`, kind: "provider_visit", at: bay.id });
      }
    }
    return out;
  }

  buildFrame(kind: "snapshot" | "delta"): StreamFrame {
    this.seq += 1;
    const staff: StaffKinematic[] = [...this.staff.values()].map((m) => this.staffKinematic(m));
    const bays: BayFrame[] = [...this.bays.values()].map((b) => ({
      bay: b.id,
      status: b.status,
      occupant: b.occupant,
      cleaning_eta: b.cleaningEta,
    }));
    const chips: PatientChip[] = [...this.patients.values()].map((p) => ({
      patient: p.id,
      esi: p.esi,
      at_node: p.atNode,
      stage: p.stage,
      waited: this.simTimeUs - p.stageSinceUs,
    }));
    const queues: QueueFrame[] = (["waiting_triage", "waiting_bay"] as const).map((stage) => {
      const waiting = this.waitingByStage(stage);
      return { stage, depth: waiting.length, head: waiting.slice(0, 5).map((p) => p.id) };
    });
    const events = this.pendingEvents;
    this.pendingEvents = [];
    return {
      run: this.runId,
      sim_time: this.simTimeUs,
      seq: this.seq,
      kind,
      state: this.state === "created" ? "paused" : (this.state as StreamFrame["state"]),
      speed: this.speed,
      staff,
      bays,
      queues,
      patients: chips,
      pending_tasks: this.pendingTasks(),
      events,
      kpi_preview: null,
    };
  }

  // ------------------------------------------------------------------ metrics

  metrics(): KpiVector {
    const values: Record<string, number | null> = {};
    for (const key of KPI_KEYS) {
      values[key] = null;
    }
    const elapsedUs = Math.max(this.simTimeUs, STEP_US);
    values["completions_per_week"] = this.completions;
    values["wip_end_of_week"] = this.patients.size;
    values["door_to_triage_s_mean"] = mean(this.doorToTriageS);
    values["door_to_triage_s_p90"] = percentile([...this.doorToTriageS].sort((a, b) => a - b), 0.9);
    values["door_to_provider_s_mean"] = mean(this.doorToProviderS);
    values["door_to_provider_s_p90"] = percentile(
      [...this.doorToProviderS].sort((a, b) => a - b),
      0.9,
    );
    for (const esi of [1, 2, 3, 4, 5] as const) {
      const los = this.losSByEsi[esi] ?? [];
      values[`los_s_mean_by_esi_${esi}`] = mean(los);
      values[`los_s_p90_by_esi_${esi}`] = percentile([...los].sort((a, b) => a - b), 0.9);
    }
    values["staff_minutes_walked"] = this.fracUs.walk / (60 * S);
    // Denominated by the bays actually IN SERVICE, not by the floor's geometry:
    // closing bays (by capacity slider or operator override) must not read as
    // "utilization fell", which is the opposite of what closing them did.
    const openBays = [...this.bays.values()].filter((b) => b.status !== "closed").length;
    values["bay_utilization"] = Math.min(
      1,
      this.occupiedBayUs / (Math.max(openBays, 1) * elapsedUs),
    );
    values["turnaround_time_s_mean"] = mean(this.turnaroundS);
    values["boarding_time_s_mean"] = mean(this.boardingS);
    const utilOf = (role: StaffRole): number => {
      const members = [...this.staff.values()].filter((m) => m.role === role);
      if (members.length === 0) {
        return 0;
      }
      return Math.min(
        1,
        members.reduce((a, m) => a + m.busyUs, 0) / (members.length * elapsedUs),
      );
    };
    values["provider_util"] = utilOf("physician");
    values["nurse_util"] = utilOf("nurse");
    const staffSeconds = this.staff.size * elapsedUs;
    const frac = (v: number): number => v / Math.max(staffSeconds, 1);
    const walk = frac(this.fracUs.walk);
    const care = frac(this.fracUs.direct_care);
    const clean = frac(this.fracUs.cleaning);
    const doc = frac(this.fracUs.documentation);
    values["staff_frac_walk"] = walk;
    values["staff_frac_direct_care"] = care;
    values["staff_frac_cleaning"] = clean;
    values["staff_frac_documentation"] = doc;
    values["staff_frac_idle"] = Math.max(0, 1 - walk - care - clean - doc);
    return { values };
  }

  bottleneck(): BottleneckReport {
    const resources: ResourceWait[] = (
      ["triage", "provider", "nurse", "imaging", "lab", "housekeeping"] as const
    ).map((resource) => {
      const totalUs = this.waitSums[resource];
      const n = this.waitCounts[resource];
      return {
        resource,
        total_wait_s: totalUs / S,
        n_requests: n,
        mean_wait_s: n > 0 ? totalUs / S / n : 0,
        share_of_cycle: 0,
      };
    });
    const total = resources.reduce((a, r) => a + r.total_wait_s, 0);
    const shared = resources.map((r) => ({
      ...r,
      share_of_cycle: total > 0 ? r.total_wait_s / total : 0,
    }));
    // The mock always computes a finite share (see above), so `?? 0` is a
    // type-level floor for the nullable wire type, never a live code path.
    const binding = shared.reduce(
      (a, b) => ((b.share_of_cycle ?? 0) > (a.share_of_cycle ?? 0) ? b : a),
      { ...(shared[0] as ResourceWait) },
    );
    const byRole: Record<string, number> = {};
    for (const role of ["physician", "nurse", "tech", "porter", "housekeeping"] as const) {
      const busy = [...this.staff.values()].filter((m) => m.role === role).map((m) => m.busyUs);
      byRole[role] = gini(busy);
    }
    return {
      binding: binding.resource,
      resources: shared,
      total_cycle_s: total,
      gini_by_role: byRole,
      gini_overall: gini([...this.staff.values()].map((m) => m.busyUs)),
    };
  }

  // ----------------------------------------------------------------- overrides

  private currentPlan(): Plan {
    const items: PlanItem[] = [];
    for (const bay of this.bays.values()) {
      if (bay.status === "occupied" && bay.occupant !== null) {
        items.push({
          stable_id: `assign:${bay.occupant}`,
          kind: "assign_bay",
          patient: bay.occupant,
          bay: bay.id,
        });
      }
      if (bay.status === "cleaning") {
        items.push({ stable_id: `clean:${bay.id}`, kind: "clean", bay: bay.id });
      }
    }
    return { items };
  }

  /**
   * Mirror of the real pipeline: compile → validate → apply-or-reject.
   * Violations use the same kinds and detail phrasing as core.validation so
   * the OverridePanel renders realistic verbatim reasons. Atomic: a rejection
   * mutates nothing.
   */
  applyOverride(action: OperatorAction, pin: boolean): OverrideOutcome {
    const violations: Violation[] = [];
    const reject = (): OverrideOutcome => ({ status: "rejected", violations });

    switch (action.kind) {
      case "reassign": {
        const patient = this.patients.get(action.patient);
        const bay = this.bays.get(action.bay);
        const fixture = this.bayFixture(action.bay);
        if (patient === undefined) {
          violations.push({ kind: "unknown_entity", detail: "unknown patient", entity: action.patient });
        }
        if (bay === undefined || fixture === undefined) {
          violations.push({ kind: "unknown_entity", detail: "unknown bay", entity: action.bay });
        }
        if (patient === undefined || bay === undefined || fixture === undefined) {
          return reject();
        }
        if (bay.status !== "free" && bay.occupant !== patient.id) {
          violations.push({
            kind: "bay_incompatible",
            detail: `bay not free (status ${bay.status})`,
            entity: bay.id,
          });
          if (bay.status === "occupied") {
            violations.push({ kind: "double_booked", detail: "2 patients on one bay", entity: bay.id });
          }
        }
        if (!allowedZoneTypes(patient.esi).includes(fixture.zone_type)) {
          violations.push({
            kind: "bay_incompatible",
            detail: `esi ${patient.esi} not allowed in ${fixture.zone_type}`,
            entity: bay.id,
          });
        }
        if (patient.isolation && !fixture.isolation_capable) {
          violations.push({
            kind: "isolation_violated",
            detail: "patient requires an isolation-capable bay",
            entity: bay.id,
          });
        }
        if (violations.length > 0) {
          return reject();
        }
        const old = patient.bay === null ? null : this.bays.get(patient.bay);
        if (old !== undefined && old !== null && old.occupant === patient.id) {
          old.status = "free";
          old.occupant = null;
        }
        this.placePatient(patient, bay, "operator");
        patient.pinned = pin;
        if (pin) {
          this.pins.add(patient.id);
        }
        break;
      }
      case "close_bay": {
        const bay = this.bays.get(action.bay);
        if (bay === undefined) {
          violations.push({ kind: "unknown_entity", detail: "unknown bay", entity: action.bay });
          return reject();
        }
        if (bay.status !== "free") {
          // Closing an occupied bay strands its assign_bay item — the operator
          // must reassign first; the engine never evicts (doc 07 §4.2).
          violations.push({
            kind: "bay_incompatible",
            detail: `bay not free (status ${bay.status})`,
            entity: bay.id,
          });
          return reject();
        }
        bay.status = "closed";
        break;
      }
      case "block_edge": {
        const [a, b] = action.edge;
        const known = this.edgeUs.has(`${a}>${b}`) || this.edgeUs.has(`${b}>${a}`);
        if (!known) {
          violations.push({ kind: "unknown_entity", detail: "unknown edge", entity: `${a}>${b}` });
          return reject();
        }
        this.blockedEdges.add(`${a}>${b}`);
        break;
      }
      case "reroute": {
        const staff = this.staff.get(action.staff);
        if (staff === undefined) {
          violations.push({ kind: "unknown_entity", detail: "unknown staff", entity: action.staff });
          return reject();
        }
        const [taskKind, node] = action.task.split(":") as [string, NodeId | undefined];
        const required: StaffRole =
          taskKind === "cleaning" ? "housekeeping" : taskKind === "provider_visit" ? "physician" : "nurse";
        if (staff.role !== required) {
          violations.push({
            kind: "staff_lacks_skill",
            detail: `task needs role ${required}, staff is ${staff.role}`,
            entity: staff.id,
          });
          return reject();
        }
        if (node !== undefined && this.layout.graph.nodes.some((n) => n.id === node)) {
          this.dispatchStaff(staff, {
            kind: required === "housekeeping" ? "cleaning" : required === "physician" ? "provider_visit" : "nurse_visit",
            node,
            durationUs: 300 * S,
          });
        }
        break;
      }
      case "bump_priority":
      case "expedite_discharge": {
        const patient = this.patients.get(action.patient);
        if (patient === undefined) {
          violations.push({ kind: "unknown_entity", detail: "unknown patient", entity: action.patient });
          return reject();
        }
        if (action.kind === "expedite_discharge" && patient.stage === "in_bay") {
          patient.remainingUs = Math.min(patient.remainingUs, 120 * S);
        }
        if (action.kind === "bump_priority") {
          // Reorder only — never touch stageSinceUs, or the chip's displayed
          // wait and the boarding metric would both balloon (finding #7).
          patient.priority = action.priority;
        }
        break;
      }
      case "expedite_clean": {
        const bay = this.bays.get(action.bay);
        if (bay === undefined) {
          violations.push({ kind: "unknown_entity", detail: "unknown bay", entity: action.bay });
          return reject();
        }
        if (bay.status === "cleaning") {
          bay.remainingUs = Math.min(bay.remainingUs, 60 * S);
          bay.cleaningEta = this.simTimeUs + bay.remainingUs;
        }
        break;
      }
    }

    return { status: "applied", plan: this.currentPlan(), applied_at: this.simTimeUs };
  }
}
