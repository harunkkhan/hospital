/**
 * Pure reducer folding StreamFrames into the client's world view.
 *
 * Semantics (doc 07 §7.2/§7.3):
 * - `snapshot` frames REPLACE the whole view (the reconnection protocol is
 *   "re-snapshot", never "replay from seq").
 * - `delta` frames UPSERT the entities they carry, keyed by id. Entity
 *   removal is event-driven: a `discharge_completed` retires the patient
 *   chip. The server-authoritative world is truth; anything stale here is
 *   corrected by the next snapshot.
 * - A `seq` gap on a delta marks the view `desynced` and applies nothing —
 *   the stream layer must reconnect, which yields a fresh snapshot.
 * - The `events` tail is never dropped: it is appended to a bounded ring.
 */

import type {
  BayFrame,
  BayId,
  EventEnvelope,
  KpiVector,
  PatientChip,
  PatientId,
  PendingTask,
  QueueFrame,
  RunId,
  RunState,
  SimTime,
  StaffId,
  StaffKinematic,
  StreamFrame,
} from "../api/types";

export const EVENT_RING_CAPACITY = 250;

export interface WorldView {
  run: RunId | null;
  simTime: SimTime;
  /** Last applied frame seq; -1 before the first frame. */
  seq: number;
  state: RunState | null;
  speed: number;
  staff: Readonly<Record<StaffId, StaffKinematic>>;
  bays: Readonly<Record<BayId, BayFrame>>;
  queues: readonly QueueFrame[];
  patients: Readonly<Record<PatientId, PatientChip>>;
  /** Reroutable tasks at the last applied frame (real ids from the sim). */
  pendingTasks: readonly PendingTask[];
  /** Bounded ring of recent events, oldest first. */
  events: readonly EventEnvelope[];
  kpiPreview: KpiVector | null;
  /** True when a seq gap was detected — the stream must re-snapshot. */
  desynced: boolean;
}

export function initialWorld(): WorldView {
  return {
    run: null,
    simTime: 0,
    seq: -1,
    state: null,
    speed: 1,
    staff: {},
    bays: {},
    queues: [],
    patients: {},
    pendingTasks: [],
    events: [],
    kpiPreview: null,
    desynced: false,
  };
}

function keyBy<T>(items: readonly T[], key: (item: T) => string): Record<string, T> {
  const out: Record<string, T> = {};
  for (const item of items) {
    out[key(item)] = item;
  }
  return out;
}

function appendEvents(
  ring: readonly EventEnvelope[],
  incoming: readonly EventEnvelope[],
): readonly EventEnvelope[] {
  if (incoming.length === 0) {
    return ring;
  }
  const merged = [...ring, ...incoming];
  return merged.length > EVENT_RING_CAPACITY ? merged.slice(-EVENT_RING_CAPACITY) : merged;
}

/** Patients whose terminal event arrived in this frame are retired from the map. */
function departedPatients(events: readonly EventEnvelope[]): ReadonlySet<PatientId> {
  const gone = new Set<PatientId>();
  for (const env of events) {
    if (env.event.kind === "discharge_completed") {
      gone.add(env.event.patient);
    }
  }
  return gone;
}

export function applyFrame(world: WorldView, frame: StreamFrame): WorldView {
  if (frame.kind === "snapshot") {
    return {
      run: frame.run,
      simTime: frame.sim_time,
      seq: frame.seq,
      state: frame.state,
      speed: frame.speed,
      staff: keyBy(frame.staff, (s) => s.staff),
      bays: keyBy(frame.bays, (b) => b.bay),
      queues: frame.queues,
      patients: keyBy(frame.patients, (p) => p.patient),
      pendingTasks: frame.pending_tasks ?? [],
      events: appendEvents(world.run === frame.run ? world.events : [], frame.events),
      kpiPreview: frame.kpi_preview ?? null,
      desynced: false,
    };
  }

  // Stale or duplicate delta — the view already reflects a later frame.
  if (frame.seq <= world.seq) {
    return world;
  }

  // Gap: a delta we cannot safely merge. Apply nothing; flag for re-snapshot.
  if (world.seq >= 0 && frame.seq !== world.seq + 1) {
    return world.desynced ? world : { ...world, desynced: true };
  }

  // A delta before any snapshot is equally unsafe to merge.
  if (world.seq < 0) {
    return world.desynced ? world : { ...world, desynced: true };
  }

  const gone = departedPatients(frame.events);
  const patients: Record<PatientId, PatientChip> = {};
  for (const [id, chip] of Object.entries(world.patients)) {
    if (!gone.has(id)) {
      patients[id] = chip;
    }
  }
  for (const chip of frame.patients) {
    if (!gone.has(chip.patient)) {
      patients[chip.patient] = chip;
    }
  }

  return {
    run: frame.run,
    simTime: frame.sim_time,
    seq: frame.seq,
    state: frame.state,
    speed: frame.speed,
    staff: frame.staff.length > 0 ? { ...world.staff, ...keyBy(frame.staff, (s) => s.staff) } : world.staff,
    bays: frame.bays.length > 0 ? { ...world.bays, ...keyBy(frame.bays, (b) => b.bay) } : world.bays,
    queues: frame.queues.length > 0 || frame.patients.length > 0 ? frame.queues : world.queues,
    patients,
    // Present (even empty) is authoritative; omitted keeps the last known set.
    pendingTasks: frame.pending_tasks ?? world.pendingTasks,
    events: appendEvents(world.events, frame.events),
    kpiPreview: frame.kpi_preview ?? world.kpiPreview,
    desynced: false,
  };
}
