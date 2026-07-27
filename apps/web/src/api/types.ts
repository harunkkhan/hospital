/**
 * The API wire contract, hand-typed from docs/plan/07-apps-console.md §3/§7
 * and the frozen pydantic contracts in hospital.core.
 *
 * This module is deliberately the ONLY place wire shapes are defined on the
 * web side. The API will later commit a generated JSON-schema artifact
 * (`schema.gen.ts` via openapi-typescript) and a contract-drift check will
 * compare against it — keep every wire type here so the swap is one import
 * edit, and never let a component redefine a contract shape inline.
 *
 * Pydantic `RootModel` newtypes flatten on the wire: every typed id is a bare
 * string and every SimTime/Duration is an int (microseconds). The aliases
 * below are documentation, not nominal types — branding, if ever needed,
 * happens in an adapter over this module, never inside it.
 */

// ---------------------------------------------------------------------------
// ids + time (core.ids / core.time — RootModel newtypes, flat on the wire)
// ---------------------------------------------------------------------------

export type PatientId = string;
export type BayId = string;
export type ZoneId = string;
export type NodeId = string;
export type StaffId = string;
export type TaskId = string;
export type RunId = string;

/** Absolute sim instant, integer microseconds since run start. */
export type SimTime = number;
/** Elapsed sim span, integer microseconds. */
export type Duration = number;

// ---------------------------------------------------------------------------
// enums (core.enums — StrEnum values verbatim; EsiAcuity is an IntEnum)
// ---------------------------------------------------------------------------

/** Emergency Severity Index. 1 = MOST critical, 5 = least (the sign trap). */
export type EsiAcuity = 1 | 2 | 3 | 4 | 5;

export type ArrivalMode = "walk_in" | "ambulance";

export type ZoneType =
  | "triage"
  | "general"
  | "resus_trauma"
  | "fast_track"
  | "observation"
  | "imaging"
  | "lab";

export type BayStatus = "free" | "occupied" | "cleaning" | "closed";

export type StaffRole = "physician" | "nurse" | "tech" | "porter" | "housekeeping";

export type DispositionKind = "discharge" | "admit" | "transfer";

export type Activity =
  | "triage"
  | "provider_visit"
  | "nurse_visit"
  | "imaging"
  | "lab"
  | "documentation"
  | "discharge"
  | "cleaning"
  | "transport";

// ---------------------------------------------------------------------------
// layout (core.entities / core.graph — static, fetched once per run)
// ---------------------------------------------------------------------------

export interface RouteNode {
  id: NodeId;
  label: string;
  /** Viz/interpolation only — NEVER pathfinding (edge `seconds` is authority). */
  x_cm: number;
  y_cm: number;
}

export interface RouteEdge {
  a: NodeId;
  b: NodeId;
  /** Centimetres (Distance newtype). */
  distance: number;
  /** Traversal time in MICROSECONDS (Duration newtype, despite the name). */
  seconds: Duration;
  bidirectional: boolean;
}

export interface RouteGraph {
  nodes: readonly RouteNode[];
  edges: readonly RouteEdge[];
}

export interface Zone {
  id: ZoneId;
  zone_type: ZoneType;
  capacity: number;
}

export interface Bay {
  id: BayId;
  zone: ZoneId;
  zone_type: ZoneType;
  node: NodeId;
  serving_station: NodeId;
  isolation_capable: boolean;
  equipment: readonly string[];
}

export interface FloorLayout {
  graph: RouteGraph;
  zones: readonly Zone[];
  bays: readonly Bay[];
  stations: readonly NodeId[];
  entrances: readonly NodeId[];
  imaging_nodes: readonly NodeId[];
  lab_nodes: readonly NodeId[];
}

// ---------------------------------------------------------------------------
// events (core.events — streamed VERBATIM in frames; discriminated on `kind`)
// ---------------------------------------------------------------------------

interface Ev {
  occurred_at: SimTime;
}

export type DecidedBy = "solver" | "operator" | "baseline";

export type Event =
  | (Ev & { kind: "patient_arrived"; patient: PatientId; mode: ArrivalMode })
  | (Ev & { kind: "triage_started"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "triage_completed"; patient: PatientId; esi: EsiAcuity })
  | (Ev & { kind: "bay_requested"; patient: PatientId })
  | (Ev & { kind: "bay_assigned"; patient: PatientId; bay: BayId; by: DecidedBy })
  | (Ev & {
      kind: "patient_moved";
      patient: PatientId;
      edge: [NodeId, NodeId];
      seconds: Duration;
    })
  | (Ev & { kind: "provider_visit_started"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "provider_visit_completed"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "nurse_visit_started"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "nurse_visit_completed"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "test_ordered"; patient: PatientId; activity: Activity })
  | (Ev & { kind: "test_resulted"; patient: PatientId; activity: Activity })
  | (Ev & { kind: "documentation_started"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "documentation_completed"; patient: PatientId; staff: StaffId })
  | (Ev & { kind: "disposition_decided"; patient: PatientId; disposition: DispositionKind })
  | (Ev & { kind: "discharge_started"; patient: PatientId })
  | (Ev & { kind: "discharge_completed"; patient: PatientId })
  | (Ev & { kind: "bay_cleaning_started"; bay: BayId; staff: StaffId })
  | (Ev & { kind: "bay_cleaning_completed"; bay: BayId; staff: StaffId })
  | (Ev & { kind: "staff_moved"; staff: StaffId; edge: [NodeId, NodeId]; seconds: Duration })
  | (Ev & { kind: "staff_idle"; staff: StaffId; at: NodeId })
  | (Ev & { kind: "disruption_injected"; disruption: string; detail: string })
  | (Ev & { kind: "vitals_sampled"; patient: PatientId; news2: number })
  | (Ev & { kind: "deterioration_detected"; patient: PatientId; news2: number })
  | (Ev & { kind: "emergency_raised"; patient: PatientId });

export interface EventEnvelope {
  event: Event;
  sequence: number;
  caused_by: number | null;
}

// ---------------------------------------------------------------------------
// KPI contract (core.kpi — closed key set; empty strata are NaN, never omitted)
// ---------------------------------------------------------------------------

const LOS_KEYS = (["mean", "p90"] as const).flatMap((stat) =>
  ([1, 2, 3, 4, 5] as const).map((k) => `los_s_${stat}_by_esi_${k}` as const),
);

/** The closed KPI contract — mirrors core.kpi.KPI_KEYS (27 keys). */
export const KPI_KEYS = [
  "completions_per_week",
  "wip_end_of_week",
  "door_to_triage_s_mean",
  "door_to_triage_s_p90",
  "door_to_provider_s_mean",
  "door_to_provider_s_p90",
  ...LOS_KEYS,
  "staff_minutes_walked",
  "bay_utilization",
  "turnaround_time_s_mean",
  "boarding_time_s_mean",
  "provider_util",
  "nurse_util",
  "staff_frac_walk",
  "staff_frac_direct_care",
  "staff_frac_cleaning",
  "staff_frac_documentation",
  "staff_frac_idle",
] as const;

export type KpiKey = (typeof KPI_KEYS)[number];

/**
 * A KPI reading. `Mapping[str, float]` generates to an index signature, so
 * membership in KPI_KEYS is NOT enforced by TS — panels must treat keys as
 * ⊆ KPI_KEYS, present-or-absent, and guard NaN/null for empty strata.
 */
export interface KpiVector {
  values: Readonly<Record<string, number | null>>;
}

// ---------------------------------------------------------------------------
// plan / validation (core.seam / core.validation — the override contract)
// ---------------------------------------------------------------------------

export type PlanItemKind =
  | "assign_bay"
  | "sequence"
  | "dispatch"
  | "clean"
  | "discharge"
  | "staffing";

export interface PlanItem {
  stable_id: string;
  kind: PlanItemKind;
  patient?: PatientId | null;
  bay?: BayId | null;
  staff?: StaffId | null;
  task?: TaskId | null;
  priority?: number | null;
  route?: readonly NodeId[] | null;
  order?: readonly string[] | null;
}

export interface Plan {
  items: readonly PlanItem[];
}

export type ViolationKind =
  | "unknown_entity"
  | "bay_incompatible"
  | "capacity_exceeded"
  | "isolation_violated"
  | "staff_lacks_skill"
  | "double_booked"
  | "precedence_violated";

/** A single rule breach — rendered VERBATIM by the OverridePanel. */
export interface Violation {
  kind: ViolationKind;
  detail: string;
  entity: string;
}

// ---------------------------------------------------------------------------
// run lifecycle (POST /runs, GET /runs/{id})
// ---------------------------------------------------------------------------

export type Arm = "baseline" | "optimized";
export type RunState = "created" | "playing" | "paused" | "stepping" | "finished";

export interface ScenarioRef {
  id: string;
}

export interface ScenarioInline {
  base: string;
  overrides: Readonly<Record<string, number>>;
}

export interface RunRequest {
  scenario: ScenarioRef | ScenarioInline;
  seed: number;
  arm: Arm;
  /** Spins a CRN shadow arm under the SAME seed, for /compare. */
  compare_to?: Arm | null;
  start?: "paused" | "playing";
}

export interface RunHandle {
  run: RunId;
  arm: Arm;
  seed: number;
  horizon: SimTime;
  state: RunState;
  sim_time: SimTime;
  stream_url: string;
  shadow?: RunId | null;
}

// ---------------------------------------------------------------------------
// playback control (POST /runs/{id}/control)
// ---------------------------------------------------------------------------

export type ControlAction = "play" | "pause" | "step" | "speed";
export type StepGranularity = "decision" | "event" | "tick";

export interface ControlCommand {
  action: ControlAction;
  /** For "speed": sim-µs per wall-ms. Pacing only — never sampling. */
  multiplier?: number | null;
  granularity?: StepGranularity;
  count?: number;
}

export interface SessionState {
  run: RunId;
  state: RunState;
  sim_time: SimTime;
  speed: number;
  horizon: SimTime;
}

// ---------------------------------------------------------------------------
// operator overrides (POST /runs/{id}/override)
// ---------------------------------------------------------------------------

/**
 * The action vocabulary (doc 07 §4.1). Five compile to PlanItem edits; the
 * two availability actions (close_bay / block_edge) compile to
 * ValidationContext deltas. Discriminated on `kind`.
 */
export type OperatorAction =
  | { kind: "reassign"; patient: PatientId; bay: BayId }
  | { kind: "bump_priority"; patient: PatientId; priority: number }
  | { kind: "reroute"; staff: StaffId; task: TaskId }
  | { kind: "expedite_clean"; bay: BayId; priority?: number }
  | { kind: "expedite_discharge"; patient: PatientId; priority?: number }
  | { kind: "close_bay"; bay: BayId }
  | { kind: "block_edge"; edge: [NodeId, NodeId] };

export type OperatorActionKind = OperatorAction["kind"];

export interface OverrideRequest {
  action: OperatorAction;
  /** Hold the decision against the solver's next re-solve (pin registry). */
  pin?: boolean;
}

export interface OverrideAccepted {
  status: "applied";
  /** The merged, validated core.seam.Plan now in force. */
  plan: Plan;
  applied_at: SimTime;
}

export interface OverrideRejected {
  status: "rejected";
  /** Verbatim core.validation.Violation list — never repaired or softened. */
  violations: readonly Violation[];
}

export type OverrideOutcome = OverrideAccepted | OverrideRejected;

// ---------------------------------------------------------------------------
// metrics / compare (GET /runs/{id}/metrics, /compare, /bottleneck)
// ---------------------------------------------------------------------------

export interface KpiContrast {
  key: string;
  baseline: number;
  optimized: number;
  /** baseline - optimized. Sign is honest; direction-of-good is per key. */
  delta: number;
  ci_lo: number;
  ci_hi: number;
  significant: boolean;
}

export interface CompareResponse {
  baseline_run: RunId;
  optimized_run: RunId;
  /** 1 == live single-seed point delta (CIs degenerate — render "n=1, no CI"). */
  replications: number;
  contrasts: readonly KpiContrast[];
}

/** Projection of hospital.analysis.bottleneck.BottleneckReport. */
export interface ResourceWait {
  resource: string;
  total_wait_s: number;
  n_requests: number;
  mean_wait_s: number;
  share_of_cycle: number;
}

export interface BottleneckReport {
  binding: string;
  resources: readonly ResourceWait[];
  total_cycle_s: number;
  gini_by_role: Readonly<Record<string, number>>;
  gini_overall: number;
}

// ---------------------------------------------------------------------------
// scenarios (GET/POST /scenarios)
// ---------------------------------------------------------------------------

export interface ScenarioSummary {
  id: string;
  name: string;
  horizon: SimTime;
  note: string;
}

export interface ScenarioCreateRequest {
  base: string;
  overrides: Readonly<Record<string, number>>;
}

export interface ScenarioCreated {
  id: string;
}

// ---------------------------------------------------------------------------
// stream frames (GET /runs/{id}/stream — doc 07 §7.2, owned by api/stream.py)
// ---------------------------------------------------------------------------

/** Rendering-shaped staff kinematics — NOT a domain type. */
export interface StaffKinematic {
  staff: StaffId;
  role: StaffRole;
  /** Resting at a node ... */
  at_node: NodeId | null;
  /** ... or traversing an edge. */
  edge: [NodeId, NodeId] | null;
  /** 0..1 along the edge, for smooth interpolation. */
  edge_progress: number;
  activity: Activity | "idle";
  current_task: TaskId | null;
}

export interface BayFrame {
  bay: BayId;
  status: BayStatus;
  occupant: PatientId | null;
  cleaning_eta: SimTime | null;
}

export interface QueueFrame {
  stage: string;
  depth: number;
  /** First-N for chip rendering. */
  head: readonly PatientId[];
}

export interface PatientChip {
  patient: PatientId;
  esi: EsiAcuity;
  at_node: NodeId | null;
  stage: string;
  waited: Duration;
}

/**
 * A unit of pending work an operator can reroute a staff member onto. The
 * `id` is OPAQUE (the sim mints it, e.g. `task_000001`) — the console must
 * echo it back verbatim in a `reroute` override and MUST NOT synthesize its
 * own from kind+bay, or the id will never match a real task. `kind`/`at` are
 * for labelling the picker only.
 */
export interface PendingTask {
  id: TaskId;
  kind: Activity;
  at: NodeId;
}

export type FrameKind = "snapshot" | "delta";

export interface StreamFrame {
  run: RunId;
  sim_time: SimTime;
  /** Monotonic per run — the client's gap detector; gap ⇒ re-snapshot. */
  seq: number;
  kind: FrameKind;
  state: Exclude<RunState, "created">;
  speed: number;
  staff: readonly StaffKinematic[];
  bays: readonly BayFrame[];
  queues: readonly QueueFrame[];
  patients: readonly PatientChip[];
  /** core.events verbatim: events since the previous frame (never dropped). */
  events: readonly EventEnvelope[];
  /**
   * Reroutable tasks live at this instant. Present (even empty) ⇒ authoritative;
   * omitted on a delta ⇒ unchanged. A frame that never carries it leaves the
   * reroute picker empty, which the OverridePanel disables.
   */
  pending_tasks?: readonly PendingTask[] | null;
  kpi_preview?: KpiVector | null;
}
