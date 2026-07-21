# Simulation Spec — What the ER Digital Twin Represents

This is the **domain / capability specification** for the Hospital
Operations-Research Simulator. It describes *what* the system models — the
emergency-room world, the entities in it, how patients flow, what gets
optimized, and what gets measured. It is deliberately implementation-agnostic:
the *how* (the engine's typed seam, determinism model, solver contract, and
package layout) lives in the companion [`PROGRAMMING_MODEL_SPEC.md`](./PROGRAMMING_MODEL_SPEC.md).

---

## 1. Purpose

A digital-twin **operations-research simulator** of a single, large
(~100,000 sq ft) hospital **Emergency Room**, run over a **one-week** horizon.

Hospitals cannot legally optimize most clinical variables. What they *can*
optimize is **operations** — how people and beds move, are placed, are cleaned,
and are scheduled. On a large ER floor, enormous amounts of time evaporate into
staff walking long distances, patients placed far from the staff who serve them,
bays sitting dirty and unavailable after a patient leaves, slow discharge
paperwork, and mismatched staffing. Because time-per-procedure is unpredictable,
the floor is run reactively and space is wasted.

The simulator runs an **unoptimized BASELINE** arm against an **OPTIMIZED** arm
over *identical* randomness and quantifies the difference. It is the falsifiable
proof that better operations — not better medicine — save patient-time and
increase throughput. The end product is an **operator's console** in the browser
where a solver auto-pilots the floor and a human can intervene in real time.

### Goals

- **G1 — Prove the value headless first.** A reproducible engine that runs
  BASELINE vs OPTIMIZED over one week and emits a metrics report showing time
  saved and extra completed treatments.
- **G2 — Optimize the full operational lever set:** placement, sequencing,
  routing/dispatch, bed turnaround, discharge, paperwork, staff scheduling.
- **G3 — An operator's console in the browser:** observe the live floor, let the
  solver drive, and allow real-time human overrides through the same validated
  decision seam the solver uses.
- **G4 — Predict, then optimize:** statistical and ML forecasts (service time,
  arrivals/surge, deterioration) that feed the solver's inputs and retrain as
  data accumulates.
- **G5 — Analysis as a first-class output:** where the time goes, what the
  binding constraint is, how utilized staff are, and a statistically honest
  baseline-vs-optimized comparison.

---

## 2. Scope and non-goals (v1)

**In scope (v1 / M1):** a single ER floor modeled as a spatial graph; a
one-week patient workload; the full ER patient journey; the operational levers
above; deterministic baseline-vs-optimized comparison with confidence intervals.

**Non-goals (v1), each deferred to a named milestone:**

- **No admission gatekeeping, diversion, or refusal.** Patients are *always*
  accepted; there is **no "left without being seen"** — patients have infinite
  patience. Throughput is a consequence of good operations, never of turning
  people away.
- **No dollar/cost model yet.** Cost translation is deferred until analytics are
  observable (M4b); v1 optimizes and reports *time*, not money.
- **No other floors and no elevators.** Single ER floor; vertical movement and
  inter-floor transport arrive in M4.
- **No clinical decision-making.** We model *operations*, not diagnosis or
  treatment choice. Acuity, complaint, and workup needs are inputs, not outputs.

---

## 3. The ER domain model

Everything crossing a boundary is a frozen, validated value (see the programming
model spec). This section describes the *meaning* of each.

### 3.1 Units and time

- **Time** (`SimTime`, `Duration`) is integer **microseconds** from a scenario
  epoch. Float timestamps are never stored or compared, so runs are bit-stable.
  Helpers express human units: `seconds()`, `minutes()`, `hours()`, and an
  `OperatingWeek` covering the 7-day horizon.
- **Distance** is integer **centimetres**; a `WalkSpeed` converts an edge's
  distance into a traversal `Duration`.

### 3.2 Typed identities

Each entity carries a distinct typed ID — `PatientId`, `BayId`, `ZoneId`,
`StaffId`, `NodeId`, `TaskId`, `RunId`. They are not interchangeable strings: a
`BayId` can never be passed where a `PatientId` is expected. This makes whole
classes of wiring bugs impossible before the simulation ever runs.

### 3.3 Entities

- **Patient** — `arrival_time`, `arrival_mode ∈ {walk_in, ambulance}`,
  `esi_acuity ∈ 1..5` (Emergency Severity Index; 1 = most critical),
  `complaint` category, `isolation_required`, and a `WorkupNeeds` bundle
  (labs?, imaging kind?, procedure count, expected provider visits). A derived
  `care_deadline` encodes a soft SLA by acuity.
- **Bay / Zone** — a treatment space with a `zone_type ∈ {triage, general,
  resus_trauma, fast_track, observation, imaging, lab}`, a `node_id` locating it
  on the floor graph, a `status ∈ {free, occupied, cleaning, closed}`,
  `capabilities` (equipment, isolation-capable), and a `serving_station` (the
  nurse station that covers it).
- **StaffMember** — a `role ∈ {physician, nurse, tech, porter, housekeeping}`, a
  `home_station`, a set of `skills`, and an `on_shift` flag driven by the
  schedule. Staff are **movable agents with graph positions**, not counters —
  their travel is the thing the simulator measures.
- **FloorLayout** — the floor graph plus the bay/zone/station inventory.

---

## 4. Floor and spatial model

- **Graph** `G = (V, E)`. Nodes are bays, triage rooms, imaging suites, the lab,
  nurse stations, the waiting room, the ambulance bay, the walk-in entrance, and
  corridor junctions. Edges are physical corridors carrying `distance_cm` and a
  precomputed traversal `seconds` at walk speed, with a `bidirectional` flag.
- **Scale matters.** The graph is sized so the floor area is ≈ **100,000 sq ft**,
  which means cross-floor traversals take tens of seconds to minutes. **Movement
  is a first-order cost even on one floor** — which is precisely why placement
  and routing can move the metrics.
- **Pathfinding** is deterministic shortest-path with stable tie-breaks on
  `(seconds, distance, node_id)`. It supports live rerouting around
  `blocked_edges` and `closed_nodes` — a corridor cordoned off, a zone locked
  down. Distances are memoized.
- **Movement is physical.** Staff and transported patients traverse edges one at
  a time and emit movement events; **nothing teleports**. This is what lets the
  simulator measure staff-minutes walked and prove placement/routing gains.
- **Layout is randomness-free.** The floor is built deterministically from scale
  parameters (zone counts, bays per zone, station coverage, corridor topology).
  Randomness enters *only* at simulated sampling points, never at construction.

---

## 5. Patient flow

Each patient is an independent process moving through nine stages. Every
transition emits a typed event to the append-only log (§8).

1. **Arrive** (walk-in or ambulance) and enqueue for registration/triage.
2. **Triage** — an acuity is assigned (ambulances may pre-assign); triage events
   are emitted.
3. **Wait for a bay** in an acuity-priority queue. A patient never abandons the
   queue (infinite patience).
4. **Bay assigned** by the decision layer — the patient is escorted or walks the
   graph to the bay (physical traversal; staff may be required).
5. **Provider evaluation** — a physician is dispatched (travel + service time).
6. **Workup loop** — labs and imaging (transport to the imaging/lab node, queue
   for that resource, result delay) plus nurse visits, repeated per the
   patient's `WorkupNeeds`.
7. **Disposition** — `discharge`, `admit`, or `transfer`.
8. **Discharge process** — documentation/paperwork time, then the patient
   leaves (a *completion*). An `admit` becomes **boarding**: the patient waits,
   then leaves the modeled floor in v1, with boarding time recorded.
9. **Bay turnaround** — a housekeeping cleaning task; on completion the bay
   returns to `free`.

### Resources and agents

Bays are capacity-1 resources; imaging, lab, and triage are capacity-N. Staff
are movable agents with graph positions. Same-instant events are ordered by
fixed priority tiers (completions before decision ticks before disruptions) so
ties never introduce nondeterminism.

### Service-time model

`ServiceTimes.sample(activity, acuity, complaint)` draws from short-mean
lognormal/gamma distributions (it is an ER) via a dedicated random substream, so
adding a new activity never perturbs existing draws (see the CRN model in the
programming spec).

---

## 6. Workload, horizon, and disruptions

### 6.1 Workload generation

The one-week arrival stream is deterministic given a seed: **Poisson arrivals**
modulated by a **time-of-day × day-of-week** intensity surface; an ESI acuity
mix; a complaint mix; a walk-in/ambulance split; and per-patient `WorkupNeeds`.
Every draw is keyed by content so the *same* realized week can be replayed
across arms (common random numbers).

### 6.2 One-week horizon and work-in-progress

The run covers `[0, 1 week]`. Queues, occupied bays, and in-progress patients
carry across day boundaries. At the week boundary, incomplete patients are
counted as **work-in-progress (WIP), not completions** — so a policy cannot look
good by hiding backlog. `completions_per_week` counts only patients who reached
a disposition-out event *within* the window. A tested invariant holds every run:
`arrivals == completions + wip_end_of_week` (no patient is created or lost).

### 6.3 Disruptions

Exogenous stressors are injected identically into both arms: ambulance surge,
staff absence, bay/zone closure (a closed node), and imaging outage. They exist
to stress-test the difference between BASELINE and OPTIMIZED under adversity, not
to add noise.

---

## 7. Operational levers — what gets optimized

The OPTIMIZED arm improves the levers a hospital can legally control. Each is a
distinct decision the decision layer makes and the physics layer executes.

- **Placement (bay/zone assignment).** Assign waiting patients to compatible
  free bays to minimize expected downstream travel — the walking implied by a
  patient's future provider/nurse visits and transports — weighted by acuity
  urgency, subject to zone/isolation/equipment compatibility and capacity.
- **Sequencing / triage prioritization.** Choose who is seen next as providers
  free up, using an acuity-weighted priority with **anti-starvation** so low
  acuity patients are eventually served (consistent with "never turn away").
- **Routing and dispatch.** Assign each task (provider visit, transport,
  cleanup) to the nearest qualified idle staff, solving a small assignment
  problem when several tasks and staff are free rather than acting greedily; a
  staff member's queued visits are sequenced as a short route over the graph.
- **Bed turnaround.** Prioritize cleaning of the dirty bays that unblock the most
  high-acuity demand soonest.
- **Discharge and documentation.** Prioritize discharges that free scarce bays,
  and schedule documentation into low-load windows so paperwork does not steal
  capacity during peaks.
- **Staff scheduling.** Choose staffing per role per shift-block to cover demand
  at minimum staff-hours. In v1 this is a **tunable scenario input** (you set
  staffing; the simulator measures); later it is solved from the demand forecast.

The BASELINE arm makes the same *kinds* of decisions with simple reactive rules
(first-available bay, nearest-idle dispatch, FIFO-within-acuity, no lookahead).
The two arms differ only in decision quality, never in the physics.

---

## 8. The event log — the single source of truth

Every fact the simulation produces is a typed event appended to an immutable,
byte-stable log (JSONL). The event vocabulary (discriminated on a `kind` tag)
covers arrivals, triage, bay request/assignment, movement (per edge traversal),
provider and nurse visits, test order/result, documentation, disposition,
discharge, bay cleaning, staff movement and idling, injected disruptions, and
(M3) vitals sampling, deterioration detection, and emergency escalation. Each is
wrapped with `occurred_at`, a monotonic `sequence`, and an optional causal link.

**All analysis reads this log and nothing else.** There is exactly one event
schema; the simulation writes it, and every downstream consumer (analysis,
console streaming, movement-trace export, forecasting features) reads it.

---

## 9. Analysis — the first-class output

From the event log, analysis produces the report and (M2) live indicators.

- **KPI fold** (`O(events)`): `completions_per_week`; door-to-triage and
  door-to-provider times (mean + percentiles, overall and **by acuity**);
  length-of-stay (exit − arrival, by acuity, right-censored WIP excluded);
  `staff_minutes_walked`; `bay_utilization` (warmup-windowed); `turnaround_time`;
  `boarding_time` for admits; and `wip_end_of_week`.
- **Wait decomposition** — split each patient's stay into stages (door→triage,
  triage→bay, bay→provider, workup wait, disposition→discharge) so it is visible
  *where the time actually goes*, including time lost to turnaround, discharge,
  and paperwork.
- **Bottleneck detection** — the resource/zone with the largest share of cycle
  time spent waiting in its queue is the binding constraint; also work
  concentration across staff (are two nurses doing everything?).
- **Staff-utilization decomposition** — every staff-second is classified into
  `{walk, direct_care, cleaning, documentation, idle}` and the fractions sum to
  one, revealing whether "optimized" converted walking into care.
- **Baseline-vs-optimized comparison** — N replications under common random
  numbers; per-KPI paired difference `baseline − optimized`; **percentile
  bootstrap** confidence intervals with Bonferroni correction across KPIs; a
  `significant` flag per contrast. This is the statistically honest "time saved /
  extra completions" headline.

---

## 10. Forecasting (M3) — predict, then optimize

The "predict" half of predict→optimize. Models train on simulation-generated
(and later real) data and feed the solver's inputs:

- **Arrival intensity** `λ(hour-of-day, day-of-week)` driving staffing and
  capacity planning, plus a short-horizon surge forecaster.
- **Per-(acuity, complaint) service-time and length-of-stay models** that replace
  static solver parameters with data-driven estimates.
- **Deterioration classifier** — a rolling synthetic-vitals window scored against
  a NEWS2-style early-warning threshold; crossing it raises an emergency and
  dispatches the nearest qualified staff via the routing oracle.

The retraining loop closes: the simulation emits labeled data, models are fit and
versioned, the solver consumes predictions as inputs, and the system re-runs to
measure whether predictions actually improved outcomes on held-out weeks.

---

## 11. Operator console (M2)

A browser application over an HTTP API lets a human watch and steer a live run:

- **FloorMap** — the ~100k-sq-ft ER rendered with zones, bays colored by status,
  patient chips colored by acuity, and staff dots animated along real graph edges
  from streamed positions.
- **Playback controls** — play/pause/step/speed/scrub, with sim-time decoupled
  from wall-clock.
- **KPI and bottleneck panels** — live indicators computed from the event log.
- **Override panel** — pick an entity, choose an action (reassign a patient to a
  bay, bump a priority, hold/close a bay, reroute staff, expedite a cleanup or
  discharge), and see it **accepted or rejected with reasons**. An operator
  override is just another decision through the same validated seam the solver
  uses — never silently repaired.
- **Compare view** — baseline-vs-optimized delta tiles with confidence intervals.

---

## 12. Capability roadmap (milestones)

- **M1 — Single ER floor, headless.** The full engine: domain model, solver
  levers, SimPy twin over a one-week horizon with CRN and disruptions, analysis,
  and a headless CLI. *Accepted when* a reference run produces `metrics.json` in
  which OPTIMIZED beats BASELINE on acuity-weighted time, staff-minutes walked,
  and weekly completions with significant confidence intervals, committed as a
  golden result.
- **M2 — Operator console.** The API and browser app: live run streaming,
  playback, validated overrides, and the compare view.
- **M3 — Vitals, forecasting, emergency response.** Synthetic vitals, the
  statistical and ML forecasts, the retraining loop, and emergency dispatch on
  deterioration.
- **M4 — Scale and cost.** Additional floor types as configurations,
  elevators/stairs and inter-floor transport and ED boarding, hospital-wide
  placement — then the deferred cost/money layer translating saved time and
  utilization into dollars.
