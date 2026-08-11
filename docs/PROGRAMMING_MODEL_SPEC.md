# Programming Model Spec — How the Simulator Is Built

This is the **engine / contract specification** for the Hospital
Operations-Research Simulator. It describes *how* the system is programmed: the
hard decision↔physics seam, the frozen typed values that cross it, the
common-random-number determinism model, the validate-never-repair guarantee, the
solver `Protocol` + registry, the single objective definition, and the package
architecture that keeps every concept in exactly one place. The domain it models
— the ER, its entities, and what gets optimized — lives in the companion
[`SIMULATION_SPEC.md`](./SIMULATION_SPEC.md).

The engineering discipline here is deliberate: it is what makes runs
deterministic, parallelizable, comparable, and safe to hand to a human operator.

---

## 1. The decision↔physics seam

The single most important architectural rule.

- The **decision layer** — a solver *or* a human operator — proposes a `Plan`.
- The **physics layer** — the SimPy digital twin — executes mechanical truth and
  advances time.

The seam between them is hard and one-directional:

- The decision layer **can never mutate simulator state** or read hidden facts.
  It sees only an immutable `DecisionInput` projection and returns a `Plan`.
- The engine **never silently repairs** a bad plan. It validates the plan and, if
  it violates any rule, **rejects** it and reports the violations. No partial
  application, no "fixing up."

This one rule is what makes the whole system tractable: runs become
deterministic and parallelizable (the physics is a pure function of the seam
traffic and the seed), and the **operator override becomes a first-class, safe
input** — a human is just another policy speaking through the same seam.

### Seam value types

- **`DecisionInput`** — an immutable projection of current floor state (patients
  waiting, bays, staff positions, pending tasks) plus the batch of events since
  the last decision point. No hidden fields.
- **`Plan`** — the complete set of currently-revisable decisions: bay
  assignments, the sequencing order, dispatch/route assignments,
  turnaround/discharge priorities, and (at shift boundaries) staffing. Each
  `PlanItem` carries a `stable_id` so plans diff cleanly across re-solves.
- **`DecisionResponse`** — `keep` or `replace(Plan)`, plus an optional next-wake
  directive.

The same `Plan`/`Violation` types are produced by the solver *and* by operator
overrides — there is no separate "override" path with different rules.

---

## 2. Everything at the boundary is a frozen, typed value

No loose dicts ever cross the seam. The boundary vocabulary is built from a small
set of primitives:

- **`FrozenModel`** — a pydantic v2 base with `frozen=True`, `extra="forbid"`,
  and validated defaults. Every boundary value derives from it. Immutability is
  what lets values be shared across processes and hashed for provenance.
- **Typed IDs** — `PatientId`, `BayId`, `ZoneId`, `StaffId`, `NodeId`, `TaskId`,
  `RunId` are each a distinct `RootModel[str]`, so cross-type misuse (`BayId`
  where a `PatientId` is expected) fails typecheck, not at runtime.
- **Integer-scaled units** — time in microseconds, distance in centimetres. Float
  timestamps are never stored or compared, which is a precondition for
  byte-stable logs.
- **Discriminated-union events** — every event is tagged with a
  `kind: Literal[...]`, so exhaustiveness is checkable and adding a new event
  kind forces every consumer to handle it.

`pyright` in **strict** mode enforces this across package boundaries — which only
works because every distribution ships a `py.typed` marker (otherwise downstream
packages would see `Any` and the typed-ID safety net would silently evaporate).

---

## 3. Determinism and common random numbers (CRN)

Determinism is a **tested property, not a hope** — it is the entire point of the
design and it is falsifiable: identical inputs must produce a byte-identical
event log.

- **All randomness comes from content-addressable substreams.** A single seed
  seeds a `RandomStreams` object; every draw asks for a substream keyed by
  content — e.g. `substream("service_time", patient_id, "provider_visit", n)` —
  via a blake2b-keyed `SeedSequence`. Draws are therefore reproducible and
  *independent* of the order in which code requests them.
- **Adding a draw never perturbs existing draws.** Because substreams are keyed
  by content rather than allocated sequentially, introducing a new sampled
  quantity leaves every other realized value untouched.
- **World randomness is isolated from policy/operator randomness.** The realized
  week (arrivals, service times, workups) is drawn from world substreams; any
  randomness inside a policy uses separate keys. This is what makes **common
  random numbers** work: BASELINE and OPTIMIZED see the *identical* realized
  week, so their measured difference is pure signal, not sampling noise.

**Tested invariants:** two runs of the same `(scenario, arm, seed)` produce
identical `EventLog.to_jsonl()`; serial and process-pool execution are
byte-identical; perturbing one substream key leaves every other draw unchanged.

---

## 4. Validate-never-repair — the feasibility gate

There is exactly **one** place a plan is judged, and it never mutates anything.

- **Declarative constraint rules.** Compatibility, admission, isolation,
  capacity, skill, and precedence constraints are a frozen, discriminated **rule
  vocabulary** compiled into a validator kernel — not scattered `if` statements.
  Compatibility and admission are separate kinds on purpose: the first says where
  a patient may be *worked up*, the second where they may be *admitted*, and the
  patient's care phase selects which applies. Folding wards into the first would
  make an ESI-2 eligible for a resus bay thereby eligible for an ICU bed the
  moment they finish triage — a placement no rule could then refuse, because
  nothing in the key distinguishes the two moments.
- **`validate(plan, context) -> tuple[Violation, ...]`.** Returns the (possibly
  empty) list of violations. `Violation` is a union — `UnknownEntity`,
  `BayIncompatible`, `CapacityExceeded`, `IsolationViolated`, `StaffLacksSkill`,
  `DoubleBooked`, and so on.
- **Reject, never fix.** A plan with any violation — whether from the solver or
  from an operator override — is rejected. The engine raises with the violations
  and **no state mutates**. Nothing is silently repaired.

This is simultaneously a correctness guarantee (the physics only ever executes
feasible plans) and the operator-safety mechanism (a bad human override is
refused with a clear reason, exactly like a bad solver plan).

**Defense in depth, not redundancy.** The solver may *also* check constraints
while searching, and the engine checks again on plan acceptance, and the API
checks an override — three enforcement points, **one rule source**. This apparent
redundancy is intentional and must not be "simplified" away.

---

## 5. Solvers behind a Protocol + registry

Optimization backends are interchangeable behind a small contract.

```python
class Solver(Protocol):
    def solve(self, decision_input: DecisionInput, oracle: RoutingOracle) -> SolveResult: ...
```

- **Registry.** `get_backend(name)` lazily imports and constructs a backend, so a
  fast heuristic and an OR-Tools CP-SAT backend swap interchangeably without the
  caller importing either eagerly.
- **Status on every result.** A `SolveResult` carries
  `status ∈ {OPTIMAL, FEASIBLE, HEURISTIC}`, so a caller always knows the quality
  of what it got.
- **The routing oracle is shared, not reimplemented.** `RoutingOracle` exposes
  `distance(a, b)` / `path(a, b)`; the concrete `GraphRoutingOracle` wraps the
  *one* canonical shortest-path routine plus a memo. Solver backends and sim
  policies both consume it; no backend carries its own pathfinding.
- **Provenance stamping is a single choke point.** `stamp(SolveResult, config)`
  turns a pure solve result into a wire artifact tagged with backend version and
  a canonical config hash. Every plan that reaches the physics passes through
  this one point, so every result is traceable to the exact solver and weights
  that produced it.

The abstraction earns its keep only where there are ≥2 real implementations:
placement gets a `Protocol` + registry (CP-SAT and heuristic backends);
single-implementation levers stay plain functions until a second backend
actually appears.

---

## 6. One objective, full KPI vector

- **Exactly one scalar objective.** `weighted_total(...)` is the sole scalar cost
  (acuity-weighted patient time + staff travel + penalties, integer-scaled), with
  weights living in config under a canonical hash. Scorecards and candidate
  ranking fold *through* this one function; costs are never re-added inline
  anywhere else. A tested invariant checks that every scorecard total equals
  `weighted_total(...)` recomputed from its parts.
- **The full KPI vector is always retained.** A closed, versioned `KPI_KEYS`
  contract defines every metric analysis emits. A blended objective score never
  *hides* an unsafe or infeasible outcome — the complete vector is always
  available, and a missing or extra KPI key is a test failure.
- **Ranking is explicit.** When comparing candidate optimized plans, they are
  ranked by explicit lexicographic keys (e.g. fewest uncompleted, then lowest
  acuity-weighted time), never by a hidden scalar.

---

## 7. Package architecture and dependency direction

The repository is a **`uv` workspace** (Python 3.13) plus a **`bun` workspace**
(TypeScript) in one repo. All Python code lives under a single namespace,
`hospital.*`, spread across distributions.

### 7.1 The implicit namespace package (the #1 gotcha)

Several distributions each ship `src/hospital/<subpkg>/`. For the `hospital`
namespace to span them, it **must be an implicit namespace package** (PEP 420):
there is **no `src/hospital/__init__.py` anywhere**. A stray top-level
`__init__.py` in any one distribution turns `hospital` into a *regular* package
and the other distributions' subpackages silently stop importing. Only the leaf
subpackages (`hospital/core/__init__.py`, etc.) get an `__init__.py`. Each
distribution's build declares `packages = ["src/hospital"]`, which wires this.
The structural smoke test asserts this invariant directly.

### 7.2 The distributions

| Distribution        | Namespace           | Responsibility                                               |
| ------------------- | ------------------- | ------------------------------------------------------------ |
| `hospital-core`     | `hospital.core`     | Domain model, contracts, graph, RNG/CRN, events, seam, validation, KPIs |
| `hospital-data`     | `hospital.data`     | Deterministic floor-layout and workload generators           |
| `hospital-solver`   | `hospital.solver`   | Pure optimization: `Solver` Protocol + registry, oracle, levers |
| `hospital-analysis` | `hospital.analysis` | KPI fold, decomposition, bottleneck detection, comparison stats |
| `hospital-sim`      | `hospital.sim`      | The SimPy twin: physics + policies + experiment              |
| `hospital-forecast` | `hospital.forecast` | Statistical + ML forecasting (M3)                            |
| `apps/sim-runner`   | `hospital.sim_runner` | Headless CLI composing the M1 stack                        |

### 7.3 Strict downward dependency direction

Enforced mechanically by `import-linter` (contracts in `importlinter.ini`):

```
core     -> (nothing)
data     -> core
solver   -> core
analysis -> core
sim      -> core, solver, data, analysis
forecast -> core, data
apps/*   -> the packages they compose
```

Forbidden: any leaf→leaf import not listed above, and any import of `sim` or an
app from `core`/`solver`/`analysis`. Two consequences worth stating explicitly:

- **`analysis` is mid-tier, not a leaf.** `sim.experiment` *reuses*
  `analysis.fold` and `analysis.compare` rather than duplicating the KPI fold and
  the bootstrap; the graph stays acyclic because `analysis -> core` only.
- **`forecast` literally cannot import the solver.** Because `forecast -> core,
  data` only, predictions are handed to the solver as a `core`-typed bundle
  threaded by a composition root — an architectural fact forced by the contract,
  not a stylistic choice. The deterioration monitor is likewise a `core`-owned
  `Protocol` that `sim` calls, not a `forecast` import.

Without the enforced graph, the first "just import it here" shortcut starts a
cycle, and cycles are how the one-canonical-implementation guarantee dies.

### 7.4 Two worlds, one seam

The Python `api` and the `bun` `web` app share **nothing** at build time — only
HTTP plus code-generated types. The frontend never imports Python and vice
versa; the contract is the generated types, not a shared module.

---

## 8. Reuse discipline — one canonical implementation per concept

The organizing constraint of the whole codebase: **every concept has exactly one
home, owned by the lowest package that needs it, and is imported everywhere else
— never reimplemented.** There is exactly one shortest-path routine, one RNG
substream factory, one event schema, one `validate()`, one KPI contract, one
objective definition, one workload generator, and one bootstrap routine.

Practical rules:

- **Check before writing.** If a concept already has a home, import it; if a new
  helper is needed by ≥2 packages, it belongs in `core` (or `data` for
  generation), not duplicated.
- **Backends and policies are thin.** A solver backend or a sim policy *composes*
  core/solver primitives; it must not carry its own pathfinding, RNG, event
  formatting, or validation.
- **One reader/writer for the event schema.** `sim` writes the log; `analysis`,
  the API stream, movement-trace export, and forecast features all read it
  through `core.events`. No package invents a parallel "just my events" record.
- **Public surface is the contract.** Each `__init__.py` re-exports only the
  intended public API; everything else is internal (imported by module path).
  Keeping the public surface small is what keeps the reuse contract legible and
  the import-linter contracts meaningful.

The half of this a machine can check (structural dependency direction) is
enforced by import-linter; the rest is enforced at review against the reuse
registry.

---

## 9. Testing philosophy

- **Tests are maintained code, not a default deliverable.** Test the contracts
  and the hard logic — graph, RNG/CRN, the validator, solver feasibility, the KPI
  fold, determinism — not trivial getters.
- **Reuse test builders.** Each package ships importable fixture helpers
  (`tests/_fixtures.py`); there is no `conftest.py` and no copy-pasted scenario
  construction.
- **The invariant suite is load-bearing.** Feasibility is total (the engine never
  executes a plan the validator would reject); the engine never repairs;
  work-in-progress is conserved over the week; CRN substreams are isolated; graph
  paths are consistent; runs are deterministic; every scorecard total equals the
  single objective recomputed; the KPI contract is closed; staff-second fractions
  sum to one; utilizations never exceed one and durations are never negative.
- **Golden traces and metrics.** A fixed-seed event-log slice and a committed
  `metrics.json` are checked for byte/semantic stability; a diff fails until it is
  regenerated and reviewed as a deliberate semantic change, never accepted as
  silent churn.

### CI gates

On every pull request and push to `main`, all of the following must pass:
`uv sync --locked`; `ruff check .` and `ruff format --check .`; `pyright`
(strict) over `src` and `tests`; `pytest -q` (the full suite, goldens and
properties); and `lint-imports` (the dependency-direction contracts). The
frontend gates (`bun install --frozen-lockfile`, `tsc --noEmit`, `bun test`, and
the TypeScript↔pydantic contract-drift check) run once the web app exists.

---

## 10. Build sequence

Each step ends green before the next begins:

0. **Scaffold** — the workspaces, per-package builds, tooling config, empty
   namespace trees, and these two specs. *Done when* `uv sync` and `pyright` are
   clean on the empty packages and the structural smoke test asserts the layout.
1. **`hospital-core`** — the frozen contract everything imports.
2. **`hospital-data`** — deterministic floor + workload generation.
3. **`hospital-sim` physics** — world/executor/service-times/resources with event
   emission; patients move, no decisions yet.
4. **Baseline policies + `run_replication`** — the first end-to-end one-week run.
5. **`hospital-analysis`** — the KPI fold, decomposition, and comparison.
6. **`hospital-solver`** — oracle → objective → placement → sequencing → dispatch
   → turnaround/discharge, with the validator self-check and stamping.
7. **Optimized policies + comparison** — both arms under CRN; committed golden
   `metrics.json`. **← M1.**
8. **`apps/api` + `apps/web`** — streaming, playback, overrides, compare view.
   **← M2.**
9. **`hospital-forecast`** — features, models, retraining loop, emergency
   dispatch. **← M3.**
10. **Scale to more floors**, then the **cost/money layer**. **← M4. ✅**
