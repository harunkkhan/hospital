# Hospital Operations-Research Simulator

A digital-twin **operations-research simulator** of a large (~100,000 sq ft) hospital
**Emergency Room**, run over a one-week horizon. Patients are never turned away; the
system optimizes the **operational** levers a hospital can legally control — patient
placement, movement/routing, bed turnaround, discharge, paperwork, and staff
scheduling — to minimize acuity-weighted patient time and maximize completed
treatments per week.

It runs an **unoptimized baseline** against an **optimized** arm over identical
randomness (common random numbers) and quantifies the difference, then exposes an
**operator's console** in the browser where a solver auto-pilots and a human can
intervene in real time through the same validated decision seam.

## Architecture

A `uv` (Python 3.13) + `bun` (TypeScript) monorepo under the single `hospital.*`
namespace, with a strict downward dependency direction enforced by `import-linter`:

```
core      pure domain model, contracts, graph, RNG/CRN, events, seam, validation, KPIs
data      deterministic floor-layout + workload generators
solver    pure optimization (OR-Tools CP-SAT placement, routing oracle, dispatch, …)
analysis  KPI fold, wait decomposition, bottleneck detection, baseline-vs-optimized stats
sim       SimPy digital twin (physics + policies + experiment)  →  depends on core/solver/data/analysis
forecast  statistical + ML forecasting that feeds the solver     →  depends on core/data
apps/     sim-runner CLI, FastAPI operator API, React console
```

## Status

All four milestones are built: **M1** headless engine, **M2** operator console, **M3**
forecasting/ML with emergency dispatch, **M4** multi-floor scale and the cost layer. Staff
scheduling — the last lever that was an input — is now solved from the arrival forecast.

The project is **not finished**, and the gaps are specific rather than a general disclaimer:

- **M1 meets two of its three acceptance criteria.** The committed golden shows OPTIMIZED
  beating BASELINE on acuity-weighted time and staff-minutes walked; weekly completions is
  flat and non-significant, because the reference floor is demand-limited and no decision can
  add a completion there. Deliberately not re-sited to a congested floor to make the test
  pass — see `tests/goldens/test_golden_metrics.py` for the measurement and the reasoning.
- **The optimized arm trades deadline compliance for its objective win.** It breaches acuity
  care deadlines by ~38 more patient-hours a week than the naive baseline, significantly, and
  that contrast is pinned in the same golden. The spec's remedy for such a trade is that it
  stays visible, not that the objective be re-weighted to erase it.
- **One half of a §7 lever is inert.** "Schedule documentation into low-load windows so
  paperwork does not steal capacity during peaks" has its mechanism — `FloorLoad.is_peak` at a
  0.8 threshold — but every production caller passes the neutral `FloorLoad()` default, whose
  0.0 utilization can never cross it. Documentation is therefore always in the promoted band,
  for discharge and for dispatch's priority-augmented urgency alike. It is closable within the
  seam's rules: `StaffState` already carries `busy_until`/`current_task`, so instantaneous
  utilization is derivable from the projection rather than being a hidden field.
- **Two roles/inputs the model carries but never exercises.** No flow creates a task requiring
  a `TECH`, so the six that `er_floor.yaml` staffs are paid to be idle (measured at 0.00
  role-minutes per patient); and `WorkupNeeds.procedures` is sampled per patient and never
  simulated, which is consistent with §5's workup loop but leaves the field dangling.

## Development

```sh
uv sync            # provision Python 3.13 + all workspace packages
uv run ruff check . && uv run ruff format --check .
uv run pyright                       # strict, over src and tests
uv run pytest -q
uv run lint-imports                  # the 8 dependency-direction contracts
```

The browser console (`apps/web`) is a `bun` workspace with its own gates:

```sh
bun install --frozen-lockfile
bun run --cwd apps/web tsc --noEmit
bun test                             # includes the TypeScript<->pydantic drift check
```

A pydantic contract change must be followed by
`uv run python -m hospital.api.codegen dump-schema --out apps/api/schema/openapi.json`,
or the drift gates fail — which is them working.
