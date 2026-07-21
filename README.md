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

Under active construction. See the milestone plan in the build sequence
(`M1` headless engine → `M2` operator console → `M3` forecasting/ML → `M4` scale + cost).

## Development

```sh
uv sync            # provision Python 3.13 + all workspace packages
uv run ruff check .
uv run pyright
uv run pytest -q
```
