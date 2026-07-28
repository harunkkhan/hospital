"""The run resource, live KPIs, paired comparison, and the scenario store.

Thin wire projections over ``hospital.analysis``/``hospital.data`` output
(doc 07 §3.1/§3.5-§3.7). The API owns no KPI math and no scenario model:

* ``/metrics`` folds the session's ``EventLog`` up to the current ``sim_time``
  through the one ``analysis.fold.compute_kpis`` — a partial-week ``KpiVector``
  whose empty strata are NaN by the D8 convention (serialized as ``null``). The
  fold *window* ends at that same cut, never at the horizon (:func:`_live_window`),
  so a rate KPI is normalized by elapsed time rather than by intended time.
* ``/compare`` is a projection of ``analysis.compare.paired_bootstrap`` output,
  never a recompute; a live paired run is a single-seed point delta
  (``replications == 1``, CIs degenerate -> ``null``).
* ``/scenarios`` stores and lists ``hospital.data.scenario.Scenario`` — overrides
  are compiled by ``api.sliders`` (the console's named knobs plus literal dotted
  paths) and validated by ``data.scenario.apply_overlay``; a bad override is a
  data-layer validation error surfaced as 422, not an API-invented check.

CRN (doc 07 nuances 7.2): ``compare_to`` builds the shadow arm from the same
``(scenario, seed)`` — each ``RunSession`` seeds its own ``RandomStreams(seed)``
and regenerates the identical workload/layout, so the two arms face the same
realized week and the delta is policy signal, not weather.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import model_validator

from hospital.analysis import compute_kpis, paired_bootstrap
from hospital.api.overrides import PinRegistry
from hospital.api.sessions import RunSession, SessionRegistry, require_session
from hospital.api.sliders import compile_overrides
from hospital.core import (
    KPI_KEYS,
    Duration,
    EventLog,
    FloorLayout,
    FrozenModel,
    KpiVector,
    OperatingWeek,
    RunId,
    SimTime,
    hours,
)
from hospital.data.scenario import Scenario, load_scenario
from hospital.sim.policies.factory import Arm

if TYPE_CHECKING:
    from fastapi import FastAPI

router = APIRouter()

# Live paired compare is a single-seed point delta: CIs are degenerate at
# n_pairs == 1 (NaN -> null on the wire), so a small bootstrap suffices.
_LIVE_COMPARE_N_BOOT = 500


class ScenarioRef(FrozenModel):
    """A stored scenario, by id."""

    id: str


class ScenarioInline(FrozenModel):
    """A stored base scenario plus slider overrides.

    A key is either one of the console's named knobs (``api.sliders.SLIDER_KEYS``
    — e.g. ``"workload.arrival_rate_multiplier"``, which is *relative* and so
    could never be a leaf value) or a literal dotted path into the ``Scenario``
    document (e.g. ``"workload.base_rate_per_hour"``, whose value replaces the
    addressed leaf). Both compile through ``api.sliders.compile_overrides`` into
    an overlay ``data.scenario.apply_overlay`` validates.
    """

    base: str
    overrides: Mapping[str, float]


class RunRequest(FrozenModel):
    """``POST /runs`` body (doc 07 §3.1)."""

    scenario: ScenarioRef | ScenarioInline
    seed: int
    arm: Arm
    compare_to: Literal["baseline", "optimized"] | None = None
    start: Literal["paused", "playing"] = "paused"

    @model_validator(mode="after")
    def _shadow_is_the_other_arm(self) -> RunRequest:
        if self.compare_to is not None and self.compare_to == self.arm:
            raise ValueError("compare_to must name the OTHER arm (a paired contrast)")
        return self


class RunHandle(FrozenModel):
    """The run resource representation."""

    run: RunId
    arm: Arm
    seed: int
    horizon: SimTime
    state: Literal["created", "playing", "paused", "stepping", "finished"]
    sim_time: SimTime
    stream_url: str
    shadow: RunId | None = None


class KpiContrast(FrozenModel):
    """One key's baseline-vs-optimized contrast — a projection of
    ``analysis.compare.Contrast`` (``significant`` is mirrored, never re-derived).

    Every number here is nullable **on the wire** because every one of them can be
    NaN in the analysis output, and pydantic writes NaN as JSON ``null``: an empty
    stratum (no ESI-1 patient arrived yet) has no mean to contrast, and a live
    single-seed pair has degenerate CIs at ``n_pairs == 1``. Typing them as plain
    numbers made the schema — and the generated TypeScript — promise a figure that
    is routinely absent, which is how a console ends up rendering a confidence
    bound it was never given.
    """

    key: str
    baseline: float | None
    optimized: float | None
    delta: float | None
    ci_lo: float | None
    ci_hi: float | None
    significant: bool


class CompareResponse(FrozenModel):
    baseline_run: RunId
    optimized_run: RunId
    replications: int
    contrasts: tuple[KpiContrast, ...]


class ScenarioSummary(FrozenModel):
    id: str
    name: str
    horizon: SimTime
    note: str


class ScenarioCreated(FrozenModel):
    id: str


class ScenarioStore:
    """Lifespan-scoped scenario storage: YAML-seeded plus operator-derived.

    Validation and (de)serialization are ``hospital.data.scenario``'s — the
    store never defines a scenario model.
    """

    def __init__(self) -> None:
        self._scenarios: dict[str, Scenario] = {}
        self._notes: dict[str, str] = {}
        self._counter = itertools.count(1)

    @classmethod
    def from_dir(cls, directory: Path) -> ScenarioStore:
        store = cls()
        if directory.is_dir():
            for path in sorted(directory.glob("*.yaml")):
                store.register(path.stem, load_scenario(path), note=f"loaded from {path.name}")
        return store

    def register(self, scenario_id: str, scenario: Scenario, *, note: str) -> None:
        self._scenarios[scenario_id] = scenario
        self._notes[scenario_id] = note

    def get(self, scenario_id: str) -> Scenario | None:
        return self._scenarios.get(scenario_id)

    def summaries(self) -> tuple[ScenarioSummary, ...]:
        return tuple(
            ScenarioSummary(
                id=scenario_id,
                name=scenario.name,
                horizon=scenario.workload.horizon.end,
                note=self._notes[scenario_id],
            )
            for scenario_id, scenario in self._scenarios.items()
        )

    def derive(self, base_id: str, overrides: Mapping[str, float]) -> str:
        """Apply console sliders / dotted-path overrides to a base and store the result.

        Raises ``KeyError`` for an unknown base and ``ValueError`` (or a
        pydantic ``ValidationError``) when the data layer rejects the overlay.
        """
        base = self.get(base_id)
        if base is None:
            raise KeyError(base_id)
        derived = compile_overrides(base, overrides)
        scenario_id = f"{base_id}-v{next(self._counter)}"
        while scenario_id in self._scenarios:
            scenario_id = f"{base_id}-v{next(self._counter)}"
        self.register(
            scenario_id, derived, note=f"derived from {base_id} ({len(overrides)} override(s))"
        )
        return scenario_id


def _registry(app: FastAPI) -> SessionRegistry:
    return cast("SessionRegistry", app.state.registry)


def _scenarios(app: FastAPI) -> ScenarioStore:
    return cast("ScenarioStore", app.state.scenarios)


def _resolve_scenario(store: ScenarioStore, ref: ScenarioRef | ScenarioInline) -> Scenario:
    if isinstance(ref, ScenarioRef):
        scenario = store.get(ref.id)
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"unknown scenario: {ref.id}")
        return scenario
    base = store.get(ref.base)
    if base is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {ref.base}")
    try:
        return compile_overrides(base, ref.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid scenario overrides: {exc}") from exc


def _handle(session: RunSession) -> RunHandle:
    return RunHandle(
        run=session.run_id,
        arm=session.arm,
        seed=session.seed,
        horizon=session.horizon.end,
        state=session.state,
        sim_time=session.sim_time,
        stream_url=f"/runs/{session.run_id.root}/stream",
        shadow=session.shadow,
    )


def _live_window(session: RunSession, cut: SimTime) -> tuple[OperatingWeek, Duration]:
    """The fold window for a run observed only up to ``cut``.

    The window must END at ``cut``, because that is where the log ends. Folding
    the FULL horizon over a partial log is what makes a live reading wrong: every
    rate- and duration-normalized KPI divides by the measurement window's length
    (``bay_utilization``, ``provider_util``, ``nurse_util``, the ``staff_frac_*``
    split, ``completions_per_week``), so a tenth of the way into a run they read
    about a tenth of their true value — a console reporting 4% provider
    utilization on a floor that is in fact saturated. ``wip_end_of_week`` reads
    against the wrong instant for the same reason.

    Warmup scales with the *observed* span under the same rule the headless
    scorecard applies to a full horizon (24h, or a quarter of the span for a short
    scenario). That is what makes the live series usable from early on and still
    exact at the end: at ``cut == horizon.end`` this returns precisely the
    headless window and warmup, so the final live vector equals the one the
    goldens pin for the same ``(scenario, seed, arm)``.
    """
    span = cut.root - session.horizon.start.root
    window = OperatingWeek(start=session.horizon.start, end=cut)
    return window, Duration(min(hours(24).root, span // 4))


def _observed_cut(session: RunSession) -> SimTime:
    """How far this session's log is complete — and 409 if that is nowhere yet.

    A run parked at ``t=0`` has a zero-length window, which is not a partial
    measurement but the absence of one (``compute_kpis`` rejects it outright).
    Saying so is honest; the alternative — quietly widening the window to the
    horizon — is the very thing this cut exists to prevent.
    """
    cut = session.sim_time
    if cut.root <= session.horizon.start.root:
        raise HTTPException(
            status_code=409,
            detail="no elapsed sim time to fold yet — advance the run first",
        )
    return cut


def _fold_session(session: RunSession, log: EventLog, cut: SimTime) -> KpiVector:
    window, warmup = _live_window(session, cut)
    return compute_kpis(log, session.layout, session.roster, window=window, warmup=warmup)


def _log_prefix(log: EventLog, cut: SimTime) -> EventLog:
    """The envelopes strictly before ``cut`` — the paired-fold cut for /compare.

    Strict, matching the half-open ``[start, cut)`` window the prefix is folded
    through, so the index and the window agree on the boundary instant.
    """
    out = EventLog()
    for envelope in log:
        if envelope.event.occurred_at.root < cut.root:
            out.append(envelope.event, caused_by=envelope.caused_by)
    return out


def _pydantic_json(model: FrozenModel) -> Response:
    """Serialize through pydantic (NaN -> null) — starlette's json.dumps rejects NaN."""
    return Response(content=model.model_dump_json(), media_type="application/json")


# --------------------------------------------------------------------- /runs
@router.post("/runs", status_code=201, response_model=RunHandle)
async def create_run(body: RunRequest, request: Request) -> RunHandle:
    """Create a run session at ``t=0`` (paused unless ``start="playing"``);
    ``compare_to`` spins a same-seed shadow arm so CRN holds."""
    app = cast("FastAPI", request.app)
    registry = _registry(app)
    scenario = _resolve_scenario(_scenarios(app), body.scenario)
    run_id = registry.mint_run_id(scenario.name, body.arm, body.seed)
    session = RunSession(run_id, scenario, body.arm, body.seed, pins=PinRegistry())
    registry.add(session)
    shadow: RunSession | None = None
    if body.compare_to is not None:
        shadow_id = registry.mint_run_id(scenario.name, body.compare_to, body.seed)
        shadow = RunSession(shadow_id, scenario, body.compare_to, body.seed, pins=PinRegistry())
        shadow.shadow_of = run_id
        session.shadow = shadow_id
        registry.add(shadow)
    if body.start == "playing":
        async with session.lock:
            session.play()
        if shadow is not None:
            async with shadow.lock:
                shadow.play()
    return _handle(session)


@router.get("/runs/{run_id}", response_model=RunHandle)
async def get_run(run_id: str, request: Request) -> RunHandle:
    return _handle(require_session(cast("FastAPI", request.app), run_id))


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(run_id: str, request: Request) -> Response:
    app = cast("FastAPI", request.app)
    removed = await _registry(app).teardown(run_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return Response(status_code=204)


@router.get("/runs/{run_id}/layout", response_model=FloorLayout)
async def get_layout(run_id: str, request: Request) -> Response:
    """The run's static floor geometry (doc 07 §3.8): route graph, zones, bays,
    stations, entrances — the ``core.entities.FloorLayout`` verbatim. Fetched once
    on connect; the per-tick stream frame carries only the mutable projection."""
    session = require_session(cast("FastAPI", request.app), run_id)
    return _pydantic_json(session.layout)


@router.get("/runs/{run_id}/metrics", response_model=KpiVector)
async def get_metrics(run_id: str, request: Request) -> Response:
    """The live ``core.kpi.KpiVector``: the one fold over the log so far.

    Folded through the observed cut (:func:`_live_window`), so the vector is a
    reading of the sim time that has actually elapsed rather than of the horizon
    the run is heading for.
    """
    session = require_session(cast("FastAPI", request.app), run_id)
    async with session.lock:
        cut = _observed_cut(session)
        kpis = _fold_session(session, session.log, cut)
    return _pydantic_json(kpis)


@router.get("/runs/{run_id}/compare", response_model=CompareResponse)
async def get_compare(run_id: str, request: Request) -> Response:
    """Baseline-vs-optimized deltas via the one ``analysis`` bootstrap.

    Both paired logs are folded at the same cut (the lagging arm's ``sim_time``)
    so the contrast compares the same realized window under CRN.
    """
    app = cast("FastAPI", request.app)
    session = require_session(app, run_id)
    if session.shadow is None:
        raise HTTPException(
            status_code=409,
            detail="run has no paired shadow arm (create it with compare_to)",
        )
    shadow = _registry(app).get(session.shadow.root)
    if shadow is None:
        raise HTTPException(status_code=409, detail="paired shadow session no longer exists")
    baseline, optimized = (session, shadow) if session.arm == "baseline" else (shadow, session)
    async with session.lock, shadow.lock:
        # The lagging arm's cut, for both: the arms are driven independently, so
        # only their common prefix is a like-for-like window under CRN.
        cut = SimTime(min(_observed_cut(session).root, _observed_cut(shadow).root))
        baseline_kpis = _fold_session(baseline, _log_prefix(baseline.log, cut), cut)
        optimized_kpis = _fold_session(optimized, _log_prefix(optimized.log, cut), cut)
    result = paired_bootstrap([baseline_kpis], [optimized_kpis], n_boot=_LIVE_COMPARE_N_BOOT)
    contrasts = tuple(
        KpiContrast(
            key=key,
            baseline=contrast.baseline_mean,
            optimized=contrast.optimized_mean,
            delta=contrast.diff_mean,
            ci_lo=contrast.ci_lo,
            ci_hi=contrast.ci_hi,
            significant=contrast.significant,
        )
        for key in KPI_KEYS
        for contrast in (result.contrasts[key],)
    )
    return _pydantic_json(
        CompareResponse(
            baseline_run=baseline.run_id,
            optimized_run=optimized.run_id,
            replications=result.n_reps,
            contrasts=contrasts,
        )
    )


# ---------------------------------------------------------------- /scenarios
@router.get("/scenarios", response_model=tuple[ScenarioSummary, ...])
async def list_scenarios(request: Request) -> tuple[ScenarioSummary, ...]:
    return _scenarios(cast("FastAPI", request.app)).summaries()


@router.get("/scenarios/{scenario_id}", response_model=Scenario)
async def get_scenario(scenario_id: str, request: Request) -> Response:
    """The stored ``hospital.data.scenario.Scenario``, verbatim."""
    scenario = _scenarios(cast("FastAPI", request.app)).get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {scenario_id}")
    return _pydantic_json(scenario)


@router.post("/scenarios", status_code=201, response_model=ScenarioCreated)
async def create_scenario(body: ScenarioInline, request: Request) -> ScenarioCreated:
    """Validate slider overrides through ``data.scenario`` and store the result."""
    store = _scenarios(cast("FastAPI", request.app))
    try:
        scenario_id = store.derive(body.base, body.overrides)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {body.base}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid scenario overrides: {exc}") from exc
    return ScenarioCreated(id=scenario_id)


__all__ = [
    "CompareResponse",
    "KpiContrast",
    "RunHandle",
    "RunRequest",
    "ScenarioCreated",
    "ScenarioInline",
    "ScenarioRef",
    "ScenarioStore",
    "ScenarioSummary",
    "router",
]
