"""``/metrics`` is the one fold; ``/compare`` is the one bootstrap, projected.

No KPI math in the API (doc 07 §3.5-§3.6): the live vector must equal
``analysis.fold.compute_kpis`` over the session log, and every ``KpiContrast``
must mirror ``analysis.compare.paired_bootstrap`` output.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from _api_fixtures import (
    create_run,
    make_app,
    run_to_finish,
    session_of,
    step,
    wait_until_finished,
)
from fastapi.testclient import TestClient

from hospital.analysis import compute_kpis
from hospital.core import KPI_KEYS, Duration, OperatingWeek, SimTime, hours

if TYPE_CHECKING:
    from pathlib import Path

    from hospital.api.sessions import RunSession


def _wire_value_equals(wire: float | None, folded: float) -> bool:
    """NaN crosses the wire as null (the D8 empty-stratum convention)."""
    if wire is None:
        return math.isnan(folded)
    return wire == folded


def _expected_window(session: RunSession, cut: SimTime) -> tuple[OperatingWeek, Duration]:
    """The window a live fold must use: ends at the observed cut, warmup scaled to it."""
    span = cut.root - session.horizon.start.root
    return (
        OperatingWeek(start=session.horizon.start, end=cut),
        Duration(min(hours(24).root, span // 4)),
    )


def test_metrics_returns_the_closed_kpi_contract(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        step(client, handle["run"], granularity="decision", count=10)

        response = client.get(f"/runs/{handle['run']}/metrics")
        assert response.status_code == 200
        values: dict[str, Any] = response.json()["values"]
        assert set(values) == set(KPI_KEYS), "keys must be exactly KPI_KEYS"

        session = session_of(app, handle["run"])
        window, warmup = _expected_window(session, session.sim_time)
        folded = compute_kpis(
            session.log,
            session.layout,
            session.roster,
            window=window,
            warmup=warmup,
        )
        for key in KPI_KEYS:
            assert _wire_value_equals(values[key], folded.values[key]), key


def test_live_metrics_fold_through_the_observed_cut_not_the_horizon(tmp_path: Path) -> None:
    """A live vector must measure the sim time that has elapsed, not the intended horizon.

    Folding a partial log against the FULL horizon is wrong in **both** directions,
    which is why the fix is the window and not a scale factor:

    * *censoring* — warmup is measured from the window start, so a horizon-width
      window on a young run puts the whole observed prefix before the measurement
      window and every wait/LOS KPI comes back NaN. The console renders those as
      em dashes: waits it has actually observed, reported as unmeasured.
    * *extrapolation* — ``fold`` closes a still-open bay occupancy at
      ``window.end`` (``fold.py``: ``end = cyc.clean_start ... else window.end``),
      so a horizon-width window bills occupancy for sim time that has not been
      simulated, and ``bay_utilization`` reads HIGH.

    Both are asserted below as properties of this fixture rather than pinned
    numbers, and the API's own vector is required to equal the cut fold key for key
    — which is the regression itself: before the fix it equalled the horizon fold.
    """
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        step(client, handle["run"], granularity="decision", count=5)
        session = session_of(app, handle["run"])
        cut = session.sim_time
        assert 0 < cut.root < session.horizon.end.root, "need a genuinely partial observation"

        live: dict[str, Any] = client.get(f"/runs/{handle['run']}/metrics").json()["values"]

        window, warmup = _expected_window(session, cut)
        cut_fold = compute_kpis(
            session.log, session.layout, session.roster, window=window, warmup=warmup
        )
        for key in KPI_KEYS:
            assert _wire_value_equals(live[key], cut_fold.values[key]), key

        horizon_warmup = Duration(min(hours(24).root, session.horizon.end.root // 4))
        horizon_fold = compute_kpis(
            session.log,
            session.layout,
            session.roster,
            window=session.horizon,
            warmup=horizon_warmup,
        )

        # Censoring: the horizon window's warmup swallows the entire observed run,
        # so a wait the cut fold measures is NaN there.
        assert cut.root < horizon_warmup.root, "fixture must stop inside the horizon warmup"
        assert math.isnan(horizon_fold.values["door_to_triage_s_mean"])
        assert not math.isnan(cut_fold.values["door_to_triage_s_mean"])

        # Extrapolation: the horizon window bills more occupied bay-seconds than the
        # run has elapsed bay-seconds in total -- occupancy that was never simulated.
        n_bays = len(session.layout.bays)
        horizon_measured_s = (session.horizon.end.root - horizon_warmup.root) / 1e6
        billed_bay_s = horizon_fold.values["bay_utilization"] * n_bays * horizon_measured_s
        elapsed_bay_s = n_bays * (cut.root - session.horizon.start.root) / 1e6
        assert billed_bay_s > elapsed_bay_s, "expected the horizon window to extrapolate"
        # The cut fold cannot: its numerator closes at the cut.
        cut_measured_s = (cut.root - window.start.root - warmup.root) / 1e6
        assert cut_fold.values["bay_utilization"] * n_bays * cut_measured_s <= elapsed_bay_s


def test_metrics_before_any_elapsed_sim_time_is_a_409(tmp_path: Path) -> None:
    """A run parked at t=0 has no measurable window — and says so.

    Widening the window to the horizon to manufacture a 200 here is exactly the
    bug: it would report a full week's worth of denominators over an empty log.
    """
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        assert client.get(f"/runs/{handle['run']}/metrics").status_code == 409

        # Work can happen AT t=0 (arrivals, the first decision), so "the run has
        # advanced" is not "sim time has elapsed" -- step until the clock moves.
        session = session_of(app, handle["run"])
        while session.sim_time.root == session.horizon.start.root:
            step(client, handle["run"], granularity="tick")
        response = client.get(f"/runs/{handle['run']}/metrics")
        assert response.status_code == 200, response.text
        values: dict[str, Any] = response.json()["values"]
        assert set(values) == set(KPI_KEYS)
        # Empty strata are null on the wire, never omitted.
        assert values["los_s_mean_by_esi_1"] is None


def test_compare_projects_the_paired_bootstrap(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, arm="optimized", compare_to="baseline", seed=9)
        run_to_finish(client, handle["run"])
        # Both arms, not just the primary: /compare folds at the LAGGING arm's cut,
        # so a shadow still being driven would make the API's fold a prefix of the
        # full-log fold this test re-derives.
        wait_until_finished(client, handle["shadow"])

        response = client.get(f"/runs/{handle['run']}/compare")
        assert response.status_code == 200
        body = response.json()
        assert body["baseline_run"] == handle["shadow"]
        assert body["optimized_run"] == handle["run"]
        assert body["replications"] == 1
        contrasts: list[dict[str, Any]] = body["contrasts"]
        assert [c["key"] for c in contrasts] == list(KPI_KEYS)

        baseline = session_of(app, handle["shadow"])
        optimized = session_of(app, handle["run"])
        # Both arms are finished, so the common cut IS the horizon end -- which is
        # where the live window and the headless one coincide exactly.
        baseline_window, baseline_warmup = _expected_window(baseline, baseline.sim_time)
        optimized_window, optimized_warmup = _expected_window(optimized, optimized.sim_time)
        assert baseline_window.end == baseline.horizon.end
        baseline_kpis = compute_kpis(
            baseline.log,
            baseline.layout,
            baseline.roster,
            window=baseline_window,
            warmup=baseline_warmup,
        )
        optimized_kpis = compute_kpis(
            optimized.log,
            optimized.layout,
            optimized.roster,
            window=optimized_window,
            warmup=optimized_warmup,
        )
        for contrast in contrasts:
            key = contrast["key"]
            assert _wire_value_equals(contrast["baseline"], baseline_kpis.values[key]), key
            assert _wire_value_equals(contrast["optimized"], optimized_kpis.values[key]), key
            expected_delta = baseline_kpis.values[key] - optimized_kpis.values[key]
            assert _wire_value_equals(contrast["delta"], expected_delta), key
            # A single-seed live pair: CIs are degenerate and nothing is significant.
            assert contrast["ci_lo"] is None
            assert contrast["ci_hi"] is None
            assert contrast["significant"] is False


def test_compare_without_a_shadow_is_a_409(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        response = client.get(f"/runs/{handle['run']}/compare")
        assert response.status_code == 409
