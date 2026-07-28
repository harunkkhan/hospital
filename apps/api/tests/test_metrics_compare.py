"""``/metrics`` is the one fold; ``/compare`` is the one bootstrap, projected.

No KPI math in the API (doc 07 §3.5-§3.6): the live vector must equal
``analysis.fold.compute_kpis`` over the session log, and every ``KpiContrast``
must mirror ``analysis.compare.paired_bootstrap`` output.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from _api_fixtures import create_run, make_app, run_to_finish, session_of, step
from fastapi.testclient import TestClient

from hospital.analysis import compute_kpis
from hospital.core import KPI_KEYS, Duration, hours

if TYPE_CHECKING:
    from pathlib import Path

    from hospital.api.sessions import RunSession


def _wire_value_equals(wire: float | None, folded: float) -> bool:
    """NaN crosses the wire as null (the D8 empty-stratum convention)."""
    if wire is None:
        return math.isnan(folded)
    return wire == folded


def _expected_warmup(session: RunSession) -> Duration:
    span = session.horizon.end.root - session.horizon.start.root
    return Duration(min(hours(24).root, span // 4))


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
        folded = compute_kpis(
            session.log,
            session.layout,
            session.roster,
            window=session.horizon,
            warmup=_expected_warmup(session),
        )
        for key in KPI_KEYS:
            assert _wire_value_equals(values[key], folded.values[key]), key


def test_metrics_of_a_fresh_run_is_a_valid_partial_vector(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client)
        response = client.get(f"/runs/{handle['run']}/metrics")
        assert response.status_code == 200
        values: dict[str, Any] = response.json()["values"]
        assert set(values) == set(KPI_KEYS)
        assert values["completions_per_week"] == 0.0
        # Empty strata are null on the wire, never omitted.
        assert values["los_s_mean_by_esi_1"] is None


def test_compare_projects_the_paired_bootstrap(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as client:
        handle = create_run(client, arm="optimized", compare_to="baseline", seed=9)
        run_to_finish(client, handle["run"])

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
        baseline_kpis = compute_kpis(
            baseline.log,
            baseline.layout,
            baseline.roster,
            window=baseline.horizon,
            warmup=_expected_warmup(baseline),
        )
        optimized_kpis = compute_kpis(
            optimized.log,
            optimized.layout,
            optimized.roster,
            window=optimized.horizon,
            warmup=_expected_warmup(optimized),
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
