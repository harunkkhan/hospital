"""``fold.compute_kpis`` — the closed 27-key contract, hand-computed exact values."""

from __future__ import annotations

import math

from _analysis_fixtures import build_sample_log, tiny_layout, tiny_roster

from hospital.analysis.fold import compute_kpis
from hospital.core import KPI_KEYS, EventLog, hours


def test_emits_exactly_kpi_keys() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    vec = compute_kpis(log, layout, roster, warmup=hours(0))
    assert set(vec.values.keys()) == set(KPI_KEYS)
    assert len(vec.values) == 27


def test_hand_computed_values_match() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    vec = compute_kpis(log, layout, roster, warmup=hours(0))
    v = vec.values

    # P1 completes (discharge), P2 arrives but stays WIP -> arrivals=2, completions=1.
    assert v["completions_per_week"] == 1.0
    assert v["wip_end_of_week"] == 1.0

    # door_to_triage: P1 ts0-a=60, P2 ts0-a=50.
    assert math.isclose(v["door_to_triage_s_mean"], (60.0 + 50.0) / 2)
    assert math.isclose(v["door_to_triage_s_p90"], 50.0 + 0.9 * (60.0 - 50.0))

    # door_to_provider: P1 pv-a=600, P2 pv-a=300.
    assert math.isclose(v["door_to_provider_s_mean"], (600.0 + 300.0) / 2)
    assert math.isclose(v["door_to_provider_s_p90"], 300.0 + 0.9 * (600.0 - 300.0))

    # LOS by ESI: only P1 (ESI3, completed) contributes; P2 is WIP -> excluded.
    assert math.isclose(v["los_s_mean_by_esi_3"], 1500.0)
    assert math.isclose(v["los_s_p90_by_esi_3"], 1500.0)

    # Empty ESI strata (no completed patients) are NaN, never omitted/dropped.
    for k in (1, 2, 4, 5):
        assert math.isnan(v[f"los_s_mean_by_esi_{k}"])
        assert math.isnan(v[f"los_s_p90_by_esi_{k}"])

    # staff_minutes_walked: phys 20+30=50s, nurse 15s, hk 25s -> 90s total.
    assert math.isclose(v["staff_minutes_walked"], 90.0 / 60.0)

    # bay_utilization: bay-1 occupied [480,1560)=1080s; bay-2 occupied
    # [2200, window.end) (still open, WIP) = 604800-2200=602600s; bay-3 unused.
    occupied = (1560.0 - 480.0) + (604800.0 - 2200.0)
    assert math.isclose(v["bay_utilization"], occupied / (len(layout.bays) * 604800.0))

    # turnaround_time_s_mean: only bay-1's cycle has a clean_end (bay-2 never
    # cleaned): clean_end(1860) - disposition_decided(1200) = 660s.
    assert math.isclose(v["turnaround_time_s_mean"], 660.0)

    # No ADMIT patients in this log -> boarding_time_s_mean is NaN (empty cohort).
    assert math.isnan(v["boarding_time_s_mean"])

    # staff_frac_* sum to 1.0 to machine epsilon (residual construction).
    frac_total = math.fsum(v[k] for k in KPI_KEYS if k.startswith("staff_frac_"))
    assert math.isclose(frac_total, 1.0, abs_tol=1e-9)


def test_arrivals_equal_completions_plus_wip() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    vec = compute_kpis(log, layout, roster, warmup=hours(0))
    v = vec.values
    arrivals = 2.0  # p1 + p2
    assert arrivals == v["completions_per_week"] + v["wip_end_of_week"]


def test_empty_log_all_nan_or_zero_without_raising() -> None:
    log = EventLog()
    layout = tiny_layout()
    roster = tiny_roster()
    vec = compute_kpis(log, layout, roster, warmup=hours(0))
    v = vec.values
    assert v["completions_per_week"] == 0.0
    assert v["wip_end_of_week"] == 0.0
    for k in (1, 2, 3, 4, 5):
        assert math.isnan(v[f"los_s_mean_by_esi_{k}"])
    assert math.isnan(v["door_to_triage_s_mean"])
    assert v["bay_utilization"] == 0.0
    frac_total = math.fsum(v[k] for k in KPI_KEYS if k.startswith("staff_frac_"))
    assert math.isclose(frac_total, 1.0, abs_tol=1e-9)
