"""``report.fold_arm``/``build_metrics``/``write_metrics`` — smoke + shape tests."""

from __future__ import annotations

import json
from pathlib import Path

from _analysis_fixtures import build_sample_log, tiny_layout, tiny_roster

from hospital.analysis.compare import paired_bootstrap
from hospital.analysis.fold import compute_kpis
from hospital.analysis.report import build_metrics, fold_arm, write_metrics
from hospital.core import KPI_KEYS, EsiAcuity, EventLog, hours


def test_fold_arm_and_build_metrics_round_trip(tmp_path: Path) -> None:
    layout = tiny_layout()
    roster = tiny_roster()
    logs = [build_sample_log(), build_sample_log()]  # 2 identical replications

    baseline = fold_arm(logs, layout, roster, warmup=hours(0))
    optimized = fold_arm(logs, layout, roster, warmup=hours(0))
    assert baseline.n_reps == 2
    assert set(baseline.kpis.values.keys()) == set(KPI_KEYS)

    raw_baseline = [compute_kpis(log, layout, roster, warmup=hours(0)) for log in logs]
    raw_optimized = [compute_kpis(log, layout, roster, warmup=hours(0)) for log in logs]
    comparison = paired_bootstrap(raw_baseline, raw_optimized, n_boot=200, seed=7)
    # Identical arms -> no key should show a significant difference.
    assert not any(c.significant for c in comparison.contrasts.values())

    metrics = build_metrics("test_scenario", 42, baseline, optimized, comparison)
    assert metrics.schema_version == "1"
    assert set(metrics.arms.keys()) == {"baseline", "optimized"}
    assert len(metrics.contrasts) == len(KPI_KEYS)

    text = metrics.to_json()
    assert text.endswith("\n")

    path = tmp_path / "metrics.json"
    write_metrics(metrics, path)
    assert path.read_text() == text


def test_fold_arm_with_no_patient_time_has_no_binding_constraint() -> None:
    """Regression (finding #6, rep-averaged path): all-NaN shares across every
    replication must average to an empty ``binding``, not the alphabetically
    first resource."""
    arm = fold_arm([EventLog(), EventLog()], tiny_layout(), tiny_roster(), warmup=hours(0))
    assert arm.bottleneck.binding == ""


def test_metrics_json_serializes_nan_as_null() -> None:
    """Regression (finding #1): empty ESI strata and <2-pair contrasts are NaN
    in memory (D8) but must serialize as JSON ``null`` — a bare ``NaN`` token
    makes ``JSON.parse`` reject the whole metrics.json artifact."""
    layout = tiny_layout()
    roster = tiny_roster()
    logs = [build_sample_log()]  # single replication -> n_pairs=1 -> NaN CIs
    arm = fold_arm(logs, layout, roster, warmup=hours(0))
    raw = [compute_kpis(log, layout, roster, warmup=hours(0)) for log in logs]
    comparison = paired_bootstrap(raw, raw, n_boot=20, seed=1)
    metrics = build_metrics("s", 1, arm, arm, comparison)

    text = metrics.to_json()
    assert "NaN" not in text
    assert "Infinity" not in text

    def _reject_constant(name: str) -> float:
        raise AssertionError(f"non-JSON constant in metrics.json: {name}")

    parsed = json.loads(text, parse_constant=_reject_constant)
    # A known-empty stratum (no ESI-1 patients in the sample log) is null...
    assert parsed["arms"]["baseline"]["kpis"]["values"]["los_s_mean_by_esi_1"] is None
    # ...and so is a <2-pair contrast CI.
    assert parsed["contrasts"]["completions_per_week"]["ci_lo"] is None


def test_acuity_weighted_headline_only_when_weights_passed(tmp_path: Path) -> None:
    layout = tiny_layout()
    roster = tiny_roster()
    logs = [build_sample_log()]
    baseline = fold_arm(logs, layout, roster, warmup=hours(0))
    optimized = fold_arm(logs, layout, roster, warmup=hours(0))
    raw = [compute_kpis(log, layout, roster, warmup=hours(0)) for log in logs]
    comparison = paired_bootstrap(raw, raw, n_boot=50, seed=1)

    metrics_no_weights = build_metrics("s", 1, baseline, optimized, comparison)
    assert "acuity_weighted_time_saved_s" not in metrics_no_weights.headline

    metrics_weighted = build_metrics(
        "s", 1, baseline, optimized, comparison, acuity_weights={EsiAcuity.ESI3: 2.0}
    )
    assert "acuity_weighted_time_saved_s" in metrics_weighted.headline
