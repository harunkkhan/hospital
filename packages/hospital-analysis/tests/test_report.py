"""``report.fold_arm``/``build_metrics``/``write_metrics`` — smoke + shape tests."""

from __future__ import annotations

from pathlib import Path

from _analysis_fixtures import build_sample_log, tiny_layout, tiny_roster

from hospital.analysis.compare import paired_bootstrap
from hospital.analysis.fold import compute_kpis
from hospital.analysis.report import build_metrics, fold_arm, write_metrics
from hospital.core import KPI_KEYS, EsiAcuity, hours


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
