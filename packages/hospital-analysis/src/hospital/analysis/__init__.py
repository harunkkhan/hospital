"""hospital.analysis — KPI fold, decomposition, bottleneck detection, comparison stats.

A pure, stateless reader of the one canonical ``core.events.EventLog`` (doc 05
§1): it turns a run's event stream into the closed ``core.kpi.KpiVector``,
three decompositions (waits / bottleneck / utilization), and the baseline-vs-
optimized statistics. Depends on ``core`` only. This ``__init__`` re-exports
ONLY the public surface (doc 00 §3) — everything else is import-by-module.
"""

from __future__ import annotations

from hospital.analysis.bottleneck import BottleneckReport, ResourceWait, detect_bottleneck, gini
from hospital.analysis.compare import (
    WEIGHTED_OBJECTIVE_KEY,
    ComparisonResult,
    Contrast,
    paired_bootstrap,
    paired_scalar_contrast,
)
from hospital.analysis.fold import compute_kpis
from hospital.analysis.report import ArmSummary, Metrics, build_metrics, fold_arm, write_metrics
from hospital.analysis.utilization import (
    StaffSecondBudget,
    UtilizationReport,
    classify_staff_seconds,
    utilization_report,
)
from hospital.analysis.waits import (
    BayTurnaroundProfile,
    PatientWaitProfile,
    StageAggregate,
    StageSeconds,
    WaitDecomposition,
    decompose_waits,
)

__all__ = [
    "WEIGHTED_OBJECTIVE_KEY",
    "ArmSummary",
    "BayTurnaroundProfile",
    "BottleneckReport",
    "ComparisonResult",
    "Contrast",
    "Metrics",
    "PatientWaitProfile",
    "ResourceWait",
    "StaffSecondBudget",
    "StageAggregate",
    "StageSeconds",
    "UtilizationReport",
    "WaitDecomposition",
    "build_metrics",
    "classify_staff_seconds",
    "compute_kpis",
    "decompose_waits",
    "detect_bottleneck",
    "fold_arm",
    "gini",
    "paired_bootstrap",
    "paired_scalar_contrast",
    "utilization_report",
    "write_metrics",
]
