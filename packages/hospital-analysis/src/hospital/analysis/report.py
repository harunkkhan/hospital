"""``fold_arm``/``build_metrics``/``write_metrics`` — assemble ``metrics.json``.

Composes §4.1-4.5 (doc 05 §4.6): no statistics or KPI computation of its own —
the mean-diffs/CIs come from ``compare``, the KPIs from ``fold``. Rep-averaging
is NaN-aware throughout (an empty-strata NaN in some reps must not poison the
whole average, and must never be treated as 0), so the "no data" signal flows
coherently from a single replication's empty ESI stratum all the way to a NaN
headline entry. ``write_metrics`` emits sorted-key, ``\\n``-terminated JSON —
byte-stable, golden-trace friendly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from hospital.analysis._index import build_index
from hospital.analysis._stats import DEFAULT_WARMUP, DEFAULT_WINDOW
from hospital.analysis.bottleneck import BottleneckReport, ResourceWait, detect_bottleneck
from hospital.analysis.compare import ComparisonResult, Contrast
from hospital.analysis.fold import compute_kpis
from hospital.analysis.utilization import (
    StaffSecondBudget,
    UtilizationReport,
    utilization_report,
)
from hospital.analysis.waits import StageAggregate, WaitDecomposition, decompose_waits
from hospital.core import (
    KPI_KEYS,
    Duration,
    EsiAcuity,
    EventLog,
    FloorLayout,
    FrozenModel,
    KpiVector,
    OperatingWeek,
    StaffId,
    StaffMember,
)

__all__ = ["ArmSummary", "Metrics", "build_metrics", "fold_arm", "write_metrics"]

SCHEMA_VERSION = "1"
# build_metrics's horizon_s/warmup_s keyword defaults (doc 05 §3's build_metrics
# signature has no window/warmup parameters at all — see the docstring below).
_DEFAULT_HORIZON_S: float = DEFAULT_WINDOW.end.root / 1_000_000
_DEFAULT_WARMUP_S: float = DEFAULT_WARMUP.to_seconds()

_STAFF_FRAC_ZERO_KEYS: tuple[str, ...] = (
    "staff_frac_walk",
    "staff_frac_direct_care",
    "staff_frac_cleaning",
    "staff_frac_documentation",
)


def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return math.fsum(vals) / len(vals) if vals else float("nan")


def _empty_kpi_values() -> dict[str, float]:
    values = dict.fromkeys(KPI_KEYS, float("nan"))
    for key in _STAFF_FRAC_ZERO_KEYS:
        values[key] = 0.0
    values["staff_frac_idle"] = 1.0
    return values


def _avg_kpi(vectors: Sequence[KpiVector]) -> KpiVector:
    if not vectors:
        return KpiVector(values=_empty_kpi_values())
    values = {key: _nanmean([v.values[key] for v in vectors]) for key in KPI_KEYS}
    return KpiVector(values=values)


def _avg_stage_aggregate(aggs: Sequence[StageAggregate]) -> StageAggregate:
    return StageAggregate(
        mean_s=_nanmean([a.mean_s for a in aggs]),
        p90_s=_nanmean([a.p90_s for a in aggs]),
        total_s=_nanmean([a.total_s for a in aggs]),
        share_of_los=_nanmean([a.share_of_los for a in aggs]),
    )


def _avg_waits(decomps: Sequence[WaitDecomposition]) -> WaitDecomposition:
    if not decomps:
        return WaitDecomposition(per_patient=(), per_bay_cycle=(), stage_means={}, by_esi={})
    stage_names: set[str] = set()
    for d in decomps:
        stage_names.update(d.stage_means.keys())
    stage_means = {
        name: _avg_stage_aggregate([d.stage_means[name] for d in decomps if name in d.stage_means])
        for name in stage_names
    }
    by_esi: dict[int, dict[str, StageAggregate]] = {}
    for k in (1, 2, 3, 4, 5):
        names: set[str] = set()
        for d in decomps:
            names.update(d.by_esi.get(k, {}).keys())
        by_esi[k] = {
            name: _avg_stage_aggregate(
                [d.by_esi[k][name] for d in decomps if name in d.by_esi.get(k, {})]
            )
            for name in names
        }
    # Per-replication raw traces (per_patient/per_bay_cycle) have no natural
    # rep-average; the aggregated stage_means/by_esi tables above — what
    # metrics.json actually serializes — are proper NaN-aware field averages.
    return WaitDecomposition(
        per_patient=(), per_bay_cycle=(), stage_means=stage_means, by_esi=by_esi
    )


def _bottleneck_sort_key(rw: ResourceWait) -> tuple[float, str]:
    share = rw.share_of_cycle
    # NaN shares (no observed patient-time in any rep) rank LAST, as in
    # bottleneck.detect_bottleneck.
    return (-share if not math.isnan(share) else float("inf"), rw.resource)


def _avg_bottleneck(reports: Sequence[BottleneckReport]) -> BottleneckReport:
    if not reports:
        return BottleneckReport(
            binding="",
            resources=(),
            total_cycle_s=float("nan"),
            gini_by_role={},
            gini_overall=float("nan"),
        )
    names: set[str] = set()
    for r in reports:
        names.update(rw.resource for rw in r.resources)
    resources: list[ResourceWait] = []
    for name in names:
        entries = [rw for r in reports for rw in r.resources if rw.resource == name]
        resources.append(
            ResourceWait(
                resource=name,
                total_wait_s=_nanmean([e.total_wait_s for e in entries]),
                n_requests=round(_nanmean([float(e.n_requests) for e in entries])),
                mean_wait_s=_nanmean([e.mean_wait_s for e in entries]),
                share_of_cycle=_nanmean([e.share_of_cycle for e in entries]),
            )
        )
    resources.sort(key=_bottleneck_sort_key)
    # As in detect_bottleneck: no finite share -> no binding constraint.
    binding = next((rw.resource for rw in resources if not math.isnan(rw.share_of_cycle)), "")
    total_cycle_s = _nanmean([r.total_cycle_s for r in reports])

    role_names: set[str] = set()
    for r in reports:
        role_names.update(r.gini_by_role.keys())
    gini_by_role = {
        role: _nanmean([r.gini_by_role[role] for r in reports if role in r.gini_by_role])
        for role in role_names
    }
    gini_overall = _nanmean([r.gini_overall for r in reports])

    return BottleneckReport(
        binding=binding,
        resources=tuple(resources),
        total_cycle_s=total_cycle_s,
        gini_by_role=gini_by_role,
        gini_overall=gini_overall,
    )


def _avg_utilization(reports: Sequence[UtilizationReport]) -> UtilizationReport:
    if not reports:
        return UtilizationReport(per_staff=(), fractions={}, util_by_role={})
    staff_ids: list[StaffId] = []
    seen: set[StaffId] = set()
    for r in reports:
        for b in r.per_staff:
            if b.staff not in seen:
                seen.add(b.staff)
                staff_ids.append(b.staff)
    per_staff: list[StaffSecondBudget] = []
    for sid in staff_ids:
        entries = [b for r in reports for b in r.per_staff if b.staff == sid]
        per_staff.append(
            StaffSecondBudget(
                staff=sid,
                role=entries[0].role,
                on_shift_s=_nanmean([e.on_shift_s for e in entries]),
                walk_s=_nanmean([e.walk_s for e in entries]),
                direct_care_s=_nanmean([e.direct_care_s for e in entries]),
                cleaning_s=_nanmean([e.cleaning_s for e in entries]),
                documentation_s=_nanmean([e.documentation_s for e in entries]),
                idle_s=_nanmean([e.idle_s for e in entries]),
            )
        )
    frac_names: set[str] = set()
    for r in reports:
        frac_names.update(r.fractions.keys())
    fractions = {
        name: _nanmean([r.fractions[name] for r in reports if name in r.fractions])
        for name in frac_names
    }
    role_names: set[str] = set()
    for r in reports:
        role_names.update(r.util_by_role.keys())
    util_by_role = {
        name: _nanmean([r.util_by_role[name] for r in reports if name in r.util_by_role])
        for name in role_names
    }
    return UtilizationReport(
        per_staff=tuple(per_staff), fractions=fractions, util_by_role=util_by_role
    )


class ArmSummary(FrozenModel):
    kpis: KpiVector
    waits: WaitDecomposition
    bottleneck: BottleneckReport
    utilization: UtilizationReport
    n_reps: int


class Metrics(FrozenModel):
    schema_version: str
    scenario: str
    seed: int
    horizon_s: float
    warmup_s: float
    arms: Mapping[str, ArmSummary]
    contrasts: Mapping[str, Contrast]
    headline: Mapping[str, float]

    def to_json(self) -> str:
        return _dump_json(self)


def _jsonify(value: object) -> object:
    """Map non-finite floats (NaN/Inf) to ``None`` at the serialization boundary.

    The in-memory NaN convention (D8: empty strata are NaN, never omitted) is
    untouched — but a bare ``NaN`` token in ``metrics.json`` is not JSON and
    ``JSON.parse`` rejects the whole file, so NaN/Inf become ``null`` on disk.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return {key: _jsonify(val) for key, val in mapping.items()}
    if isinstance(value, list):
        return [_jsonify(val) for val in cast("list[object]", value)]
    return value


def _dump_json(metrics: Metrics) -> str:
    data = _jsonify(metrics.model_dump(mode="json"))
    # allow_nan=False: if a non-finite value ever slips past _jsonify again,
    # raise loudly rather than emit a file JSON.parse cannot read.
    return json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"


def fold_arm(
    logs: Sequence[EventLog],
    layout: FloorLayout,
    roster: tuple[StaffMember, ...],
    *,
    window: OperatingWeek = DEFAULT_WINDOW,
    warmup: Duration = DEFAULT_WARMUP,
) -> ArmSummary:
    """Fold ``N`` replication logs of one arm into a rep-averaged ``ArmSummary``."""
    kpi_vectors: list[KpiVector] = []
    wait_decomps: list[WaitDecomposition] = []
    bottlenecks: list[BottleneckReport] = []
    utils: list[UtilizationReport] = []
    for log in logs:
        idx = build_index(log, layout, roster)
        kpi_vectors.append(
            compute_kpis(log, layout, roster, window=window, warmup=warmup, index=idx)
        )
        wait_decomps.append(decompose_waits(log, layout, window=window, warmup=warmup, index=idx))
        bottlenecks.append(
            detect_bottleneck(log, layout, roster, window=window, warmup=warmup, index=idx)
        )
        utils.append(utilization_report(log, roster, window=window, warmup=warmup, index=idx))

    return ArmSummary(
        kpis=_avg_kpi(kpi_vectors),
        waits=_avg_waits(wait_decomps),
        bottleneck=_avg_bottleneck(bottlenecks),
        utilization=_avg_utilization(utils),
        n_reps=len(logs),
    )


def build_metrics(
    scenario: str,
    seed: int,
    baseline: ArmSummary,
    optimized: ArmSummary,
    comparison: ComparisonResult,
    *,
    acuity_weights: Mapping[EsiAcuity, float] | None = None,
    horizon_s: float = _DEFAULT_HORIZON_S,
    warmup_s: float = _DEFAULT_WARMUP_S,
) -> Metrics:
    """Pair the two arms with the ``ComparisonResult`` and compute the headline.

    ``horizon_s``/``warmup_s`` default to the reference one-week/24h scenario;
    doc 05 §3's ``build_metrics`` signature has no ``window``/``warmup``
    parameters of its own (only the two arms + comparison), so these are
    surfaced as explicit, overridable keyword defaults rather than silently
    hardcoded (a deviation from the literal signature, documented in the final
    report).
    """
    b_kpi = baseline.kpis.values
    o_kpi = optimized.kpis.values
    headline: dict[str, float] = {
        "extra_completions_per_week": (
            o_kpi["completions_per_week"] - b_kpi["completions_per_week"]
        ),
        "staff_minutes_walked_saved": (
            b_kpi["staff_minutes_walked"] - o_kpi["staff_minutes_walked"]
        ),
    }

    los_reductions: list[float] = []
    for k in (1, 2, 3, 4, 5):
        key = f"los_s_mean_by_esi_{k}"
        bv, ov = b_kpi[key], o_kpi[key]
        if not (math.isnan(bv) or math.isnan(ov)):
            los_reductions.append(bv - ov)
    headline["los_s_mean_reduction_overall"] = _nanmean(los_reductions)

    if acuity_weights is not None:
        total = 0.0
        any_term = False
        for k in (1, 2, 3, 4, 5):
            esi = EsiAcuity(k)
            if esi not in acuity_weights:
                continue
            key = f"los_s_mean_by_esi_{k}"
            bv, ov = b_kpi[key], o_kpi[key]
            if math.isnan(bv) or math.isnan(ov):
                continue
            total += acuity_weights[esi] * (bv - ov)
            any_term = True
        if any_term:
            headline["acuity_weighted_time_saved_s"] = total

    return Metrics(
        schema_version=SCHEMA_VERSION,
        scenario=scenario,
        seed=seed,
        horizon_s=horizon_s,
        warmup_s=warmup_s,
        arms={"baseline": baseline, "optimized": optimized},
        contrasts=dict(comparison.contrasts),
        headline=headline,
    )


def write_metrics(metrics: Metrics, path: Path) -> None:
    path.write_text(_dump_json(metrics))
