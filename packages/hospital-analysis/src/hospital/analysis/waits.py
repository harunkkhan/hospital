"""``decompose_waits`` — per-patient LOS decomposition into seven tiling stages.

Adjacent-milestone differences tile ``[a, de]`` exactly (doc 05 §4.2 / nuance
5.4): ``Σ(seven stages) == (de - a)/1e6 == los_s``, to the microsecond. A
patient missing any milestone up to ``de`` (still WIP / censored, or the log
never completed the chain) is excluded from tiling — the censored residual is
accounted for exactly once, in ``fold.compute_kpis``'s ``wip_end_of_week``, and
never smeared across a fabricated stage here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from hospital.analysis._index import EventIndex, PatientTrace, build_index
from hospital.analysis._stats import (
    DEFAULT_WARMUP,
    DEFAULT_WINDOW,
    mean,
    measurement_window,
    percentile,
)
from hospital.core import (
    BayId,
    DispositionKind,
    Duration,
    EsiAcuity,
    EventLog,
    FloorLayout,
    FrozenModel,
    OperatingWeek,
    PatientId,
)

__all__ = [
    "BayTurnaroundProfile",
    "PatientWaitProfile",
    "StageAggregate",
    "StageSeconds",
    "WaitDecomposition",
    "decompose_waits",
]

_STAGE_NAMES: tuple[str, ...] = (
    "wait_triage",
    "svc_triage",
    "wait_bay",
    "wait_provider",
    "workup_service",
    "workup_wait",
    "paperwork_or_boarding",
)


class StageSeconds(FrozenModel):
    """The seven stages that tile ``[a, de]`` exactly — fields sum to ``los_s``."""

    wait_triage: float
    svc_triage: float
    wait_bay: float
    wait_provider: float
    workup_service: float
    workup_wait: float
    paperwork_or_boarding: float


class PatientWaitProfile(FrozenModel):
    patient: PatientId
    esi: EsiAcuity
    disposition: DispositionKind | None
    los_s: float
    stages: StageSeconds


class BayTurnaroundProfile(FrozenModel):
    """The bay's own post-disposition cycle — a parallel timeline, not part of LOS."""

    bay: BayId
    hold_to_vacate_s: float
    wait_housekeeper_s: float
    cleaning_s: float
    turnaround_s: float


class StageAggregate(FrozenModel):
    mean_s: float
    p90_s: float
    total_s: float
    share_of_los: float


class WaitDecomposition(FrozenModel):
    per_patient: tuple[PatientWaitProfile, ...]
    per_bay_cycle: tuple[BayTurnaroundProfile, ...]
    stage_means: Mapping[str, StageAggregate]
    by_esi: Mapping[int, Mapping[str, StageAggregate]]


def _tile_patient(p: PatientTrace) -> PatientWaitProfile | None:
    if (
        p.esi is None
        or p.triage_start is None
        or p.triage_end is None
        or p.bay_arrival is None
        or p.provider_start is None
        or p.disposition_time is None
        or p.exit is None
    ):
        return None

    wait_triage = (p.triage_start - p.arrival).to_seconds()
    svc_triage = (p.triage_end - p.triage_start).to_seconds()
    wait_bay = (p.bay_arrival - p.triage_end).to_seconds()
    wait_provider = (p.provider_start - p.bay_arrival).to_seconds()

    workup_service = 0.0
    for iv in (*p.provider_intervals, *p.nurse_intervals):
        # iv.end is None = interval still open at the horizon; it cannot fall
        # inside [pv, dd] of a COMPLETED patient, so it never joins the tiling.
        if iv.end is not None and iv.start >= p.provider_start and iv.end <= p.disposition_time:
            workup_service += (iv.end - iv.start).to_seconds()
    workup_wait = (p.disposition_time - p.provider_start).to_seconds() - workup_service
    paperwork_or_boarding = (p.exit - p.disposition_time).to_seconds()

    stages = StageSeconds(
        wait_triage=wait_triage,
        svc_triage=svc_triage,
        wait_bay=wait_bay,
        wait_provider=wait_provider,
        workup_service=workup_service,
        workup_wait=workup_wait,
        paperwork_or_boarding=paperwork_or_boarding,
    )
    los_s = (p.exit - p.arrival).to_seconds()
    return PatientWaitProfile(
        patient=p.patient, esi=p.esi, disposition=p.disposition, los_s=los_s, stages=stages
    )


def _stage_aggregates(profiles: Sequence[PatientWaitProfile]) -> dict[str, StageAggregate]:
    per_stage: dict[str, list[float]] = {name: [] for name in _STAGE_NAMES}
    for prof in profiles:
        for name in _STAGE_NAMES:
            per_stage[name].append(getattr(prof.stages, name))
    totals = {name: math.fsum(vals) for name, vals in per_stage.items()}
    grand_total = math.fsum(totals.values())
    result: dict[str, StageAggregate] = {}
    for name in _STAGE_NAMES:
        vals = per_stage[name]
        share = totals[name] / grand_total if grand_total > 0.0 else float("nan")
        result[name] = StageAggregate(
            mean_s=mean(vals),
            p90_s=percentile(vals, 0.90),
            total_s=totals[name],
            share_of_los=share,
        )
    return result


def decompose_waits(
    log: EventLog,
    layout: FloorLayout,
    *,
    window: OperatingWeek = DEFAULT_WINDOW,
    warmup: Duration = DEFAULT_WARMUP,
    index: EventIndex | None = None,
) -> WaitDecomposition:
    # decompose_waits's public signature (doc 05 §3) carries no roster — it never
    # reads staff traces, only patient milestones and bay cycles — so building a
    # fresh index here uses an empty roster tuple.
    idx = index if index is not None else build_index(log, layout, ())
    m = measurement_window(window, warmup)

    per_patient: list[PatientWaitProfile] = []
    for p in idx.patients.values():
        prof = _tile_patient(p)
        if prof is not None:
            per_patient.append(prof)

    c_wait_patients = {p.patient for p in idx.patients.values() if m.contains(p.arrival)}
    cohort = [prof for prof in per_patient if prof.patient in c_wait_patients]

    stage_means = _stage_aggregates(cohort)
    by_esi: dict[int, dict[str, StageAggregate]] = {}
    for k in (1, 2, 3, 4, 5):
        by_esi[k] = _stage_aggregates([prof for prof in cohort if int(prof.esi) == k])

    per_bay_cycle: list[BayTurnaroundProfile] = []
    for bay, cycles in idx.bays.items():
        for cyc in cycles:
            if (
                cyc.disposition_time is None
                or cyc.exit is None
                or cyc.clean_start is None
                or cyc.clean_end is None
            ):
                continue
            hold_to_vacate_s = (cyc.exit - cyc.disposition_time).to_seconds()
            wait_housekeeper_s = (cyc.clean_start - cyc.exit).to_seconds()
            cleaning_s = (cyc.clean_end - cyc.clean_start).to_seconds()
            turnaround_s = (cyc.clean_end - cyc.disposition_time).to_seconds()
            per_bay_cycle.append(
                BayTurnaroundProfile(
                    bay=bay,
                    hold_to_vacate_s=hold_to_vacate_s,
                    wait_housekeeper_s=wait_housekeeper_s,
                    cleaning_s=cleaning_s,
                    turnaround_s=turnaround_s,
                )
            )

    return WaitDecomposition(
        per_patient=tuple(per_patient),
        per_bay_cycle=tuple(per_bay_cycle),
        stage_means=stage_means,
        by_esi=by_esi,
    )
