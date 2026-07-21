"""``compute_kpis`` — fold one ``EventLog`` into the closed ``core.kpi.KpiVector``.

Emits EXACTLY the 27 ``core.kpi.KPI_KEYS`` (doc 05 §4.1 / D8): an empty cohort
(e.g. an ESI stratum with zero completed patients) is ``NaN``, never omitted,
so ``KpiVector``'s closed contract always validates. Counts
(``completions_per_week``, ``wip_end_of_week``) use the FULL window; every
other key uses the post-warmup measurement window ``M`` — the two-window split
is a deliberate anti-gaming choice (doc 05 nuance 5.3), not an accident.
"""

from __future__ import annotations

from hospital.analysis._index import EventIndex, build_index
from hospital.analysis._stats import (
    DEFAULT_WARMUP,
    DEFAULT_WINDOW,
    clip_seconds,
    mean,
    measurement_window,
    percentile,
)
from hospital.analysis.utilization import utilization_report
from hospital.core import (
    DispositionKind,
    Duration,
    EventLog,
    FloorLayout,
    KpiVector,
    OperatingWeek,
    StaffMember,
    ZeroTimeCycle,
)

__all__ = ["compute_kpis"]


def compute_kpis(
    log: EventLog,
    layout: FloorLayout,
    roster: tuple[StaffMember, ...],
    *,
    window: OperatingWeek = DEFAULT_WINDOW,
    warmup: Duration = DEFAULT_WARMUP,
    index: EventIndex | None = None,
) -> KpiVector:
    idx = index if index is not None else build_index(log, layout, roster)
    m = measurement_window(window, warmup)
    if m.duration() <= Duration(0):
        raise ZeroTimeCycle("measurement window is zero-length or negative (warmup >= horizon)")

    patients = list(idx.patients.values())

    values: dict[str, float] = {}

    # 1-2: full-window counts (deliberately NOT clipped to M, doc nuance 5.3).
    arrivals_by_end = sum(1 for p in patients if p.arrival <= window.end)
    completions_by_end = sum(1 for p in patients if p.exit is not None and p.exit <= window.end)
    values["completions_per_week"] = float(
        sum(1 for p in patients if p.exit is not None and window.contains(p.exit))
    )
    values["wip_end_of_week"] = float(arrivals_by_end - completions_by_end)

    # Cohorts (defined once, reused): C_wait = arrival in M.
    c_wait = [p for p in patients if m.contains(p.arrival)]

    # 3-4: door_to_triage
    triage_samples = [
        (p.triage_start - p.arrival).to_seconds() for p in c_wait if p.triage_start is not None
    ]
    values["door_to_triage_s_mean"] = mean(triage_samples)
    values["door_to_triage_s_p90"] = percentile(triage_samples, 0.90)

    # 5-6: door_to_provider
    provider_samples = [
        (p.provider_start - p.arrival).to_seconds() for p in c_wait if p.provider_start is not None
    ]
    values["door_to_provider_s_mean"] = mean(provider_samples)
    values["door_to_provider_s_p90"] = percentile(provider_samples, 0.90)

    # 7-16: los_s_{mean,p90}_by_esi_{1..5} — C_los(k), right-censored WIP excluded.
    for k in (1, 2, 3, 4, 5):
        los_samples = [
            (p.exit - p.arrival).to_seconds() for p in c_wait if p.esi == k and p.exit is not None
        ]
        values[f"los_s_mean_by_esi_{k}"] = mean(los_samples)
        values[f"los_s_p90_by_esi_{k}"] = percentile(los_samples, 0.90)

    # 21-27 (and the shared numerator for #17): the ONE utilization computation.
    util = utilization_report(log, roster, window=window, warmup=warmup, index=idx)
    values["staff_minutes_walked"] = sum(b.walk_s for b in util.per_staff) / 60.0
    values["provider_util"] = util.util_by_role.get("provider_util", float("nan"))
    values["nurse_util"] = util.util_by_role.get("nurse_util", float("nan"))
    values.update(util.fractions)

    # 18: bay_utilization = occupied-bay-seconds / (N_bays * |M|), warmup-windowed.
    n_bays = len(layout.bays)
    if n_bays == 0:
        values["bay_utilization"] = float("nan")
    else:
        occupied = 0.0
        for cycles in idx.bays.values():
            for cyc in cycles:
                if cyc.bay_arrival is None:
                    continue
                end = cyc.clean_start if cyc.clean_start is not None else window.end
                occupied += clip_seconds(cyc.bay_arrival, end, m)
        values["bay_utilization"] = occupied / (n_bays * m.duration().to_seconds())

    # 19: turnaround_time_s_mean = mean(clean_end - disposition_decided), clean_end in M.
    turnaround_samples = [
        (cyc.clean_end - cyc.disposition_time).to_seconds()
        for cycles in idx.bays.values()
        for cyc in cycles
        if cyc.clean_end is not None
        and cyc.disposition_time is not None
        and m.contains(cyc.clean_end)
    ]
    values["turnaround_time_s_mean"] = mean(turnaround_samples)

    # 20: boarding_time_s_mean = mean(exit - disposition) for ADMIT, disposition in M.
    boarding_samples = [
        (p.exit - p.disposition_time).to_seconds()
        for p in patients
        if p.disposition == DispositionKind.ADMIT
        and p.disposition_time is not None
        and m.contains(p.disposition_time)
        and p.exit is not None
    ]
    values["boarding_time_s_mean"] = mean(boarding_samples)

    return KpiVector(values=values)
