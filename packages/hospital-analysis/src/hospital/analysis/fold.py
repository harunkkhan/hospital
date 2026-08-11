"""``compute_kpis`` — fold one ``EventLog`` into the closed ``core.kpi.KpiVector``.

Emits EXACTLY the 31 ``core.kpi.KPI_KEYS`` (doc 05 §4.1 / D8): an empty cohort
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
    care_deadline_for,
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

    # 1-2: full-window counts (deliberately NOT clipped to M, doc nuance 5.3),
    # under the half-open [start, end) convention applied CONSISTENTLY: an
    # arrival or exit at exactly window.end is outside the week — excluded from
    # completions_per_week (via window.contains) AND from these counts, so flow
    # conservation (arrivals == completions + wip) holds at the boundary.
    arrivals_by_end = sum(1 for p in patients if p.arrival < window.end)
    completions_by_end = sum(1 for p in patients if p.exit is not None and p.exit < window.end)
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

    # 31: deadline_breach_hours_total — patient-hours waited past `care_deadline` for a
    # provider, the extensive and *priceable* form of "was this patient seen in time".
    #
    # Acuity-weighted by construction rather than by a separate weight: the deadline itself
    # is the acuity term (immediate for ESI-1, two hours for ESI-5), so the same hour of
    # waiting breaches sooner for a sicker patient. That is why this is the right thing to
    # price and `door_to_provider_s_mean` is not — a mean cannot be multiplied by a rate, and
    # an unweighted one treats an ESI-1's hour and an ESI-5's as the same loss.
    #
    # Censored at the window end, like `boarding_hours_total` and for the same reason: a
    # patient still waiting when the week ends contributes the hours they actually waited.
    # Counting only those who were eventually seen would make a gridlocked week — where the
    # longest waits never resolve — report *less* breach than a calm one.
    breach_s = 0.0
    for p in c_wait:
        if p.esi is None:
            continue  # never triaged, so no acuity and no deadline to breach
        deadline = care_deadline_for(p.arrival, p.esi)
        seen = p.provider_start if p.provider_start is not None else window.end
        if seen > deadline:
            breach_s += clip_seconds(deadline, seen, m)
    values["deadline_breach_hours_total"] = breach_s / 3600.0

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
    # 28: the shared denominator behind every staff_frac_*, kept as hours. A fraction
    # cannot be priced; the hours it is a fraction of can.
    values["staff_hours_paid"] = util.on_shift_s / 3600.0
    values["provider_util"] = util.util_by_role.get("provider_util", float("nan"))
    values["nurse_util"] = util.util_by_role.get("nurse_util", float("nan"))
    values.update(util.fractions)

    # 18 + 29: bay_utilization = occupied-bay-seconds / (N_bays * |M|), warmup-windowed,
    # and the same numerator kept in hours. The ratio alone cannot be priced, and it is
    # also ambiguous once a building has floors: 60% of 76 ED bays and 60% of 150
    # hospital bays are the same number and very different hospitals.
    n_bays = len(layout.bays)
    occupied = 0.0
    for cycles in idx.bays.values():
        for cyc in cycles:
            if cyc.bay_arrival is None:
                continue
            end = cyc.clean_start if cyc.clean_start is not None else window.end
            occupied += clip_seconds(cyc.bay_arrival, end, m)
    values["bay_hours_occupied"] = occupied / 3600.0
    values["bay_utilization"] = (
        float("nan") if n_bays == 0 else occupied / (n_bays * m.duration().to_seconds())
    )

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
    # `(decided, exit)` pairs rather than the patient records, so the non-None
    # disposition time survives into both metrics below as a narrowed value.
    boarded = [
        (p.disposition_time, p.exit)
        for p in patients
        if p.disposition == DispositionKind.ADMIT
        and p.disposition_time is not None
        and m.contains(p.disposition_time)
    ]
    values["boarding_time_s_mean"] = mean(
        [(exit_at - decided).to_seconds() for decided, exit_at in boarded if exit_at is not None]
    )

    # 30: boarding_hours_total — the same wait, CENSORED at the horizon instead of
    # conditioned on reaching a bed, and summed rather than averaged.
    #
    # The mean above is the M1 metric and stays exactly as it was, but it answers a
    # question that stopped being safe when wards gained finite capacity: it drops every
    # patient who never got a bed. Under a full hospital those are the longest waits, so
    # excluding them makes a gridlocked week report a *shorter* mean boarding time than a
    # roomy one. Counting the hours a still-boarding patient has actually waited when the
    # week ends is the version that cannot be gamed by running out of beds.
    values["boarding_hours_total"] = (
        sum(
            clip_seconds(decided, exit_at if exit_at is not None else window.end, m)
            for decided, exit_at in boarded
        )
        / 3600.0
    )

    return KpiVector(values=values)
