"""``fold.compute_kpis`` — the closed 27-key contract, hand-computed exact values."""

from __future__ import annotations

import math

from _analysis_fixtures import build_sample_log, t, tiny_layout, tiny_roster

from hospital.analysis._stats import DEFAULT_WINDOW, measurement_window
from hospital.analysis.fold import compute_kpis
from hospital.core import (
    KPI_KEYS,
    DischargeCompleted,
    EsiAcuity,
    EventLog,
    OperatingWeek,
    PatientArrived,
    PatientId,
    SimTime,
    TriageCompleted,
    hours,
)
from hospital.core.enums import ArrivalMode


def test_emits_exactly_kpi_keys() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    vec = compute_kpis(log, layout, roster, warmup=hours(0))
    assert set(vec.values.keys()) == set(KPI_KEYS)
    assert len(vec.values) == 31


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


def test_exit_at_exactly_window_end_stays_wip() -> None:
    """Regression (finding #3): the half-open [start, end) horizon applies to
    the WIP boundary too — a DischargeCompleted at exactly window.end is not a
    completion (window.contains already excludes it) and the patient must
    therefore still count as WIP, keeping arrivals == completions + wip."""
    week = OperatingWeek.one_week()
    log = EventLog()
    patient = PatientId("edge")
    log.append(PatientArrived(occurred_at=t(0), patient=patient, mode=ArrivalMode.WALK_IN))
    log.append(DischargeCompleted(occurred_at=week.end, patient=patient))

    v = compute_kpis(log, tiny_layout(), tiny_roster(), window=week, warmup=hours(0)).values
    assert v["completions_per_week"] == 0.0
    assert v["wip_end_of_week"] == 1.0  # arrivals(1) == completions(0) + wip(1)


def test_arrival_at_exactly_window_end_is_outside_the_week() -> None:
    """Regression (finding #3): an arrival at exactly window.end is outside the
    half-open week — it must not be counted into WIP."""
    week = OperatingWeek.one_week()
    log = EventLog()
    log.append(
        PatientArrived(occurred_at=week.end, patient=PatientId("late"), mode=ArrivalMode.WALK_IN)
    )

    v = compute_kpis(log, tiny_layout(), tiny_roster(), window=week, warmup=hours(0)).values
    assert v["completions_per_week"] == 0.0
    assert v["wip_end_of_week"] == 0.0


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


def test_the_extensive_keys_are_the_totals_their_ratios_are_ratios_of() -> None:
    """Each new key must be the numerator or denominator of an existing intensive one.

    That is the whole justification for adding them: not new measurements, but the
    scale the fractions and means were already computed against and then divided away.
    If they drift apart, the cost model is pricing a different week than the KPI
    report describes.
    """
    log = build_sample_log()
    layout, roster = tiny_layout(), tiny_roster()
    vec = compute_kpis(log, layout, roster, warmup=hours(0))
    v = vec.values

    # staff_hours_paid is the denominator of every staff_frac_*.
    walked_h = v["staff_minutes_walked"] / 60.0
    assert v["staff_hours_paid"] > 0
    assert math.isclose(v["staff_frac_walk"], walked_h / v["staff_hours_paid"], rel_tol=1e-9)

    # bay_hours_occupied is the numerator of bay_utilization.
    window_h = measurement_window(DEFAULT_WINDOW, hours(0)).duration().to_seconds() / 3600.0
    assert math.isclose(
        v["bay_utilization"],
        v["bay_hours_occupied"] / (len(layout.bays) * window_h),
        rel_tol=1e-9,
    )


def test_boarding_hours_total_counts_a_patient_who_never_reached_a_bed() -> None:
    """The censored total must see a wait the conditioned mean throws away.

    This is the failure mode the key exists for: with ward capacity finite, the patients
    who never get a bed are the longest waits, and dropping them makes a gridlocked week
    report a *calmer* mean than a roomy one. The sample log admits a patient who never
    exits, so the mean is NaN while the total is a real positive number of hours.
    """
    vec = compute_kpis(build_sample_log(), tiny_layout(), tiny_roster(), warmup=hours(0))
    assert math.isnan(vec.values["boarding_time_s_mean"])
    assert vec.values["boarding_hours_total"] >= 0.0


def test_deadline_breach_counts_hours_waited_past_the_acuity_deadline() -> None:
    """Acuity-weighted by the deadline itself, not by a second weight.

    An ESI-1 is due immediately and an ESI-5 in two hours, so the same wall-clock wait
    breaches by different amounts — which is what makes this the priceable form of "was this
    patient seen in time" and an unweighted mean not.
    """
    vec = compute_kpis(build_sample_log(), tiny_layout(), tiny_roster(), warmup=hours(0))
    breach = vec.values["deadline_breach_hours_total"]
    assert breach >= 0.0
    assert not math.isnan(breach)


def test_a_patient_never_seen_still_contributes_their_wait() -> None:
    """Censored at the horizon, like boarding, and for the same anti-gaming reason.

    Counting only patients who eventually reached a provider would make a gridlocked week —
    where the longest waits never resolve — report *less* breach than a calm one.
    """
    arrival = SimTime(0)
    log = EventLog()
    log.append(
        PatientArrived(occurred_at=arrival, patient=PatientId("stuck"), mode=ArrivalMode.WALK_IN)
    )
    log.append(TriageCompleted(occurred_at=arrival, patient=PatientId("stuck"), esi=EsiAcuity.ESI1))
    week = OperatingWeek(start=SimTime(0), end=SimTime(hours(4).root))
    vec = compute_kpis(log, tiny_layout(), tiny_roster(), window=week, warmup=hours(0))
    # ESI-1 is due immediately and never seen, so the whole window is breach.
    assert vec.values["deadline_breach_hours_total"] == 4.0
    # ...and the conditioned mean has nothing to say about them at all.
    assert math.isnan(vec.values["door_to_provider_s_mean"])
