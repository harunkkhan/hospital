"""``utilization.classify_staff_seconds``/``utilization_report`` — residual partition."""

from __future__ import annotations

import math

from _analysis_fixtures import (
    HOUSEKEEPER,
    NURSE,
    PHYSICIAN,
    build_sample_log,
    t,
    tiny_layout,
    tiny_roster,
)

from hospital.analysis._index import build_index
from hospital.analysis.utilization import classify_staff_seconds, utilization_report
from hospital.core import (
    EventLog,
    OperatingWeek,
    PatientArrived,
    PatientId,
    ProviderVisitStarted,
    StaffMember,
    hours,
)
from hospital.core.enums import ArrivalMode


def test_fractions_sum_to_one() -> None:
    log = build_sample_log()
    roster = tiny_roster()
    report = utilization_report(log, roster, warmup=hours(0))
    total = math.fsum(report.fractions.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_no_negative_durations_and_idle_is_residual() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    idx = build_index(log, layout, roster)
    window = OperatingWeek.one_week()
    budgets = classify_staff_seconds(idx, roster, window=window, warmup=hours(0))
    for b in budgets:
        assert b.walk_s >= 0.0
        assert b.direct_care_s >= 0.0
        assert b.cleaning_s >= 0.0
        assert b.documentation_s >= 0.0
        assert b.idle_s >= 0.0
        residual = b.on_shift_s - b.walk_s - b.direct_care_s - b.cleaning_s - b.documentation_s
        assert math.isclose(b.idle_s, residual, rel_tol=1e-9, abs_tol=1e-6)


def test_hand_computed_per_staff_budgets() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    idx = build_index(log, layout, roster)
    window = OperatingWeek.one_week()
    budgets = {
        b.staff: b for b in classify_staff_seconds(idx, roster, window=window, warmup=hours(0))
    }

    phys = budgets[PHYSICIAN]
    assert math.isclose(phys.walk_s, 50.0)  # 20 + 30
    # P1's completed visit (300s) + P2's visit still open at the horizon,
    # clipped to the window end (finding #2): 604800 - 2300 = 602500s.
    assert math.isclose(phys.direct_care_s, 300.0 + (604800.0 - 2300.0))

    nurse = budgets[NURSE]
    assert math.isclose(nurse.walk_s, 15.0)
    # triage(p1)=240 + triage(p2)=50 + nurse_visit(p1)=50
    assert math.isclose(nurse.direct_care_s, 340.0)
    assert math.isclose(nurse.documentation_s, 30.0)

    hk = budgets[HOUSEKEEPER]
    assert math.isclose(hk.walk_s, 25.0)
    assert math.isclose(hk.cleaning_s, 300.0)


def test_open_service_at_horizon_counts_as_direct_care_not_idle() -> None:
    """Regression (finding #2): a provider visit still open at run end must be
    clipped at the measurement horizon, not lost from the classification — a
    horizon-censored run would otherwise count in-progress service as idle and
    understate utilization."""
    log = EventLog()
    patient = PatientId("wip")
    log.append(PatientArrived(occurred_at=t(0), patient=patient, mode=ArrivalMode.WALK_IN))
    log.append(ProviderVisitStarted(occurred_at=t(100), patient=patient, staff=PHYSICIAN))

    report = utilization_report(log, tiny_roster(), warmup=hours(0))
    phys = next(b for b in report.per_staff if b.staff == PHYSICIAN)
    week_s = 7 * 24 * 3600.0
    assert math.isclose(phys.direct_care_s, week_s - 100.0)
    assert math.isclose(phys.idle_s, 100.0 - phys.walk_s)
    assert report.util_by_role["provider_util"] > 0.99


def test_empty_roster_falls_back_to_all_idle() -> None:
    log = build_sample_log()
    roster: tuple[StaffMember, ...] = ()
    report = utilization_report(log, roster, warmup=hours(0))
    assert report.fractions["staff_frac_idle"] == 1.0
    total = math.fsum(report.fractions.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)
