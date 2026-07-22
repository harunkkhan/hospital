"""``bottleneck.detect_bottleneck``/``gini`` — binding constraint + Gini."""

from __future__ import annotations

import math

from _analysis_fixtures import build_bottleneck_log, tiny_layout, tiny_roster

from hospital.analysis.bottleneck import detect_bottleneck, gini
from hospital.core import EventLog, hours


def test_engineered_queue_zone_is_binding_and_tops_resources() -> None:
    log = build_bottleneck_log()
    layout = tiny_layout()
    roster = tiny_roster()
    report = detect_bottleneck(log, layout, roster, warmup=hours(0))

    assert report.binding == "bay:resus_trauma"
    assert report.resources[0].resource == "bay:resus_trauma"
    # Fully ranked, descending by share_of_cycle.
    shares = [r.share_of_cycle for r in report.resources]
    assert shares == sorted(shares, reverse=True)

    resus = next(r for r in report.resources if r.resource == "bay:resus_trauma")
    general = next(r for r in report.resources if r.resource == "bay:general")
    assert math.isclose(resus.total_wait_s, 600.0)
    assert math.isclose(general.total_wait_s, 60.0)
    assert resus.n_requests == 1
    assert general.n_requests == 1


def test_share_of_cycle_in_unit_interval() -> None:
    log = build_bottleneck_log()
    layout = tiny_layout()
    roster = tiny_roster()
    report = detect_bottleneck(log, layout, roster, warmup=hours(0))
    for r in report.resources:
        if not math.isnan(r.share_of_cycle):
            assert 0.0 <= r.share_of_cycle <= 1.0


def test_no_patient_time_means_no_binding_constraint() -> None:
    """Regression (finding #6): with no patient-time in the window every
    share_of_cycle is NaN — there is no binding constraint, and reporting the
    alphabetically-first resource as binding would be an arbitrary verdict."""
    report = detect_bottleneck(EventLog(), tiny_layout(), tiny_roster(), warmup=hours(0))
    assert report.binding == ""
    assert report.total_cycle_s == 0.0
    assert all(math.isnan(r.share_of_cycle) for r in report.resources)


def test_gini_zero_for_equal_loads() -> None:
    assert gini([10.0, 10.0, 10.0]) == 0.0


def test_gini_near_one_when_one_does_everything() -> None:
    assert math.isclose(gini([0.0, 0.0, 100.0]), 1.0)


def test_gini_monotonic_as_load_concentrates() -> None:
    even = gini([34.0, 33.0, 33.0])
    mid = gini([50.0, 50.0, 0.0])
    extreme = gini([100.0, 0.0, 0.0])
    assert even < mid < extreme


def test_gini_degenerate_cases() -> None:
    assert gini([]) == 0.0
    assert gini([5.0]) == 0.0
    assert gini([0.0, 0.0, 0.0]) == 0.0
