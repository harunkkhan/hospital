"""Time: microsecond banker's rounding, no-drift, and half-open week windows."""

from __future__ import annotations

from hospital.core import (
    MICROS_PER_SEC,
    Duration,
    OperatingWeek,
    SimTime,
    TimeWindow,
    hours,
    minutes,
    round_micros,
    seconds,
)


def test_exact_unit_conversions() -> None:
    assert seconds(1).root == MICROS_PER_SEC
    assert seconds(3).root == 3 * MICROS_PER_SEC
    assert minutes(1).root == 60 * MICROS_PER_SEC
    assert hours(1).root == 3_600 * MICROS_PER_SEC


def test_bankers_rounding_half_to_even() -> None:
    # 0.5 µs -> 0 (even), 1.5 µs -> 2 (even), 2.5 µs -> 2 (even).
    assert seconds(0.0000005).root == 0
    assert seconds(0.0000015).root == 2
    assert seconds(0.0000025).root == 2
    assert round_micros(0.5) == 0
    assert round_micros(1.5) == 2
    assert round_micros(2.5) == 2
    assert round_micros(3.5) == 4


def test_no_drift_across_many_small_conversions() -> None:
    # 1000 increments of 1 ms sum exactly to 1 s — no per-conversion drift.
    total = Duration(0)
    for _ in range(1000):
        total = total + seconds(0.001)
    assert total == seconds(1)
    assert total.root == MICROS_PER_SEC


def test_typed_time_algebra() -> None:
    t0 = SimTime(0)
    t1 = SimTime(500)
    delta = t1 - t0
    assert isinstance(delta, Duration)
    assert delta.root == 500
    assert (t0 + delta) == t1
    assert (delta + delta).root == 1000
    assert (delta - Duration(200)).root == 300


def test_operating_week_is_half_open() -> None:
    week = OperatingWeek.one_week()
    assert week.start.root == 0
    assert week.end.root == hours(7 * 24).root
    assert week.contains(SimTime(0))
    assert week.contains(SimTime(week.end.root - 1))
    # The exact end instant is NOT contained — a discharge at `end` stays WIP.
    assert not week.contains(week.end)


def test_time_window_half_open_and_duration() -> None:
    w = TimeWindow(start=SimTime(100), end=SimTime(400))
    assert w.duration().root == 300
    assert w.contains(SimTime(100))
    assert w.contains(SimTime(399))
    assert not w.contains(SimTime(400))
    assert not w.contains(SimTime(50))
