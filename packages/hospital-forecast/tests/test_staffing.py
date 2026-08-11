"""Arrival intensity -> role demand -> a solved roster: the predict-then-staff join."""

from __future__ import annotations

import itertools
import math

import pytest

from hospital.core import Duration, OperatingWeek, SimTime, StaffRole, TimeWindow, hours, minutes
from hospital.forecast.arrivals import ArrivalIntensityModel, intensity_from_rates
from hospital.forecast.staffing import block_grid, role_demand

_WEEK = OperatingWeek.one_week()
_HOURS_IN_WEEK = 7 * 24


def _flat(rate_per_hour: float) -> ArrivalIntensityModel:
    """A perfectly flat weekly intensity — one λ per hour, all equal."""
    return intensity_from_rates(
        dict.fromkeys(range(_HOURS_IN_WEEK), rate_per_hour),
        resolution=hours(1),
        n_bins=_HOURS_IN_WEEK,
    )


def test_the_block_grid_spans_the_week_without_gaps_or_overlap() -> None:
    """Both sides of the join index into this grid, so it has to be a real partition."""
    blocks = block_grid(_WEEK, hours(4))
    assert blocks[0].start == _WEEK.start
    assert blocks[-1].end == _WEEK.end
    for earlier, later in itertools.pairwise(blocks):
        assert earlier.end == later.start


def test_a_week_that_does_not_divide_evenly_keeps_its_last_partial_block() -> None:
    """Dropping the remainder would leave the end of the week unstaffed by construction."""
    blocks = block_grid(_WEEK, hours(5))  # 168 is not a multiple of 5
    assert blocks[-1].end == _WEEK.end
    assert blocks[-1].end.root - blocks[-1].start.root < hours(5).root
    covered = sum(b.end.root - b.start.root for b in blocks)
    assert covered == _WEEK.end.root - _WEEK.start.root


def test_a_zero_block_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        block_grid(_WEEK, Duration(0))


def test_demand_is_the_offered_load_rounded_up() -> None:
    """Hand-computed: 6 arrivals/h x 30 provider-min each, over a 1h block, is 3 providers."""
    blocks = block_grid(_WEEK, hours(1))
    demand = role_demand(
        _flat(6.0),
        blocks,
        _WEEK,
        minutes_per_patient={StaffRole.PHYSICIAN: 30.0},
    )
    assert demand[(StaffRole.PHYSICIAN, 0)] == 3
    assert all(v == 3 for v in demand.values())  # flat intensity -> flat demand


def test_rounding_up_never_understaffs_the_mean() -> None:
    """Flooring would produce a roster that provably cannot serve the forecast."""
    blocks = block_grid(_WEEK, hours(1))
    # 5 arrivals/h x 25 min = 125 role-min per 60-min block = 2.083... -> 3
    demand = role_demand(_flat(5.0), blocks, _WEEK, minutes_per_patient={StaffRole.NURSE: 25.0})
    assert demand[(StaffRole.NURSE, 0)] == 3
    assert demand[(StaffRole.NURSE, 0)] >= math.ceil(5.0 * 25.0 / 60.0)


def test_a_busier_forecast_demands_more_staff() -> None:
    """The whole point of driving staffing from a forecast rather than a constant."""
    blocks = block_grid(_WEEK, hours(1))
    budget = {StaffRole.NURSE: 30.0}
    quiet = role_demand(_flat(2.0), blocks, _WEEK, minutes_per_patient=budget)
    busy = role_demand(_flat(20.0), blocks, _WEEK, minutes_per_patient=budget)
    assert busy[(StaffRole.NURSE, 0)] > quiet[(StaffRole.NURSE, 0)]


def test_a_time_varying_forecast_produces_a_time_varying_roster_need() -> None:
    """A flat demand curve would make the whole exercise pointless.

    The night hours here are quiet and the day hours busy, and the demand must track that
    per block — this is the peak the covering MIP later has to cover without over-staffing
    the trough.
    """
    rates = {h: (12.0 if 8 <= (h % 24) < 20 else 1.0) for h in range(_HOURS_IN_WEEK)}
    model = intensity_from_rates(rates, resolution=hours(1), n_bins=_HOURS_IN_WEEK)
    blocks = block_grid(_WEEK, hours(1))
    demand = role_demand(model, blocks, _WEEK, minutes_per_patient={StaffRole.NURSE: 30.0})
    day = demand[(StaffRole.NURSE, 9)]
    night = demand[(StaffRole.NURSE, 3)]
    assert day > night >= 1


def test_safety_buys_headroom_and_is_never_implicit() -> None:
    """Staffing exactly at offered load has unbounded expected wait; that is the caller's
    call to make, not a fudge factor buried in the conversion."""
    blocks = block_grid(_WEEK, hours(1))
    budget = {StaffRole.NURSE: 30.0}
    base = role_demand(_flat(6.0), blocks, _WEEK, minutes_per_patient=budget)
    padded = role_demand(_flat(6.0), blocks, _WEEK, minutes_per_patient=budget, safety=1.5)
    assert padded[(StaffRole.NURSE, 0)] > base[(StaffRole.NURSE, 0)]

    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="safety"):
            role_demand(_flat(6.0), blocks, _WEEK, minutes_per_patient=budget, safety=bad)


def test_an_undescribed_role_demands_nobody() -> None:
    """A role the caller said nothing about gets no invented figure."""
    blocks = block_grid(_WEEK, hours(1))
    demand = role_demand(
        _flat(6.0),
        blocks,
        _WEEK,
        minutes_per_patient={StaffRole.NURSE: 30.0, StaffRole.PORTER: 0.0},
    )
    assert (StaffRole.NURSE, 0) in demand
    assert not any(role is StaffRole.PORTER for role, _ in demand)
    assert not any(role is StaffRole.HOUSEKEEPING for role, _ in demand)


def test_demand_is_deterministic() -> None:
    blocks = block_grid(_WEEK, hours(2))
    budget = {StaffRole.NURSE: 30.0, StaffRole.PHYSICIAN: 20.0}
    model = _flat(7.0)
    assert role_demand(model, blocks, _WEEK, minutes_per_patient=budget) == role_demand(
        model, blocks, _WEEK, minutes_per_patient=budget
    )


def test_a_short_block_is_skipped_rather_than_dividing_by_zero() -> None:
    """A degenerate window contributes nothing instead of raising deep in the arithmetic."""
    zero = TimeWindow(start=SimTime(0), end=SimTime(0))
    assert (
        role_demand(_flat(6.0), (zero,), _WEEK, minutes_per_patient={StaffRole.NURSE: 30.0}) == {}
    )
    # ...and a sub-hour block still demands proportionally.
    short = TimeWindow(start=SimTime(0), end=SimTime(minutes(30).root))
    demand = role_demand(_flat(6.0), (short,), _WEEK, minutes_per_patient={StaffRole.NURSE: 30.0})
    assert demand[(StaffRole.NURSE, 0)] == 3  # 3 arrivals in 30 min x 30 min / 30 min
