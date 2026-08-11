"""The money layer: what a week costs, and what the arithmetic refuses to claim (M4b)."""

from __future__ import annotations

import math

import pytest
from _analysis_fixtures import full_kpi_values

from hospital.analysis.cost import WeeklyCost, walking_cost
from hospital.core import CostModel, CostRates, KpiVector

# Deliberately round, deliberately not defaults: these are a *test's* prices, and
# `CostRates` ships none of its own so no number here can be mistaken for the model's
# opinion about what nursing costs.
RATES = CostRates(
    staff_hour_cents=6_000,
    bay_hour_cents=1_000,
    boarding_hour_cents=5_000,
    wip_carry_cents=20_000,
    completion_revenue_cents=100_000,
)


def _kpis(**overrides: float) -> KpiVector:
    base = full_kpi_values(
        staff_hours_paid=1_000.0,
        bay_hours_occupied=500.0,
        boarding_hours_total=100.0,
        wip_end_of_week=10.0,
        completions_per_week=800.0,
        staff_minutes_walked=6_000.0,
    )
    base.update(overrides)
    return KpiVector(values=base)


def test_weekly_cost_implements_the_core_protocol() -> None:
    """`core.cost` declared the shape in M1; this is the implementation arriving."""
    assert isinstance(WeeklyCost(rates=RATES), CostModel)


def test_the_total_is_exactly_the_sum_of_its_terms() -> None:
    """A breakdown that does not add up is worse than no breakdown."""
    model = WeeklyCost(rates=RATES)
    kpis = _kpis()
    parts = model.breakdown(kpis)
    assert sum(parts.values()) == model.price(kpis).root
    assert parts == {
        "labour": 6_000_000,  # 1000 h x $60
        "capacity": 500_000,  # 500 bed-h x $10
        "boarding": 500_000,  # 100 boarded-h x $50
        "backlog": 200_000,  # 10 carried x $200
        "revenue": -80_000_000,  # 800 completions x $1000, earned
    }
    assert model.price(kpis).root == -72_800_000


def test_walking_is_priced_inside_labour_and_not_added_twice() -> None:
    """The one double-count this model could plausibly make, pinned shut.

    Staff-minutes walked are paid minutes: they are already inside `staff_hours_paid`.
    `walking_cost` answers "what did the transit cost" as a decomposition, so it must be
    a strict part of the labour term and must not move the total.
    """
    model = WeeklyCost(rates=RATES)
    kpis = _kpis()
    walk = walking_cost(kpis, RATES)
    assert walk.root == 600_000  # 100 h walked x $60
    assert 0 < walk.root < model.breakdown(kpis)["labour"]

    # Walking more, with every other KPI held fixed, does not change the priced total.
    walked_more = _kpis(staff_minutes_walked=12_000.0)
    assert model.price(walked_more) == model.price(kpis)
    assert walking_cost(walked_more, RATES).root == 2 * walk.root


def test_boarding_is_what_makes_a_crowded_week_cost_more() -> None:
    """The M4 mechanism has to show up in the money, or pricing it was pointless."""
    model = WeeklyCost(rates=RATES)
    calm = model.price(_kpis(boarding_hours_total=20.0)).root
    crowded = model.price(_kpis(boarding_hours_total=400.0)).root
    assert crowded > calm
    assert crowded - calm == (400 - 20) * RATES.boarding_hour_cents


def test_carrying_backlog_is_not_free() -> None:
    """Otherwise a policy books a cheap week by deferring everyone into the next one."""
    model = WeeklyCost(rates=RATES)
    finished = model.price(_kpis(wip_end_of_week=0.0)).root
    deferred = model.price(_kpis(wip_end_of_week=50.0)).root
    assert deferred > finished


def test_a_week_with_no_data_has_no_price() -> None:
    """NaN means "no data", and coercing it to zero would report a free week."""
    model = WeeklyCost(rates=RATES)
    with pytest.raises(ValueError, match="staff_hours_paid"):
        model.price(_kpis(staff_hours_paid=math.nan))


def test_zero_rates_price_a_week_at_nothing() -> None:
    """The degenerate configuration is well-defined rather than a special case.

    It is also what a scenario gets if it opts into cost and then declines to say what
    anything costs — a zero, not a plausible-looking invented figure.
    """
    free = CostRates(
        staff_hour_cents=0,
        bay_hour_cents=0,
        boarding_hour_cents=0,
        wip_carry_cents=0,
        completion_revenue_cents=0,
    )
    assert WeeklyCost(rates=free).price(_kpis()).root == 0


def test_rates_must_be_stated_and_non_negative() -> None:
    """No defaults, by design: a built-in dollar figure would be quoted as fact."""
    with pytest.raises(ValueError):
        CostRates()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        CostRates(
            staff_hour_cents=-1,
            bay_hour_cents=0,
            boarding_hour_cents=0,
            wip_carry_cents=0,
            completion_revenue_cents=0,
        )
