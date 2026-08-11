"""Cost model interface and rate vocabulary (M4b).

Cost is a pure function of a :class:`~hospital.core.kpi.KpiVector`, which keeps money
entirely out of ``sim``/``solver`` (clean separation, and why cost could wait until the
analytics were observable). This module owns the vocabulary — :class:`Money`, the
:class:`CostRates` a scenario states, and the :class:`CostModel` Protocol; the
arithmetic that folds a week into a number lives in ``analysis.cost``, beside the KPI
fold it prices.

The risk this module flagged from M1 came true exactly as written: some cost drivers
needed signals absent from ``KPI_KEYS``. Every resource KPI through M3 was a *fraction*
or a *mean*, and neither can be multiplied into money — "6% of staff time was spent
walking" does not say how many hours. ``core.kpi.EXTENSIVE_KEYS`` was the versioned
answer, added deliberately rather than worked around.

**No default rates, and that is the design.** What an hour of nursing or a boarded hour
costs is a property of a particular hospital in a particular market, not of a simulator,
and a plausible-looking built-in default would be quoted back as though the model knew.
Every field of :class:`CostRates` is required, so a scenario either states its own prices
or reports time only — which is what every scenario committed so far does.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field, RootModel

from hospital.core.kpi import KpiVector
from hospital.core.models import FrozenModel


class Money(RootModel[int]):
    """A monetary amount in integer minor units (e.g. cents).

    Integer, like every other quantity that crosses a boundary here: a week's cost is
    summed from millions of terms, and float accumulation would make the total depend on
    summation order — the same reason time is microseconds and distance centimetres.
    """

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.root)


class CostRates(FrozenModel):
    """What a hospital pays, per unit of the thing the KPI fold measures.

    All in integer minor units (cents), all required — see the module docstring on why
    there are no defaults. Each rate names the ``KPI_KEYS`` entry it multiplies, so the
    contract between the fold and the price is legible in one place and a KPI that is
    renamed cannot leave a rate quietly multiplying nothing.
    """

    # x staff_hours_paid — the whole wage bill, blended across roles. Walking, direct
    # care, cleaning, and idle are all *inside* this: the staff_frac_* decomposition
    # says how it was spent, and pricing those separately would double-count it.
    staff_hour_cents: int = Field(ge=0)
    # x bay_hours_occupied — the marginal cost of holding a bed (consumables, linen,
    # utilities, turnover), not its capital cost.
    bay_hour_cents: int = Field(ge=0)
    # x boarding_hours_total — what an admitted patient held in the ED costs beyond the
    # bed itself: the crowding it imposes on everyone else. A penalty, not a purchase.
    boarding_hour_cents: int = Field(ge=0)
    # x wip_end_of_week — work carried past the horizon rather than finished in it.
    # Non-zero is what stops a policy looking cheap by deferring everyone to next week.
    wip_carry_cents: int = Field(ge=0)
    # x completions_per_week — earned, so it enters the total negatively. Set to 0 to
    # price pure operating cost with no revenue side.
    completion_revenue_cents: int = Field(ge=0)


@runtime_checkable
class CostModel(Protocol):
    """Prices a KPI reading into :class:`Money`. Implemented by ``analysis.cost``."""

    def price(self, kpis: KpiVector) -> Money: ...


__all__ = ["CostModel", "CostRates", "Money"]
