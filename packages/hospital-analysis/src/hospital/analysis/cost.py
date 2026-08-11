"""``WeeklyCost`` — the deferred money layer, folding a KPI week into a number (M4b).

The last thing the roadmap asked for: translate saved time and utilization into dollars.
It lands here rather than in ``core`` because pricing is *analysis* — it reads the KPI
vector the fold produces and nothing else — and it lands after everything else because
until the extensive keys existed there was nothing multipliable to price. A fraction and
a mean are both unpriceable; ``core.kpi.EXTENSIVE_KEYS`` is what made this module
possible, and adding them was the versioned KPI change ``core.cost`` predicted in M1.

The model is a weekly operating P&L, lower is better::

    total = labour + capacity + boarding + waiting + carried backlog - revenue

Four deliberate choices, each of which could reasonably have gone the other way:

* **Labour is the whole wage bill, priced once.** ``staff_hours_paid`` already contains
  the walking, the direct care, the cleaning, and the idling; the ``staff_frac_*`` keys
  say how it was *spent*, and pricing any of them on top would double-count hours the
  hospital pays for exactly once. :func:`walking_cost` exists to answer "what did the
  transit inside that bill cost" and is deliberately NOT a term in the total.
* **Staffing is a scenario input in v1, so labour is identical across arms and cancels
  in any paired comparison.** That is not a defect to hide — it is the honest statement
  that a policy which cannot change the roster cannot change the wage bill. The cost
  *difference* between two arms comes from the beds they hold, the boarding they cause,
  the backlog they carry, and the patients they finish.
* **Boarding is a penalty, not a purchase.** The bed an admitted patient occupies is
  already paid for under ``bay_hours_occupied``; ``boarding_hour_cents`` prices the
  crowding their holding an ED bay imposes on everyone else. Counting it twice would be
  wrong; counting it not at all would make the whole M4 mechanism free.
* **Waiting is priced, because not pricing it inverted the model's advice.** With labour and
  beds on the books and patient delay absent, the cheapest roster is always the thinnest
  one — measured, in the staffing loop, as a 53% worse door-to-provider being *rewarded*.
  ``deadline_breach_hours_total`` is the term that closes it, and it is the right one to
  price because the deadline is already acuity-relative: an hour of an ESI-1 waiting
  breaches where an hour of an ESI-5 does not. Boarding stays separate — that is delay
  *after* a disposition, in a bed, and double-counting it as breach would charge one wait
  twice.
* **Revenue is subtracted, not maximized separately.** A single scalar is what makes
  cost rankable against the objective; a two-vector "cost and revenue" answer would need
  a rate to trade them, which is the thing we would be pretending not to choose.

Rounding is banker's-free and explicit: every term is computed in cents as an integer
via :func:`round`, then summed. Hours are floats out of the fold, so a term is rounded
once at its own boundary rather than the total being rounded at the end — that keeps a
term's printed value equal to its contribution.

NaN is contagious and refused rather than coerced. A KPI that is NaN means "no data",
and a hospital week with no data has no honest price; silently reading it as zero would
report a free week.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hospital.core import CostRates, KpiVector, Money

__all__ = ["CostBreakdown", "WeeklyCost", "walking_cost"]


class CostBreakdown(dict[str, int]):
    """The priced terms, in cents, keyed by name. Sums to the total by construction."""

    @property
    def total(self) -> Money:
        return Money(sum(self.values()))


def _finite(kpis: KpiVector, key: str) -> float:
    value = kpis.values[key]
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"cannot price a week whose {key} is {value}")
    return value


@dataclass(frozen=True)
class WeeklyCost:
    """A :class:`~hospital.core.CostModel` over one week's :class:`KpiVector`."""

    rates: CostRates

    def breakdown(self, kpis: KpiVector) -> CostBreakdown:
        """Every term separately, so a total can be read rather than trusted.

        Revenue is negative by construction: the total is a cost, and a week that earns
        more than it spends is a negative number rather than a separate sign convention
        the caller has to remember.
        """
        r = self.rates
        return CostBreakdown(
            {
                "labour": round(_finite(kpis, "staff_hours_paid") * r.staff_hour_cents),
                "capacity": round(_finite(kpis, "bay_hours_occupied") * r.bay_hour_cents),
                "boarding": round(_finite(kpis, "boarding_hours_total") * r.boarding_hour_cents),
                "waiting": round(
                    _finite(kpis, "deadline_breach_hours_total") * r.deadline_breach_hour_cents
                ),
                "backlog": round(_finite(kpis, "wip_end_of_week") * r.wip_carry_cents),
                "revenue": -round(
                    _finite(kpis, "completions_per_week") * r.completion_revenue_cents
                ),
            }
        )

    def price(self, kpis: KpiVector) -> Money:
        """The single scalar — implements ``core.CostModel``."""
        return self.breakdown(kpis).total


def walking_cost(kpis: KpiVector, rates: CostRates) -> Money:
    """What the walking inside the wage bill cost — a decomposition, not a term.

    The headline the whole project exists to produce: staff-minutes walked are paid
    minutes that produced no care, so this is "saved time, in dollars" stated directly.
    It is deliberately not part of :meth:`WeeklyCost.price` — those minutes are already
    inside ``labour`` — and reporting it beside the total is what keeps that clear.
    """
    hours = _finite(kpis, "staff_minutes_walked") / 60.0
    return Money(round(hours * rates.staff_hour_cents))
