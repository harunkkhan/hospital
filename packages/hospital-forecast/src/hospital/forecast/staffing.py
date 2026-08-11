"""Arrival intensity -> per-role staffing demand — the predict half of predict-then-staff.

The spec's §10 says the arrival model drives "staffing and capacity planning", and §7 says
staff scheduling is "later solved from the demand forecast". This module is the join: it
turns λ(hour-of-day, day-of-week) into the ``{(role, block): headcount}`` mapping
``solver.scheduling.solve_coverage`` covers.

It lives in ``forecast`` rather than beside the MIP because ``forecast -> core, data`` and
**cannot import the solver** — an architectural fact, not a preference. The output is
therefore a plain mapping of core types, threaded to the solver by a composition root.

### The conversion, and what it does and does not claim

Offered load. Expected arrivals in a block, times the minutes one patient consumes of a
role, is role-minutes of work; divided by the block's own minutes, that is the headcount
needed to keep up. ``ArrivalIntensityModel.expected_arrivals`` is additive over adjacent
windows by construction, which is exactly what makes a block's expectation well-defined.

**This is a deterministic-flow approximation and it is a floor, not a target.** It staffs
the *mean*. Real queues are variable, and a server pool staffed exactly at its offered load
has unbounded expected wait — the classic result that makes square-root-staffing rules
exist. Rather than bury a fudge factor, ``safety`` is an explicit multiplier the caller
chooses and the scenario records: at 1.0 you get the mean-load floor and should expect
queues, and the surge forecaster's own quantile band (§10) is the other, data-driven way to
buy the same headroom.

Rounding is **up**, and only at the end. Half a nurse cannot be rostered, and flooring
would produce a roster that provably cannot serve the mean.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from hospital.core import Duration, OperatingWeek, SimTime, StaffRole, TimeWindow
from hospital.forecast.arrivals import ArrivalIntensityModel

__all__ = ["block_grid", "role_demand"]

_MICROS_PER_MINUTE = 60 * 1_000_000


def block_grid(week: OperatingWeek, block: Duration) -> tuple[TimeWindow, ...]:
    """Partition ``week`` into equal ``block``-long windows, the last one truncated.

    The grid both sides of the join must agree on: ``role_demand`` keys its output by
    index into it and ``solve_coverage`` takes it verbatim, so neither can invent its own
    idea of what block 7 spans. A final partial block is kept rather than dropped — a week
    that does not divide evenly still has to be staffed to its end.
    """
    if block.root <= 0:
        raise ValueError("block duration must be positive")
    out: list[TimeWindow] = []
    cursor = week.start.root
    while cursor < week.end.root:
        end = min(cursor + block.root, week.end.root)
        out.append(TimeWindow(start=SimTime(cursor), end=SimTime(end)))
        cursor = end
    return tuple(out)


def role_demand(
    model: ArrivalIntensityModel,
    blocks: tuple[TimeWindow, ...],
    week: OperatingWeek,
    *,
    minutes_per_patient: Mapping[StaffRole, float],
    safety: float = 1.0,
) -> dict[tuple[StaffRole, int], int]:
    """Headcount per ``(role, block index)`` implied by the forecast arrival rate.

    ``minutes_per_patient`` is how long one patient occupies one member of a role across
    their whole stay — a per-role service-time budget, which the caller owns because it
    belongs to the *physics* (``sim``'s service-time table), and ``forecast`` cannot import
    that any more than it can import the solver.

    A role with a zero or absent budget demands nobody, rather than defaulting to some
    invented figure: a role the caller did not describe is one this conversion has nothing
    to say about.
    """
    if safety <= 0.0 or not math.isfinite(safety):
        raise ValueError(f"safety must be a positive finite multiplier, got {safety}")
    demand: dict[tuple[StaffRole, int], int] = {}
    for index, block in enumerate(blocks):
        block_minutes = (block.end.root - block.start.root) / _MICROS_PER_MINUTE
        if block_minutes <= 0:
            continue
        arrivals = model.expected_arrivals(block, week)
        for role in sorted(minutes_per_patient, key=lambda r: r.value):
            per_patient = minutes_per_patient[role]
            if per_patient <= 0.0:
                continue
            needed = arrivals * per_patient * safety / block_minutes
            headcount = math.ceil(needed - 1e-9)
            if headcount > 0:
                demand[(role, index)] = headcount
    return demand
