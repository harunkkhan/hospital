"""The Scenario Lab's vocabulary: every knob compiles to the field it names.

Judged by the *effect* on the derived ``Scenario``, never by the overlay the
alias wrote: a key that validates but changes nothing is exactly the failure a
schema-shaped assertion misses. A headcount is therefore judged through
``realize_staff`` — the committed ``er_floor`` schedules 14 nurses in its blocks
and carries 7 in its defaults, so the spec a slider writes and the roster a run
realizes are genuinely different questions.
"""

from __future__ import annotations

from collections import Counter

import pytest
from _api_fixtures import api_facility, api_workload

from hospital.api.sliders import compile_overrides
from hospital.core import SimTime, StaffRole, TimeWindow, hours
from hospital.data.layout import generate_floor
from hospital.data.scenario import Scenario, ShiftBlock, StaffingSpec, realize_staff


def _roster(scenario: Scenario) -> Counter[StaffRole]:
    """What the engine will actually put on the floor for the whole horizon."""
    horizon = scenario.workload.horizon
    window = TimeWindow(start=horizon.start, end=horizon.end)
    layout = generate_floor(scenario.facility)
    return Counter(m.role for m in realize_staff(scenario.staffing, layout, window))


def _shift_scenario() -> Scenario:
    """``er_floor``'s shape in miniature: blocks that schedule, defaults that do not.

    The committed reference carries a *lower* ``default_counts`` than its blocks
    (7 nurses vs 14), and ``realize_staff`` takes the max over overlapping blocks
    — so writing only the defaults realizes the base's staffing, unchanged.
    """
    window = TimeWindow(start=SimTime(0), end=SimTime(hours(2).root))
    return Scenario(
        name="api_shifted",
        seed=7,
        facility=api_facility(),
        workload=api_workload(rate_per_hour=6.0, horizon_hours=2),
        staffing=StaffingSpec(
            blocks=(
                ShiftBlock(
                    window=window,
                    role_counts={
                        StaffRole.PHYSICIAN: 4,
                        StaffRole.NURSE: 14,
                        StaffRole.TECH: 6,
                        StaffRole.PORTER: 3,
                        StaffRole.HOUSEKEEPING: 3,
                    },
                ),
            ),
            default_counts={StaffRole.NURSE: 7, StaffRole.PHYSICIAN: 3},
        ),
    )


@pytest.mark.parametrize(
    ("role", "count"),
    [
        (StaffRole.PHYSICIAN, 5),
        (StaffRole.NURSE, 9),
        (StaffRole.TECH, 4),
        (StaffRole.PORTER, 2),
        (StaffRole.HOUSEKEEPING, 6),
    ],
)
def test_every_headcount_slider_changes_the_realized_roster(role: StaffRole, count: int) -> None:
    """Every role ``realize_staff`` can roster is adjustable, not just the clinical two.

    Housekeeping and porters are the ones a capacity question actually needs:
    they turn bays over and move people, so they gate bed availability rather
    than clinical throughput.
    """
    base = _shift_scenario()
    derived = compile_overrides(base, {f"staffing.{role.value}_count": count})
    realized = _roster(derived)
    assert realized[role] == count
    # Untouched roles keep the base's realized counts.
    for other, before in _roster(base).items():
        if other is not role:
            assert realized[other] == before


def test_every_headcount_slider_survives_the_others() -> None:
    """All five at once: each rebuilds the whole ``blocks`` tuple, which replaces
    wholesale under ``_deep_merge`` — merged as independent fragments, four of
    the five would silently vanish."""
    base = _shift_scenario()
    wanted = {
        StaffRole.PHYSICIAN: 8,
        StaffRole.NURSE: 20,
        StaffRole.TECH: 2,
        StaffRole.PORTER: 5,
        StaffRole.HOUSEKEEPING: 4,
    }
    derived = compile_overrides(
        base, {f"staffing.{role.value}_count": n for role, n in wanted.items()}
    )
    assert _roster(derived) == Counter(wanted)


def test_a_fractional_headcount_is_refused_rather_than_truncated() -> None:
    with pytest.raises(ValueError, match="whole, finite count"):
        compile_overrides(_shift_scenario(), {"staffing.tech_count": 2.5})
