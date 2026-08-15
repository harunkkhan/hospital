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
from _api_fixtures import api_facility, api_scenario, api_staffing, api_workload

from hospital.api.sliders import compile_overrides
from hospital.core import SimTime, StaffRole, TimeWindow, ZoneType, hours
from hospital.data.layout import generate_floor
from hospital.data.scenario import (
    FacilitySpec,
    Scenario,
    ShiftBlock,
    StaffingSpec,
    ZoneQuota,
    realize_staff,
)


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


# ------------------------------------------------------------------------ capacity: bays
def _quota(scenario: Scenario, zone_type: ZoneType) -> int:
    return sum(q.bays for q in scenario.facility.zones if q.zone_type is zone_type)


def _repeated_zone_scenario() -> Scenario:
    """A floor that allocates ``general`` twice — the shape ``er_floor`` really has."""
    return Scenario(
        name="api_two_general",
        seed=7,
        facility=FacilitySpec(
            target_area_sqft=25_000,
            zones=(
                ZoneQuota(zone_type=ZoneType.GENERAL, bays=6, isolation_bays=2),
                ZoneQuota(zone_type=ZoneType.GENERAL, bays=2),
                ZoneQuota(zone_type=ZoneType.FAST_TRACK, bays=1),
            ),
            imaging_suites=1,
            lab_stations=1,
            triage_rooms=1,
        ),
        workload=api_workload(rate_per_hour=6.0, horizon_hours=2),
        staffing=api_staffing(),
    )


@pytest.mark.parametrize(
    ("key", "zone_type"),
    [
        ("facility.general_bays", ZoneType.GENERAL),
        ("facility.resus_bays", ZoneType.RESUS_TRAUMA),
        ("facility.fast_track_bays", ZoneType.FAST_TRACK),
    ],
)
def test_every_bay_slider_resizes_the_zone_it_names(key: str, zone_type: ZoneType) -> None:
    base = api_scenario()
    derived = compile_overrides(base, {key: 7})
    assert _quota(derived, zone_type) == 7
    # Only that zone type moved.
    for other in (ZoneType.GENERAL, ZoneType.RESUS_TRAUMA, ZoneType.FAST_TRACK):
        if other is not zone_type:
            assert _quota(derived, other) == _quota(base, other)


def test_bay_sliders_survive_each_other() -> None:
    """Same tuple-replacement hazard as the headcounts, on ``facility.zones``."""
    derived = compile_overrides(
        api_scenario(),
        {"facility.general_bays": 5, "facility.resus_bays": 3, "facility.fast_track_bays": 2},
    )
    assert _quota(derived, ZoneType.GENERAL) == 5
    assert _quota(derived, ZoneType.RESUS_TRAUMA) == 3
    assert _quota(derived, ZoneType.FAST_TRACK) == 2


def test_a_repeated_zone_type_is_apportioned_not_duplicated() -> None:
    """The slider states the floor's TOTAL for the type, across every zone of it.

    Writing the number into each matching zone would have doubled a floor that
    allocates ``general`` twice — which the committed ``er_floor`` does.
    """
    base = _repeated_zone_scenario()
    assert _quota(base, ZoneType.GENERAL) == 8

    grown = compile_overrides(base, {"facility.general_bays": 12})
    assert [q.bays for q in grown.facility.zones if q.zone_type is ZoneType.GENERAL] == [9, 3]
    assert _quota(grown, ZoneType.GENERAL) == 12


def test_restating_a_zone_types_current_size_rewrites_nothing() -> None:
    """The fixed point the panel's "= base" indicator depends on."""
    base = _repeated_zone_scenario()
    restated = compile_overrides(base, {"facility.general_bays": _quota(base, ZoneType.GENERAL)})
    assert restated.facility.zones == base.facility.zones


def test_shrinking_a_zone_shrinks_its_isolation_subset() -> None:
    """``isolation_bays <= bays`` is a ZoneQuota invariant; the subset follows the
    zone rather than rejecting an edit the operator cannot act on."""
    base = _repeated_zone_scenario()
    shrunk = compile_overrides(base, {"facility.general_bays": 1})
    general = [q for q in shrunk.facility.zones if q.zone_type is ZoneType.GENERAL]
    assert [q.bays for q in general] == [1, 0]
    assert [q.isolation_bays for q in general] == [1, 0]


def test_a_missing_zone_is_opened_only_when_asked_for_bays() -> None:
    """The tiny fixture has no observation zone, so the slider must create one —
    and must not litter the tuple with an empty quota for a request of zero."""
    base = api_scenario()
    assert _quota(base, ZoneType.OBSERVATION) == 0

    opened = compile_overrides(base, {"facility.observation_bays": 4})
    assert _quota(opened, ZoneType.OBSERVATION) == 4

    untouched = compile_overrides(base, {"facility.observation_bays": 0})
    assert untouched.facility.zones == base.facility.zones


def test_a_floor_may_not_be_emptied_of_bays() -> None:
    """Not an API rule: ``FacilitySpec`` requires at least one bay in total."""
    with pytest.raises(ValueError, match="at least one bay"):
        compile_overrides(
            api_scenario(),
            {
                "facility.general_bays": 0,
                "facility.resus_bays": 0,
                "facility.fast_track_bays": 0,
            },
        )
