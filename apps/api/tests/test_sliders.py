"""The Scenario Lab's vocabulary: every knob compiles to (and reads back from)
the field it names.

**Compile** is judged by the *effect* on the derived ``Scenario``, never by the
overlay the alias wrote: a key that validates but changes nothing is exactly the
failure a schema-shaped assertion misses. A headcount is therefore judged through
``realize_staff`` — the committed ``er_floor`` schedules 14 nurses in its blocks
and carries 7 in its defaults, so the spec a slider writes and the roster a run
realizes are genuinely different questions.

**Read** is judged the same way and for the same reason: the catalogue's value
for a knob has to be what the base really has, or the panel opens half a floor
away from the scenario it claims to describe and the operator's first drag is a
blind edit.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest
from _api_fixtures import (
    DEFAULT_SCENARIO_ID,
    api_facility,
    api_scenario,
    api_staffing,
    api_workload,
    make_app,
)
from fastapi.testclient import TestClient

from hospital.api.sliders import SLIDER_KEYS, catalogue, compile_overrides
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

if TYPE_CHECKING:
    from pathlib import Path


def _reads(base: Scenario) -> dict[str, float]:
    """The catalogue's value for every published knob, keyed by knob key."""
    return {knob.key: knob.value for knob in catalogue(base)}


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


# --------------------------------------------------------------------- demand: the mix
def test_the_demand_sliders_reach_the_workload_fields_they_rename() -> None:
    base = api_scenario()
    isolating = compile_overrides(base, {"workload.isolation_share": 0.3})
    assert isolating.workload.isolation_fraction == 0.3
    # ...and nothing else in the mix moved with it.
    assert isolating.workload.ambulance_fraction == base.workload.ambulance_fraction
    assert isolating.workload.esi_mix == base.workload.esi_mix

    assert (
        compile_overrides(base, {"workload.ambulance_share": 0.75}).workload.ambulance_fraction
        == 0.75
    )
    doubled = compile_overrides(base, {"workload.arrival_rate_multiplier": 2.0})
    assert doubled.workload.base_rate_per_hour == base.workload.base_rate_per_hour * 2.0


@pytest.mark.parametrize(
    ("alias", "literal"),
    [
        ("workload.isolation_share", "workload.isolation_fraction"),
        ("workload.ambulance_share", "workload.ambulance_fraction"),
        ("staffing.tech_count", "staffing.default_counts"),
        ("facility.general_bays", "facility.zones"),
    ],
)
def test_an_alias_and_a_literal_path_onto_the_same_leaf_are_refused(
    alias: str, literal: str
) -> None:
    """Two names for one field cannot both be honored, and resolving it by dict
    order would make the derived scenario depend on JSON key order."""
    with pytest.raises(ValueError, match="already sets"):
        compile_overrides(api_scenario(), {alias: 0.5, literal: 0.5})


@pytest.mark.parametrize(
    "overrides",
    [
        # A share is a probability: the bounds are ``WorkloadSpec``'s, not ours.
        {"workload.isolation_share": 1.4},
        {"workload.ambulance_share": -0.1},
        # A headcount is a non-negative whole number (`range(-n)` is silently empty).
        {"staffing.housekeeping_count": -1},
        # A bay count likewise.
        {"facility.resus_bays": -2},
        {"facility.general_bays": 1.5},
    ],
)
def test_out_of_range_values_are_data_layer_rejections(overrides: dict[str, float]) -> None:
    """No slider bound lives in this module — every one of these is ``data``
    saying no, surfaced by the caller as a 422."""
    with pytest.raises(ValueError):
        compile_overrides(api_scenario(), overrides)


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


# --------------------------------------------------------------------- read: the inverse
def test_the_catalogue_covers_every_alias() -> None:
    """Nothing translated is left unpublished — the panel's list IS this one."""
    published = set(_reads(api_scenario()))
    assert set(SLIDER_KEYS) <= published
    # ...and the knobs that need no alias are published too, from their literal paths.
    assert {
        "facility.imaging_suites",
        "facility.lab_stations",
        "facility.triage_rooms",
    } <= published


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("workload.ambulance_share", 0.75),
        ("workload.isolation_share", 0.3),
        ("staffing.physician_count", 5),
        ("staffing.nurse_count", 9),
        ("staffing.tech_count", 4),
        ("staffing.porter_count", 2),
        ("staffing.housekeeping_count", 6),
        ("facility.general_bays", 7),
        ("facility.observation_bays", 4),
        ("facility.resus_bays", 3),
        ("facility.fast_track_bays", 2),
        ("facility.triage_rooms", 5),
        ("facility.imaging_suites", 4),
        ("facility.lab_stations", 3),
    ],
)
def test_every_knob_reads_back_the_value_it_was_set_to(key: str, value: float) -> None:
    """``read`` is a true inverse of ``compile``, for translated and literal knobs alike."""
    derived = compile_overrides(_shift_scenario(), {key: value})
    assert _reads(derived)[key] == pytest.approx(value)


def test_the_relative_knob_always_reads_one() -> None:
    """A multiplier has no stored value: the scaling is already inside
    ``base_rate_per_hour``. Against ANY scenario — base or derived — the honest
    reading is "nothing scaled yet", which is what the panel needs to open on."""
    base = api_scenario()
    assert _reads(base)["workload.arrival_rate_multiplier"] == 1.0
    doubled = compile_overrides(base, {"workload.arrival_rate_multiplier": 2.0})
    assert _reads(doubled)["workload.arrival_rate_multiplier"] == 1.0
    assert doubled.workload.base_rate_per_hour == base.workload.base_rate_per_hour * 2.0


def test_headcounts_read_the_roster_the_engine_will_build_not_the_defaults() -> None:
    """``er_floor``'s exact trap: blocks schedule 14 nurses, defaults carry 7."""
    base = _shift_scenario()
    reads = _reads(base)
    realized = _roster(base)
    for role in StaffRole:
        assert reads[f"staffing.{role.value}_count"] == float(realized[role])
    # The specific number a defaults-reader would have reported instead.
    assert reads["staffing.nurse_count"] == 14.0
    assert base.staffing.default_counts[StaffRole.NURSE] == 7


def test_writing_a_knob_back_at_its_own_value_changes_nothing_observable() -> None:
    """A knob at rest is a no-op — the property behind the panel's "= base" mark.

    Scoped honestly: the fractions and the zone tuple come back byte-identical,
    and the headcounts come back identical *as realized*. A headcount alias also
    rewrites ``default_counts`` and flattens per-shift variation by design, so a
    document-level fixed point is not on offer for those and claiming one here
    would be the lie. (It is also why the panel submits only the knobs that
    actually moved.)
    """
    base = _repeated_zone_scenario()
    reads = _reads(base)
    facility_and_workload = [key for key in reads if key.startswith(("facility.", "workload."))]
    assert compile_overrides(base, {key: reads[key] for key in facility_and_workload}) == base

    shifted = _shift_scenario()
    staffed = compile_overrides(
        shifted,
        {key: value for key, value in _reads(shifted).items() if key.startswith("staffing.")},
    )
    assert _roster(staffed) == _roster(shifted)


# ------------------------------------------------------------------------- the endpoint
def test_catalogue_endpoint_publishes_grouped_knobs_read_against_the_base(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, {DEFAULT_SCENARIO_ID: _shift_scenario()})) as client:
        response = client.get(f"/scenarios/{DEFAULT_SCENARIO_ID}/sliders")
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        assert body["scenario"] == DEFAULT_SCENARIO_ID

        knobs: list[dict[str, Any]] = body["knobs"]
        by_key = {knob["key"]: knob for knob in knobs}
        assert set(by_key) == set(_reads(_shift_scenario()))
        assert {knob["group"] for knob in knobs} == {"demand", "staffing", "capacity"}
        # The two roles that gate bed turnaround are supply, not clinical staffing.
        assert by_key["staffing.housekeeping_count"]["group"] == "capacity"
        assert by_key["staffing.porter_count"]["group"] == "capacity"

        for knob in knobs:
            assert knob["label"] != ""
            assert knob["unit"] != ""
            assert knob["step"] > 0
            assert knob["min"] < knob["max"]
        # ...and the values are the base's own, not a published default.
        assert by_key["staffing.nurse_count"]["value"] == 14.0
        assert by_key["workload.arrival_rate_multiplier"]["value"] == 1.0


def test_catalogue_endpoint_404s_for_an_unknown_base(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        assert client.get("/scenarios/nope/sliders").status_code == 404


def test_catalogue_tracks_a_derived_scenario(tmp_path: Path) -> None:
    """Re-read against the variant a ``POST /scenarios`` produced, it shows where
    that variant now sits — the catalogue is derived, never stored."""
    with TestClient(make_app(tmp_path)) as client:
        created = client.post(
            "/scenarios",
            json={"base": DEFAULT_SCENARIO_ID, "overrides": {"facility.fast_track_bays": 9}},
        )
        assert created.status_code == 201, created.text
        derived_id = created.json()["id"]
        knobs: list[dict[str, Any]] = client.get(f"/scenarios/{derived_id}/sliders").json()["knobs"]
        by_key = {knob["key"]: knob for knob in knobs}
        assert by_key["facility.fast_track_bays"]["value"] == 9.0


def test_a_value_outside_a_published_range_is_still_the_data_layers_call(tmp_path: Path) -> None:
    """The catalogue's min/max are affordances: nothing re-checks them server-side,
    so a value past a range is accepted or refused purely on ``data``'s terms."""
    with TestClient(make_app(tmp_path)) as client:
        knobs: list[dict[str, Any]] = client.get(
            f"/scenarios/{DEFAULT_SCENARIO_ID}/sliders"
        ).json()["knobs"]
        ceiling = next(k for k in knobs if k["key"] == "facility.general_bays")["max"]

        past_the_range = client.post(
            "/scenarios",
            json={
                "base": DEFAULT_SCENARIO_ID,
                "overrides": {"facility.general_bays": ceiling + 10},
            },
        )
        assert past_the_range.status_code == 201, past_the_range.text

        # ...whereas a value the MODEL rejects is a 422, from the model.
        refused = client.post(
            "/scenarios",
            json={"base": DEFAULT_SCENARIO_ID, "overrides": {"workload.isolation_share": 1.4}},
        )
        assert refused.status_code == 422


def test_the_lab_launches_a_run_from_every_knob_at_once(tmp_path: Path) -> None:
    """The console's own re-run path: inline sliders straight into ``POST /runs``."""
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/runs",
            json={
                "scenario": {
                    "base": DEFAULT_SCENARIO_ID,
                    "overrides": {
                        "workload.arrival_rate_multiplier": 1.5,
                        "workload.ambulance_share": 0.4,
                        "workload.isolation_share": 0.2,
                        "staffing.physician_count": 3,
                        "staffing.nurse_count": 4,
                        "staffing.tech_count": 2,
                        "staffing.porter_count": 2,
                        "staffing.housekeeping_count": 2,
                        "facility.general_bays": 4,
                        "facility.observation_bays": 2,
                        "facility.resus_bays": 2,
                        "facility.fast_track_bays": 2,
                        "facility.triage_rooms": 2,
                        "facility.imaging_suites": 2,
                        "facility.lab_stations": 2,
                    },
                },
                "seed": 11,
                "arm": "optimized",
            },
        )
        assert response.status_code == 201, response.text
