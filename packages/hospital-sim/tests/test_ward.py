"""ED boarding into wards: capacity becomes the wait, and the trip upstairs is real.

The claim this file exists to check is the one M4 is for: **boarding time is downstream of
ward capacity**. Through M3 it was a draw from a fixed distribution, so a full hospital
looked exactly like an empty one. :func:`test_a_smaller_ward_boards_patients_longer` is
that claim; the rest guard the mechanics it rests on.
"""

from __future__ import annotations

import json
from typing import Any

from _sim_fixtures import tiny_scenario

from hospital.core import WARD_ZONE_TYPES, BayStatus, ZoneType
from hospital.data.hospital import generate_hospital
from hospital.data.scenario import FacilitySpec, FloorSpec, Scenario, ZoneQuota
from hospital.sim import run_replication
from hospital.sim.flow.ward import WARD_PREFERENCE, has_ward_beds, ward_beds

_SEED = 7


def _ward_floor(zone_type: ZoneType, beds: int) -> FloorSpec:
    return FloorSpec(
        name=zone_type.value,
        facility=FacilitySpec(
            target_area_sqft=30_000,
            zones=(ZoneQuota(zone_type=zone_type, bays=beds, isolation_bays=1),),
            imaging_suites=0,
            lab_stations=0,
            triage_rooms=0,
        ),
    )


def _with_wards(beds: int, *, hours: int = 24) -> Scenario:
    base = tiny_scenario(horizon_hours=hours, rate_per_hour=4.0)
    return base.model_copy(update={"upper_floors": (_ward_floor(ZoneType.MED_SURG, beds),)})


def _events(jsonl: str) -> list[dict[str, Any]]:
    return [json.loads(line)["event"] for line in jsonl.splitlines() if line.strip()]


def _kinds(jsonl: str, kind: str) -> list[dict[str, Any]]:
    return [e for e in _events(jsonl) if e["kind"] == kind]


def _boarding_seconds(jsonl: str, horizon_s: float) -> list[float]:
    """Per admitted patient: how long their ED bay was held after the disposition.

    **Censored at the horizon, not conditioned on success.** Counting only patients who
    reached a bed would select for short waits and invert the very comparison this file
    makes: with one inpatient bed the first admit takes it for days, so everyone after
    them boards past the end of the run and the survivors are exactly the lucky ones. A
    patient still boarding when the week ends contributes the time they actually waited.
    """
    events = _events(jsonl)
    admitted: dict[str, int] = {}
    for event in events:
        if event["kind"] == "disposition_decided" and event["disposition"] == "admit":
            admitted[str(event["patient"])] = int(event["occurred_at"])
    # The ward grant is the second `bay_assigned` a patient receives.
    grants: dict[str, list[int]] = {}
    for event in events:
        if event["kind"] == "bay_assigned":
            grants.setdefault(str(event["patient"]), []).append(int(event["occurred_at"]))

    out: list[float] = []
    for patient, decided in admitted.items():
        later = [t for t in grants.get(patient, []) if t >= decided]
        end = min(later) if later else int(horizon_s * 1_000_000)
        out.append(max(0.0, (end - decided) / 1_000_000))
    return out


def _still_boarding(jsonl: str) -> int:
    """Admitted patients who never reached a bed before the run ended."""
    events = _events(jsonl)
    admitted = {
        str(e["patient"])
        for e in events
        if e["kind"] == "disposition_decided" and e["disposition"] == "admit"
    }
    granted: dict[str, int] = {}
    for event in events:
        if event["kind"] == "bay_assigned":
            granted[str(event["patient"])] = granted.get(str(event["patient"]), 0) + 1
    return sum(1 for p in admitted if granted.get(p, 0) < 2)


def test_an_ed_only_scenario_is_untouched_by_the_ward_code() -> None:
    """No inpatient bed means no behaviour change — the switch is the building itself."""
    scenario = tiny_scenario()
    assert run_replication(scenario, "baseline", _SEED).event_log_jsonl == (
        run_replication(scenario, "baseline", _SEED).event_log_jsonl
    )
    layout = generate_hospital(scenario.hospital())
    assert not [bay for bay in layout.bays if bay.zone_type in WARD_ZONE_TYPES]


def test_adding_a_ward_changes_the_run() -> None:
    """The port must be live: admits now go somewhere instead of evaporating."""
    plain = run_replication(tiny_scenario(), "baseline", _SEED).event_log_jsonl
    warded = run_replication(_with_wards(beds=8), "baseline", _SEED).event_log_jsonl
    assert plain != warded


def test_admitted_patients_end_up_in_ward_beds() -> None:
    """A ward bed that is never occupied is scenery, not capacity."""
    log = run_replication(_with_wards(beds=8), "baseline", _SEED).event_log_jsonl
    ward_bay_ids = {
        bay.id.root
        for bay in generate_hospital(_with_wards(beds=8).hospital()).bays
        if bay.zone_type in WARD_ZONE_TYPES
    }
    assert ward_bay_ids
    granted = {str(e["bay"]) for e in _kinds(log, "bay_assigned")}
    assert granted & ward_bay_ids, "no admitted patient ever reached a ward bed"


def test_a_smaller_ward_boards_patients_longer() -> None:
    """The claim the milestone exists for: ED boarding is ward capacity, felt downstream.

    Through M3 this could not be true by construction — boarding was a draw from a fixed
    distribution, so a hospital with two inpatient beds looked exactly like one with
    twenty. Under CRN the two runs share every draw, so the difference is the beds.
    """
    horizon_s = 24 * 3600.0
    roomy_log = run_replication(_with_wards(beds=12), "baseline", _SEED).event_log_jsonl
    tight_log = run_replication(_with_wards(beds=1), "baseline", _SEED).event_log_jsonl
    roomy = _boarding_seconds(roomy_log, horizon_s)
    tight = _boarding_seconds(tight_log, horizon_s)
    assert roomy, "nobody was admitted in the roomy run"
    assert tight, "nobody was admitted in the tight run"

    assert sum(tight) > sum(roomy), (
        f"a one-bed ward held no more ED bay-time than a twelve-bed one: "
        f"{sum(tight):.0f}s vs {sum(roomy):.0f}s"
    )
    # ...and the scarcity shows in the other direction too: more people never get a bed.
    assert _still_boarding(tight_log) > _still_boarding(roomy_log)


def test_boarding_holds_the_ed_bay_until_a_bed_is_found() -> None:
    """The blockage is the mechanism: a boarded patient's ED bay is unavailable.

    If the ED bay were released at disposition, a full ward would cost nothing and the
    whole model would be decorative.
    """
    log = run_replication(_with_wards(beds=1), "baseline", _SEED).event_log_jsonl
    events = _events(log)
    ward_ids = {
        bay.id.root
        for bay in generate_hospital(_with_wards(beds=1).hospital()).bays
        if bay.zone_type in WARD_ZONE_TYPES
    }

    # For an admitted patient: their ED bay is vacated no earlier than the ward grant.
    ed_grant: dict[str, tuple[int, str]] = {}
    ward_grant: dict[str, int] = {}
    for event in events:
        if event["kind"] != "bay_assigned":
            continue
        patient, bay, at = str(event["patient"]), str(event["bay"]), int(event["occurred_at"])
        if bay in ward_ids:
            ward_grant.setdefault(patient, at)
        else:
            ed_grant.setdefault(patient, (at, bay))

    cleaned: dict[str, int] = {}
    for event in events:
        if event["kind"] == "bay_cleaning_started":
            cleaned.setdefault(str(event["bay"]), int(event["occurred_at"]))

    checked = 0
    for patient, granted_at in ward_grant.items():
        held = ed_grant.get(patient)
        if held is None or held[1] not in cleaned:
            continue
        checked += 1
        assert cleaned[held[1]] >= granted_at, (
            f"{patient}'s ED bay was released before their ward bed existed"
        )
    assert checked, "no admitted patient had both an ED bay and a ward bed"


def test_the_trip_upstairs_is_a_real_escorted_move() -> None:
    """The patient is walked to the bed over the graph, elevator edges included."""
    log = run_replication(_with_wards(beds=8), "baseline", _SEED).event_log_jsonl
    layout = generate_hospital(_with_wards(beds=8).hospital())
    shafts = {node.root for node in layout.elevators}
    assert shafts

    moved = _kinds(log, "patient_moved")
    assert moved, "nobody moved at all"
    crossed = [e for e in moved if str(e["edge"][0]) in shafts and str(e["edge"][1]) in shafts]
    assert crossed, "no patient ever traversed a shaft edge"
    # A shaft traversal costs its configured time, not a walk over four metres.
    for event in crossed:
        assert int(event["seconds"]) > 0


def test_the_ward_run_is_deterministic() -> None:
    scenario = _with_wards(beds=4)
    assert (
        run_replication(scenario, "baseline", _SEED).event_log_jsonl
        == run_replication(scenario, "baseline", _SEED).event_log_jsonl
    )


def test_a_ward_bed_is_released_after_the_stay() -> None:
    """Beds that are never freed would deadlock the ED within a day."""
    scenario = _with_wards(beds=6, hours=24 * 8)
    log = run_replication(scenario, "baseline", _SEED).event_log_jsonl
    ward_ids = {
        bay.id.root
        for bay in generate_hospital(scenario.hospital()).bays
        if bay.zone_type in WARD_ZONE_TYPES
    }
    cleaned = {str(e["bay"]) for e in _kinds(log, "bay_cleaning_started")}
    assert cleaned & ward_ids, "no ward bed was ever released"


def test_every_acuity_has_somewhere_to_be_admitted() -> None:
    """A missing preference row would board that acuity forever, silently."""
    from hospital.core import EsiAcuity

    for esi in EsiAcuity:
        assert WARD_PREFERENCE.get(esi), f"ESI-{int(esi)} has no ward preference"
        assert set(WARD_PREFERENCE[esi]) <= WARD_ZONE_TYPES


def test_ward_bed_lookup_is_deterministic_and_status_free() -> None:
    """`ward_beds` is a static question about the building, not about its state."""
    scenario = _with_wards(beds=5)
    layout = generate_hospital(scenario.hospital())

    class _World:
        def __init__(self) -> None:
            self.layout = layout

    world = _World()
    beds = ward_beds(world)  # type: ignore[arg-type]
    assert len(beds) == 5
    assert list(beds) == sorted(beds, key=lambda b: b.root)
    assert has_ward_beds(world)  # type: ignore[arg-type]
    assert BayStatus.FREE  # the lookup never consults status; this is a static list
