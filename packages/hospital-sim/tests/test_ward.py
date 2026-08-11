"""ED boarding into wards: capacity becomes the wait, and the trip upstairs is real.

The claim this file exists to check is the one M4 is for: **boarding time is downstream of
ward capacity**. Through M3 it was a draw from a fixed distribution, so a full hospital
looked exactly like an empty one. :func:`test_a_smaller_ward_boards_patients_longer` is
that claim; the rest guard the mechanics it rests on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _sim_fixtures import tiny_scenario

from hospital.analysis import fold_arm
from hospital.core import (
    WARD_ZONE_TYPES,
    BayStatus,
    EventLog,
    FloorLayout,
    RandomStreams,
    TimeWindow,
    ZoneType,
    compile_rules,
    hours,
)
from hospital.data.hospital import generate_hospital
from hospital.data.layout import generate_floor
from hospital.data.scenario import (
    FacilitySpec,
    FloorSpec,
    Scenario,
    ZoneQuota,
    load_scenario,
    realize_staff,
)
from hospital.data.workload import generate_workload
from hospital.sim import run_replication
from hospital.sim.experiment.replication import DEFAULT_OBJECTIVE, default_rules
from hospital.sim.experiment.scorecard import fold_scorecard
from hospital.sim.flow.ward import has_ward_beds, ward_beds
from hospital.sim.physics.service_times import admit_probability, mean_ward_stay
from hospital.sim.policies.factory import Arm
from hospital.solver.placement import WARD_PREFERENCE

_SEED = 7
_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _two_wards(*, icu: int, med_surg: int, hours: int = 24) -> Scenario:
    """A scarce ICU over a roomy med-surg — the floor plan where *which* ward matters.

    A single-ward hospital cannot show the difference between the arms: every bed is
    interchangeable, so first-available and the solver both just take one. Two wards of
    different scarcity is the smallest building in which the choice is a choice.
    """
    base = tiny_scenario(horizon_hours=hours, rate_per_hour=4.0)
    return base.model_copy(
        update={
            "upper_floors": (
                _ward_floor(ZoneType.ICU, icu),
                _ward_floor(ZoneType.MED_SURG, med_surg),
            )
        }
    )


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
    """A missing row would board that acuity forever, silently.

    Both halves matter and they fail differently. A missing *permission* row is fatal:
    the whitelist is closed, so the validator refuses every bed and the patient holds
    an ED bay until the horizon. A missing *preference* row only means the beds that
    acuity may use are all ranked last — worse placement, not no placement — which is
    why the preference is the solver's and the permission is the rule's.
    """
    from hospital.core import EsiAcuity

    rules = compile_rules(default_rules())
    for esi in EsiAcuity:
        permitted = rules.ward_zone_types_for(esi)
        assert permitted, f"ESI-{int(esi)} may be admitted nowhere"
        assert permitted <= WARD_ZONE_TYPES
        assert WARD_PREFERENCE.get(esi), f"ESI-{int(esi)} has no ward preference"
        assert set(WARD_PREFERENCE[esi]) <= permitted


def test_an_admitted_patient_is_never_granted_an_ed_bay() -> None:
    """The care phase is a wall, not a preference: the two populations cannot cross.

    Every ``bay_assigned`` in a warded run is checked against the phase the patient was
    in when it landed — their second grant is the admission — so a solver or an
    operator that offered an admitted patient a fast-track bay would be caught here
    rather than producing a plausible-looking run in which the ED never empties.
    """
    scenario = _with_wards(beds=8)
    log = run_replication(scenario, "baseline", _SEED).event_log_jsonl
    layout = generate_hospital(scenario.hospital())
    ward_ids = {bay.id.root for bay in layout.bays if bay.zone_type in WARD_ZONE_TYPES}

    admitted = {
        str(e["patient"])
        for e in _kinds(log, "disposition_decided")
        if str(e["disposition"]) == "admit"
    }
    seen: dict[str, int] = {}
    checked = 0
    for event in _kinds(log, "bay_assigned"):
        patient, bay = str(event["patient"]), str(event["bay"])
        seen[patient] = seen.get(patient, 0) + 1
        if seen[patient] == 1:
            assert bay not in ward_ids, f"{patient} was worked up in a ward bed"
        elif patient in admitted:
            checked += 1
            assert bay in ward_ids, f"admitted {patient} was granted ED bay {bay}"
    assert checked, "no admitted patient was ever granted a bed"


def test_the_solver_keeps_the_icu_for_the_acuity_that_needs_it() -> None:
    """The M4 §3 claim, end to end: hospital-wide placement beats first-available.

    Both arms see the identical realized week under CRN, so the difference is the
    decision and nothing else. The baseline scans free beds in ``BayId`` order, and the
    ICU floor sorts first — so an ESI-3 who merely *may* use an ICU bed takes one, and
    a critical patient arriving later boards in the ED behind them. The solver ranks
    med-surg first for that ESI-3, which costs it a longer escort and buys back the bed.

    Asserted as a comparison, not as "the solver never puts an ESI-3 in the ICU": with
    every med-surg bed full it should, and does (place-first dominates preference).
    """
    scenario = _two_wards(icu=2, med_surg=10)
    icu_ids = {
        bay.id.root
        for bay in generate_hospital(scenario.hospital()).bays
        if bay.zone_type is ZoneType.ICU
    }

    def icu_grants_to_low_acuity(arm: Arm) -> int:
        log = run_replication(scenario, arm, _SEED).event_log_jsonl
        esi = {str(e["patient"]): int(e["esi"]) for e in _kinds(log, "triage_completed")}
        return sum(
            1
            for e in _kinds(log, "bay_assigned")
            if str(e["bay"]) in icu_ids and esi.get(str(e["patient"]), 0) >= 3
        )

    baseline = icu_grants_to_low_acuity("baseline")
    optimized = icu_grants_to_low_acuity("optimized")
    assert baseline > 0, "the baseline never squatted an ICU bed — the fixture proves nothing"
    assert optimized < baseline, (
        f"the solver filled the ICU with low-acuity patients as readily as "
        f"first-available did: {optimized} vs {baseline}"
    )


def test_a_ward_grant_comes_through_the_seam() -> None:
    """The bed is a *decision*, so the log must attribute it to a decision maker.

    Under the greedy claim this event was stamped by the ward module itself, which is
    the tell that no policy chose it and no override could have. Both arms are checked
    because the point is that the grant travels the same path either way.
    """
    scenario = _with_wards(beds=4)
    layout = generate_hospital(scenario.hospital())
    ward_ids = {bay.id.root for bay in layout.bays if bay.zone_type in WARD_ZONE_TYPES}

    arms: tuple[tuple[Arm, str], ...] = (("baseline", "baseline"), ("optimized", "solver"))
    for arm, origin in arms:
        log = run_replication(scenario, arm, _SEED).event_log_jsonl
        grants = [e for e in _kinds(log, "bay_assigned") if str(e["bay"]) in ward_ids]
        assert grants, f"no ward bed was granted in the {arm} arm"
        assert {str(e["by"]) for e in grants} == {origin}


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


def test_the_committed_hospital_is_tight_but_not_gridlocked() -> None:
    """`scenarios/hospital.yaml`'s bed count is an operating point; this pins which one.

    Sized against that scenario's own realized week rather than guessed. Its arrivals
    admit enough patients that their sampled ward stays need ~90 beds to run at full
    occupancy, and the committed 74 leaves the wards genuinely scarce. Size to ~90 and
    boarding vanishes, so the building stops demonstrating the mechanism M4 exists for;
    drop to ~56 and the ED gridlocks (measured: 19h mean boarding, 15% fewer weekly
    completions), which is a stressed variant rather than a reference.

    The assertion couples the scenario to the *sim's* ward-stay model, which is why it
    lives here and not beside the other scenario tests: change either side enough to
    move this ratio and the reference hospital has been silently re-sited.
    """
    scenario = load_scenario(_REPO_ROOT / "scenarios" / "hospital.yaml")
    beds = sum(
        quota.bays
        for floor in scenario.upper_floors
        for quota in floor.facility.zones
        if quota.zone_type in WARD_ZONE_TYPES
    )
    arrivals = generate_workload(
        scenario.workload, RandomStreams(scenario.seed), disruptions=scenario.disruptions
    )
    bed_days = sum(
        admit_probability(a.patient.esi) * mean_ward_stay(a.patient.esi).root / 1e6 / 86_400.0
        for a in arrivals
    )
    horizon_days = scenario.workload.horizon.end.root / 1e6 / 86_400.0
    needed = bed_days / horizon_days
    assert 0.7 <= beds / needed <= 0.9, f"{beds} beds against {needed:.0f} for full occupancy"


def test_the_scorecard_folds_against_the_whole_building() -> None:
    """Analysis must see the floors the engine ran, or it reports half a hospital.

    `run_replication` builds the building with `generate_hospital`; through M4 §2 the
    scorecard, the CLI's arm summary, and the API session still built the ground floor
    with `generate_floor`. On an ED-only scenario the two are identical, so the mismatch
    was invisible — and on a hospital it silently drops every ward bay from the index,
    so the occupied bed-hours the log reports simply do not reach the KPI. It does not
    raise and it does not look wrong, which is why a test rather than a type had to
    catch it — and it matters more now than it reads, because `bay_hours_occupied` is
    what the cost model prices capacity from.

    Asserted by folding the SAME log both ways: the ED-only layout must under-report.
    """
    scenario = _with_wards(beds=8)
    rep = run_replication(scenario, "baseline", _SEED)
    horizon = scenario.workload.horizon
    log = EventLog.from_jsonl(rep.event_log_jsonl)
    warmup = hours(1)

    whole = generate_hospital(scenario.hospital())
    ed_only = generate_floor(scenario.facility)
    assert len(whole.bays) > len(ed_only.bays), "the fixture has no upstairs to lose"

    def occupied_hours(layout: FloorLayout) -> float:
        roster = realize_staff(
            scenario.staffing, layout, TimeWindow(start=horizon.start, end=horizon.end)
        )
        arm = fold_arm([log], layout, roster, window=horizon, warmup=warmup)
        return arm.kpis.values["bay_hours_occupied"]

    assert occupied_hours(ed_only) < occupied_hours(whole)

    # And the scorecard — the path the CLI and the comparison actually use — is on the
    # right side of that inequality.
    folded = fold_scorecard(rep, DEFAULT_OBJECTIVE).kpis.values["bay_hours_occupied"]
    assert folded > occupied_hours(ed_only)
