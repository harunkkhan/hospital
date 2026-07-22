"""``waits.decompose_waits`` — the seven-stage tiling invariant + WIP exclusion."""

from __future__ import annotations

import math

from _analysis_fixtures import build_sample_log, tiny_layout

from hospital.analysis.waits import decompose_waits
from hospital.core import PatientId, hours


def test_stages_sum_to_los_for_fully_observed_patient() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    decomp = decompose_waits(log, layout, warmup=hours(0))

    profiles = {p.patient: p for p in decomp.per_patient}
    p1 = profiles[PatientId("p1")]
    stages = p1.stages
    stage_sum = (
        stages.wait_triage
        + stages.svc_triage
        + stages.wait_bay
        + stages.wait_provider
        + stages.workup_service
        + stages.workup_wait
        + stages.paperwork_or_boarding
    )
    assert math.isclose(stage_sum, p1.los_s, rel_tol=1e-9, abs_tol=1e-6)
    assert math.isclose(p1.los_s, 1500.0)

    # Hand-computed exact stage values (see _fixtures.build_sample_log docstring).
    assert math.isclose(stages.wait_triage, 60.0)
    assert math.isclose(stages.svc_triage, 240.0)
    assert math.isclose(stages.wait_bay, 180.0)
    assert math.isclose(stages.wait_provider, 120.0)
    assert math.isclose(stages.workup_service, 350.0)
    assert math.isclose(stages.workup_wait, 250.0)
    assert math.isclose(stages.paperwork_or_boarding, 300.0)


def test_wip_patient_excluded_from_tiling() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    decomp = decompose_waits(log, layout, warmup=hours(0))
    patients_present = {p.patient for p in decomp.per_patient}
    assert PatientId("p1") in patients_present
    assert PatientId("p2") not in patients_present  # still WIP -> no `de` -> untileable


def test_bay_turnaround_substages_sum_to_turnaround() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    decomp = decompose_waits(log, layout, warmup=hours(0))
    assert len(decomp.per_bay_cycle) == 1  # only bay-1's cycle fully completed cleaning
    cyc = decomp.per_bay_cycle[0]
    assert math.isclose(
        cyc.hold_to_vacate_s + cyc.wait_housekeeper_s + cyc.cleaning_s, cyc.turnaround_s
    )
    assert math.isclose(cyc.hold_to_vacate_s, 300.0)
    assert math.isclose(cyc.wait_housekeeper_s, 60.0)
    assert math.isclose(cyc.cleaning_s, 300.0)
    assert math.isclose(cyc.turnaround_s, 660.0)


def test_share_of_los_sums_to_one_per_cohort() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    decomp = decompose_waits(log, layout, warmup=hours(0))
    total_share = math.fsum(agg.share_of_los for agg in decomp.stage_means.values())
    assert math.isclose(total_share, 1.0, abs_tol=1e-9)


def test_no_negative_stage_durations() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    decomp = decompose_waits(log, layout, warmup=hours(0))
    for prof in decomp.per_patient:
        s = prof.stages
        for value in (
            s.wait_triage,
            s.svc_triage,
            s.wait_bay,
            s.wait_provider,
            s.workup_service,
            s.workup_wait,
            s.paperwork_or_boarding,
        ):
            assert value >= 0.0
