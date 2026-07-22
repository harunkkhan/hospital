"""``_index.build_index`` — the single-pass reconstruction, milestone correctness."""

from __future__ import annotations

from _analysis_fixtures import (
    HOUSEKEEPER,
    NURSE,
    PHYSICIAN,
    build_sample_log,
    tiny_layout,
    tiny_roster,
)

from hospital.analysis._index import build_index
from hospital.core import BayId, DispositionKind, EsiAcuity, PatientId


def test_patient_milestones_reconstructed_correctly() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    idx = build_index(log, layout, roster)

    p1 = idx.patients[PatientId("p1")]
    assert p1.arrival.root == 0
    assert p1.esi == EsiAcuity.ESI3
    assert p1.triage_start is not None and p1.triage_start.root == 60 * 1_000_000
    assert p1.triage_end is not None and p1.triage_end.root == 300 * 1_000_000
    assert p1.bay == BayId("bay-1")
    assert p1.bay_arrival is not None and p1.bay_arrival.root == 480 * 1_000_000
    assert p1.provider_start is not None and p1.provider_start.root == 600 * 1_000_000
    assert p1.disposition == DispositionKind.DISCHARGE
    assert p1.exit is not None and p1.exit.root == 1500 * 1_000_000
    assert len(p1.provider_intervals) == 1
    assert len(p1.nurse_intervals) == 1
    assert len(p1.documentation_intervals) == 1

    p2 = idx.patients[PatientId("p2")]
    assert p2.exit is None  # still WIP
    assert p2.disposition is None
    assert p2.provider_start is not None  # started but not completed
    # Regression (finding #2): a *_started still open at the end of the log is
    # WIP service — preserved as an OPEN interval (end=None), never dropped.
    assert len(p2.provider_intervals) == 1
    assert p2.provider_intervals[0].end is None
    assert p2.provider_intervals[0].start.root == 2300 * 1_000_000


def test_bay_cycle_reconstruction() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    idx = build_index(log, layout, roster)

    bay1_cycles = idx.bays[BayId("bay-1")]
    assert len(bay1_cycles) == 1
    cyc = bay1_cycles[0]
    assert cyc.clean_start is not None and cyc.clean_start.root == 1560 * 1_000_000
    assert cyc.clean_end is not None and cyc.clean_end.root == 1860 * 1_000_000
    assert cyc.clean_staff == HOUSEKEEPER

    bay2_cycles = idx.bays[BayId("bay-2")]
    assert len(bay2_cycles) == 1
    assert bay2_cycles[0].clean_start is None  # never cleaned (still WIP)


def test_staff_traces_present_for_whole_roster() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    idx = build_index(log, layout, roster)
    assert set(idx.staff.keys()) == {PHYSICIAN, NURSE, HOUSEKEEPER}
    assert len(idx.staff[PHYSICIAN].walk_intervals) == 2
    assert len(idx.staff[HOUSEKEEPER].cleaning_intervals) == 1
