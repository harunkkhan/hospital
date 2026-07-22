"""``_index.build_index`` — the single-pass reconstruction, milestone correctness."""

from __future__ import annotations

import pytest
from _analysis_fixtures import (
    HOUSEKEEPER,
    NURSE,
    PHYSICIAN,
    build_sample_log,
    t,
    tiny_layout,
    tiny_roster,
)

from hospital.analysis._index import build_index
from hospital.core import (
    Activity,
    BayId,
    DispositionKind,
    DocumentationCompleted,
    EsiAcuity,
    Event,
    EventLog,
    NurseVisitCompleted,
    PatientId,
    ProviderVisitCompleted,
    TriageCompleted,
    TriageStarted,
    ZeroTimeCycle,
)

# Aliased so pytest does not try to collect the "Test*"-named event class.
from hospital.core import TestResulted as _TestResulted


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


@pytest.mark.parametrize(
    "event",
    [
        ProviderVisitCompleted(occurred_at=t(100), patient=PatientId("ghost"), staff=PHYSICIAN),
        NurseVisitCompleted(occurred_at=t(100), patient=PatientId("ghost"), staff=NURSE),
        DocumentationCompleted(occurred_at=t(100), patient=PatientId("ghost"), staff=NURSE),
        _TestResulted(occurred_at=t(100), patient=PatientId("ghost"), activity=Activity.LAB),
    ],
    ids=["provider", "nurse", "documentation", "test"],
)
def test_unmatched_completion_raises_malformed_log_error(event: Event) -> None:
    """Regression (finding #5): a ``*Completed``/``TestResulted`` with no open
    matching start is a causally-impossible, corrupt log — raised as a typed
    error, never silently dropped into a plausible-but-wrong undercount."""
    log = EventLog()
    log.append(event)
    with pytest.raises(ZeroTimeCycle):
        build_index(log, tiny_layout(), tiny_roster())


def test_sliced_log_anchors_synthetic_trace_at_first_observed_event() -> None:
    """Regression (finding #4): a log slice that starts after a patient's
    ``PatientArrived`` must anchor the synthesized trace at the first event
    actually observed — anchoring at SimTime(0) fabricates pre-slice waiting
    time and inflates every LOS/cycle measure."""
    log = EventLog()
    patient = PatientId("pre-slice")
    log.append(TriageStarted(occurred_at=t(5000), patient=patient, staff=NURSE))
    log.append(TriageCompleted(occurred_at=t(5100), patient=patient, esi=EsiAcuity.ESI3))

    idx = build_index(log, tiny_layout(), tiny_roster())
    trace = idx.patients[patient]
    assert trace.arrival.root == 5000 * 1_000_000  # first observed event, not epoch


def test_staff_traces_present_for_whole_roster() -> None:
    log = build_sample_log()
    layout = tiny_layout()
    roster = tiny_roster()
    idx = build_index(log, layout, roster)
    assert set(idx.staff.keys()) == {PHYSICIAN, NURSE, HOUSEKEEPER}
    assert len(idx.staff[PHYSICIAN].walk_intervals) == 2
    assert len(idx.staff[HOUSEKEEPER].cleaning_intervals) == 1
