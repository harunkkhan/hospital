"""Events: byte-stable JSONL round-trip, discriminator dispatch, ordering."""

from __future__ import annotations

import json

from hospital.core import (
    ArrivalMode,
    BayAssigned,
    BayId,
    Duration,
    EsiAcuity,
    EventLog,
    NodeId,
    PatientArrived,
    PatientId,
    SimTime,
    StaffId,
    StaffMoved,
    TriageCompleted,
)


def _sample_log() -> EventLog:
    log = EventLog()
    s0 = log.append(
        PatientArrived(
            occurred_at=SimTime(100), patient=PatientId("p1"), mode=ArrivalMode.AMBULANCE
        )
    )
    # Same timestamp as arrival — sequence disambiguates the µs collision.
    log.append(
        StaffMoved(
            occurred_at=SimTime(100),
            staff=StaffId("s1"),
            edge=(NodeId("a"), NodeId("b")),
            seconds=Duration(5_000),
        )
    )
    log.append(
        TriageCompleted(occurred_at=SimTime(50), patient=PatientId("p1"), esi=EsiAcuity.ESI2),
        caused_by=s0,
    )
    log.append(
        BayAssigned(
            occurred_at=SimTime(300), patient=PatientId("p1"), bay=BayId("bay-2"), by="solver"
        )
    )
    return log


def test_jsonl_round_trip_is_byte_stable() -> None:
    log = _sample_log()
    text = log.to_jsonl()
    reparsed = EventLog.from_jsonl(text)
    assert reparsed.to_jsonl() == text
    assert len(reparsed) == len(log)


def test_discriminator_dispatch_reconstructs_types() -> None:
    log = EventLog.from_jsonl(_sample_log().to_jsonl())
    kinds = [type(env.event).__name__ for env in log]
    assert kinds == ["PatientArrived", "StaffMoved", "TriageCompleted", "BayAssigned"]
    first = next(iter(log)).event
    assert isinstance(first, PatientArrived)
    assert first.mode == ArrivalMode.AMBULANCE


def test_sequence_is_monotonic_and_caused_by_preserved() -> None:
    log = _sample_log()
    assert [env.sequence for env in log] == [0, 1, 2, 3]
    reparsed = EventLog.from_jsonl(log.to_jsonl())
    triage = next(env for env in reparsed if isinstance(env.event, TriageCompleted))
    assert triage.caused_by == 0


def test_ordered_by_occurred_at_then_sequence() -> None:
    ordered = _sample_log().ordered()
    keys = [(env.event.occurred_at.root, env.sequence) for env in ordered]
    assert keys == sorted(keys)
    # occurred_at 50 (seq2) < 100 (seq0) < 100 (seq1) < 300 (seq3)
    assert keys == [(50, 2), (100, 0), (100, 1), (300, 3)]


def test_jsonl_has_no_floats() -> None:
    text = _sample_log().to_jsonl()
    for line in text.splitlines():
        payload = json.loads(line)
        event = payload["event"]
        assert isinstance(payload["sequence"], int)
        assert isinstance(event["occurred_at"], int)
        if "seconds" in event:
            assert isinstance(event["seconds"], int)
        assert "e-" not in line and "e+" not in line
