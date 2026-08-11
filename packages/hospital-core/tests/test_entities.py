"""Frozen entity descriptors: the derived ``care_deadline`` and what it must not become."""

from __future__ import annotations

import json

from _fixtures import patient

from hospital.core import (
    CARE_SLA_BY_ACUITY,
    EsiAcuity,
    NodeId,
    Patient,
    SimTime,
    StaffId,
    StaffMember,
    StaffRole,
    TimeWindow,
    hours,
    minutes,
)


def test_every_acuity_has_an_sla_and_sicker_means_sooner() -> None:
    """A missing row would raise mid-run; a non-monotone table would invert the point.

    The acuity-inversion trap this repo names elsewhere (higher urgency = *lower* ESI
    number) applies here too: the deadline must tighten as acuity rises, so ESI-1 has the
    earliest deadline and ESI-5 the latest.
    """
    assert set(CARE_SLA_BY_ACUITY) == set(EsiAcuity)
    slas = [CARE_SLA_BY_ACUITY[EsiAcuity(k)].root for k in (1, 2, 3, 4, 5)]
    assert slas == sorted(slas), "SLA must be non-decreasing as ESI number rises"
    assert CARE_SLA_BY_ACUITY[EsiAcuity.ESI1].root == 0  # immediate


def test_care_deadline_is_arrival_plus_the_acuity_sla() -> None:
    p = patient("p", EsiAcuity.ESI3).model_copy(update={"arrival_time": SimTime(minutes(5).root)})
    assert p.care_deadline == SimTime(minutes(35).root)  # 5 + 30


def test_a_sicker_patient_arriving_later_can_still_be_due_sooner() -> None:
    """The deadline is absolute, so acuity and arrival time genuinely trade off.

    This is the property that makes it usable as a measurement reference at all: an ESI-1
    who walked in twenty minutes after an ESI-5 is still due first.
    """
    late_critical = patient("crit", EsiAcuity.ESI1).model_copy(
        update={"arrival_time": SimTime(minutes(20).root)}
    )
    early_minor = patient("minor", EsiAcuity.ESI5).model_copy(update={"arrival_time": SimTime(0)})
    assert late_critical.care_deadline < early_minor.care_deadline


def test_care_deadline_is_derived_and_never_serialized() -> None:
    """It is a property, so it adds no byte to any log, golden trace, or wire contract.

    A stored field would be a second source of truth for something two frozen values
    already imply — and would have moved every committed golden the day it landed.
    """
    p = patient("p", EsiAcuity.ESI2)
    assert "care_deadline" not in p.model_dump()
    assert "care_deadline" not in json.loads(p.model_dump_json())
    assert "care_deadline" not in Patient.model_json_schema()["properties"]
    # Still reachable, and stable across reads (no hidden "now").
    assert p.care_deadline == p.care_deadline


# --- the shift schedule behind §3.3's "on_shift flag driven by the schedule" ---


def _shift(start_h: int, end_h: int) -> TimeWindow:
    return TimeWindow(start=SimTime(hours(start_h).root), end=SimTime(hours(end_h).root))


def _member(*shifts: TimeWindow) -> StaffMember:
    return StaffMember(
        id=StaffId("s1"),
        role=StaffRole.NURSE,
        home_station=NodeId("stat"),
        skills=frozenset(),
        shifts=shifts,
    )


def test_no_schedule_means_always_on_duty() -> None:
    """The pre-shift-awareness behaviour, and the reason nothing was re-baselined.

    A roster realized without ``shift_aware`` carries no windows, and must read as on duty
    at every instant — that is what makes every scenario written before schedules existed
    byte-identical.
    """
    always = _member()
    for hour in (0, 3, 12, 167):
        assert always.on_shift(SimTime(hours(hour).root))
    assert always.next_shift_start(SimTime(0)) is None
    assert always.on_duty_until(SimTime(0)) is None


def test_a_scheduled_member_is_off_duty_outside_their_shifts() -> None:
    day = _member(_shift(7, 19), _shift(31, 43))
    assert day.on_shift(SimTime(hours(12).root))
    assert not day.on_shift(SimTime(hours(3).root))
    assert not day.on_shift(SimTime(hours(25).root))
    assert day.on_shift(SimTime(hours(36).root))
    # Half-open, like every other window here: the end instant is already off duty.
    assert day.on_shift(SimTime(hours(18).root))
    assert not day.on_shift(SimTime(hours(19).root))


def test_next_shift_start_is_when_an_off_duty_member_becomes_available() -> None:
    """What the engine projects as ``busy_until``, so no policy needs to know about shifts."""
    day = _member(_shift(7, 19), _shift(31, 43))
    assert day.next_shift_start(SimTime(0)) == SimTime(hours(7).root)
    assert day.next_shift_start(SimTime(hours(20).root)) == SimTime(hours(31).root)
    # Nothing left in the horizon: unavailable for the remainder of it.
    assert day.next_shift_start(SimTime(hours(50).root)) is None


def test_on_duty_until_reports_the_end_of_the_covering_shift() -> None:
    day = _member(_shift(7, 19))
    assert day.on_duty_until(SimTime(hours(12).root)) == SimTime(hours(19).root)
    assert day.on_duty_until(SimTime(hours(3).root)) is None
