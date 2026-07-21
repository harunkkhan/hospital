"""INTERNAL single-pass reconstruction: ``EventLog`` -> ``EventIndex``.

Not public (doc 05 §2): ``fold``, ``waits``, ``bottleneck``, and ``utilization``
all read this ONE index rather than re-scanning the log four times (anti-dup,
doc 00 §5.2) — the only way to guarantee all four decompositions observe
*identical* milestones for the same patient/bay/staff.

Ordering is the log's own canonical ``(occurred_at, sequence)`` order
(``EventLog.ordered()``, doc D8) — µs collisions are common, so "first
``ProviderVisitStarted`` fixes ``pv``" is only well-defined under that total
order.

Start/complete pairing is a per-``(patient, activity)`` open-stack: push on
``*_started``/``*_ordered``, pop on the matching ``*_completed``/``*_resulted``
(never positional). An unmatched ``*_started`` still open at the end of the log
is a legitimate WIP service interval and is simply left out of the closed
interval list (callers clip open work at the horizon where relevant); a
``*_completed`` with no matching open ``*_started`` is a causally-impossible,
corrupt log and raises :class:`~hospital.core.errors.ZeroTimeCycle`... no wait
see module docstring below for the actual guard.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from hospital.core import (
    Activity,
    BayAssigned,
    BayCleaningCompleted,
    BayCleaningStarted,
    BayId,
    BayRequested,
    DischargeCompleted,
    DischargeStarted,
    DispositionDecided,
    DispositionKind,
    DocumentationCompleted,
    DocumentationStarted,
    EsiAcuity,
    EventLog,
    FloorLayout,
    NodeId,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientArrived,
    PatientId,
    PatientMoved,
    ProviderVisitCompleted,
    ProviderVisitStarted,
    SimTime,
    StaffId,
    StaffMember,
    StaffMoved,
    StaffRole,
    TestOrdered,
    TestResulted,
    TriageCompleted,
    TriageStarted,
    ZeroTimeCycle,
    ZoneType,
)

__all__ = [
    "BayCycle",
    "EventIndex",
    "PatientTrace",
    "ServiceInterval",
    "StaffTrace",
    "TestInterval",
    "build_index",
]


@dataclass(frozen=True, slots=True)
class ServiceInterval:
    """A closed ``[start, end]`` service interval, optionally staff-attributed."""

    kind: str  # "triage" | "provider_visit" | "nurse_visit" | "documentation"
    start: SimTime
    end: SimTime
    staff: StaffId | None


@dataclass(frozen=True, slots=True)
class TestInterval:
    """A ``TestOrdered -> TestResulted`` sojourn (imaging/lab contention proxy)."""

    activity: Activity
    start: SimTime
    end: SimTime


@dataclass(frozen=True, slots=True)
class PatientTrace:
    """Per-patient milestones + service intervals. Milestones are ``None`` if unreached."""

    patient: PatientId
    arrival: SimTime  # a
    esi: EsiAcuity | None = None
    triage_start: SimTime | None = None  # ts0
    triage_end: SimTime | None = None  # ts1
    bay_requested_at: SimTime | None = None
    bay: BayId | None = None
    bay_ready: SimTime | None = None  # br (BayAssigned)
    bay_arrival: SimTime | None = None  # ba (physically in bay)
    provider_start: SimTime | None = None  # pv (first ProviderVisitStarted)
    nurse_start: SimTime | None = None  # first NurseVisitStarted
    disposition_time: SimTime | None = None  # dd
    disposition: DispositionKind | None = None
    discharge_start: SimTime | None = None  # ds
    exit: SimTime | None = None  # de (DischargeCompleted)
    triage_interval: ServiceInterval | None = None
    provider_intervals: tuple[ServiceInterval, ...] = ()
    nurse_intervals: tuple[ServiceInterval, ...] = ()
    documentation_intervals: tuple[ServiceInterval, ...] = ()
    test_intervals: tuple[TestInterval, ...] = ()


@dataclass(frozen=True, slots=True)
class BayCycle:
    """One occupy-to-clean cycle of a bay (may be open/incomplete at the horizon)."""

    bay: BayId
    zone_type: ZoneType | None
    occupant: PatientId | None
    bay_arrival: SimTime | None  # ba
    disposition_time: SimTime | None  # dd
    exit: SimTime | None  # de
    clean_start: SimTime | None
    clean_end: SimTime | None
    clean_staff: StaffId | None


@dataclass(frozen=True, slots=True)
class StaffTrace:
    """Per-staff walk + cleaning intervals (direct-care/documentation live on patients)."""

    staff: StaffId
    role: StaffRole
    walk_intervals: tuple[tuple[SimTime, SimTime], ...] = ()
    cleaning_intervals: tuple[tuple[SimTime, SimTime], ...] = ()


@dataclass(frozen=True, slots=True)
class EventIndex:
    """The single reconstruction shared by ``fold``/``waits``/``bottleneck``/``utilization``."""

    patients: Mapping[PatientId, PatientTrace]
    bays: Mapping[BayId, tuple[BayCycle, ...]]
    staff: Mapping[StaffId, StaffTrace]


@dataclass(slots=True)
class _MutablePatient:
    patient: PatientId
    arrival: SimTime
    esi: EsiAcuity | None = None
    triage_start: SimTime | None = None
    triage_end: SimTime | None = None
    triage_staff: StaffId | None = None
    bay_requested_at: SimTime | None = None
    bay: BayId | None = None
    bay_ready: SimTime | None = None
    bay_arrival: SimTime | None = None
    provider_start: SimTime | None = None
    nurse_start: SimTime | None = None
    disposition_time: SimTime | None = None
    disposition: DispositionKind | None = None
    discharge_start: SimTime | None = None
    exit: SimTime | None = None
    provider_intervals: list[ServiceInterval] = field(default_factory=list[ServiceInterval])
    nurse_intervals: list[ServiceInterval] = field(default_factory=list[ServiceInterval])
    documentation_intervals: list[ServiceInterval] = field(default_factory=list[ServiceInterval])
    test_intervals: list[TestInterval] = field(default_factory=list[TestInterval])


@dataclass(slots=True)
class _MutableBayCycle:
    bay: BayId
    zone_type: ZoneType | None
    occupant: PatientId | None
    bay_arrival: SimTime | None = None
    disposition_time: SimTime | None = None
    exit: SimTime | None = None
    clean_start: SimTime | None = None
    clean_end: SimTime | None = None
    clean_staff: StaffId | None = None

    def freeze(self) -> BayCycle:
        return BayCycle(
            bay=self.bay,
            zone_type=self.zone_type,
            occupant=self.occupant,
            bay_arrival=self.bay_arrival,
            disposition_time=self.disposition_time,
            exit=self.exit,
            clean_start=self.clean_start,
            clean_end=self.clean_end,
            clean_staff=self.clean_staff,
        )


def build_index(log: EventLog, layout: FloorLayout, roster: tuple[StaffMember, ...]) -> EventIndex:
    """The one linear pass: ``EventLog`` -> per-patient/bay/staff traces.

    Raises :class:`~hospital.core.errors.ZeroTimeCycle` for a causally
    impossible bay cycle (``BayCleaningCompleted`` before its own
    ``DispositionDecided``, or a negative cleaning interval) — guards a
    malformed log at the source rather than letting a negative duration
    silently poison every downstream sum.
    """
    bay_zone_type: dict[BayId, ZoneType] = {b.id: b.zone_type for b in layout.bays}
    bay_node: dict[BayId, NodeId] = {b.id: b.node for b in layout.bays}

    patients: dict[PatientId, _MutablePatient] = {}
    bay_cycles: dict[BayId, list[BayCycle]] = defaultdict(list)
    current_cycle: dict[BayId, _MutableBayCycle] = {}
    staff_walk: dict[StaffId, list[tuple[SimTime, SimTime]]] = defaultdict(list)
    staff_clean: dict[StaffId, list[tuple[SimTime, SimTime]]] = defaultdict(list)

    open_provider: dict[PatientId, list[tuple[SimTime, StaffId]]] = defaultdict(list)
    open_nurse: dict[PatientId, list[tuple[SimTime, StaffId]]] = defaultdict(list)
    open_doc: dict[PatientId, list[tuple[SimTime, StaffId]]] = defaultdict(list)
    open_test: dict[tuple[PatientId, Activity], list[SimTime]] = defaultdict(list)

    def patient_of(pid: PatientId) -> _MutablePatient:
        p = patients.get(pid)
        if p is None:
            # A visit/attribute event for a patient never explicitly "arrived" in
            # this log slice (e.g. arrival before the log window). Synthesize a
            # trace anchored at this instant so downstream code has a milestone
            # to branch on rather than crashing on a missing key.
            p = _MutablePatient(patient=pid, arrival=SimTime(0))
            patients[pid] = p
        return p

    for env in log.ordered():
        e = env.event
        t = e.occurred_at

        if isinstance(e, PatientArrived):
            patients[e.patient] = _MutablePatient(patient=e.patient, arrival=t)
        elif isinstance(e, TriageStarted):
            p = patient_of(e.patient)
            p.triage_start = t
            p.triage_staff = e.staff
        elif isinstance(e, TriageCompleted):
            p = patient_of(e.patient)
            p.triage_end = t
            p.esi = e.esi
        elif isinstance(e, BayRequested):
            p = patient_of(e.patient)
            if p.bay_requested_at is None:
                p.bay_requested_at = t
        elif isinstance(e, BayAssigned):
            p = patient_of(e.patient)
            p.bay = e.bay
            p.bay_ready = t
            current_cycle[e.bay] = _MutableBayCycle(
                bay=e.bay, zone_type=bay_zone_type.get(e.bay), occupant=e.patient
            )
        elif isinstance(e, PatientMoved):
            p = patients.get(e.patient)
            if p is not None and p.bay is not None and bay_node.get(p.bay) == e.edge[1]:
                p.bay_arrival = t
                cyc = current_cycle.get(p.bay)
                if cyc is not None and cyc.occupant == e.patient:
                    cyc.bay_arrival = t
        elif isinstance(e, ProviderVisitStarted):
            p = patient_of(e.patient)
            if p.provider_start is None:
                p.provider_start = t
            open_provider[e.patient].append((t, e.staff))
        elif isinstance(e, ProviderVisitCompleted):
            stack = open_provider[e.patient]
            if stack:
                start, staff = stack.pop()
                patient_of(e.patient).provider_intervals.append(
                    ServiceInterval(kind="provider_visit", start=start, end=t, staff=staff)
                )
        elif isinstance(e, NurseVisitStarted):
            p = patient_of(e.patient)
            if p.nurse_start is None:
                p.nurse_start = t
            open_nurse[e.patient].append((t, e.staff))
        elif isinstance(e, NurseVisitCompleted):
            stack = open_nurse[e.patient]
            if stack:
                start, staff = stack.pop()
                patient_of(e.patient).nurse_intervals.append(
                    ServiceInterval(kind="nurse_visit", start=start, end=t, staff=staff)
                )
        elif isinstance(e, DocumentationStarted):
            open_doc[e.patient].append((t, e.staff))
        elif isinstance(e, DocumentationCompleted):
            stack = open_doc[e.patient]
            if stack:
                start, staff = stack.pop()
                patient_of(e.patient).documentation_intervals.append(
                    ServiceInterval(kind="documentation", start=start, end=t, staff=staff)
                )
        elif isinstance(e, TestOrdered):
            open_test[(e.patient, e.activity)].append(t)
        elif isinstance(e, TestResulted):
            stack = open_test[(e.patient, e.activity)]
            if stack:
                start = stack.pop()
                patient_of(e.patient).test_intervals.append(
                    TestInterval(activity=e.activity, start=start, end=t)
                )
        elif isinstance(e, DispositionDecided):
            p = patient_of(e.patient)
            p.disposition_time = t
            p.disposition = e.disposition
            if p.bay is not None and (cyc := current_cycle.get(p.bay)) is not None:
                cyc.disposition_time = t
        elif isinstance(e, DischargeStarted):
            patient_of(e.patient).discharge_start = t
        elif isinstance(e, DischargeCompleted):
            p = patient_of(e.patient)
            p.exit = t
            if p.bay is not None and (cyc := current_cycle.get(p.bay)) is not None:
                cyc.exit = t
        elif isinstance(e, BayCleaningStarted):
            cyc = current_cycle.get(e.bay)
            if cyc is not None:
                cyc.clean_start = t
        elif isinstance(e, BayCleaningCompleted):
            cyc = current_cycle.get(e.bay)
            if cyc is not None:
                if cyc.disposition_time is not None and t < cyc.disposition_time:
                    raise ZeroTimeCycle(
                        f"bay {e.bay}: cleaning completed before its disposition_decided"
                    )
                if cyc.clean_start is not None and t < cyc.clean_start:
                    raise ZeroTimeCycle(f"bay {e.bay}: negative-length cleaning interval")
                cyc.clean_end = t
                cyc.clean_staff = e.staff
                if cyc.clean_start is not None:
                    staff_clean[e.staff].append((cyc.clean_start, t))
                bay_cycles[e.bay].append(cyc.freeze())
                del current_cycle[e.bay]
        elif isinstance(e, StaffMoved):
            # occurred_at is the edge ARRIVAL instant (nuance E); the walk
            # occupied [occurred_at - seconds, occurred_at]. SimTime's type
            # algebra has no SimTime-Duration, only SimTime+Duration, so the
            # start instant is `t + (-seconds)`.
            staff_walk[e.staff].append((t + (-e.seconds), t))
        # StaffIdle / DisruptionInjected / Vitals* / Deterioration* / EmergencyRaised
        # carry no structured payload this package folds on (no staff-affecting
        # window on DisruptionInjected, no idle-*end* marker on StaffIdle) —
        # deliberately not consumed here (see analysis nuances, judgment call).

    # Any bay cycle still open at the end of the log is legitimate WIP occupancy
    # (patient/bay still mid-cycle at the horizon) — keep it, un-cleaned.
    for cyc in current_cycle.values():
        bay_cycles[cyc.bay].append(cyc.freeze())

    # ba fallback: if bay-arrival couldn't be resolved from PatientMoved, fall
    # back to the BayAssigned decision instant (slightly overstates occupancy).
    frozen_patients: dict[PatientId, PatientTrace] = {}
    for pid, p in patients.items():
        bay_arrival = p.bay_arrival if p.bay_arrival is not None else p.bay_ready
        triage_interval = (
            ServiceInterval(
                kind="triage", start=p.triage_start, end=p.triage_end, staff=p.triage_staff
            )
            if p.triage_start is not None and p.triage_end is not None
            else None
        )
        frozen_patients[pid] = PatientTrace(
            patient=pid,
            arrival=p.arrival,
            esi=p.esi,
            triage_start=p.triage_start,
            triage_end=p.triage_end,
            bay_requested_at=p.bay_requested_at,
            bay=p.bay,
            bay_ready=p.bay_ready,
            bay_arrival=bay_arrival,
            provider_start=p.provider_start,
            nurse_start=p.nurse_start,
            disposition_time=p.disposition_time,
            disposition=p.disposition,
            discharge_start=p.discharge_start,
            exit=p.exit,
            triage_interval=triage_interval,
            provider_intervals=tuple(p.provider_intervals),
            nurse_intervals=tuple(p.nurse_intervals),
            documentation_intervals=tuple(p.documentation_intervals),
            test_intervals=tuple(p.test_intervals),
        )

    frozen_bays: dict[BayId, tuple[BayCycle, ...]] = {
        bay: tuple(cycles) for bay, cycles in bay_cycles.items()
    }

    frozen_staff: dict[StaffId, StaffTrace] = {
        member.id: StaffTrace(
            staff=member.id,
            role=member.role,
            walk_intervals=tuple(staff_walk.get(member.id, ())),
            cleaning_intervals=tuple(staff_clean.get(member.id, ())),
        )
        for member in roster
    }

    return EventIndex(patients=frozen_patients, bays=frozen_bays, staff=frozen_staff)
