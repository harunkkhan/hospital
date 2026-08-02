"""Importable synthetic-log builders for ``hospital-forecast`` (no ``conftest``; D10).

``forecast`` may import only ``core`` and ``data``, so these tests cannot ask
``sim`` for an ``EventLog``. They synthesize one instead — which is better for
this package anyway: the generator here knows the **true** arrival rates and the
**true** service-time parameters it wrote, so a fit can be scored against ground
truth rather than against whatever the engine happened to produce.

The synthetic floor is deliberately simple (arrive → triage → bay → provider
visit(s) → nurse visit(s) → tests → documentation → discharge) but it emits real
``core.events`` in canonical order, so every extractor is exercised on the same
event vocabulary a real run produces.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from hospital.core import (
    Activity,
    ArrivalMode,
    BayId,
    DispositionKind,
    Duration,
    EsiAcuity,
    EventLog,
    NodeId,
    OperatingWeek,
    Patient,
    PatientId,
    RandomStreams,
    SimTime,
    StaffId,
    WorkupNeeds,
    ZoneType,
    hours,
    minutes,
    sample_lognormal,
)
from hospital.core.events import (
    BayAssigned,
    BayCleaningCompleted,
    BayCleaningStarted,
    DischargeCompleted,
    DischargeStarted,
    DispositionDecided,
    DocumentationCompleted,
    DocumentationStarted,
    NurseVisitCompleted,
    NurseVisitStarted,
    PatientArrived,
    ProviderVisitCompleted,
    ProviderVisitStarted,
    StaffIdle,
    TestOrdered,
    TestResulted,
    TriageCompleted,
    TriageStarted,
)

# Ground truth the fits must recover. Keyed the same way `service_time` keys its
# table: (activity, esi, complaint).
TRUE_SERVICE: Final[dict[tuple[Activity, EsiAcuity, str], tuple[float, float]]] = {
    (Activity.PROVIDER_VISIT, EsiAcuity.ESI2, "chest_pain"): (900.0, 0.45),
    (Activity.PROVIDER_VISIT, EsiAcuity.ESI3, "chest_pain"): (700.0, 0.40),
    (Activity.PROVIDER_VISIT, EsiAcuity.ESI4, "chest_pain"): (450.0, 0.35),
    (Activity.PROVIDER_VISIT, EsiAcuity.ESI2, "abdominal"): (1020.0, 0.50),
    (Activity.PROVIDER_VISIT, EsiAcuity.ESI3, "abdominal"): (780.0, 0.42),
    (Activity.PROVIDER_VISIT, EsiAcuity.ESI4, "abdominal"): (500.0, 0.38),
    (Activity.NURSE_VISIT, EsiAcuity.ESI2, "chest_pain"): (420.0, 0.40),
    (Activity.NURSE_VISIT, EsiAcuity.ESI3, "chest_pain"): (360.0, 0.38),
    (Activity.NURSE_VISIT, EsiAcuity.ESI4, "chest_pain"): (300.0, 0.35),
    (Activity.NURSE_VISIT, EsiAcuity.ESI2, "abdominal"): (450.0, 0.42),
    (Activity.NURSE_VISIT, EsiAcuity.ESI3, "abdominal"): (390.0, 0.40),
    (Activity.NURSE_VISIT, EsiAcuity.ESI4, "abdominal"): (330.0, 0.36),
}

_TRIAGE: Final[tuple[float, float]] = (240.0, 0.30)
_DOCUMENTATION: Final[tuple[float, float]] = (480.0, 0.35)
_CLEANING: Final[tuple[float, float]] = (600.0, 0.25)
_COMPLAINTS: Final[tuple[str, ...]] = ("chest_pain", "abdominal")
_STATION: Final[NodeId] = NodeId("station_0")

# A weekday/weekend by hour-of-day arrival shape with an obvious afternoon peak,
# so a fitted intensity has something to be right or wrong about.
_HOURLY_SHAPE: Final[tuple[float, ...]] = (
    0.5, 0.4, 0.35, 0.3, 0.3, 0.4, 0.7, 1.0,
    1.3, 1.5, 1.6, 1.6, 1.5, 1.5, 1.6, 1.7,
    1.8, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.6,
)  # fmt: skip
_DOW_SHAPE: Final[tuple[float, ...]] = (1.15, 1.0, 0.95, 0.95, 1.05, 1.2, 1.25)


class SynthWeek(NamedTuple):
    """One synthesized week plus the ground truth used to build it."""

    log: EventLog
    roster: dict[PatientId, Patient]
    week: OperatingWeek
    base_rate_per_hour: float

    def true_lambda(self, hour_of_week: int) -> float:
        """The λ actually used for ``hour_of_week`` — what a fit must recover."""
        return (
            self.base_rate_per_hour
            * _HOURLY_SHAPE[hour_of_week % 24]
            * _DOW_SHAPE[(hour_of_week // 24) % 7]
        )


def _draw_patient(streams: RandomStreams, index: int, at: SimTime, week_index: int) -> Patient:
    key = ("synth", "patient", index)
    esi_g = streams.substream(*key, "esi")
    complaint_g = streams.substream(*key, "complaint")
    workup_g = streams.substream(*key, "workup")
    mode_g = streams.substream(*key, "mode")
    esi = EsiAcuity(int(esi_g.choice([2, 3, 3, 3, 4, 4])))
    complaint = str(complaint_g.choice(_COMPLAINTS))
    imaging = (ZoneType.IMAGING,) if workup_g.random() < 0.35 else ()
    return Patient(
        # Week-scoped: pooling several weeks must not collide two patients into
        # one, which would silently corrupt every per-patient join.
        id=PatientId(f"pt_{week_index:02d}_{index:05d}"),
        arrival_time=at,
        arrival_mode=ArrivalMode.AMBULANCE if mode_g.random() < 0.2 else ArrivalMode.WALK_IN,
        esi=esi,
        complaint=complaint,
        isolation_required=bool(workup_g.random() < 0.08),
        workup=WorkupNeeds(
            provider_visits=1 + int(workup_g.random() < 0.4),
            nurse_visits=1 + int(workup_g.random() < 0.5),
            imaging=imaging,
            labs=int(workup_g.integers(0, 3)),
            procedures=0,
        ),
    )


def _service(
    streams: RandomStreams, key: tuple[str | int, ...], params: tuple[float, float]
) -> Duration:
    mean_s, cv = params
    return sample_lognormal(streams.substream(*key), mean_s, cv)


def synth_week(
    *,
    seed: int = 4,
    days: int = 7,
    base_rate_per_hour: float = 2.4,
    week_index: int = 0,
) -> SynthWeek:
    """Synthesize one week of events from a known λ profile and known service times.

    ``week_index`` shifts every RNG key, so successive weeks are independent
    draws from the *same* generating process — which is what the rolling-origin
    holdout protocol needs (train on earlier weeks, score on a later one).
    """
    streams = RandomStreams(seed + 1_000 * week_index)
    week = OperatingWeek(start=SimTime(0), end=SimTime(hours(24 * days).root))
    log = EventLog()
    roster: dict[PatientId, Patient] = {}

    triage_staff = StaffId("staff_triage_000")
    provider = StaffId("staff_physician_000")
    nurse = StaffId("staff_nurse_000")

    # Draw arrival instants per hour from the known intensity.
    arrivals: list[tuple[SimTime, Patient]] = []
    index = 0
    for hour in range(24 * days):
        rate = base_rate_per_hour * _HOURLY_SHAPE[hour % 24] * _DOW_SHAPE[(hour // 24) % 7]
        g = streams.substream("synth", "arrivals", hour)
        for _ in range(int(g.poisson(rate))):
            offset = int(g.random() * hours(1).root)
            at = SimTime(hours(hour).root + offset)
            patient = _draw_patient(streams, index, at, week_index)
            arrivals.append((at, patient))
            roster[patient.id] = patient
            index += 1
    arrivals.sort(key=lambda pair: pair[0].root)

    # Sized so the floor genuinely contends at peak: offered load is roughly
    # 3.6 arrivals/h at ~1h each, so six bays queue in the afternoon and clear
    # overnight. A roomier pool would leave `queue_len_by_esi` identically zero
    # and the congestion features would be untested dead weight.
    bays = [BayId(f"bay_{i:02d}") for i in range(6)]
    free_at: dict[BayId, int] = {b: 0 for b in bays}

    # Shift presence: a handful of staff check in each hour, so `staff_on_shift`
    # (distinct staff seen in the trailing hour) varies instead of sitting at 0.
    for hour in range(24 * days):
        on_duty = 3 if 8 <= hour % 24 < 20 else 2
        for who in range(on_duty):
            log.append(
                StaffIdle(
                    occurred_at=SimTime(hours(hour).root + who),
                    staff=StaffId(f"staff_nurse_{who:03d}"),
                    at=_STATION,
                )
            )

    for order, (at, patient) in enumerate(arrivals):
        key = ("synth", "svc", week_index, order)
        log.append(PatientArrived(occurred_at=at, patient=patient.id, mode=patient.arrival_mode))

        triage_start = SimTime(at.root + minutes(2).root)
        log.append(TriageStarted(occurred_at=triage_start, patient=patient.id, staff=triage_staff))
        triage_dur = _service(streams, (*key, "triage"), _TRIAGE)
        triaged_at = SimTime(triage_start.root + triage_dur.root)
        log.append(TriageCompleted(occurred_at=triaged_at, patient=patient.id, esi=patient.esi))

        bay = min(bays, key=lambda b: free_at[b])
        placed_at = SimTime(max(triaged_at.root, free_at[bay]))
        log.append(BayAssigned(occurred_at=placed_at, patient=patient.id, bay=bay, by="baseline"))

        cursor = placed_at.root
        for visit in range(patient.workup.provider_visits):
            params = TRUE_SERVICE.get(
                (Activity.PROVIDER_VISIT, patient.esi, patient.complaint), (700.0, 0.4)
            )
            log.append(
                ProviderVisitStarted(
                    occurred_at=SimTime(cursor), patient=patient.id, staff=provider
                )
            )
            duration = _service(streams, (*key, "provider", visit), params)
            cursor += duration.root
            log.append(
                ProviderVisitCompleted(
                    occurred_at=SimTime(cursor), patient=patient.id, staff=provider
                )
            )

        for visit in range(patient.workup.nurse_visits):
            params = TRUE_SERVICE.get(
                (Activity.NURSE_VISIT, patient.esi, patient.complaint), (360.0, 0.38)
            )
            log.append(
                NurseVisitStarted(occurred_at=SimTime(cursor), patient=patient.id, staff=nurse)
            )
            duration = _service(streams, (*key, "nurse", visit), params)
            cursor += duration.root
            log.append(
                NurseVisitCompleted(occurred_at=SimTime(cursor), patient=patient.id, staff=nurse)
            )

        for zone in patient.workup.imaging:
            del zone
            log.append(
                TestOrdered(
                    occurred_at=SimTime(cursor), patient=patient.id, activity=Activity.IMAGING
                )
            )
            cursor += minutes(25).root
            log.append(
                TestResulted(
                    occurred_at=SimTime(cursor), patient=patient.id, activity=Activity.IMAGING
                )
            )
        for lab in range(patient.workup.labs):
            del lab
            log.append(
                TestOrdered(occurred_at=SimTime(cursor), patient=patient.id, activity=Activity.LAB)
            )
            cursor += minutes(18).root
            log.append(
                TestResulted(occurred_at=SimTime(cursor), patient=patient.id, activity=Activity.LAB)
            )

        log.append(
            DocumentationStarted(occurred_at=SimTime(cursor), patient=patient.id, staff=provider)
        )
        cursor += _service(streams, (*key, "doc"), _DOCUMENTATION).root
        log.append(
            DocumentationCompleted(occurred_at=SimTime(cursor), patient=patient.id, staff=provider)
        )
        log.append(
            DispositionDecided(
                occurred_at=SimTime(cursor),
                patient=patient.id,
                disposition=DispositionKind.DISCHARGE,
            )
        )
        log.append(DischargeStarted(occurred_at=SimTime(cursor), patient=patient.id))
        cursor += minutes(6).root
        exit_at = SimTime(cursor)
        log.append(DischargeCompleted(occurred_at=exit_at, patient=patient.id))

        log.append(BayCleaningStarted(occurred_at=exit_at, bay=bay, staff=nurse))
        clean_done = exit_at.root + _service(streams, (*key, "clean"), _CLEANING).root
        log.append(BayCleaningCompleted(occurred_at=SimTime(clean_done), bay=bay, staff=nurse))
        free_at[bay] = clean_done

    return SynthWeek(log=log, roster=roster, week=week, base_rate_per_hour=base_rate_per_hour)


def synth_weeks(n: int, *, seed: int = 4, days: int = 7, rate: float = 2.4) -> list[SynthWeek]:
    """``n`` independent weeks from the same process — the rolling-origin corpus."""
    return [
        synth_week(seed=seed, days=days, base_rate_per_hour=rate, week_index=i) for i in range(n)
    ]
