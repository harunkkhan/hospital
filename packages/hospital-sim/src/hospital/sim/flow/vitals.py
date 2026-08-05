"""Vitals monitoring and the emergency escalation it can trigger (doc 06 §3/§7).

One SimPy process per monitored patient. The trajectory itself is **pre-sampled**
by ``data.generate_vitals`` — a pure function of ``(seed, patient id)`` — and this
process only *reveals* it, one reading per cadence tick. That split is the same
construction/sampling rule the rest of the engine follows: ``data`` draws, ``sim``
plays the draw out in time.

It also means the vitals a patient will have do not depend on how often anyone looks:
``data.vitals`` advances its latent walk on a fixed grid and keys the measurement noise
by elapsed time, so halving ``cadence`` returns twice the readings rather than a
different patient. ``span`` is not in that guarantee — it places the deterioration
onset, so it is a construction parameter (see :func:`~hospital.data.vitals.generate_vitals`)
and any comparison across arms has to hold it fixed.

Each tick appends a ``VitalsSampled`` stamped with its NEWS2 total (the rubric
lives in ``core.vitals`` precisely so this module can reach it). If a
:class:`~hospital.core.seam.RiskMonitor` is injected, the reading is offered to
it; on an escalating verdict **the engine** — never the monitor — appends
``DeteriorationDetected`` then ``EmergencyRaised``, and raises a top-priority
resuscitation task for dispatch to answer.

**Strictly opt-in.** ``run_replication`` starts no vitals process unless a
:class:`VitalsWatch` is passed. A run without one appends no ``VitalsSampled`` and
is byte-identical to the M1/M2 engine — which is what keeps the existing goldens
and the API's byte-identity tests meaningful rather than merely re-baselined.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from hospital.core import (
    Duration,
    EsiAcuity,
    FrozenModel,
    StaffRole,
    minutes,
    news2_score,
)
from hospital.core.enums import Activity
from hospital.core.events import (
    DeteriorationDetected,
    EmergencyRaised,
    VitalsSampled,
)
from hospital.data.vitals import generate_vitals
from hospital.sim.physics.executor import PriorityTier

if TYPE_CHECKING:
    from collections.abc import Generator

    import simpy

    from hospital.core import EventLog, Patient, RandomStreams, RiskMonitor
    from hospital.sim.physics.executor import TaskExecutor
    from hospital.sim.physics.world import World

# How long a raised emergency occupies the responder. Short and fixed: this is the
# immediate bedside response, not the workup that follows it.
EMERGENCY_RESPONSE: Final[Duration] = minutes(8)


class VitalsWatch(FrozenModel):
    """Opt-in vitals monitoring for a run.

    ``span`` bounds how far past arrival a patient is sampled. It is a cap, not a
    prediction of the stay: the process also stops the moment the patient is
    discharged, so a short visit does not keep emitting readings for a bed that
    is already being cleaned.
    """

    cadence: Duration = minutes(5)
    span: Duration = minutes(360)
    # Only patients at or above this acuity are monitored (ESI is inverted:
    # ESI1 is the sickest). Monitoring every walk-in would bury the signal in
    # readings nobody would act on.
    monitor_at_or_above: EsiAcuity = EsiAcuity.ESI3


def _raise_emergency(
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    patient: Patient,
    news2: int,
) -> None:
    """Log the escalation and put a top-priority task in front of dispatch.

    The engine writes both events — the monitor cannot (nuance 1.4: one writer).
    The task is a real unit of work at the patient's location, so the *existing*
    dispatch policy answers it by its own rules; boosting it is what makes it
    jump the queue, rather than a separate emergency code path that would bypass
    the seam entirely.
    """
    now = executor.now()
    event_log.append(DeteriorationDetected(occurred_at=now, patient=patient.id, news2=news2))
    event_log.append(EmergencyRaised(occurred_at=now, patient=patient.id))

    task = world.add_task(
        kind="provider_visit",
        patient=patient.id,
        at=world.patient_at(patient.id),
        required_role=StaffRole.PHYSICIAN,
        activity=Activity.PROVIDER_VISIT,
        duration=EMERGENCY_RESPONSE,
        esi=EsiAcuity.ESI1,
    )
    world.boost_task(task.spec.id)
    # Ask for a decision now rather than waiting for the next scheduled tick: an
    # emergency that sits until the policy happens to wake is not an emergency.
    world.request_decision()


def vitals_process(
    env: simpy.Environment,
    world: World,
    executor: TaskExecutor,
    event_log: EventLog,
    patient: Patient,
    *,
    streams: RandomStreams,
    watch: VitalsWatch,
    stay: simpy.Process | None = None,
    monitor: RiskMonitor | None = None,
) -> Generator[simpy.Event, object]:
    """Reveal one patient's pre-sampled vitals, escalating at most once.

    ``stay`` is the patient's own flow process. Monitoring ends when it does —
    the world never forgets a patient, so their process finishing is the only
    honest signal that they have actually left, and without it a discharged
    patient would keep emitting readings for a bed already being cleaned.
    """
    del env
    stream = generate_vitals(patient, streams, until=watch.span, cadence=watch.cadence)
    escalated = False
    previous = 0

    for sample in stream.samples:
        step = sample.elapsed.root - previous
        previous = sample.elapsed.root
        if step > 0:
            yield executor.delay(Duration(step), PriorityTier.COMPLETION)
        if stay is not None and stay.processed:
            return

        scored = news2_score(sample)
        event = VitalsSampled(occurred_at=executor.now(), patient=patient.id, news2=scored.total)
        event_log.append(event)

        if monitor is None or escalated:
            continue
        assessment = monitor.observe(event, sample)
        if assessment is not None and assessment.escalate:
            # Once only. A model that stays above threshold for an hour would
            # otherwise raise a fresh emergency every tick, and twelve identical
            # pages are how a real alarm gets ignored.
            escalated = True
            _raise_emergency(world, executor, event_log, patient, scored.total)


__all__ = ["EMERGENCY_RESPONSE", "VitalsWatch", "vitals_process"]
