"""``generate_workload`` — the one arrival generator (registry: ``data.workload``).

``sim.experiment`` calls this; ``sim`` has none of its own (doc 00 §4). Unlike
``scenario.py``/``layout.py``, this module *samples* — every draw is
``streams.substream("workload", ...)`` (doc 02 §2.3). The crux is
content-addressable key locality: arrival times are drawn **per hour-bin**, and
every patient attribute is keyed by its own ``(h, k)`` stem (the hour and
within-hour index — never the global ``sequence``). Perturbing one input
(``hourly_profile[14]``, ``complaint_mix``, …) therefore re-draws only what
depends on it; every other patient's draws are bit-identical. This is the CRN
property that makes a scenario edit an isolated experimental variable rather
than a reshuffle of the week.

Surge (a ``kind="surge"`` ``DisruptionEvent``) adds *extra* arrivals on a
separate, independently-keyed domain (``"surge"`` / ``"surge_<attr>"``) so the
base week is bit-identical with or without a surge overlay — additive, never a
perturbation of the base draws.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hospital.core import (
    Duration,
    FrozenModel,
    OperatingWeek,
    Patient,
    PatientId,
    RandomStreams,
    SimTime,
    TimeWindow,
    hours,
    sample_poisson_arrivals,
)
from hospital.data.distributions import (
    bernoulli,
    sample_arrival_mode,
    sample_complaint,
    sample_esi,
    sample_workup,
)
from hospital.data.scenario import DisruptionSpec, WorkloadSpec

_HOUR_US = hours(1).root

_KeyFor = Callable[[str], tuple[str | int, ...]]


class PatientArrival(FrozenModel):
    """One realized arrival: its content-address key stem plus the ground-truth patient."""

    sequence: int
    hour_index: int
    within_hour: int
    patient: Patient


def _num_hours(horizon: OperatingWeek) -> int:
    return (horizon.end - horizon.start).root // _HOUR_US


def _hour_window(horizon: OperatingWeek, h: int) -> TimeWindow:
    return TimeWindow(start=horizon.start + hours(h), end=horizon.start + hours(h + 1))


def _lambda(spec: WorkloadSpec, h: int) -> float:
    hod = h % 24
    dow = (h // 24) % 7
    return spec.base_rate_per_hour * spec.hourly_profile[hod] * spec.dow_profile[dow]


def _draw_patient(
    streams: RandomStreams, spec: WorkloadSpec, key_for: _KeyFor, t: SimTime, pid: PatientId
) -> Patient:
    """Sample one patient's attributes, each on its own content-addressed substream."""
    esi = sample_esi(streams.substream(*key_for("esi")), spec.esi_mix)
    complaint = sample_complaint(streams.substream(*key_for("complaint")), spec.complaint_mix)
    mode = sample_arrival_mode(streams.substream(*key_for("mode")), spec.ambulance_fraction)
    isolation = bernoulli(streams.substream(*key_for("isolation")), spec.isolation_fraction)
    esi_scale = spec.esi_workup_scale.get(esi, 1.0)

    def workup_substream(component: str) -> np.random.Generator:
        # Each workup component (provider/nurse/imaging_<zone>/labs/procedures)
        # gets its own content-addressed key, so a profile edit to one
        # component never shifts another component's draws for this patient.
        return streams.substream(*key_for(f"workup_{component}"))

    workup = sample_workup(workup_substream, spec.workups[complaint], esi_scale=esi_scale)
    return Patient(
        id=pid,
        arrival_time=t,
        arrival_mode=mode,
        esi=esi,
        complaint=complaint,
        isolation_required=isolation,
        workup=workup,
    )


def _base_arrivals(
    spec: WorkloadSpec, streams: RandomStreams, num_hours: int
) -> list[tuple[SimTime, int, int, Patient]]:
    records: list[tuple[SimTime, int, int, Patient]] = []
    for h in range(num_hours):
        g = streams.substream("workload", "arrivals", h)
        window = _hour_window(spec.horizon, h)
        times = sample_poisson_arrivals(g, rate_per_hour=_lambda(spec, h), window=window)
        for k, t in enumerate(times):
            pid = PatientId(f"p_{h:03d}_{k:02d}")

            def key_for(attr: str, h: int = h, k: int = k) -> tuple[str | int, ...]:
                return ("workload", attr, h, k)

            records.append((t, h, k, _draw_patient(streams, spec, key_for, t, pid)))
    return records


def _hours_overlapping(
    horizon: OperatingWeek, at: SimTime, duration: Duration, num_hours: int
) -> list[int]:
    """Every hour-bin index whose window overlaps ``[at, at + duration)``."""
    end = at + duration
    return [
        h
        for h in range(num_hours)
        if (window := _hour_window(horizon, h)).start < end and at < window.end
    ]


def _surge_arrivals(
    spec: WorkloadSpec,
    streams: RandomStreams,
    disruptions: DisruptionSpec | None,
    num_hours: int,
) -> list[tuple[SimTime, int, int, Patient]]:
    if disruptions is None:
        return []
    records: list[tuple[SimTime, int, int, Patient]] = []
    for i, ev in enumerate(disruptions.events):
        if ev.kind != "surge":
            continue
        extra_factor = max(0.0, (ev.magnitude if ev.magnitude is not None else 1.0) - 1.0)
        if extra_factor <= 0.0:
            continue
        ev_end = ev.at + ev.duration
        for h in _hours_overlapping(spec.horizon, ev.at, ev.duration, num_hours):
            g = streams.substream("workload", "surge", i, h)
            hour = _hour_window(spec.horizon, h)
            # Sample only the intersection [ev.at, ev.at + ev.duration) ∩ hour —
            # a surge starting or ending mid-hour must not spill extra arrivals
            # across the whole overlapping hour.
            window = TimeWindow(
                start=SimTime(max(hour.start.root, ev.at.root)),
                end=SimTime(min(hour.end.root, ev_end.root)),
            )
            lam = _lambda(spec, h) * extra_factor
            times = sample_poisson_arrivals(g, rate_per_hour=lam, window=window)
            for k, t in enumerate(times):
                pid = PatientId(f"s{i}_{h:03d}_{k:02d}")

                def key_for(attr: str, i: int = i, h: int = h, k: int = k) -> tuple[str | int, ...]:
                    return ("workload", f"surge_{attr}", i, h, k)

                records.append((t, h, k, _draw_patient(streams, spec, key_for, t, pid)))
    return records


def generate_workload(
    spec: WorkloadSpec, streams: RandomStreams, *, disruptions: DisruptionSpec | None = None
) -> tuple[PatientArrival, ...]:
    """The one workload generator: a bit-reproducible realized week of arrivals.

    Sorted by ``arrival_time`` with a dense ``sequence`` assigned only after the
    base and any surge arrivals are merged — ``sequence`` is never itself a
    draw-key, so an early edit never renumbers (and re-draws) later patients.
    """
    num_hours = _num_hours(spec.horizon)
    records = _base_arrivals(spec, streams, num_hours)
    records.extend(_surge_arrivals(spec, streams, disruptions, num_hours))
    records.sort(key=lambda item: item[0].root)
    return tuple(
        PatientArrival(sequence=i, hour_index=h, within_hour=k, patient=patient)
        for i, (_t, h, k, patient) in enumerate(records)
    )


__all__ = ["PatientArrival", "generate_workload"]
