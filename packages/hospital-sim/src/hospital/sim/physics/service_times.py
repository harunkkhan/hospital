"""One sampler, content-addressed for CRN (doc 04 §3.3).

The load-bearing invariant: every draw is keyed by *what* is being drawn —
``("world", <domain>, patient, activity, index)`` — never by *when* an arm asks.
"Patient P's 2nd nurse visit" therefore draws the identical duration in BASELINE
and OPTIMIZED even though the arms reach it at different wall-clock moments;
that is the mechanism that makes the paired diff pure decision signal. The
repetition ``index`` is derived from the pre-sampled ``WorkupNeeds`` (fixed and
identical across arms), never from how many visits *this arm* has dispatched.

``ServiceTimes`` owns no sampler of its own: the draw is
``core.rng.sample_lognormal`` (banker's-rounded to integer µs by the one shared
rule). This module owns only the wiring and the frozen ``(mean, cv)`` lookup.

A missing table row is a hard ``KeyError`` — a silently defaulted service time
is an un-tuned realism hole no test would flag. The v1 table is keyed
``(activity, esi)``; the ``complaint`` dimension is accepted (contract shape)
but not yet discriminated (judgment call, flagged in the build report).

Disposition and boarding are also *world* randomness (they must match across
arms for the same patient), so their content-addressed draws live here too
(🟡 A5: the discharge/admit/transfer mix is a fixed per-acuity table in v1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from hospital.core import (
    MICROS_PER_SEC,
    Activity,
    DispositionKind,
    Duration,
    EsiAcuity,
    Patient,
    PatientId,
    RandomStreams,
    sample_categorical,
    sample_lognormal,
)

# Base ER-short service means (seconds) and coefficients of variation.
_BASE_MEAN_CV: Final[dict[Activity, tuple[float, float]]] = {
    Activity.TRIAGE: (300.0, 0.35),
    Activity.PROVIDER_VISIT: (720.0, 0.50),
    Activity.NURSE_VISIT: (480.0, 0.45),
    Activity.IMAGING: (1_080.0, 0.40),
    Activity.LAB: (300.0, 0.40),
    Activity.DOCUMENTATION: (540.0, 0.40),
    Activity.CLEANING: (720.0, 0.30),
    Activity.DISCHARGE: (240.0, 0.30),
}

# Sicker patients take longer (cleaning is acuity-independent).
_ESI_MEAN_SCALE: Final[dict[EsiAcuity, float]] = {
    EsiAcuity.ESI1: 1.6,
    EsiAcuity.ESI2: 1.3,
    EsiAcuity.ESI3: 1.0,
    EsiAcuity.ESI4: 0.85,
    EsiAcuity.ESI5: 0.7,
}

# Post-service result turnaround (seconds, cv): radiology read / lab reporting.
# This time elapses OFF the machine — holding a resource for it would turn the
# 2-station lab into the floor's binding constraint at reference load.
_RESULT_DELAY_MEAN_CV: Final[dict[Activity, tuple[float, float]]] = {
    Activity.IMAGING: (1_500.0, 0.35),
    Activity.LAB: (2_400.0, 0.35),
}

# How long a lab sample physically occupies an analyzer station (seconds, cv).
_ANALYZER_MEAN_CV: Final[tuple[float, float]] = (600.0, 0.30)

# 🟡 A5 — the disposition mix per acuity (kind-sorted so the categorical draw is
# a pure function of content, never of mapping construction order).
_DISPOSITION_MIX: Final[dict[EsiAcuity, dict[DispositionKind, float]]] = {
    EsiAcuity.ESI1: {
        DispositionKind.ADMIT: 0.80,
        DispositionKind.DISCHARGE: 0.10,
        DispositionKind.TRANSFER: 0.10,
    },
    EsiAcuity.ESI2: {
        DispositionKind.ADMIT: 0.50,
        DispositionKind.DISCHARGE: 0.45,
        DispositionKind.TRANSFER: 0.05,
    },
    EsiAcuity.ESI3: {
        DispositionKind.ADMIT: 0.25,
        DispositionKind.DISCHARGE: 0.72,
        DispositionKind.TRANSFER: 0.03,
    },
    EsiAcuity.ESI4: {
        DispositionKind.ADMIT: 0.05,
        DispositionKind.DISCHARGE: 0.94,
        DispositionKind.TRANSFER: 0.01,
    },
    EsiAcuity.ESI5: {
        DispositionKind.ADMIT: 0.02,
        DispositionKind.DISCHARGE: 0.97,
        DispositionKind.TRANSFER: 0.01,
    },
}

_BOARDING_MEAN_S: Final[float] = 7_200.0
_BOARDING_CV: Final[float] = 0.5

# Inpatient length of stay once a bed is occupied, by the acuity that got them admitted.
# Days, not hours: a ward bed turns over on a completely different timescale from an ED
# bay, which is exactly why a handful of beds can block an ED all week.
_WARD_STAY_MEAN_S: Final[Mapping[EsiAcuity, float]] = {
    EsiAcuity.ESI1: 4.0 * 86_400.0,
    EsiAcuity.ESI2: 3.0 * 86_400.0,
    EsiAcuity.ESI3: 2.0 * 86_400.0,
    EsiAcuity.ESI4: 1.5 * 86_400.0,
    EsiAcuity.ESI5: 1.0 * 86_400.0,
}
_WARD_STAY_CV: Final[float] = 0.7


@dataclass(frozen=True)
class ServiceTable:
    """The frozen ``(activity, esi) -> (mean_s, cv)`` lookup. Missing row = KeyError."""

    rows: Mapping[tuple[Activity, EsiAcuity], tuple[float, float]]

    def lookup(self, activity: Activity, esi: EsiAcuity, complaint: str) -> tuple[float, float]:
        """``(mean_s, cv)`` for the service — never a silent default.

        ``complaint`` is part of the contract shape but not yet a lookup
        dimension in v1 (one table row per activity/acuity pair).
        """
        row = self.rows.get((activity, esi))
        if row is None:
            raise KeyError(f"no service-time row for ({activity.value}, esi={int(esi)})")
        return row


def default_service_table() -> ServiceTable:
    """The reference ER-short-means table over every ``(activity, esi)`` pair."""
    rows: dict[tuple[Activity, EsiAcuity], tuple[float, float]] = {}
    for activity, (mean_s, cv) in _BASE_MEAN_CV.items():
        for esi in EsiAcuity:
            scale = 1.0 if activity is Activity.CLEANING else _ESI_MEAN_SCALE[esi]
            rows[(activity, esi)] = (mean_s * scale, cv)
    return ServiceTable(rows=rows)


class ServiceTimes:
    """Content-addressed service-duration draws over one ``RandomStreams``."""

    def __init__(self, streams: RandomStreams, table: ServiceTable) -> None:
        self._streams = streams
        self._table = table

    def sample(
        self,
        activity: Activity,
        esi: EsiAcuity,
        complaint: str,
        *,
        patient: PatientId,
        index: int = 0,
    ) -> Duration:
        """One service duration, keyed ``(patient, activity, index)`` — call-order free."""
        g = self._streams.substream(
            "world", "service_time", str(patient), activity.value, int(index)
        )
        mean_s, cv = self._table.lookup(activity, esi, complaint)
        return sample_lognormal(g, mean_s, cv)

    def result_delay(self, activity: Activity, *, patient: PatientId, index: int = 0) -> Duration:
        """Off-machine turnaround from service end to ``TestResulted`` (read/reporting)."""
        row = _RESULT_DELAY_MEAN_CV.get(activity)
        if row is None:
            raise KeyError(f"no result-delay row for {activity.value}")
        g = self._streams.substream(
            "world", "result_delay", str(patient), activity.value, int(index)
        )
        mean_s, cv = row
        return sample_lognormal(g, mean_s, cv)

    def analyzer_time(self, *, patient: PatientId, index: int = 0) -> Duration:
        """How long one sample holds a lab analyzer station (its own CRN domain)."""
        g = self._streams.substream("world", "lab_analyzer", str(patient), int(index))
        mean_s, cv = _ANALYZER_MEAN_CV
        return sample_lognormal(g, mean_s, cv)


def sample_disposition(streams: RandomStreams, patient: Patient) -> DispositionKind:
    """The patient's terminal disposition — keyed on the patient alone (🟡 A5).

    Content-addressed on the patient id so both arms decide the same fate for
    the same person; the mix is iterated in sorted-kind order for determinism.
    """
    mix = _DISPOSITION_MIX[patient.esi]
    ordered = {kind: mix[kind] for kind in sorted(mix, key=lambda k: k.value)}
    g = streams.substream("world", "disposition", str(patient.id))
    return sample_categorical(g, ordered)


def sample_ward_stay(streams: RandomStreams, patient: Patient) -> Duration:
    """How long an admitted patient occupies their inpatient bed.

    World randomness, content-addressed on the patient like every other draw here, so the
    same patient stays the same length under both arms of a comparison.
    """
    g = streams.substream("world", "ward_stay", str(patient.id))
    return sample_lognormal(g, _WARD_STAY_MEAN_S[patient.esi], _WARD_STAY_CV)


def sample_boarding_delay(streams: RandomStreams, patient: Patient) -> Duration:
    """How long an admitted patient boards (holds the bay) before leaving the floor."""
    g = streams.substream("world", "boarding", str(patient.id))
    return sample_lognormal(g, _BOARDING_MEAN_S, _BOARDING_CV)


def admit_probability(esi: EsiAcuity) -> float:
    """P(disposition = ADMIT) for an ``esi`` patient — the demand side of bed capacity."""
    return _DISPOSITION_MIX[esi][DispositionKind.ADMIT]


def mean_ward_stay(esi: EsiAcuity) -> Duration:
    """The mean inpatient stay for an ``esi`` patient — the supply side of bed capacity.

    Public alongside :func:`admit_probability` because sizing a ward is a real question
    to ask of the model and not only of a run: the two together turn a week of arrivals
    into the bed-days it implies, which is how ``scenarios/hospital.yaml`` was sized and
    how the test that pins its operating point checks it is still true. Reading the
    private tables from outside would make that arithmetic silently wrong the day either
    distribution moves.
    """
    return Duration(round(_WARD_STAY_MEAN_S[esi] * MICROS_PER_SEC))


__all__ = [
    "ServiceTable",
    "ServiceTimes",
    "admit_probability",
    "default_service_table",
    "mean_ward_stay",
    "sample_boarding_delay",
    "sample_disposition",
    "sample_ward_stay",
]
