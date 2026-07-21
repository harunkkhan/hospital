"""Vitals trajectory sampling — **M3**, not implemented in M1 (doc 02 §2.4).

The full model (acuity-baseline draw, NEWS2-informed deterioration schedule,
per-tick content-addressable measurement noise) is deferred to milestone 3,
where it feeds ``forecast.deterioration`` training labels. The M3 signature is
declared now — with its supporting frozen models — so downstream milestones can
depend on the *shape* of this API before the sampling logic lands; calling it
today raises :class:`NotImplementedError` rather than silently returning a
placeholder trajectory.
"""

from __future__ import annotations

from hospital.core import Duration, FrozenModel, Patient, PatientId, RandomStreams


class VitalsSample(FrozenModel):
    """One integer-scaled vitals reading at ``elapsed`` since arrival (M3)."""

    elapsed: Duration
    hr: int
    spo2: int
    sbp: int
    dbp: int
    temp_c_x10: int
    rr: int


class VitalsStream(FrozenModel):
    """A patient's full sampled vitals trajectory plus its deterioration label (M3)."""

    patient: PatientId
    samples: tuple[VitalsSample, ...]
    deteriorates: bool
    onset: Duration | None = None


def generate_vitals(
    patient: Patient, streams: RandomStreams, *, until: Duration, cadence: Duration
) -> VitalsStream:
    """Sample a per-patient vitals trajectory up to ``until`` at ``cadence`` (M3).

    Not implemented in M1/M2 — the signature is fixed so callers can be written
    against it now. Raises :class:`NotImplementedError` unconditionally.
    """
    raise NotImplementedError("M3")


__all__ = ["VitalsSample", "VitalsStream", "generate_vitals"]
