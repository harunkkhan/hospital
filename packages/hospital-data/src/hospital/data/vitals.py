"""Vitals trajectory sampling — the deterioration label source (doc 02 §2.4, M3).

Each patient gets a physiological baseline drawn from their acuity, a slow
random walk around it, and — for the subset that deteriorates — a monotone
drift toward instability starting at a sampled ``onset``. Measurements are then
observed through additive noise, so the recorded stream is *noisy observations
of a latent trajectory* rather than the trajectory itself. That gap is the whole
point: ``forecast.deterioration`` has to predict the latent event from the noisy
view, which is only a real learning problem if the noise exists.

Determinism (nuance 1.8): every draw comes from a content-addressed
``streams.substream("world", "vitals", patient, …)`` key, so a patient's
trajectory is a pure function of ``(seed, patient id)`` — independent of how
many other patients were sampled first, and identical across arms under CRN.

**And independent of the cadence it is sampled at.** The latent walk advances on a
fixed internal grid of :data:`LATENT_STEP`, and both the walk and the observation
noise are keyed by *elapsed time* rather than by tick index. Keying by index made
looking more often change the patient: at a five-minute cadence the reading at ten
minutes came from ``walk/2``, at ten minutes from ``walk/1``, so the same instant in
the same patient's stream held different physiology depending on the monitor's
settings. Observation must not move the thing observed.

``until`` is the exception, and deliberately so: ``onset`` is placed as a fraction of
it (see :func:`generate_vitals`), which makes it a *construction* parameter rather
than a viewing window. Callers comparing arms must hold it fixed.

Ground truth is carried explicitly: :attr:`VitalsStream.deteriorates` and
:attr:`VitalsStream.onset` are the *labels*, not observable signals. Training
code may read them; the online monitor may not (it sees only ``VitalsSampled``
events). Keeping the label on the stream — rather than inferring it from the
samples — is what makes "did the classifier find it?" a well-posed question.

Units are integer-scaled to keep the model exactly reproducible: temperature is
tenths of a degree C (``temp_c_x10``), everything else is a whole unit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

from hospital.core import (
    Duration,
    EsiAcuity,
    FrozenModel,
    Patient,
    PatientId,
    RandomStreams,
    VitalsReading,
)

if TYPE_CHECKING:
    import numpy as np

_MICROS_PER_MINUTE: Final[int] = 60 * 1_000_000


class VitalsSample(VitalsReading):
    """A :class:`~hospital.core.VitalsReading` stamped with its offset from arrival.

    Extends the core value type rather than restating its six fields, so a sample
    *is* a reading and can be handed straight to a ``RiskMonitor`` — one shape,
    no conversion step to get wrong.
    """

    elapsed: Duration


class VitalsStream(FrozenModel):
    """A patient's full sampled vitals trajectory plus its deterioration label (M3)."""

    patient: PatientId
    samples: tuple[VitalsSample, ...]
    deteriorates: bool
    onset: Duration | None = None


class _Baseline(NamedTuple):
    """A patient's latent starting physiology, before walk/drift/noise."""

    hr: float
    spo2: float
    sbp: float
    dbp: float
    temp_c_x10: float
    rr: float


# Population means by acuity. ESI-1 arrives already deranged and ESI-5 near
# normal, so acuity alone carries real signal -- which is exactly why the
# classifier must be scored against a NEWS2-only baseline (doc 06 §15) rather
# than against chance.
_BASELINE_BY_ESI: Final[dict[EsiAcuity, _Baseline]] = {
    EsiAcuity.ESI1: _Baseline(hr=118.0, spo2=91.0, sbp=98.0, dbp=60.0, temp_c_x10=381.0, rr=26.0),
    EsiAcuity.ESI2: _Baseline(hr=106.0, spo2=94.0, sbp=110.0, dbp=68.0, temp_c_x10=378.0, rr=22.0),
    EsiAcuity.ESI3: _Baseline(hr=92.0, spo2=96.0, sbp=122.0, dbp=74.0, temp_c_x10=372.0, rr=18.0),
    EsiAcuity.ESI4: _Baseline(hr=82.0, spo2=97.0, sbp=126.0, dbp=78.0, temp_c_x10=369.0, rr=16.0),
    EsiAcuity.ESI5: _Baseline(hr=76.0, spo2=98.0, sbp=124.0, dbp=78.0, temp_c_x10=368.0, rr=14.0),
}

# Between-patient spread of the baseline (SD, in each vital's own units).
_BASELINE_SD: Final[_Baseline] = _Baseline(
    hr=9.0, spo2=1.6, sbp=12.0, dbp=8.0, temp_c_x10=4.0, rr=2.0
)

# Per-step random-walk SD of the LATENT state (physiological drift), where a step is
# one `LATENT_STEP`.
_WALK_SD: Final[_Baseline] = _Baseline(hr=2.2, spo2=0.35, sbp=2.6, dbp=1.8, temp_c_x10=1.1, rr=0.55)

# The fixed grid the latent walk advances on, independent of the cadence anyone views
# it at. Five minutes because that is what `_WALK_SD` above is calibrated against: the
# drift accumulates per step, so re-gridding without re-tuning would rescale every
# trajectory by the square root of the change. It also means a five-minute view is
# byte-identical to what this generator produced before the walk was keyed by time.
#
# A view *finer* than this sees the latent state hold still between grid points while
# the measurement noise still varies per instant — a plateau, not a frozen patient.
LATENT_STEP: Final[Duration] = Duration(5 * _MICROS_PER_MINUTE)

# Measurement noise (SD) added on top of the latent state at observation time.
# This is what the monitor actually sees, and why a single reading is weak
# evidence while a window of them is not.
_NOISE_SD: Final[_Baseline] = _Baseline(hr=3.0, spo2=0.7, sbp=4.0, dbp=3.0, temp_c_x10=1.5, rr=0.9)

# Full-severity deterioration displacement, reached `_RAMP` after onset.
_DETERIORATION_SHIFT: Final[_Baseline] = _Baseline(
    hr=42.0, spo2=-9.0, sbp=-30.0, dbp=-18.0, temp_c_x10=14.0, rr=14.0
)
_RAMP: Final[Duration] = Duration(25 * _MICROS_PER_MINUTE)

# Before onset, a patient who is going to crash drifts subtly for this long, and
# reaches this fraction of the full displacement by the moment of onset.
#
# Without a prodrome the label would be *unpredictable by construction*: onset is
# drawn independently of the trajectory, so pre-onset vitals would carry no signal
# at all and "detect deterioration early" would be an impossible task dressed up
# as a hard one. Real decompensation is foreshadowed; this is the modelled version
# of that, and it is what makes the classifier's job well-posed rather than rigged
# in either direction.
_PRODROME: Final[Duration] = Duration(20 * _MICROS_PER_MINUTE)
_PRODROME_SEVERITY: Final[float] = 0.22

# P(a patient deteriorates at all), by acuity.
_DETERIORATION_RISK: Final[dict[EsiAcuity, float]] = {
    EsiAcuity.ESI1: 0.34,
    EsiAcuity.ESI2: 0.18,
    EsiAcuity.ESI3: 0.07,
    EsiAcuity.ESI4: 0.02,
    EsiAcuity.ESI5: 0.01,
}

# Physiological clamps: a sampled vital never leaves the range a real monitor
# could display, so a long tail cannot produce a negative pulse.
_LIMITS: Final[dict[str, tuple[int, int]]] = {
    "hr": (25, 220),
    "spo2": (50, 100),
    "sbp": (50, 220),
    "dbp": (25, 140),
    "temp_c_x10": (330, 425),
    "rr": (4, 60),
}


def _clamp(name: str, value: float) -> int:
    """Round to the unit a monitor displays, then hold inside physiological limits.

    ``round`` is banker's rounding, the same tie rule ``core.time`` uses, so a
    vitals tick and a duration never disagree about how a .5 resolves.
    """
    low, high = _LIMITS[name]
    return max(low, min(high, round(value)))


def _severity(elapsed: Duration, onset: Duration | None) -> float:
    """How far into deterioration this instant is, on ``[0, 1]``.

    Three phases: flat until ``_PRODROME`` before onset, a shallow climb to
    ``_PRODROME_SEVERITY`` at onset, then the full ramp over ``_RAMP``. The
    prodrome is what makes early detection *possible but not trivial* — the signal
    is faint and buried in measurement noise for a while before it is
    unmistakable, which is exactly the window the classifier is meant to exploit.
    """
    if onset is None:
        return 0.0
    if elapsed.root >= onset.root:
        after = min(1.0, (elapsed.root - onset.root) / _RAMP.root)
        return _PRODROME_SEVERITY + (1.0 - _PRODROME_SEVERITY) * after
    lead = onset.root - elapsed.root
    if lead >= _PRODROME.root:
        return 0.0
    return _PRODROME_SEVERITY * (1.0 - lead / _PRODROME.root)


def _observe(
    g: np.random.Generator, latent: _Baseline, elapsed: Duration, onset: Duration | None
) -> VitalsSample:
    """Add deterioration displacement and measurement noise to the latent state."""
    severity = _severity(elapsed, onset)
    noise = _NOISE_SD
    shift = _DETERIORATION_SHIFT
    return VitalsSample(
        elapsed=elapsed,
        hr=_clamp("hr", latent.hr + severity * shift.hr + g.normal(0.0, noise.hr)),
        spo2=_clamp("spo2", latent.spo2 + severity * shift.spo2 + g.normal(0.0, noise.spo2)),
        sbp=_clamp("sbp", latent.sbp + severity * shift.sbp + g.normal(0.0, noise.sbp)),
        dbp=_clamp("dbp", latent.dbp + severity * shift.dbp + g.normal(0.0, noise.dbp)),
        temp_c_x10=_clamp(
            "temp_c_x10",
            latent.temp_c_x10 + severity * shift.temp_c_x10 + g.normal(0.0, noise.temp_c_x10),
        ),
        rr=_clamp("rr", latent.rr + severity * shift.rr + g.normal(0.0, noise.rr)),
    )


def _draw_baseline(g: np.random.Generator, esi: EsiAcuity) -> _Baseline:
    mean = _BASELINE_BY_ESI[esi]
    sd = _BASELINE_SD
    return _Baseline(*(float(g.normal(m, s)) for m, s in zip(mean, sd, strict=True)))


def _walk(g: np.random.Generator, latent: _Baseline) -> _Baseline:
    """One tick of latent physiological drift (a random walk, not observation noise)."""
    return _Baseline(*(v + float(g.normal(0.0, s)) for v, s in zip(latent, _WALK_SD, strict=True)))


def generate_vitals(
    patient: Patient, streams: RandomStreams, *, until: Duration, cadence: Duration
) -> VitalsStream:
    """Sample a per-patient vitals trajectory up to ``until``, viewed at ``cadence``.

    Pure construction from ``(seed, patient.id)``: no wall clock, no global
    state, and no dependence on sampling order. ``until`` is inclusive of the
    final on-cadence tick, so a stream always carries at least the arrival
    reading (``elapsed == 0``).

    ``cadence`` is purely a *view* onto the trajectory. The latent walk advances on the
    fixed :data:`LATENT_STEP` grid and the observation noise is keyed by elapsed time,
    so the reading at a given instant is the same whether the monitor looks every
    minute or every half hour — sampling twice as often returns twice as many readings,
    not a different patient.

    Whether the patient deteriorates, and when, is drawn here and reported on the
    returned stream as ground truth. Onset is confined to ``[0.15, 0.85]`` of
    ``until``, which is what makes that argument a construction parameter and not a
    window: it guarantees a deteriorating patient has both a pre-onset baseline to
    contrast against and enough post-onset time to be detectable, so a "missed"
    detection is a real miss rather than a labelling artefact. The cost is that two
    spans give two different patients, so anything comparing arms must fix it.
    """
    if cadence.root <= 0:
        raise ValueError("cadence must be positive")
    if until.root < 0:
        raise ValueError("until must be non-negative")

    pid = patient.id.root
    risk = _DETERIORATION_RISK[patient.esi]
    deteriorates = bool(streams.substream("world", "vitals", pid, "deteriorates").random() < risk)

    onset: Duration | None = None
    if deteriorates and until.root > 0:
        fraction = float(streams.substream("world", "vitals", pid, "onset").uniform(0.15, 0.85))
        onset = Duration(int(until.root * fraction))

    latent = _draw_baseline(streams.substream("world", "vitals", pid, "baseline"), patient.esi)
    ticks = int(until.root // cadence.root)
    samples: list[VitalsSample] = []
    walked = 0
    for index in range(ticks + 1):
        elapsed = Duration(index * cadence.root)
        # Advance the latent walk along the fixed internal grid up to this instant, so
        # the state at `elapsed` is the same however coarsely it is being viewed.
        target = elapsed.root // LATENT_STEP.root
        while walked < target:
            walked += 1
            latent = _walk(streams.substream("world", "vitals", pid, "walk", walked), latent)
        observed = streams.substream("world", "vitals", pid, "observe", elapsed.root)
        samples.append(_observe(observed, latent, elapsed, onset))

    return VitalsStream(
        patient=patient.id,
        samples=tuple(samples),
        deteriorates=deteriorates,
        onset=onset,
    )


def deterioration_label(stream: VitalsStream, at: Duration, *, horizon: Duration) -> bool:
    """Ground truth for "does onset fall in ``(at, at + horizon]``?" (doc 06 §13-5).

    The classifier's target is a *future* crossing, not the present state, so the
    label is defined here — beside the generator that owns the truth — rather
    than re-derived from samples wherever a training frame is built.
    """
    if not stream.deteriorates or stream.onset is None:
        return False
    return at.root < stream.onset.root <= at.root + horizon.root


__all__ = [
    "VitalsSample",
    "VitalsStream",
    "deterioration_label",
    "generate_vitals",
]
