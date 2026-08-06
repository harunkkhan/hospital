"""``generate_vitals``: determinism, CRN isolation, cadence, drift, and labels (doc 02 §8).

The generator is the *label source* for ``forecast.deterioration``, so these
tests care about two things a looser suite would miss: that a patient's
trajectory is a pure function of ``(seed, patient id)`` regardless of sampling
order, and that the post-onset signal is real enough to be learnable while the
pre-onset stretch is not trivially separable.
"""

from __future__ import annotations

import statistics

from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import (
    ArrivalMode,
    Duration,
    EsiAcuity,
    Patient,
    PatientId,
    RandomStreams,
    SimTime,
    WorkupNeeds,
    hours,
    minutes,
)
from hospital.data.vitals import VitalsStream, deterioration_label, generate_vitals

_LIMITS = {
    "hr": (25, 220),
    "spo2": (50, 100),
    "sbp": (50, 220),
    "dbp": (25, 140),
    "temp_c_x10": (330, 425),
    "rr": (4, 60),
}


def _patient(pid: str = "p1", esi: EsiAcuity = EsiAcuity.ESI3) -> Patient:
    return Patient(
        id=PatientId(pid),
        arrival_time=SimTime(0),
        arrival_mode=ArrivalMode.WALK_IN,
        esi=esi,
        complaint="chest_pain",
        isolation_required=False,
        workup=WorkupNeeds(provider_visits=1, nurse_visits=1, imaging=(), labs=0, procedures=0),
    )


def _cohort(
    esi: EsiAcuity, n: int, *, seed: int = 11, until: Duration | None = None
) -> list[VitalsStream]:
    streams = RandomStreams(seed)
    span = until if until is not None else hours(4)
    return [
        generate_vitals(_patient(f"p{i:04d}", esi), streams, until=span, cadence=minutes(5))
        for i in range(n)
    ]


def test_generate_vitals_is_deterministic_given_a_seed() -> None:
    a = generate_vitals(_patient(), RandomStreams(7), until=hours(4), cadence=minutes(5))
    b = generate_vitals(_patient(), RandomStreams(7), until=hours(4), cadence=minutes(5))
    assert a == b


def test_a_patients_trajectory_does_not_depend_on_sampling_order() -> None:
    """CRN (nuance 1.8): the draw is content-addressed by patient, not by call order.

    Two arms sample different patient sets in different orders; if the ordering
    leaked in, the same patient would get different vitals under each arm and the
    comparison would measure the generator, not the policy.
    """
    streams = RandomStreams(7)
    alone = generate_vitals(_patient("focus"), streams, until=hours(4), cadence=minutes(5))
    for i in range(50):
        generate_vitals(_patient(f"noise{i}"), streams, until=hours(2), cadence=minutes(1))
    after_noise = generate_vitals(_patient("focus"), streams, until=hours(4), cadence=minutes(5))
    assert alone == after_noise


def test_different_seeds_give_different_trajectories() -> None:
    a = generate_vitals(_patient(), RandomStreams(1), until=hours(4), cadence=minutes(5))
    b = generate_vitals(_patient(), RandomStreams(2), until=hours(4), cadence=minutes(5))
    assert a.samples != b.samples


@settings(max_examples=25, deadline=None)
@given(
    until_min=st.integers(min_value=0, max_value=600),
    cadence_min=st.integers(min_value=1, max_value=30),
)
def test_samples_land_on_cadence_and_cover_the_span(until_min: int, cadence_min: int) -> None:
    stream = generate_vitals(
        _patient(), RandomStreams(3), until=minutes(until_min), cadence=minutes(cadence_min)
    )
    # Always at least the arrival reading, and every tick is exactly on cadence.
    assert stream.samples[0].elapsed == Duration(0)
    assert len(stream.samples) == until_min // cadence_min + 1
    for index, sample in enumerate(stream.samples):
        assert sample.elapsed == Duration(index * minutes(cadence_min).root)
        assert sample.elapsed.root <= minutes(until_min).root


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=5_000), esi=st.sampled_from(list(EsiAcuity)))
def test_every_reading_stays_physiological(seed: int, esi: EsiAcuity) -> None:
    """A long tail must never print a negative pulse — the clamps are load-bearing."""
    stream = generate_vitals(
        _patient("clamped", esi), RandomStreams(seed), until=hours(6), cadence=minutes(5)
    )
    for sample in stream.samples:
        for field, (low, high) in _LIMITS.items():
            assert low <= getattr(sample, field) <= high, (field, sample)


def test_deterioration_frequency_rises_with_acuity() -> None:
    """ESI-1 must deteriorate far more often than ESI-5 — acuity carries real signal."""
    rates = {
        esi: statistics.mean(bool(s.deteriorates) for s in _cohort(esi, 400)) for esi in EsiAcuity
    }
    ordered = [rates[esi] for esi in (EsiAcuity.ESI1, EsiAcuity.ESI2, EsiAcuity.ESI3)]
    assert ordered == sorted(ordered, reverse=True), rates
    assert rates[EsiAcuity.ESI1] > rates[EsiAcuity.ESI5] * 3, rates


def test_onset_lies_inside_the_observed_span_when_a_patient_deteriorates() -> None:
    """A label the stream cannot evidence is a labelling artefact, not a hard case."""
    for stream in _cohort(EsiAcuity.ESI1, 200):
        if stream.deteriorates:
            onset = stream.onset
            assert onset is not None
            assert 0 < onset.root < hours(4).root
        else:
            assert stream.onset is None


def test_post_onset_vitals_drift_toward_instability() -> None:
    """The deterioration displacement must survive the measurement noise.

    Asserted as a cohort contrast rather than per-sample: any single reading can
    be noise, which is exactly why the classifier gets a rolling *window*.
    """
    # `deteriorates` and `onset` are set together, but bind the onset explicitly
    # so the split below reads off one value rather than two coupled ones.
    deteriorating = [
        (s, s.onset) for s in _cohort(EsiAcuity.ESI1, 400) if s.deteriorates and s.onset is not None
    ]
    assert deteriorating, "fixture must produce deteriorating streams"
    ramp_done = minutes(25).root
    pre = [smp for s, onset in deteriorating for smp in s.samples if smp.elapsed.root < onset.root]
    post = [
        smp
        for s, onset in deteriorating
        for smp in s.samples
        if smp.elapsed.root > onset.root + ramp_done
    ]
    assert pre and post
    assert statistics.mean(x.hr for x in post) > statistics.mean(x.hr for x in pre) + 15
    assert statistics.mean(x.spo2 for x in post) < statistics.mean(x.spo2 for x in pre) - 3
    assert statistics.mean(x.rr for x in post) > statistics.mean(x.rr for x in pre) + 5


def test_non_deteriorating_streams_have_no_trend() -> None:
    """The negative class must not be separable by a trend the label never caused."""
    stable = [s for s in _cohort(EsiAcuity.ESI3, 400) if not s.deteriorates]
    assert stable
    first = statistics.mean(s.samples[0].hr for s in stable)
    last = statistics.mean(s.samples[-1].hr for s in stable)
    # A random walk wanders, but with no drift term it must not march.
    assert abs(last - first) < 8, (first, last)


def test_deterioration_label_is_a_forward_looking_window() -> None:
    """The target is a future crossing, not the present state (doc 06 §13-5)."""
    stream = next(s for s in _cohort(EsiAcuity.ESI1, 200) if s.deteriorates)
    onset = stream.onset
    assert onset is not None
    horizon = minutes(30)

    just_before = Duration(max(0, onset.root - minutes(10).root))
    assert deterioration_label(stream, just_before, horizon=horizon) is True
    # Already crossed: no longer a *prediction* target.
    assert deterioration_label(stream, Duration(onset.root + 1), horizon=horizon) is False
    # Too far out to be in the horizon.
    long_before = Duration(max(0, onset.root - minutes(90).root))
    if long_before.root + horizon.root < onset.root:
        assert deterioration_label(stream, long_before, horizon=horizon) is False


def test_a_stable_patient_is_never_labelled() -> None:
    stream = next(s for s in _cohort(EsiAcuity.ESI5, 200) if not s.deteriorates)
    for at_min in (0, 30, 90, 200):
        assert deterioration_label(stream, minutes(at_min), horizon=minutes(30)) is False


def test_looking_more_often_does_not_change_the_patient() -> None:
    """Cadence is a view, not a construction parameter (the walk is keyed by time).

    Keying the walk and the noise by *tick index* made observation move the thing
    observed: at a five-minute cadence the reading at ten minutes came from ``walk/2``,
    at a ten-minute cadence from ``walk/1``, so the same instant in the same patient's
    stream held different physiology depending on the monitor's settings.
    """
    fine = generate_vitals(
        _patient("p", EsiAcuity.ESI2), RandomStreams(7), until=hours(6), cadence=minutes(5)
    )
    coarse = generate_vitals(
        _patient("p", EsiAcuity.ESI2), RandomStreams(7), until=hours(6), cadence=minutes(30)
    )
    by_elapsed = {sample.elapsed: sample for sample in fine.samples}

    assert len(coarse.samples) < len(fine.samples)
    shared = [s for s in coarse.samples if s.elapsed in by_elapsed]
    assert len(shared) == len(coarse.samples), "the coarse grid must land on the fine one"
    for sample in shared:
        assert sample == by_elapsed[sample.elapsed], f"physiology moved at {sample.elapsed}"


def test_a_finer_view_agrees_with_the_grid_it_refines() -> None:
    """A one-minute view must reproduce every five-minute reading exactly."""
    coarse = generate_vitals(_patient("q"), RandomStreams(11), until=hours(2), cadence=minutes(5))
    fine = generate_vitals(_patient("q"), RandomStreams(11), until=hours(2), cadence=minutes(1))
    by_elapsed = {sample.elapsed: sample for sample in fine.samples}
    for sample in coarse.samples:
        assert by_elapsed[sample.elapsed] == sample, f"disagreement at {sample.elapsed}"


def test_the_deterioration_label_is_unchanged_by_the_cadence() -> None:
    """Ground truth belongs to the patient; only `until` may move it (by design)."""
    for cadence in (minutes(1), minutes(5), minutes(10), minutes(30)):
        stream = generate_vitals(
            _patient("r", EsiAcuity.ESI1), RandomStreams(3), until=hours(6), cadence=cadence
        )
        reference = generate_vitals(
            _patient("r", EsiAcuity.ESI1), RandomStreams(3), until=hours(6), cadence=minutes(5)
        )
        assert (stream.deteriorates, stream.onset) == (reference.deteriorates, reference.onset)
