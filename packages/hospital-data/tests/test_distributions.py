"""Thin ``core.rng`` wrappers: categorical stability, no new RNG idiom."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import ArrivalMode, EsiAcuity, RandomStreams, ZoneType, sample_categorical
from hospital.data.distributions import (
    WorkupSubstream,
    bernoulli,
    sample_arrival_mode,
    sample_complaint,
    sample_count,
    sample_esi,
    sample_workup,
)
from hospital.data.scenario import WorkupProfile


def _gen(seed: int, *key: str | int):
    return RandomStreams(seed).substream(*key)


def _workup_substream(seed: int, *prefix: str | int) -> WorkupSubstream:
    streams = RandomStreams(seed)

    def factory(component: str) -> np.random.Generator:
        return streams.substream(*prefix, component)

    return factory


def test_sample_esi_is_deterministic_and_within_mix() -> None:
    mix = {EsiAcuity.ESI1: 0.1, EsiAcuity.ESI2: 0.9}
    a = sample_esi(_gen(1, "esi"), mix)
    b = sample_esi(_gen(1, "esi"), mix)
    assert a == b
    assert a in mix


def test_sample_complaint_is_deterministic() -> None:
    mix = {"chest_pain": 0.5, "laceration": 0.5}
    a = sample_complaint(_gen(2, "complaint"), mix)
    b = sample_complaint(_gen(2, "complaint"), mix)
    assert a == b
    assert a in mix


def test_sample_arrival_mode_agrees_with_direct_categorical() -> None:
    seed_key = ("mode", 5)
    via_helper = sample_arrival_mode(_gen(3, *seed_key), 0.3)
    direct = sample_categorical(
        _gen(3, *seed_key), {ArrivalMode.AMBULANCE: 0.3, ArrivalMode.WALK_IN: 0.7}
    )
    assert via_helper == direct


def test_bernoulli_agrees_with_direct_categorical() -> None:
    seed_key = ("iso", 9)
    via_helper = bernoulli(_gen(4, *seed_key), 0.4)
    direct = sample_categorical(_gen(4, *seed_key), {True: 0.4, False: 0.6})
    assert via_helper == direct


def test_bernoulli_extremes_are_certain() -> None:
    assert bernoulli(_gen(1, "certain-true"), 1.0) is True
    assert bernoulli(_gen(1, "certain-false"), 0.0) is False


@settings(max_examples=25, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=50.0),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10_000),
)
def test_sample_count_never_below_minimum(mean: float, minimum: int, draw_index: int) -> None:
    g = RandomStreams(1).substream("count", draw_index)
    assert sample_count(g, mean, minimum=minimum) >= minimum


def test_sample_count_negative_mean_clamps_to_zero_lambda() -> None:
    g = RandomStreams(1).substream("neg")
    assert sample_count(g, -5.0) == 0


def _workup_profile(provider_visits_mean: float = 2.0) -> WorkupProfile:
    return WorkupProfile(
        provider_visits_mean=provider_visits_mean,
        nurse_visits_mean=3.0,
        imaging_prob={ZoneType.IMAGING: 0.5},
        labs_mean=2.0,
        procedure_prob=0.3,
    )


def test_sample_workup_provider_visits_at_least_one() -> None:
    profile = _workup_profile(provider_visits_mean=1.0)
    for i in range(20):
        needs = sample_workup(_workup_substream(1, "w", i), profile)
        assert needs.provider_visits >= 1


def test_sample_workup_esi_scale_monotonically_increases_expected_intensity() -> None:
    profile = _workup_profile(provider_visits_mean=3.0)
    n = 300
    low = [
        sample_workup(_workup_substream(1, "scale-low", i), profile, esi_scale=0.5)
        for i in range(n)
    ]
    high = [
        sample_workup(_workup_substream(1, "scale-high", i), profile, esi_scale=2.0)
        for i in range(n)
    ]
    mean_low = sum(w.provider_visits + w.nurse_visits + w.labs for w in low) / n
    mean_high = sum(w.provider_visits + w.nurse_visits + w.labs for w in high) / n
    assert mean_high > mean_low


def test_sample_workup_esi_scale_default_is_a_no_op() -> None:
    profile = _workup_profile()
    a = sample_workup(_workup_substream(1, "default"), profile)
    b = sample_workup(_workup_substream(1, "default"), profile, esi_scale=1.0)
    assert a == b


# Finding 2: sampling must be a function of the mix's content, never of the
# mapping's insertion order — otherwise dump+reload (which sorts keys) maps
# the same draw to a different outcome and breaks CRN across serialization.
def test_mix_sampling_is_insertion_order_invariant() -> None:
    forward = {
        EsiAcuity.ESI1: 0.05,
        EsiAcuity.ESI2: 0.2,
        EsiAcuity.ESI3: 0.5,
        EsiAcuity.ESI4: 0.2,
        EsiAcuity.ESI5: 0.05,
    }
    backward = dict(reversed(list(forward.items())))
    assert forward == backward  # equal content, different insertion order
    for i in range(50):
        assert sample_esi(_gen(7, "esi", i), forward) == sample_esi(_gen(7, "esi", i), backward)
    fwd_mix = {"abdominal": 0.3, "chest_pain": 0.4, "laceration": 0.3}
    bwd_mix = dict(reversed(list(fwd_mix.items())))
    for i in range(50):
        assert sample_complaint(_gen(8, "c", i), fwd_mix) == sample_complaint(
            _gen(8, "c", i), bwd_mix
        )


# Finding 3: every workup component draws on its own content-addressed
# substream — adding a zero-probability imaging entry must not shift the
# provider/nurse/lab/procedure (or other imaging) draws for the same patient.
def test_zero_probability_imaging_entry_leaves_other_draws_identical() -> None:
    base = WorkupProfile(
        provider_visits_mean=2.0,
        nurse_visits_mean=1.5,
        imaging_prob={ZoneType.IMAGING: 0.5},
        labs_mean=2.0,
        procedure_prob=0.5,
    )
    extended = WorkupProfile(
        provider_visits_mean=2.0,
        nurse_visits_mean=1.5,
        # "fast_track" sorts before "imaging", so under a single shared stream
        # this extra zero-probability draw would shift everything after it.
        imaging_prob={ZoneType.FAST_TRACK: 0.0, ZoneType.IMAGING: 0.5},
        labs_mean=2.0,
        procedure_prob=0.5,
    )
    for i in range(30):
        assert sample_workup(_workup_substream(5, "zp", i), base) == sample_workup(
            _workup_substream(5, "zp", i), extended
        )


# Finding 11: esi_scale is documented to multiply each mean — labs included.
def test_sample_workup_applies_esi_scale_to_labs() -> None:
    profile = WorkupProfile(
        provider_visits_mean=1.0,  # (mean-1)*scale == 0 -> provider fixed at 1
        nurse_visits_mean=0.0,
        imaging_prob={},
        labs_mean=2.0,
        procedure_prob=0.0,
    )
    n = 400
    unscaled = sum(sample_workup(_workup_substream(6, "labs", i), profile).labs for i in range(n))
    scaled = sum(
        sample_workup(_workup_substream(6, "labs", i), profile, esi_scale=3.0).labs
        for i in range(n)
    )
    # Expected means are 2.0 vs 6.0 per draw; require a clear separation so an
    # unscaled labs_mean (equal sums) can never pass.
    assert scaled > unscaled * 2
