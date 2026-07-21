"""Thin ``core.rng`` wrappers: categorical stability, no new RNG idiom."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import ArrivalMode, EsiAcuity, RandomStreams, ZoneType, sample_categorical
from hospital.data.distributions import (
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
        needs = sample_workup(RandomStreams(1).substream("w", i), profile)
        assert needs.provider_visits >= 1


def test_sample_workup_esi_scale_monotonically_increases_expected_intensity() -> None:
    profile = _workup_profile(provider_visits_mean=3.0)
    n = 300
    low = [
        sample_workup(RandomStreams(1).substream("scale-low", i), profile, esi_scale=0.5)
        for i in range(n)
    ]
    high = [
        sample_workup(RandomStreams(1).substream("scale-high", i), profile, esi_scale=2.0)
        for i in range(n)
    ]
    mean_low = sum(w.provider_visits + w.nurse_visits + w.labs for w in low) / n
    mean_high = sum(w.provider_visits + w.nurse_visits + w.labs for w in high) / n
    assert mean_high > mean_low


def test_sample_workup_esi_scale_default_is_a_no_op() -> None:
    profile = _workup_profile()
    a = sample_workup(RandomStreams(1).substream("default"), profile)
    b = sample_workup(RandomStreams(1).substream("default"), profile, esi_scale=1.0)
    assert a == b
