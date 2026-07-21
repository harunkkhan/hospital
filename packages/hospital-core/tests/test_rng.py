"""RNG/CRN: keyed substreams, byte-identical isolation, world vs policy split."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hospital.core import (
    Duration,
    RandomStreams,
    SimTime,
    TimeWindow,
    sample_categorical,
    sample_lognormal,
    sample_poisson_arrivals,
)


def _draw(streams: RandomStreams, *key: str | int) -> bytes:
    return streams.substream(*key).standard_normal(32).tobytes()


def test_substream_is_pure_function_of_key() -> None:
    a = RandomStreams(42)
    b = RandomStreams(42)
    assert _draw(a, "service_time", "p1", "provider_visit", 0) == _draw(
        b, "service_time", "p1", "provider_visit", 0
    )


def test_different_keys_give_different_draws() -> None:
    s = RandomStreams(42)
    assert _draw(s, "a") != _draw(s, "b")
    assert _draw(s, "service", "p1") != _draw(s, "service", "p2")


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_crn_isolation_perturbing_one_key_leaves_others_identical(seed: int) -> None:
    # BASELINE and OPTIMIZED share a seed; OPTIMIZED additionally consumes policy
    # randomness in a different order. The shared "world" draw must be identical.
    baseline = RandomStreams(seed)
    optimized = RandomStreams(seed)
    _ = optimized.substream("policy", "tiebreak", 5).random()
    _ = optimized.substream("service_time", "pX", 3).standard_normal(10)
    assert _draw(baseline, "world", "arrivals") == _draw(optimized, "world", "arrivals")


def test_world_and_policy_randomness_are_isolated() -> None:
    s = RandomStreams(7)
    assert _draw(s, "world", "arrivals") != _draw(s, "policy", "arrivals")


def test_sample_lognormal_is_deterministic_nonnegative_duration() -> None:
    g1 = RandomStreams(3).substream("service_time", "p1", "provider_visit", 0)
    g2 = RandomStreams(3).substream("service_time", "p1", "provider_visit", 0)
    d1 = sample_lognormal(g1, mean_s=300.0, cv=0.5)
    d2 = sample_lognormal(g2, mean_s=300.0, cv=0.5)
    assert isinstance(d1, Duration)
    assert d1 == d2
    assert d1.root >= 0


def test_sample_poisson_arrivals_within_window_and_deterministic() -> None:
    window = TimeWindow(start=SimTime(0), end=SimTime(7 * 24 * 3_600 * 1_000_000))
    g = RandomStreams(11).substream("world", "arrivals")
    arrivals = sample_poisson_arrivals(g, rate_per_hour=5.0, window=window)
    assert len(arrivals) > 0
    assert all(window.contains(t) for t in arrivals)
    assert list(arrivals) == sorted(arrivals, key=lambda t: t.root)
    again = sample_poisson_arrivals(
        RandomStreams(11).substream("world", "arrivals"), rate_per_hour=5.0, window=window
    )
    assert arrivals == again


def test_sample_poisson_zero_rate_is_empty() -> None:
    window = TimeWindow(start=SimTime(0), end=SimTime(1_000_000_000))
    g = RandomStreams(1).substream("world", "arrivals")
    assert sample_poisson_arrivals(g, rate_per_hour=0.0, window=window) == ()


def test_sample_categorical_respects_weights() -> None:
    g = RandomStreams(1).substream("x")
    assert sample_categorical(g, {"a": 1.0, "b": 0.0}) == "a"
    assert sample_categorical(g, {"only": 3.0}) == "only"


def test_sample_categorical_is_deterministic() -> None:
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    picks_1 = [sample_categorical(RandomStreams(5).substream("cat", i), weights) for i in range(20)]
    picks_2 = [sample_categorical(RandomStreams(5).substream("cat", i), weights) for i in range(20)]
    assert picks_1 == picks_2
    assert set(picks_1) <= {"a", "b", "c"}


def test_substream_returns_numpy_generator() -> None:
    assert isinstance(RandomStreams(0).substream("x"), np.random.Generator)
