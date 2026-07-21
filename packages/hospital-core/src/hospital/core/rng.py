"""Determinism + common random numbers (CRN) — the reason arms are comparable.

The core invariant (nuance 1.8): a draw is a pure function of ``(seed,
content-addressable key)``, **independent of the order in which code asks for
it**. ``substream("service_time", patient_id, "provider_visit", i)`` returns the
same generator whether BASELINE or OPTIMIZED requests it, so both arms see the
identical realized week and their measured difference is pure policy signal.

Mechanism: ``blake2b(seed, key) -> 256-bit digest -> SeedSequence -> Generator``.
The key is length-prefixed and type-tagged so distinct key tuples never collide.
Keys must **not** embed process/thread ids (that would break serial-vs-multiprocess
byte-identity). ``SeedSequence`` + a pinned numpy is stable across machines.

World randomness (``substream("world", …)``) is isolated from policy randomness
(``substream("policy", …)``): a policy consuming randomness cannot perturb the
weather.

Float sampler outputs are converted to integer µs with the *same* banker's
rounding rule as :mod:`hospital.core.time`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np

from hospital.core.time import Duration, SimTime, TimeWindow, round_micros, seconds

_MICROS_PER_HOUR = 3_600.0 * 1_000_000.0


def _encode_key_element(k: str | int) -> bytes:
    if isinstance(k, bool):  # bool is an int subclass; tag it distinctly for safety
        return b"b" + (b"1" if k else b"0")
    if isinstance(k, int):
        return b"i" + repr(k).encode("ascii")
    return b"s" + k.encode("utf-8")


class RandomStreams:
    """A single seed, factored into independent content-addressable substreams."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def substream(self, *key: str | int) -> np.random.Generator:
        """Return the generator for ``key`` — a pure function of ``(seed, key)``.

        Deterministic and order-independent: the same key always yields the same
        generator, and adding a new keyed draw never perturbs any existing draw.
        """
        h = hashlib.blake2b(digest_size=32)
        seed_bytes = repr(self._seed).encode("ascii")
        h.update(len(seed_bytes).to_bytes(8, "big"))
        h.update(seed_bytes)
        for element in key:
            encoded = _encode_key_element(element)
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        entropy = int.from_bytes(h.digest(), "big")
        return np.random.default_rng(np.random.SeedSequence(entropy))


def sample_lognormal(g: np.random.Generator, mean_s: float, cv: float) -> Duration:
    """Sample a lognormal service time (seconds), returned as a µs :class:`Duration`.

    Parameterized by the natural mean ``mean_s`` (seconds) and coefficient of
    variation ``cv`` (= std/mean). Uses the shared banker's rounding via
    :func:`hospital.core.time.seconds`, so a sampled service time never drifts
    against a walk time.
    """
    if mean_s <= 0.0:
        raise ValueError("mean_s must be > 0")
    if cv < 0.0:
        raise ValueError("cv must be >= 0")
    if cv == 0.0:
        return seconds(mean_s)
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean_s) - sigma2 / 2.0
    value_s = float(g.lognormal(mean=mu, sigma=sigma))
    return seconds(value_s)


def sample_poisson_arrivals(
    g: np.random.Generator, rate_per_hour: float, window: TimeWindow
) -> tuple[SimTime, ...]:
    """Homogeneous-Poisson arrival instants within the half-open ``window``.

    Exponential inter-arrival gaps at ``rate_per_hour`` (converted to per-µs);
    each instant is banker's-rounded to integer µs. A non-positive rate yields no
    arrivals. Instants are returned sorted and all satisfy ``start <= t < end``.
    """
    if rate_per_hour < 0.0:
        raise ValueError("rate_per_hour must be >= 0")
    start_us = window.start.root
    end_us = window.end.root
    if rate_per_hour == 0.0 or end_us <= start_us:
        return ()
    rate_per_us = rate_per_hour / _MICROS_PER_HOUR
    scale_us = 1.0 / rate_per_us
    arrivals: list[SimTime] = []
    t = float(start_us)
    while True:
        t += float(g.exponential(scale_us))
        if t >= end_us:
            break
        micros = round_micros(t)
        if start_us <= micros < end_us:
            arrivals.append(SimTime(micros))
    return tuple(arrivals)


def sample_categorical[T](g: np.random.Generator, weights: Mapping[T, float]) -> T:
    """Draw one key from ``weights`` proportional to its (non-negative) weight.

    Iterates ``weights`` in its own insertion order (the caller owns ordering);
    weights are normalized to probabilities. Raises on an empty mapping, a
    negative weight, or a non-positive total.
    """
    keys = list(weights.keys())
    if not keys:
        raise ValueError("weights must be non-empty")
    values = np.asarray([weights[k] for k in keys], dtype=float)
    if bool(np.any(values < 0.0)):
        raise ValueError("weights must be non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    probs = values / total
    idx = int(g.choice(len(keys), p=probs))
    return keys[idx]


__all__ = [
    "RandomStreams",
    "sample_categorical",
    "sample_lognormal",
    "sample_poisson_arrivals",
]
