"""The one percentile/mean implementation, reused by ``fold``, ``waits``, ``compare``.

Internal (``_``-prefixed): used by >=2 modules inside this package but by no
other package, so it stays here rather than being promoted to ``core`` (doc 00
§5.2). Pinning a single quantile method matters: if two modules computed a p90
KPI and a bootstrap CI with different quantile rules, a CI could exclude its
own point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from hospital.core import Duration, OperatingWeek, SimTime, TimeWindow, hours

__all__ = [
    "DEFAULT_WARMUP",
    "DEFAULT_WINDOW",
    "clip_seconds",
    "mean",
    "measurement_window",
    "percentile",
]

# Shared module-level singletons for the ``window``/``warmup`` defaults every
# public function in this package exposes (doc 05 §3) — a bare function call in
# an argument default is a lint smell (ruff B008) even when, as here, the
# result is an immutable FrozenModel/RootModel.
DEFAULT_WINDOW: Final[OperatingWeek] = OperatingWeek.one_week()
DEFAULT_WARMUP: Final[Duration] = hours(24)


def _finite(xs: Sequence[float]) -> list[float]:
    """Drop ``NaN`` entries — a ``NaN`` inside a sample breaks the sort's total order."""
    return [x for x in xs if not math.isnan(x)]


def percentile(xs: Sequence[float], q: float) -> float:
    """NumPy type-7 linear-interpolation quantile. Empty sample -> ``NaN``."""
    values = sorted(_finite(xs))
    n = len(values)
    if n == 0:
        return float("nan")
    if n == 1:
        return values[0]
    r = q * (n - 1)
    lo = math.floor(r)
    hi = min(lo + 1, n - 1)
    frac = r - lo
    return values[lo] + frac * (values[hi] - values[lo])


def mean(xs: Sequence[float]) -> float:
    """Arithmetic mean, ``NaN``-filtered. Empty sample -> ``NaN``."""
    values = _finite(xs)
    if not values:
        return float("nan")
    return math.fsum(values) / len(values)


def measurement_window(window: OperatingWeek, warmup: Duration) -> TimeWindow:
    """The post-warmup measurement window ``M = [window.start + warmup, window.end)``."""
    return TimeWindow(start=window.start + warmup, end=window.end)


def clip_seconds(start: SimTime, end: SimTime, window: TimeWindow) -> float:
    """Overlap of ``[start, end]`` with ``window``, in seconds (0.0 if disjoint)."""
    lo = max(start, window.start)
    hi = min(end, window.end)
    if hi <= lo:
        return 0.0
    return (hi - lo).to_seconds()
