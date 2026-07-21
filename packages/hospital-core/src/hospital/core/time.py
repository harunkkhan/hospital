"""Integer-microsecond time — the queue and every timestamp key on it.

The load-bearing invariant (nuance 1.3): **no float timestamp is ever stored,
compared, or queued.** Float seconds are converted to integer microseconds
exactly once, at the boundary, with **banker's rounding** (round-half-to-even),
so two "identical" runs never diverge through accumulated float drift.

Type algebra is deliberately narrow. ``SimTime`` (an absolute instant) and
``Duration`` (a signed delta) are distinct types, and only these operations are
expressible:

* ``SimTime - SimTime -> Duration``
* ``SimTime + Duration -> SimTime``
* ``Duration +/- Duration -> Duration``

``SimTime + SimTime`` is nonsense and is rejected by the type checker (there is
no such overload) and at runtime (``NotImplemented``). This kills a whole class
of "added two clocks" bugs.

Week windows use a **half-open ``[start, end)`` convention**, applied everywhere
(WIP counting, LOS censoring, warmup) so arms stay comparable.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Annotated

from pydantic import Field, RootModel

from hospital.core.models import FrozenModel

MICROS_PER_SEC = 1_000_000
_SECS_PER_MIN = 60
_SECS_PER_HOUR = 3_600
_HOURS_PER_WEEK = 7 * 24


def round_micros(micros: float) -> int:
    """Banker's-round a microsecond quantity to an integer µs.

    This is *the* rounding rule for the whole repo; every float→µs conversion
    (``seconds``/``minutes``/``hours``, ``units.walk_duration``, and the RNG
    samplers) routes through the same round-half-to-even so per-edge rounding
    can never drift into a false baseline-vs-optimized delta.
    """
    return int(Decimal(str(micros)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def _duration_from(value: float, micros_per_unit: int) -> Duration:
    # Multiply in Decimal (not float) so, e.g., minutes(0.1) has no float artifact
    # before the single half-even rounding step.
    micros = (Decimal(str(value)) * micros_per_unit).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    return Duration(int(micros))


class SimTime(RootModel[Annotated[int, Field(ge=0)]]):
    """An absolute instant in microseconds since the scenario epoch (>= 0)."""

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.root)

    def __sub__(self, other: SimTime) -> Duration:
        return Duration(self.root - other.root)

    def __add__(self, other: Duration) -> SimTime:
        return SimTime(self.root + other.root)

    def __lt__(self, other: SimTime) -> bool:
        return self.root < other.root

    def __le__(self, other: SimTime) -> bool:
        return self.root <= other.root

    def __gt__(self, other: SimTime) -> bool:
        return self.root > other.root

    def __ge__(self, other: SimTime) -> bool:
        return self.root >= other.root


class Duration(RootModel[int]):
    """A signed span in microseconds (may be zero; a delta of two instants)."""

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.root)

    def __add__(self, other: Duration) -> Duration:
        return Duration(self.root + other.root)

    def __sub__(self, other: Duration) -> Duration:
        return Duration(self.root - other.root)

    def __neg__(self) -> Duration:
        return Duration(-self.root)

    def __lt__(self, other: Duration) -> bool:
        return self.root < other.root

    def __le__(self, other: Duration) -> bool:
        return self.root <= other.root

    def __gt__(self, other: Duration) -> bool:
        return self.root > other.root

    def __ge__(self, other: Duration) -> bool:
        return self.root >= other.root

    def to_seconds(self) -> float:
        """Display-only float seconds — never fed back into logic."""
        return self.root / MICROS_PER_SEC


def seconds(n: float) -> Duration:
    """Convert float seconds to a µs :class:`Duration` (banker's-rounded once)."""
    return _duration_from(n, MICROS_PER_SEC)


def minutes(n: float) -> Duration:
    """Convert float minutes to a µs :class:`Duration` (banker's-rounded once)."""
    return _duration_from(n, _SECS_PER_MIN * MICROS_PER_SEC)


def hours(n: float) -> Duration:
    """Convert float hours to a µs :class:`Duration` (banker's-rounded once)."""
    return _duration_from(n, _SECS_PER_HOUR * MICROS_PER_SEC)


class TimeWindow(FrozenModel):
    """A half-open interval ``[start, end)`` of the sim clock."""

    start: SimTime
    end: SimTime

    def duration(self) -> Duration:
        return self.end - self.start

    def contains(self, t: SimTime) -> bool:
        """Half-open membership: ``start <= t < end``."""
        return self.start <= t < self.end


class OperatingWeek(FrozenModel):
    """The scoring week, ``[start, end)`` (half-open).

    The half-open convention is the definition of WIP/LOS censoring: a patient
    whose ``DischargeCompleted`` lands at *exactly* ``end`` is **still WIP**, not
    a completion — applied identically across arms so they stay comparable.
    """

    start: SimTime
    end: SimTime

    @classmethod
    def one_week(cls) -> OperatingWeek:
        """The reference week ``[0, 7 * 24h)`` in microseconds."""
        return cls(start=SimTime(0), end=SimTime(hours(_HOURS_PER_WEEK).root))

    def contains(self, t: SimTime) -> bool:
        """Half-open membership: ``start <= t < end``."""
        return self.start <= t < self.end


__all__ = [
    "MICROS_PER_SEC",
    "Duration",
    "OperatingWeek",
    "SimTime",
    "TimeWindow",
    "hours",
    "minutes",
    "round_micros",
    "seconds",
]
