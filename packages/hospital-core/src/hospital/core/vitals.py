"""The vitals reading value type and the NEWS2 rubric that scores it.

``core.events.VitalsSampled`` deliberately carries only ``(patient, news2)``: the
event log is a published artifact (the web console types it verbatim), and a full
physiological trace on every tick would bloat it for the sake of one consumer.
But a risk monitor needs the actual numbers, not just their NEWS2 aggregate.

So the reading is a value type owned here, and travels **alongside** the event
into :meth:`hospital.core.seam.RiskMonitor.observe`. ``data.vitals.VitalsSample``
extends it with a timestamp rather than restating the fields, so the generator's
output and the seam's input are the same six numbers by construction — there is
no second declaration to drift.

NEWS2 lives here rather than in ``forecast`` because it is a **published clinical
rubric, not a model** — a pure, table-driven function of a core value type, in the
same family as ``EsiAcuity.priority_weight``. Putting it in ``forecast`` would also
have made it unreachable from ``sim``, which must stamp ``VitalsSampled.news2`` and
cannot import ``forecast`` (the graph runs downward). One definition, three readers:
``sim`` stamps the event, ``forecast`` uses it as a feature, and a reviewer can check
it against the published chart.

Units are integer-scaled for exact reproducibility: temperature is tenths of a
degree Celsius (``temp_c_x10``), everything else a whole unit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from hospital.core.models import FrozenModel


class VitalsReading(FrozenModel):
    """One set of observed vitals, without a time. Integer-scaled throughout."""

    hr: int
    spo2: int
    sbp: int
    dbp: int
    temp_c_x10: int
    rr: int

    @property
    def temp_c(self) -> float:
        """Temperature in degrees Celsius — the scale the NEWS2 rubric is written in."""
        return self.temp_c_x10 / 10.0


Band = Literal["low", "medium", "high"]

# The NEWS2 chart, as (inclusive-low, inclusive-high, score) bands per parameter.
# Written as data so it can be read against the published table line by line.
_RESP: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 8, 3),
    (9, 11, 1),
    (12, 20, 0),
    (21, 24, 2),
    (25, math.inf, 3),
)
_SPO2_SCALE1: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 91, 3),
    (92, 93, 2),
    (94, 95, 1),
    (96, math.inf, 0),
)
_SBP: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 90, 3),
    (91, 100, 2),
    (101, 110, 1),
    (111, 219, 0),
    (220, math.inf, 3),
)
_PULSE: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 40, 3),
    (41, 50, 1),
    (51, 90, 0),
    (91, 110, 1),
    (111, 130, 2),
    (131, math.inf, 3),
)
_TEMP_C: Final[tuple[tuple[float, float, int], ...]] = (
    (-math.inf, 35.0, 3),
    (35.1, 36.0, 1),
    (36.1, 38.0, 0),
    (38.1, 39.0, 1),
    (39.1, math.inf, 2),
)

# A single parameter scoring 3 triggers the URGENT response on its own, even at a low
# total — one deranged vital is not offset by five normal ones. It is NOT the
# emergency response, which the chart reserves for an aggregate of 7 or more.
_SINGLE_PARAM_RED: Final[int] = 3
_HIGH_TOTAL: Final[int] = 7
_MEDIUM_TOTAL: Final[int] = 5

# The order `news2_sub` is reported in; also the order the features use.
NEWS2_PARAMETERS: Final[tuple[str, ...]] = (
    "resp",
    "spo2",
    "oxygen",
    "sbp",
    "pulse",
    "temp",
    "consciousness",
)


class News2Result(FrozenModel):
    """A scored NEWS2 observation: per-parameter sub-scores, total, and band.

    ``single_red`` is reported beside ``band`` because the chart's
    single-parameter-3 trigger is its own escalation *reason*. A caller that needs to
    know why a reading is urgent cannot recover that from the total alone.
    """

    sub: Mapping[str, int]
    total: int
    band: Band
    single_red: bool = False

    def ordered_sub(self) -> tuple[int, ...]:
        return tuple(self.sub[name] for name in NEWS2_PARAMETERS)


def _band_score(bands: Sequence[tuple[float, float, int]], value: float) -> int:
    for low, high, score in bands:
        if low <= value <= high:
            return score
    # Unreachable while the tables span the real line; a loud failure beats a
    # silent 0 that would read as "this vital is normal".
    raise ValueError(f"value {value} fell outside every NEWS2 band")


def news2_score(reading: VitalsReading, *, on_oxygen: bool = False) -> News2Result:
    """Score one observation against the published NEWS2 rubric (SpO2 Scale 1).

    Pure and table-driven, so it can be checked against the reference chart rather
    than against itself. ``consciousness`` is always 0: ``data.vitals`` emits no
    ACVPU level, and assuming "alert" is the honest reading of an absent
    observation — but it is a real gap, not a modelling choice.
    """
    sub = {
        "resp": _band_score(_RESP, reading.rr),
        "spo2": _band_score(_SPO2_SCALE1, reading.spo2),
        "oxygen": 2 if on_oxygen else 0,
        "sbp": _band_score(_SBP, reading.sbp),
        "pulse": _band_score(_PULSE, reading.hr),
        "temp": _band_score(_TEMP_C, reading.temp_c),
        "consciousness": 0,
    }
    total = sum(sub.values())
    single_red = max(sub.values()) >= _SINGLE_PARAM_RED
    # The RCP clinical-response chart separates three responses: total >= 7 is the
    # EMERGENCY response; total 5-6 OR any single parameter scoring 3 is the URGENT
    # one; below that is routine. Collapsing a lone 3 into "high" over-escalates --
    # one deranged vital at total 3 does not call for the same response as an
    # aggregate of 7, and treating them alike is how a band stops carrying
    # information.
    if total >= _HIGH_TOTAL:
        band: Band = "high"
    elif total >= _MEDIUM_TOTAL or single_red:
        band = "medium"
    else:
        band = "low"
    return News2Result(sub=sub, total=total, band=band, single_red=single_red)


__all__ = ["NEWS2_PARAMETERS", "Band", "News2Result", "VitalsReading", "news2_score"]
