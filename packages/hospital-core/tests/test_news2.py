"""The NEWS2 rubric, checked value-by-value against the published chart.

NEWS2 is a published clinical score, not a model, so "close enough" is not a
standard it can be held to — every band boundary is asserted at both edges. It
lives in ``core`` because ``sim`` stamps ``VitalsSampled.news2`` and cannot import
``forecast``; these tests live here with it.
"""

from __future__ import annotations

import pytest

from hospital.core import NEWS2_PARAMETERS, VitalsReading, news2_score


def _reading(**overrides: int) -> VitalsReading:
    """A NEWS2-normal adult, with the parameter under test overridden."""
    base = {"hr": 70, "spo2": 98, "sbp": 130, "dbp": 80, "temp_c_x10": 370, "rr": 16}
    return VitalsReading(**{**base, **overrides})


# --------------------------------------------------------------------- NEWS2
def test_a_normal_adult_scores_zero() -> None:
    scored = news2_score(_reading())
    assert scored.total == 0
    assert scored.band == "low"
    assert set(scored.sub) == set(NEWS2_PARAMETERS)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        # Respiration rate: <=8 -> 3, 9-11 -> 1, 12-20 -> 0, 21-24 -> 2, >=25 -> 3
        ("rr", 8, 3), ("rr", 9, 1), ("rr", 11, 1), ("rr", 12, 0), ("rr", 20, 0),
        ("rr", 21, 2), ("rr", 24, 2), ("rr", 25, 3),
        # SpO2 Scale 1: <=91 -> 3, 92-93 -> 2, 94-95 -> 1, >=96 -> 0
        ("spo2", 91, 3), ("spo2", 92, 2), ("spo2", 93, 2), ("spo2", 94, 1),
        ("spo2", 95, 1), ("spo2", 96, 0),
        # Systolic BP: <=90 -> 3, 91-100 -> 2, 101-110 -> 1, 111-219 -> 0, >=220 -> 3
        ("sbp", 90, 3), ("sbp", 91, 2), ("sbp", 100, 2), ("sbp", 101, 1),
        ("sbp", 110, 1), ("sbp", 111, 0), ("sbp", 219, 0), ("sbp", 220, 3),
        # Pulse: <=40 -> 3, 41-50 -> 1, 51-90 -> 0, 91-110 -> 1, 111-130 -> 2, >=131 -> 3
        ("hr", 40, 3), ("hr", 41, 1), ("hr", 50, 1), ("hr", 51, 0), ("hr", 90, 0),
        ("hr", 91, 1), ("hr", 110, 1), ("hr", 111, 2), ("hr", 130, 2), ("hr", 131, 3),
        # Temperature: <=35.0 -> 3, 35.1-36.0 -> 1, 36.1-38.0 -> 0, 38.1-39.0 -> 1, >=39.1 -> 2
        ("temp_c_x10", 350, 3), ("temp_c_x10", 351, 1), ("temp_c_x10", 360, 1),
        ("temp_c_x10", 361, 0), ("temp_c_x10", 380, 0), ("temp_c_x10", 381, 1),
        ("temp_c_x10", 390, 1), ("temp_c_x10", 391, 2),
    ],
)  # fmt: skip
def test_news2_matches_the_published_rubric(field: str, value: int, expected: int) -> None:
    """Every band boundary of the published chart, checked at both edges."""
    parameter = {
        "rr": "resp",
        "spo2": "spo2",
        "sbp": "sbp",
        "hr": "pulse",
        "temp_c_x10": "temp",
    }[field]
    scored = news2_score(_reading(**{field: value}))
    assert scored.sub[parameter] == expected, (field, value, scored.sub)


def test_supplemental_oxygen_adds_two() -> None:
    assert news2_score(_reading(), on_oxygen=True).sub["oxygen"] == 2
    assert news2_score(_reading(), on_oxygen=False).sub["oxygen"] == 0


def test_any_single_parameter_at_three_escalates_regardless_of_total() -> None:
    """The chart's override rule: one catastrophic vital is not averaged away."""
    scored = news2_score(_reading(spo2=88))
    assert scored.sub["spo2"] == 3
    assert scored.total == 3, "the total alone would read as low risk"
    assert scored.single_red is True
    # The RCP chart's URGENT response, not the emergency one: a lone 3 at total 3 does
    # not call for the same reaction as an aggregate of 7, and collapsing them would
    # make the band stop distinguishing anything.
    assert scored.band == "medium"
    assert news2_score(_reading()).single_red is False


def test_bands_follow_the_total_when_no_parameter_is_red() -> None:
    low = news2_score(_reading(hr=95))  # pulse 1
    assert low.total == 1 and low.band == "low"
    medium = news2_score(_reading(hr=95, rr=22, spo2=94, temp_c_x10=385))  # 1+2+1+1
    assert medium.total == 5 and medium.band == "medium"
    high = news2_score(_reading(hr=115, rr=22, spo2=92, temp_c_x10=385))  # 2+2+2+1
    assert high.total == 7 and high.band == "high"
    assert not high.single_red, "total 7 with no parameter at 3 is the emergency band"


def test_news2_is_pure_and_deterministic() -> None:
    reading = _reading(hr=105, spo2=93)
    assert news2_score(reading) == news2_score(reading)


def test_a_reading_reports_its_temperature_in_celsius() -> None:
    """The rubric's bands are written in degrees; the wire value is tenths."""
    assert VitalsReading(hr=70, spo2=98, sbp=130, dbp=80, temp_c_x10=376, rr=16).temp_c == 37.6


def test_the_rubric_spans_the_whole_real_line() -> None:
    """No reading may fall between bands — a gap would silently score as normal."""
    for extreme in (0, 1, 500):
        scored = news2_score(
            VitalsReading(
                hr=max(25, min(220, extreme)),
                spo2=max(50, min(100, extreme)),
                sbp=max(50, min(220, extreme)),
                dbp=80,
                temp_c_x10=max(330, min(425, extreme)),
                rr=max(4, min(60, extreme)),
            )
        )
        assert set(scored.sub) == set(NEWS2_PARAMETERS)
        assert scored.total >= 0
