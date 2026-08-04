"""Arrival intensity and surge: recover a known λ, stay additive, cover the band.

The fixture generates arrivals from an intensity it will tell us afterwards
(``SynthWeek.true_lambda``), so these are not "the fit ran" tests — they check
the estimate against the process that produced the data.

Per-bin comparison would be dominated by Poisson noise (a 168-bin weekly profile
over six weeks has ~20 arrivals per bin, so each bin's λ̂ carries ~22% relative
error by construction). The assertions therefore aggregate to hour-of-day, where
each estimate is backed by 42 bin-hours and the sampling error is ~7% — tight
enough that a real bias would show.
"""

from __future__ import annotations

import statistics

import pytest
from _forecast_fixtures import synth_week, synth_weeks

from hospital.core import (
    Duration,
    EventLog,
    RandomStreams,
    SimTime,
    TimeWindow,
    hours,
    minutes,
)
from hospital.forecast.arrivals import (
    SURGE_FEATURE_NAMES,
    fit_arrival_intensity,
    fit_surge_forecaster,
    intensity_from_rates,
    poisson_deviance,
    quantile_coverage,
)
from hospital.forecast.features import window_features

_WEEKS = synth_weeks(6)
_WEEK = _WEEKS[0].week
_LOGS = [w.log for w in _WEEKS]


def _by_hour_of_day(values: list[float]) -> list[float]:
    """Collapse 168 weekly bins to 24 hour-of-day means."""
    return [statistics.mean(values[hour::24]) for hour in range(24)]


def test_fit_recovers_the_generating_intensity() -> None:
    """λ̂ must track the λ the fixture actually sampled from, shape and level."""
    model = fit_arrival_intensity(_LOGS, _WEEK)
    assert len(model.rates_per_hour) == 168
    assert model.n_weeks == 6

    truth = [_WEEKS[0].true_lambda(b) for b in range(168)]
    fitted_hod = _by_hour_of_day(list(model.rates_per_hour))
    true_hod = _by_hour_of_day(truth)

    # Level: no systematic bias in either direction.
    for hour, (got, want) in enumerate(zip(fitted_hod, true_hod, strict=True)):
        assert abs(got - want) <= 0.25 * want + 0.15, (hour, got, want)

    # Shape: the daily profile is recovered, not just the average.
    assert fitted_hod.index(max(fitted_hod)) in range(14, 19), "afternoon peak"
    assert fitted_hod.index(min(fitted_hod)) in range(0, 7), "small-hours trough"


def test_more_weeks_of_exposure_tighten_the_fit() -> None:
    """The estimator must actually converge — the point of pooling weeks."""
    truth = [_WEEKS[0].true_lambda(b) for b in range(168)]

    def error(n: int) -> float:
        model = fit_arrival_intensity(_LOGS[:n], _WEEK)
        return statistics.mean(
            abs(got - want) for got, want in zip(model.rates_per_hour, truth, strict=True)
        )

    assert error(6) < error(1), "six weeks must beat one"


def test_total_expected_arrivals_matches_the_observed_count() -> None:
    """The MLE is exactly the observed rate: integrating it returns the count back."""
    model = fit_arrival_intensity(_LOGS, _WEEK)
    whole_week = TimeWindow(start=_WEEK.start, end=_WEEK.end)
    expected = model.expected_arrivals(whole_week, _WEEK)
    observed = sum(sum(row.count for row in window_features(log, _WEEK)) for log in _LOGS)
    assert expected == pytest.approx(observed / 6)


def test_expected_arrivals_is_additive_over_adjacent_windows() -> None:
    """Additivity is what lets `solver.scheduling` sum an hourly demand over a shift."""
    model = fit_arrival_intensity(_LOGS, _WEEK)
    start = SimTime(hours(30).root)
    mid = SimTime(hours(34).root)
    end = SimTime(hours(41).root)
    whole = model.expected_arrivals(TimeWindow(start=start, end=end), _WEEK)
    left = model.expected_arrivals(TimeWindow(start=start, end=mid), _WEEK)
    right = model.expected_arrivals(TimeWindow(start=mid, end=end), _WEEK)
    assert whole == pytest.approx(left + right)


def test_expected_arrivals_pro_rates_a_partial_bin() -> None:
    """Half an hour of a bin is half its λ — not a whole bin, not zero."""
    model = fit_arrival_intensity(_LOGS, _WEEK)
    start = SimTime(hours(10).root)
    half = TimeWindow(start=start, end=SimTime(start.root + minutes(30).root))
    full = TimeWindow(start=start, end=SimTime(start.root + hours(1).root))
    assert model.expected_arrivals(half, _WEEK) == pytest.approx(
        model.expected_arrivals(full, _WEEK) / 2
    )


def test_intensity_reads_the_bin_containing_the_instant() -> None:
    model = fit_arrival_intensity(_LOGS, _WEEK)
    for bin_index in (0, 5, 100, 167):
        at = SimTime(hours(bin_index).root + minutes(17).root)
        assert model.intensity(at, _WEEK) == model.rates_per_hour[bin_index]


def test_shrinkage_pulls_the_profile_toward_the_global_rate() -> None:
    """Shrinkage is a regularizer, not a rescale: it flattens without shifting the mean.

    A 168-bin profile fit on a few weeks has bins backed by a handful of arrivals;
    unshrunk they swing on sampling noise alone. The dial trades that variance for
    bias toward the pooled rate, so heavier smoothing must strictly reduce spread.
    """
    plain = fit_arrival_intensity(_LOGS, _WEEK, smoothing=0.0)
    mild = fit_arrival_intensity(_LOGS, _WEEK, smoothing=1.0)
    heavy = fit_arrival_intensity(_LOGS, _WEEK, smoothing=20.0)

    spreads = [statistics.pstdev(m.rates_per_hour) for m in (plain, mild, heavy)]
    assert spreads[0] > spreads[1] > spreads[2], spreads

    # The centre of mass barely moves -- this is shrinkage, not attenuation.
    means = [statistics.mean(m.rates_per_hour) for m in (plain, mild, heavy)]
    assert abs(means[2] - means[0]) < 0.05 * means[0], means


def test_fit_rejects_a_resolution_that_does_not_divide_the_week() -> None:
    """A ragged final bin would silently under-count its exposure."""
    for bad in (Duration(hours(5).root), Duration(hours(1).root + 1)):
        try:
            fit_arrival_intensity(_LOGS, _WEEK, resolution=bad)
        except ValueError as exc:
            assert "divide" in str(exc)
        else:  # pragma: no cover - the raise is the contract
            raise AssertionError(f"resolution {bad} must be rejected")


def test_fit_rejects_negative_smoothing() -> None:
    try:
        fit_arrival_intensity(_LOGS, _WEEK, smoothing=-1.0)
    except ValueError as exc:
        assert "smoothing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("negative smoothing must be rejected")


def test_an_empty_corpus_fits_a_zero_intensity() -> None:
    """No data is a rate of zero, not a crash and not a guess."""
    model = fit_arrival_intensity([EventLog()], _WEEK)
    assert set(model.rates_per_hour) == {0.0}
    assert model.expected_arrivals(TimeWindow(start=_WEEK.start, end=_WEEK.end), _WEEK) == 0.0


def test_fit_is_deterministic() -> None:
    a = fit_arrival_intensity(_LOGS, _WEEK, smoothing=1.0)
    b = fit_arrival_intensity(_LOGS, _WEEK, smoothing=1.0)
    assert a == b


def test_intensity_from_rates_builds_the_static_arms_model() -> None:
    """The A/B baseline arm needs the same type from scenario constants."""
    model = intensity_from_rates({0: 1.5, 3: 2.5}, resolution=hours(1), n_bins=168)
    assert model.rates_per_hour[0] == 1.5
    assert model.rates_per_hour[3] == 2.5
    assert model.rates_per_hour[7] == 0.0
    assert len(model.rates_per_hour) == 168


# ------------------------------------------------------------------- surge
_TRAIN = _LOGS[:5]
_HOLDOUT = _WEEKS[5]
_FORECASTER = fit_surge_forecaster(
    _TRAIN, _WEEK, streams=RandomStreams(17), quantile=0.9, max_lead=4
)


def _holdout_actuals() -> dict[int, int]:
    return {row.window.start.root: row.count for row in window_features(_HOLDOUT.log, _WEEK)}


def _sweep() -> tuple[list[float], list[float], list[float]]:
    """Predict from many origins across the holdout week; return (actual, mean, upper)."""
    actuals = _holdout_actuals()
    observed: list[float] = []
    mean: list[float] = []
    upper: list[float] = []
    for origin_hour in range(24, 160, 3):
        forecast = _FORECASTER.predict(
            _HOLDOUT.log, SimTime(hours(origin_hour).root), _WEEK, horizon=hours(4)
        )
        for window, m, u in zip(forecast.lead_bins, forecast.mean, forecast.upper_q, strict=True):
            observed.append(float(actuals[window.start.root]))
            mean.append(m)
            upper.append(u)
    return observed, mean, upper


def test_surge_band_covers_close_to_its_nominal_quantile() -> None:
    """A p90 band that covers 60% is a lie a staffing plan would act on."""
    observed, _, upper = _sweep()
    assert observed, "the sweep must produce forecasts"
    coverage = quantile_coverage(observed, upper)
    assert 0.78 <= coverage <= 0.99, coverage


def test_surge_band_never_sits_below_its_point_estimate() -> None:
    """The two quantile heads are fit independently and can cross on sparse data."""
    _, mean, upper = _sweep()
    assert all(u >= m for u, m in zip(upper, mean, strict=True))


def test_surge_point_forecast_beats_a_flat_baseline() -> None:
    """The model must earn its keep against "predict the weekly average, always"."""
    observed, mean, _ = _sweep()
    flat = statistics.mean(observed)
    model_error = poisson_deviance(observed, mean)
    flat_error = poisson_deviance(observed, [flat] * len(observed))
    assert model_error < flat_error, (model_error, flat_error)


def test_surge_forecast_is_shaped_and_ordered() -> None:
    forecast = _FORECASTER.predict(_HOLDOUT.log, SimTime(hours(50).root), _WEEK, horizon=hours(4))
    assert forecast.quantile == 0.9
    assert len(forecast.lead_bins) == len(forecast.mean) == len(forecast.upper_q) == 4
    assert forecast.made_at == SimTime(hours(50).root)
    starts = [w.start.root for w in forecast.lead_bins]
    assert starts == sorted(starts), "lead bins run forward in time"
    assert starts[0] == hours(51).root, "the first lead bin is strictly after the origin"
    assert all(v >= 0.0 for v in (*forecast.mean, *forecast.upper_q)), "counts are non-negative"


def test_surge_prediction_uses_only_history_up_to_now() -> None:
    """Truncating the future must not change a forecast made before it.

    The lags in a design row come from the ORIGIN bin. If they came from the
    target — which is unobserved at prediction time — deleting the rest of the
    week would move the answer.
    """
    now = SimTime(hours(60).root)
    truncated = EventLog()
    for env in _HOLDOUT.log.ordered():
        if env.event.occurred_at.root <= now.root:
            truncated.append(env.event, caused_by=env.caused_by)

    full = _FORECASTER.predict(_HOLDOUT.log, now, _WEEK, horizon=hours(4))
    partial = _FORECASTER.predict(truncated, now, _WEEK, horizon=hours(4))
    assert full.mean == partial.mean
    assert full.upper_q == partial.upper_q


def test_surge_feature_block_carries_lead_and_seasonal() -> None:
    assert "lead" in SURGE_FEATURE_NAMES
    assert "seasonal" in SURGE_FEATURE_NAMES
    assert _FORECASTER.feature_names == SURGE_FEATURE_NAMES


def test_surge_training_is_reproducible_from_the_seed() -> None:
    """`random_state` comes from the one seeded RNG, so a refit is byte-identical."""
    again = fit_surge_forecaster(_TRAIN, _WEEK, streams=RandomStreams(17), quantile=0.9, max_lead=4)
    now = SimTime(hours(72).root)
    assert again.predict(_HOLDOUT.log, now, _WEEK) == _FORECASTER.predict(_HOLDOUT.log, now, _WEEK)


def test_surge_rejects_an_impossible_quantile() -> None:
    for bad in (0.0, 1.0, -0.5, 2.0):
        try:
            fit_surge_forecaster(_TRAIN, _WEEK, streams=RandomStreams(1), quantile=bad)
        except ValueError as exc:
            assert "quantile" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"quantile {bad} must be rejected")


def test_poisson_deviance_is_zero_for_a_perfect_forecast() -> None:
    counts = [0.0, 1.0, 5.0, 12.0]
    assert poisson_deviance(counts, counts) == pytest.approx(0.0)
    assert poisson_deviance(counts, [c + 2 for c in counts]) > 0.0


def test_quantile_coverage_counts_observations_under_the_band() -> None:
    assert quantile_coverage([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 1.0
    assert quantile_coverage([1.0, 9.0], [5.0, 5.0]) == 0.5
    assert quantile_coverage([], []) == 0.0


def test_a_single_week_still_fits() -> None:
    """The smallest usable corpus must not need special handling."""
    one = synth_week(days=7, week_index=99)
    model = fit_arrival_intensity([one.log], one.week)
    assert model.n_weeks == 1
    assert sum(model.rates_per_hour) > 0.0


def test_week_boundaries_are_respected() -> None:
    """An instant outside the operating week is an error, not a silent wrap."""
    forecaster = _FORECASTER
    outside = SimTime(_WEEK.end.root + hours(5).root)
    try:
        forecaster.predict(_HOLDOUT.log, outside, _WEEK, horizon=hours(4))
    except ValueError as exc:
        assert "week" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an out-of-week origin must be rejected")


def test_poisson_deviance_refuses_negative_counts_and_rates() -> None:
    """A negative rate would make the deviance negative — better than perfect.

    Deviance is only a distance between valid Poisson quantities. Left unguarded,
    `poisson_deviance([0], [-1])` returns -2, so an impossible forecast outranks an
    exact one and every comparison built on the metric silently inverts.
    """
    for observed, predicted in (([0.0], [-1.0]), ([-1.0], [1.0]), ([2.0, 1.0], [1.0, -0.5])):
        try:
            poisson_deviance(observed, predicted)
        except ValueError as exc:
            assert "non-negative" in str(exc)
        else:  # pragma: no cover - the raise is the contract
            raise AssertionError(f"expected a rejection for {observed=} {predicted=}")

    # A zero rate against a zero count is legitimate and scores perfectly.
    assert poisson_deviance([0.0], [0.0]) == pytest.approx(0.0)
