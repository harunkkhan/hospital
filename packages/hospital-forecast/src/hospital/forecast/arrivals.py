"""Arrival intensity and short-horizon surge (doc 06 §5).

This module **estimates rates**. It never draws an arrival: sampling belongs to
``data.workload`` via ``core.rng.sample_poisson_arrivals`` (anti-duplication,
doc 00 §5.2). A fitted intensity can *drive* the generator, but the draw stays
where the seeded RNG lives.

The seasonal model is a piecewise-constant nonhomogeneous Poisson process whose
MLE is closed-form — for bin *b*, ``λ̂[b] = arrivals(b) / exposure_hours(b)``.
Optional shrinkage blends each bin toward the global rate, which matters because
a 168-bin weekly profile fit on a handful of weeks has bins with two or three
observations; unshrunk, those bins swing wildly and the "forecast" is mostly
sampling noise.

The surge head departs from doc 06 §5's "seasonal x predicted multiplier" and
predicts the **count directly**, with the seasonal rate supplied as a feature.
Same information, no division: a multiplier over a near-zero 04:00 λ is
numerically vicious and produces enormous ratios from one extra walk-in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from hospital.core import (
    Duration,
    EventLog,
    FrozenModel,
    OperatingWeek,
    RandomStreams,
    SimTime,
    TimeWindow,
    hours,
)
from hospital.core.events import PatientArrived
from hospital.forecast._estimators import GbtRegressor
from hospital.forecast.features import WINDOW_FEATURE_NAMES, WindowFeatures, window_features

if TYPE_CHECKING:
    from collections.abc import Iterable

_MICROS_PER_HOUR: Final[int] = 3_600 * 1_000_000

# Feature block for the surge head: the bin's own calendar/lag context (from
# `WindowFeatures`) plus how far ahead it is and what the seasonal fit expects.
SURGE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *WINDOW_FEATURE_NAMES,
    "lead",
    "seasonal",
)


class ArrivalIntensityModel(FrozenModel):
    """A piecewise-constant weekly intensity, one λ per bin (arrivals per hour)."""

    resolution: Duration
    rates_per_hour: tuple[float, ...]
    n_weeks: int

    def _bin_of(self, t: SimTime, week: OperatingWeek) -> int:
        offset = (t.root - week.start.root) % (len(self.rates_per_hour) * self.resolution.root)
        return int(offset // self.resolution.root)

    def intensity(self, t: SimTime, week: OperatingWeek) -> float:
        """λ (arrivals/hour) at ``t``, by its position in the operating week."""
        return self.rates_per_hour[self._bin_of(t, week)]

    def expected_arrivals(self, window: TimeWindow, week: OperatingWeek) -> float:
        """``∫ λ dt`` over ``window``, pro-rating the partial bins at each end.

        Additive over adjacent windows by construction, which is what makes it
        usable as ``solver.scheduling``'s covering demand: the expectation over a
        shift equals the sum over its hours.
        """
        total = 0.0
        cursor = window.start.root
        while cursor < window.end.root:
            index = self._bin_of(SimTime(cursor), week)
            bin_end = (
                cursor + self.resolution.root - ((cursor - week.start.root) % self.resolution.root)
            )
            span = min(bin_end, window.end.root) - cursor
            total += self.rates_per_hour[index] * span / _MICROS_PER_HOUR
            cursor += span
        return total


class SurgeForecast(FrozenModel):
    """A short-horizon count forecast per lead bin, with a staffing-safety band."""

    made_at: SimTime
    horizon: Duration
    lead_bins: tuple[TimeWindow, ...]
    mean: tuple[float, ...]
    upper_q: tuple[float, ...]
    quantile: float


def _arrival_instants(log: EventLog) -> tuple[int, ...]:
    return tuple(
        env.event.occurred_at.root for env in log.ordered() if isinstance(env.event, PatientArrived)
    )


def fit_arrival_intensity(
    logs: Sequence[EventLog],
    week: OperatingWeek,
    *,
    resolution: Duration | None = None,
    smoothing: float = 0.0,
) -> ArrivalIntensityModel:
    """Closed-form NHPP MLE over weekly bins, with optional shrinkage.

    ``smoothing`` is expressed in *pseudo-exposure hours* per bin: the estimate
    becomes ``(n_b + smoothing·μ) / (e_b + smoothing)``, so it is a prior of
    ``smoothing`` hours' worth of observation at the global rate μ. At 0 this is
    the plain MLE; large values collapse the profile toward flat. Stating it in
    the same units as the exposure is what keeps it interpretable — "worth an
    hour of average traffic" rather than an opaque dial.
    """
    if smoothing < 0.0:
        raise ValueError("smoothing must be >= 0")
    width = resolution if resolution is not None else hours(1)
    if width.root <= 0:
        raise ValueError("resolution must be positive")
    span = week.end.root - week.start.root
    if span % width.root:
        raise ValueError("resolution must divide the operating week exactly")
    n_bins = span // width.root

    counts = [0] * n_bins
    for log in logs:
        for instant in _arrival_instants(log):
            offset = instant - week.start.root
            if 0 <= offset < span:
                counts[offset // width.root] += 1

    n_weeks = max(1, len(logs))
    exposure_h = n_weeks * width.root / _MICROS_PER_HOUR
    global_rate = math.fsum(counts) / (n_bins * exposure_h) if n_bins else 0.0
    rates = tuple((count + smoothing * global_rate) / (exposure_h + smoothing) for count in counts)
    return ArrivalIntensityModel(resolution=width, rates_per_hour=rates, n_weeks=n_weeks)


def _random_state(streams: RandomStreams, name: str) -> int:
    """A reproducible sklearn ``random_state`` from the one seeded RNG (doc 06 §8)."""
    return int(streams.substream("forecast", name).integers(0, 2**31 - 1))


class SurgeForecaster:
    """Seasonal baseline plus a pair of GBT quantile heads over the residual context.

    Two heads are fit — the median and the safety quantile — because a staffing
    decision needs both "what we expect" and "what we should be able to absorb".
    Reporting only a point estimate would understaff every second week by
    construction.
    """

    def __init__(
        self,
        seasonal: ArrivalIntensityModel,
        point: GbtRegressor,
        upper: GbtRegressor,
        *,
        quantile: float,
        max_lead: int,
        feature_names: tuple[str, ...] = SURGE_FEATURE_NAMES,
    ) -> None:
        self.seasonal = seasonal
        self.feature_names = feature_names
        self.quantile = quantile
        self.max_lead = max_lead
        self._point = point
        self._upper = upper

    def _row(
        self, origin: WindowFeatures, target: WindowFeatures, lead: int, week: OperatingWeek
    ) -> list[float]:
        """One design row: the target bin's calendar, the ORIGIN's lags, the lead.

        Lags come from ``origin`` — the last bin actually observed — never from
        ``target``, which is in the future at prediction time and would otherwise
        leak its own neighbourhood into its forecast.
        """
        values = dict(target.numeric_features())
        origin_values = dict(origin.numeric_features())
        for name in ("lag_1h", "lag_2h", "lag_24h", "lag_1w", "roll_mean_3h", "roll_std_3h"):
            values[name] = origin_values[name]
        values["lead"] = float(lead)
        values["seasonal"] = self.seasonal.expected_arrivals(target.window, week)
        return [values[name] for name in self.feature_names]

    def predict(
        self, log: EventLog, now: SimTime, week: OperatingWeek, *, horizon: Duration | None = None
    ) -> SurgeForecast:
        """Forecast each lead bin after ``now`` from history at or before ``now``."""
        span = horizon if horizon is not None else hours(4)
        rows = window_features(log, week, bin_width=self.seasonal.resolution)
        by_start = {row.window.start.root: row for row in rows}
        width = self.seasonal.resolution.root
        origin_start = week.start.root + ((now.root - week.start.root) // width) * width
        origin = by_start.get(origin_start)
        if origin is None:
            raise ValueError("`now` falls outside the operating week")

        n_leads = max(1, span.root // width)
        windows: list[TimeWindow] = []
        design: list[list[float]] = []
        for lead in range(1, n_leads + 1):
            start = origin_start + lead * width
            target = by_start.get(start)
            if target is None:
                break
            windows.append(target.window)
            design.append(self._row(origin, target, min(lead, self.max_lead), week))
        if not design:
            return SurgeForecast(
                made_at=now,
                horizon=span,
                lead_bins=(),
                mean=(),
                upper_q=(),
                quantile=self.quantile,
            )

        mean = tuple(max(0.0, v) for v in self._point.predict(design))
        raw_upper = tuple(max(0.0, v) for v in self._upper.predict(design))
        # A safety band below the point estimate is not a band; the two quantile
        # heads are fit independently and can cross on sparse data.
        upper = tuple(max(u, m) for u, m in zip(raw_upper, mean, strict=True))
        return SurgeForecast(
            made_at=now,
            horizon=span,
            lead_bins=tuple(windows),
            mean=mean,
            upper_q=upper,
            quantile=self.quantile,
        )


def _surge_design(
    seasonal: ArrivalIntensityModel,
    rows: Sequence[WindowFeatures],
    week: OperatingWeek,
    max_lead: int,
) -> tuple[list[list[float]], list[float]]:
    design: list[list[float]] = []
    labels: list[float] = []
    for origin_index, origin in enumerate(rows):
        for lead in range(1, max_lead + 1):
            target_index = origin_index + lead
            if target_index >= len(rows):
                break
            target = rows[target_index]
            values = dict(target.numeric_features())
            origin_values = dict(origin.numeric_features())
            for name in ("lag_1h", "lag_2h", "lag_24h", "lag_1w", "roll_mean_3h", "roll_std_3h"):
                values[name] = origin_values[name]
            values["lead"] = float(lead)
            values["seasonal"] = seasonal.expected_arrivals(target.window, week)
            design.append([values[name] for name in SURGE_FEATURE_NAMES])
            labels.append(float(target.count))
    return design, labels


def fit_surge_forecaster(
    logs: Sequence[EventLog],
    week: OperatingWeek,
    *,
    streams: RandomStreams,
    quantile: float = 0.9,
    resolution: Duration | None = None,
    max_lead: int = 4,
    smoothing: float = 1.0,
) -> SurgeForecaster:
    """Fit the seasonal profile, then the two quantile heads over lead-time rows.

    Trained **direct multi-horizon**: one row per (origin bin, lead), with the
    lead as a feature. The alternative — a recursive one-step model rolled
    forward — would feed its own predictions back in and compound their error
    across the horizon.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    seasonal = fit_arrival_intensity(logs, week, resolution=resolution, smoothing=smoothing)

    design: list[list[float]] = []
    labels: list[float] = []
    for log in logs:
        rows = window_features(log, week, bin_width=seasonal.resolution)
        week_design, week_labels = _surge_design(seasonal, rows, week, max_lead)
        design.extend(week_design)
        labels.extend(week_labels)
    if not design:
        raise ValueError("no training rows: the logs contain no usable arrival bins")

    # The point head is a Poisson MEAN estimator, not a median. Pinball loss at
    # q=0.5 returns the conditional median, which on right-skewed counts sits
    # below the mean -- so a "mean" field fit that way would under-forecast every
    # busy hour, and a staffing plan built on it would be short by construction.
    point = GbtRegressor(loss="poisson", random_state=_random_state(streams, "surge_mean")).fit(
        design, labels
    )
    upper_head = GbtRegressor(
        quantile=quantile, random_state=_random_state(streams, "surge_upper")
    ).fit(design, labels)

    return SurgeForecaster(
        seasonal,
        point,
        upper_head,
        quantile=quantile,
        max_lead=max_lead,
    )


def poisson_deviance(observed: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Poisson deviance — the right error for counts (doc 06 §8 metrics).

    Squared error would treat a miss of 2 on a busy hour the same as on a dead
    one; deviance scales with the count, which is how arrival error actually
    hurts.

    Non-negativity is enforced rather than assumed: deviance is only a distance
    when both arguments are valid Poisson quantities, and a negative rate makes it
    go negative, which would rank an impossible forecast above a perfect one.
    """
    total = 0.0
    n = 0
    for y, mu in zip(observed, predicted, strict=True):
        # A negative rate is not a Poisson mean, and letting one through makes the
        # deviance negative -- an impossible forecast would then score BETTER than a
        # perfect one, which silently inverts every model comparison built on it.
        if mu < 0.0 or y < 0.0:
            raise ValueError(f"Poisson deviance needs non-negative counts and rates: {y=}, {mu=}")
        # The floor guards the logarithm only. Applying it to the (mu - y) term as
        # well would score a perfect forecast of an empty hour as non-zero.
        term = mu - y
        if y > 0:
            term += y * math.log(y / max(mu, 1e-9))
        total += 2.0 * term
        n += 1
    return total / n if n else 0.0


def quantile_coverage(observed: Iterable[float], upper: Iterable[float]) -> float:
    """Share of observations at or below the band — should approach the quantile."""
    values = list(zip(observed, upper, strict=True))
    if not values:
        return 0.0
    return sum(1 for y, u in values if y <= u) / len(values)


def intensity_from_rates(
    rates: Mapping[int, float], *, resolution: Duration, n_bins: int, n_weeks: int = 1
) -> ArrivalIntensityModel:
    """Build a model from explicit per-bin rates — the static/baseline arm's path."""
    return ArrivalIntensityModel(
        resolution=resolution,
        rates_per_hour=tuple(float(rates.get(i, 0.0)) for i in range(n_bins)),
        n_weeks=n_weeks,
    )


__all__ = [
    "SURGE_FEATURE_NAMES",
    "ArrivalIntensityModel",
    "SurgeForecast",
    "SurgeForecaster",
    "fit_arrival_intensity",
    "fit_surge_forecaster",
    "intensity_from_rates",
    "poisson_deviance",
    "quantile_coverage",
]
