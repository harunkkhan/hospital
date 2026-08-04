# pyright: reportMissingTypeStubs=false
# ^ scoped to this file on purpose: `sklearn` and `joblib` ship no stubs, and this
#   is the only module in the package permitted to import them. Every other module
#   type-checks strictly against the wrappers below.
"""The one untyped boundary: narrow, typed wrappers over the sklearn estimators.

``scikit-learn`` ships incomplete annotations, and the repo type-checks in strict
mode. Rather than sprinkle ``Any`` and suppressions through every model module,
the whole surface the package needs is wrapped here — three estimators and a
``joblib`` round-trip — so exactly one file talks to an untyped library and
everything above it stays strictly typed.

Internal (``_``-prefixed): used by several modules inside this package and by
nothing outside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, cast

import joblib
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# Shared defaults. Small trees and a low learning rate because the training sets
# here are thousands of rows, not millions — a deep forest would memorize a week.
DEFAULT_MAX_DEPTH: Final[int] = 4
DEFAULT_LEARNING_RATE: Final[float] = 0.08
DEFAULT_MAX_ITER: Final[int] = 200
DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 20
DEFAULT_L2: Final[float] = 1.0


class GbtSettings:
    """The tree hyperparameters both estimators take, as one bundle.

    A plain container so ``training.GbtParams`` has exactly one thing to hand down —
    while the wiring was missing, that config sat in the version hash without
    reaching a single fitted tree.
    """

    __slots__ = ("l2_regularization", "learning_rate", "max_depth", "max_iter", "min_samples_leaf")

    def __init__(
        self,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_iter: int = DEFAULT_MAX_ITER,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        l2_regularization: float = DEFAULT_L2,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization


def _as_matrix(rows: Sequence[Sequence[float]]) -> Any:
    return np.asarray(rows, dtype=float)


def _as_vector(values: Sequence[float]) -> Any:
    return np.asarray(values, dtype=float)


def _to_floats(values: Any) -> tuple[float, ...]:
    """Materialize an untyped array-like into plain floats at the boundary."""
    return tuple(float(v) for v in cast("list[Any]", values))


class GbtRegressor:
    """A gradient-boosted regressor: a conditional mean, or a conditional quantile.

    ``loss="poisson"`` is the right choice for counts — it models a log-link mean
    and keeps predictions non-negative. Passing ``quantile`` instead switches to
    pinball loss, which estimates that quantile and **not** the mean: a median
    count forecast systematically under-predicts a right-skewed arrival
    distribution, so the two are not interchangeable.
    """

    def __init__(
        self,
        *,
        random_state: int,
        quantile: float | None = None,
        loss: Literal["squared_error", "poisson"] = "squared_error",
        settings: GbtSettings | None = None,
    ) -> None:
        tuned = settings or GbtSettings()
        kwargs: dict[str, Any] = {
            "max_depth": tuned.max_depth,
            "learning_rate": tuned.learning_rate,
            "max_iter": tuned.max_iter,
            "min_samples_leaf": tuned.min_samples_leaf,
            "l2_regularization": tuned.l2_regularization,
            "random_state": random_state,
        }
        if quantile is not None:
            kwargs["loss"] = "quantile"
            kwargs["quantile"] = quantile
        else:
            kwargs["loss"] = loss
        self.quantile = quantile
        self.loss = "quantile" if quantile is not None else loss
        self._model = cast("Any", HistGradientBoostingRegressor(**kwargs))

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[float]) -> GbtRegressor:
        self._model.fit(_as_matrix(rows), _as_vector(labels))
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        if not rows:
            return ()
        return _to_floats(self._model.predict(_as_matrix(rows)))


class GbtClassifier:
    """A gradient-boosted binary classifier with an isotonic calibration stage.

    Raw boosted scores are not probabilities — they are systematically
    over-confident — and the deterioration threshold is chosen on a probability
    scale. Calibrating on a held-out fold is what makes "p >= 0.4" mean roughly
    what it says, which is the difference between a tunable alarm and a dial.
    """

    def __init__(
        self,
        *,
        random_state: int,
        settings: GbtSettings | None = None,
    ) -> None:
        tuned = settings or GbtSettings()
        self._model = cast(
            "Any",
            HistGradientBoostingClassifier(
                max_depth=tuned.max_depth,
                learning_rate=tuned.learning_rate,
                max_iter=tuned.max_iter,
                min_samples_leaf=tuned.min_samples_leaf,
                l2_regularization=tuned.l2_regularization,
                random_state=random_state,
            ),
        )
        self._calibrator: Any | None = None

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> GbtClassifier:
        self._model.fit(_as_matrix(rows), np.asarray(labels, dtype=int))
        return self

    def calibrate(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> GbtClassifier:
        """Fit isotonic calibration on a fold the classifier did not train on.

        Calibrating on the training fold would map the model's own over-confidence
        onto itself and report near-perfect reliability that does not hold.
        """
        if not rows:
            return self
        raw = self._raw_probabilities(rows)
        calibrator = cast("Any", IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0))
        calibrator.fit(_as_vector(raw), np.asarray(labels, dtype=float))
        self._calibrator = calibrator
        return self

    def _raw_probabilities(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        return tuple(
            float(row[1]) for row in cast("list[Any]", self._model.predict_proba(_as_matrix(rows)))
        )

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        if not rows:
            return ()
        raw = self._raw_probabilities(rows)
        if self._calibrator is None:
            return raw
        return tuple(
            min(1.0, max(0.0, v)) for v in _to_floats(self._calibrator.predict(_as_vector(raw)))
        )


def dump(payload: object, path: Path) -> None:
    """Persist an estimator payload (doc 06 §13-11: joblib for the sklearn backend)."""
    cast("Any", joblib).dump(payload, path)


def load(path: Path) -> Any:
    return cast("Any", joblib).load(path)


__all__ = [
    "DEFAULT_L2",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ITER",
    "DEFAULT_MIN_SAMPLES_LEAF",
    "GbtClassifier",
    "GbtRegressor",
    "GbtSettings",
    "dump",
    "load",
]
