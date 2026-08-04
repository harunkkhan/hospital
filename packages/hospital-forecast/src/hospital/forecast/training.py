"""Fit, validate on held-out weeks, and retrain champion/challenger (doc 06 §8).

Splits are **rolling-origin by week**, never random. A shuffled split would put
Tuesday's windows in training and Monday's in validation, letting the model see
the future of the very week it is scored on — the resulting number would be
higher and meaningless. Train on weeks ``[1..k]``, validate on ``k+1``, expand.

Determinism: every ``random_state`` derives from
``core.rng.substream("forecast", …)``, so the same ``(data, config)`` refits to
byte-identical predictions. The artifact version is content-addressed by
``(data_hash, config_hash)`` for the same reason.

``retrain_loop`` is monotone-guarded: a challenger replaces the champion only if
``promote_if`` says it is *better*, never on a tie. Promoting on a wash would let
a model drift away from the one that was actually validated, one no-op swap at a
time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import ConfigDict

from hospital.core import (
    Duration,
    EsiAcuity,
    EventLog,
    FrozenModel,
    OperatingWeek,
    Patient,
    PatientId,
    RandomStreams,
    minutes,
)
from hospital.data.vitals import VitalsStream, deterioration_label
from hospital.forecast.arrivals import fit_arrival_intensity, poisson_deviance
from hospital.forecast.deterioration import (
    auroc,
    brier_score,
    fit_deterioration_model,
    news2_for_features,
    operating_point,
)
from hospital.forecast.features import (
    PATIENT_FEATURE_NAMES,
    VITALS_FEATURE_NAMES,
    ComplaintEncoder,
    FeatureFrame,
    concat_frames,
    patient_features,
    to_matrix,
    vitals_window_features,
    window_features,
)
from hospital.forecast.model_store import ArtifactMeta, ModelStore, canonical_hash
from hospital.forecast.service_time import (
    fit_service_time_regressor,
    fit_service_time_table,
    los_training_frame,
    patient_los,
)

if TYPE_CHECKING:
    from hospital.forecast.deterioration import DeteriorationModel
    from hospital.forecast.service_time import ServiceTimeRegressor, ServiceTimeTable

DEFAULT_HORIZON: Final[Duration] = minutes(30)
DEFAULT_WINDOW: Final[Duration] = minutes(30)

# Four, not three: the corpus must supply a TRAIN set, a separate CALIBRATION week
# (which the isotonic calibrator and the threshold sweep both see), and a HOLDOUT week
# that nothing in fitting has touched. With three, the scored week was the calibration
# week and the report restated its own fitting fold as evidence.
MIN_WEEKS: Final[int] = 4


class WeekData(FrozenModel):
    """One week's artifacts, keyed by the run that produced it.

    Weeks stay separate all the way through training. Each runs on its own
    ``[start, end)`` timeline, so concatenating their logs would interleave
    unrelated instants and make every congestion feature a fiction.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run: str
    log: EventLog
    roster: Mapping[PatientId, Patient]
    week: OperatingWeek
    vitals: tuple[VitalsStream, ...] = ()
    acuity: Mapping[PatientId, EsiAcuity] = {}


class GbtParams(FrozenModel):
    """Tree hyperparameters. Part of the config hash, so a change is a new version."""

    max_depth: int = 4
    learning_rate: float = 0.08
    n_estimators: int = 200
    min_samples_leaf: int = 20
    l2_reg: float = 1.0


class TrainConfig(FrozenModel):
    """Everything that determines the fit, and therefore the artifact version."""

    seed: int
    horizon: Duration = DEFAULT_HORIZON
    window: Duration = DEFAULT_WINDOW
    target_sensitivity: float = 0.90
    max_false_alarm_rate: float = 0.25
    surge_quantile: float = 0.9
    smoothing: float = 1.0
    min_service_samples: int = 30
    gbt: GbtParams = GbtParams()


class ValidationReport(FrozenModel):
    """Per-model metrics on the held-out week, and the exact provenance of the split.

    ``calibration`` is recorded separately from ``train`` because the calibrator and
    the decision threshold are fitted on it. A reader has to be able to see that the
    scored week is neither — otherwise "held out" is an unverifiable claim.
    """

    per_model: Mapping[str, Mapping[str, float]]
    holdout: tuple[str, ...]
    train: tuple[str, ...]
    calibration: tuple[str, ...] = ()

    def metric(self, model: str, name: str) -> float | None:
        return self.per_model.get(model, {}).get(name)


class TrainedBundle(FrozenModel):
    """Metadata only — the fitted estimators live in the :class:`ModelStore`."""

    version: str
    data_hash: str
    config_hash: str
    metrics: ValidationReport


@dataclass(frozen=True)
class FittedModels:
    """The in-memory result of one fit, before it is persisted.

    A dataclass rather than a ``FrozenModel``: its fields are fitted estimators —
    opaque payloads, not validated values (doc 06 §2) — so there is nothing for
    pydantic to check and a validated container would only force those estimator
    types to be importable at runtime for no benefit.
    """

    service_table: ServiceTimeTable
    los_regressor: ServiceTimeRegressor
    deterioration: DeteriorationModel
    encoder: ComplaintEncoder


def rolling_origin_splits(runs: Sequence[str]) -> tuple[tuple[tuple[str, ...], str], ...]:
    """``([1..k], k+1)`` for every k — the expanding-window protocol (doc 06 §8).

    Returned as data so the caller can see exactly which weeks trained and which
    scored. The same splits feed the closed-loop harness, so "validated well" and
    "improved the sim" are measured on identically-held-out weeks.
    """
    if len(runs) < 2:
        return ()
    return tuple((tuple(runs[: k + 1]), runs[k + 1]) for k in range(len(runs) - 1))


def _vitals_frame(week: WeekData, config: TrainConfig) -> FeatureFrame | None:
    rows: list[object] = []
    labels: list[float] = []
    ids: list[str] = []
    for stream in week.vitals:
        esi = week.acuity.get(stream.patient, EsiAcuity.ESI3)
        for row in vitals_window_features(
            stream,
            esi,
            news2_for_features,
            window=config.window,
            stride=config.window,
        ):
            rows.append(row)
            ids.append(f"{stream.patient.root}@{row.window_end.root}")
            labels.append(
                float(deterioration_label(stream, row.window_end, horizon=config.horizon))
            )
    if not rows:
        return None
    return to_matrix(
        rows,  # type: ignore[arg-type]
        feature_names=VITALS_FEATURE_NAMES,
        row_ids=ids,
        labels=labels,
    )


def _los_frame(week: WeekData, encoder: ComplaintEncoder) -> FeatureFrame | None:
    rows = patient_features(week.log, week.roster, week.week)
    los = patient_los(week.log)
    if not rows or not los:
        return None
    frame = los_training_frame(rows, los, encoder, feature_names=PATIENT_FEATURE_NAMES)
    return frame if frame.matrix else None


def fit_models(
    train_weeks: Sequence[WeekData], validation: WeekData, config: TrainConfig
) -> FittedModels:
    """Fit every model on ``train_weeks``, using ``validation`` only where required.

    The deterioration classifier's calibrator and threshold are the one place the
    validation week is touched during fitting — by construction, since calibrating
    on the training fold would map the model's own over-confidence onto itself.
    """
    if not train_weeks:
        raise ValueError("no training weeks")
    streams = RandomStreams(config.seed)
    encoder = ComplaintEncoder.fit(
        patient.complaint for week in train_weeks for patient in week.roster.values()
    )

    # Each week's log paired with its OWN roster: patient ids are unique only within
    # a run, so pooling the rosters would file one week's durations under another
    # week's complaint and acuity.
    table = fit_service_time_table(
        [(w.log, w.roster) for w in train_weeks], min_samples=config.min_service_samples
    )

    los_frames = [f for f in (_los_frame(w, encoder) for w in train_weeks) if f is not None]
    if not los_frames:
        raise ValueError("no completed stays in the training weeks")
    regressor = fit_service_time_regressor(
        concat_frames(los_frames), encoder, streams=streams, quantiles=(0.9,)
    )

    train_vitals = [f for f in (_vitals_frame(w, config) for w in train_weeks) if f is not None]
    valid_vitals = _vitals_frame(validation, config)
    if not train_vitals or valid_vitals is None:
        raise ValueError("no vitals windows to fit the deterioration model on")
    deterioration = fit_deterioration_model(
        concat_frames(train_vitals),
        valid_vitals,
        streams=streams,
        horizon=config.horizon,
        target_sensitivity=config.target_sensitivity,
        max_false_alarm_rate=config.max_false_alarm_rate,
    )
    return FittedModels(
        service_table=table,
        los_regressor=regressor,
        deterioration=deterioration,
        encoder=encoder,
    )


def score_models(
    models: FittedModels,
    train_weeks: Sequence[WeekData],
    holdout: WeekData,
    config: TrainConfig,
    *,
    calibration: WeekData | None = None,
) -> ValidationReport:
    """Score every model on a week that fitting never touched (doc 06 §8 metrics).

    ``calibration`` is recorded on the report for provenance; it must NOT be the same
    week as ``holdout``, and this asserts that rather than trusting the caller.
    """
    if calibration is not None and calibration.run == holdout.run:
        raise ValueError(
            f"calibration and holdout are the same week ({holdout.run}); "
            "the report would restate a fitting fold as held-out evidence"
        )
    per_model: dict[str, dict[str, float]] = {}

    intensity = fit_arrival_intensity(
        [w.log for w in train_weeks], holdout.week, smoothing=config.smoothing
    )
    observed = [float(row.count) for row in window_features(holdout.log, holdout.week)]
    predicted = [
        intensity.expected_arrivals(row.window, holdout.week)
        for row in window_features(holdout.log, holdout.week)
    ]
    per_model["arrivals"] = {"poisson_deviance": poisson_deviance(observed, predicted)}

    los_frame = _los_frame(holdout, models.encoder)
    if los_frame is not None:
        per_model["service_time"] = {
            "mae_log": models.los_regressor.point_log_error(los_frame),
            "n": float(len(los_frame)),
        }

    vitals_frame = _vitals_frame(holdout, config)
    if vitals_frame is not None and vitals_frame.labels is not None:
        labels = [int(v) for v in vitals_frame.labels]
        probabilities = models.deterioration.classifier.predict_proba(vitals_frame.matrix)
        # Recomputed HERE, on the holdout, at the model's chosen threshold. Copying
        # `models.deterioration.sensitivity` would report the calibration fold's own
        # performance — the fold the threshold was selected on — as held-out evidence.
        sensitivity, false_alarms = operating_point(
            probabilities, labels, models.deterioration.threshold
        )
        per_model["deterioration"] = {
            "auroc": auroc(probabilities, labels),
            "brier": brier_score(probabilities, labels),
            "sensitivity": sensitivity,
            "false_alarm_rate": false_alarms,
            "threshold": models.deterioration.threshold,
            "positives": float(sum(labels)),
        }

    return ValidationReport(
        per_model=per_model,
        holdout=(holdout.run,),
        train=tuple(w.run for w in train_weeks),
        calibration=(calibration.run,) if calibration is not None else (),
    )


def data_hash(weeks: Sequence[WeekData]) -> str:
    """Content address for a training corpus — **every** input that shapes the fit.

    The log alone is not enough. The roster supplies ``complaint`` and the workup
    counts (which key the service-time table and populate the LOS features), the
    vitals streams and their labels are the entire deterioration training set, and
    the operating week sets the binning. Hashing only the log meant two corpora that
    fit visibly different models could share a version string — and
    ``ModelStore.save`` would then overwrite the earlier payload in place, leaving a
    champion whose behaviour no longer matches its own provenance.

    Ordered by run so the hash is a function of the corpus, not of the call.
    """
    payload = [
        [
            week.run,
            week.log.to_jsonl(),
            [week.week.start.root, week.week.end.root],
            # Sorted: a roster is a mapping, and its iteration order must not move
            # the hash.
            sorted([pid.root, patient.model_dump_json()] for pid, patient in week.roster.items()),
            [stream.model_dump_json() for stream in week.vitals],
            sorted([pid.root, int(esi)] for pid, esi in week.acuity.items()),
        ]
        for week in sorted(weeks, key=lambda w: w.run)
    ]
    return canonical_hash(payload)


def train_all(
    weeks: Sequence[WeekData],
    config: TrainConfig,
    store: ModelStore,
    *,
    name: str = "forecast",
    trained_at: str = "",
) -> TrainedBundle:
    """Fit, calibrate, score, and persist — on three disjoint folds in time order.

    ``trained_at`` is passed in rather than read from the clock: a wall-clock
    stamp inside the artifact would make two otherwise-identical builds differ,
    which is exactly what content-addressing exists to prevent.
    """
    if len(weeks) < MIN_WEEKS:
        raise ValueError(f"need at least {MIN_WEEKS} weeks; got {len(weeks)}")
    # Three folds, in time order: train on the earliest, calibrate + pick the
    # threshold on the second-to-last, score on the last. The scored week is the one
    # nothing in fitting has seen.
    *train_weeks, calibration, holdout = weeks
    models = fit_models(train_weeks, calibration, config)
    report = score_models(models, train_weeks, holdout, config, calibration=calibration)

    dhash = data_hash(weeks)
    chash = canonical_hash(config.model_dump(mode="json"))
    version = f"{dhash[:12]}-{chash[:12]}"
    flat = {
        f"{model}.{metric}": value
        for model, metrics in report.per_model.items()
        for metric, value in metrics.items()
    }
    store.save(
        name,
        version,
        models,
        ArtifactMeta(
            name=name,
            version=version,
            data_hash=dhash,
            config_hash=chash,
            trained_at=trained_at,
            metrics=flat,
        ),
    )
    return TrainedBundle(version=version, data_hash=dhash, config_hash=chash, metrics=report)


def validate_bundle(
    version: str,
    store: ModelStore,
    holdout: WeekData,
    train_weeks: Sequence[WeekData],
    config: TrainConfig,
    *,
    name: str = "forecast",
) -> ValidationReport:
    """Re-score a stored version against a (possibly new) held-out week."""
    models, _ = store.load(name, version)
    return score_models(models, train_weeks, holdout, config)


def retrain_loop(
    store: ModelStore,
    weeks: Sequence[WeekData],
    config: TrainConfig,
    *,
    promote_if: Callable[[ValidationReport, ValidationReport], bool],
    name: str = "forecast",
    trained_at: str = "",
) -> TrainedBundle | None:
    """Fit a challenger on the enlarged corpus; promote only if it is better.

    Returns the promoted bundle, or ``None`` when the champion is kept. The
    challenger is trained, scored, **and stored** either way — "we checked and the
    incumbent still wins" is a result worth having, and keeping the rejected
    artifact means the comparison can be re-examined later rather than re-run.
    """
    # Read the incumbent BEFORE training. `train_all` persists the challenger, and
    # `ModelStore.save` auto-promotes into an empty store -- so asking afterwards
    # would find the challenger sitting in the champion slot and report "kept"
    # for a run that had no incumbent at all.
    try:
        incumbent: ArtifactMeta | None = store.champion(name)
    except KeyError:
        incumbent = None

    challenger = train_all(weeks, config, store, name=name, trained_at=trained_at)
    if incumbent is None:
        store.promote(name, challenger.version)
        return challenger

    if incumbent.version == challenger.version:
        # Same data, same config: content addressing already resolved them to one
        # artifact, so there is nothing to compare and nothing to promote.
        return None

    champion_report = _report_from_meta(incumbent, challenger.metrics)
    if promote_if(champion_report, challenger.metrics):
        store.promote(name, challenger.version)
        return challenger
    return None


def _report_from_meta(meta: ArtifactMeta, shape: ValidationReport) -> ValidationReport:
    """Rebuild a report from a stored artifact's flattened metrics."""
    per_model: dict[str, dict[str, float]] = {}
    for flat_key, value in meta.metrics.items():
        model, _, metric = flat_key.partition(".")
        per_model.setdefault(model, {})[metric] = value
    return ValidationReport(per_model=per_model, holdout=shape.holdout, train=shape.train)


def improves_deterioration_auroc(
    champion: ValidationReport, challenger: ValidationReport, *, margin: float = 0.0
) -> bool:
    """A ready-made ``promote_if``: strictly better held-out AUROC.

    Strict, with an optional margin. A ``>=`` here would promote on every tie and
    let the live model wander away from the one that was actually validated.
    """
    old = champion.metric("deterioration", "auroc")
    new = challenger.metric("deterioration", "auroc")
    if new is None:
        return False
    if old is None:
        return True
    return new > old + margin


__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_WINDOW",
    "MIN_WEEKS",
    "FittedModels",
    "GbtParams",
    "TrainConfig",
    "TrainedBundle",
    "ValidationReport",
    "WeekData",
    "data_hash",
    "fit_models",
    "improves_deterioration_auroc",
    "retrain_loop",
    "rolling_origin_splits",
    "score_models",
    "train_all",
    "validate_bundle",
]
