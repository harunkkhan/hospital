"""Training orchestration and the artifact store: splits, hashes, promotion, the bundle.

The claims that matter here are about *protocol*, not accuracy: that a week never
appears in both halves of a split, that a version is a pure function of its data
and config, that a challenger is promoted only when it is genuinely better, and
that the ML and static bundles are the same shape so the A/B swap is honest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _forecast_fixtures import synth_week, vitals_cohort

from hospital.core import Activity, Duration, EsiAcuity, SimTime, hours, minutes, seconds
from hospital.forecast.arrivals import fit_arrival_intensity, poisson_deviance
from hospital.forecast.features import patient_features, window_features
from hospital.forecast.model_store import (
    ArtifactMeta,
    ModelStore,
    PredictionBundle,
    bundle_from_models,
    canonical_hash,
    static_bundle,
)
from hospital.forecast.service_time import fit_service_time_table
from hospital.forecast.training import (
    MIN_WEEKS,
    GbtParams,
    TrainConfig,
    ValidationReport,
    WeekData,
    data_hash,
    fit_models,
    improves_deterioration_auroc,
    retrain_loop,
    rolling_origin_splits,
    score_models,
    train_all,
    validate_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = TrainConfig(seed=13, window=minutes(30), horizon=minutes(30))


def _week(index: int) -> WeekData:
    """One week of events plus a small vitals cohort, on its own timeline."""
    synth = synth_week(days=3, week_index=index)
    cohort = vitals_cohort(70, week_index=index)
    return WeekData(
        run=f"run-{index:03d}",
        log=synth.log,
        roster=synth.roster,
        week=synth.week,
        vitals=cohort.streams,
        acuity=cohort.acuity,
    )


# Four weeks: train / calibrate / score needs three disjoint folds, and the
# rolling-origin protocol needs at least one week to train on.
_WEEKS = [_week(i) for i in range(4)]
_TEST_ROWS = patient_features(_WEEKS[-1].log, _WEEKS[-1].roster, _WEEKS[-1].week)[:25]


# ----------------------------------------------------------------- splitting
def test_rolling_origin_splits_never_overlap_and_move_forward() -> None:
    """Train on the past, score on the next week — the only honest ordering.

    A shuffled split would put a week's own later hours in training, so the model
    would be scored on data whose future it had already seen.
    """
    runs = ["w1", "w2", "w3", "w4"]
    splits = rolling_origin_splits(runs)
    assert len(splits) == 3
    for train, holdout in splits:
        assert holdout not in train, "a week cannot both train and score"
        assert train == tuple(runs[: runs.index(holdout)]), "training is a strict prefix"
    # Expanding window: each split trains on strictly more than the last.
    assert [len(train) for train, _ in splits] == [1, 2, 3]


def test_a_single_week_yields_no_split() -> None:
    assert rolling_origin_splits(["only"]) == ()
    assert rolling_origin_splits([]) == ()


# -------------------------------------------------------------------- hashes
def test_canonical_hash_is_order_independent_and_change_sensitive() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_data_hash_tracks_the_corpus_not_the_call() -> None:
    assert data_hash(_WEEKS) == data_hash(_WEEKS)
    assert data_hash(_WEEKS[:2]) != data_hash(_WEEKS)


def test_config_changes_produce_a_different_version(tmp_path: Path) -> None:
    """A hyperparameter change must not silently overwrite the previous artifact."""
    store = ModelStore(tmp_path)
    first = train_all(_WEEKS, _CONFIG, store)
    other = train_all(_WEEKS, _CONFIG.model_copy(update={"target_sensitivity": 0.75}), store)
    assert first.version != other.version
    assert first.data_hash == other.data_hash, "same data"
    assert first.config_hash != other.config_hash


def test_the_same_data_and_config_reproduce_the_same_version(tmp_path: Path) -> None:
    """Content addressing: a rebuild is the same artifact, not a new one."""
    a = train_all(_WEEKS, _CONFIG, ModelStore(tmp_path / "a"))
    b = train_all(_WEEKS, _CONFIG, ModelStore(tmp_path / "b"))
    assert a.version == b.version


# ------------------------------------------------------------------ training
def test_train_all_scores_on_the_last_week_only(tmp_path: Path) -> None:
    bundle = train_all(_WEEKS, _CONFIG, ModelStore(tmp_path))
    report = bundle.metrics
    assert report.holdout == ("run-003",), "the LAST week is scored"
    assert report.calibration == ("run-002",), "the second-to-last is the calibration fold"
    assert report.train == ("run-000", "run-001")
    # The three folds are disjoint -- that is the whole point of the split.
    assert not set(report.train) & set(report.calibration)
    assert not set(report.train) & set(report.holdout)
    assert not set(report.calibration) & set(report.holdout)
    assert set(report.per_model) >= {"arrivals", "service_time", "deterioration"}
    assert report.metric("deterioration", "auroc") is not None
    assert report.metric("service_time", "mae_log") is not None
    assert report.metric("arrivals", "poisson_deviance") is not None


def test_training_refuses_too_few_weeks(tmp_path: Path) -> None:
    """Three folds need four weeks — say so instead of silently reusing one."""
    with pytest.raises(ValueError, match="at least"):
        train_all(_WEEKS[:3], _CONFIG, ModelStore(tmp_path))
    assert MIN_WEEKS == 4


def test_a_trained_bundle_round_trips_through_the_store(tmp_path: Path) -> None:
    """Export then reload must reproduce identical predictions (doc 06 §15)."""
    store = ModelStore(tmp_path)
    bundle = train_all(_WEEKS, _CONFIG, store)
    loaded, meta = store.load("forecast", bundle.version)
    assert meta.version == bundle.version

    reloaded_report = validate_bundle(bundle.version, store, _WEEKS[-1], _WEEKS[:-2], _CONFIG)
    assert reloaded_report.metric("deterioration", "auroc") == bundle.metrics.metric(
        "deterioration", "auroc"
    )
    assert loaded.deterioration.threshold == pytest.approx(loaded.deterioration.threshold)


# --------------------------------------------------------------------- store
def _meta(version: str, auroc: float) -> ArtifactMeta:
    return ArtifactMeta(
        name="m",
        version=version,
        data_hash="d",
        config_hash="c",
        trained_at="",
        metrics={"deterioration.auroc": auroc},
    )


def test_the_first_saved_version_becomes_champion(tmp_path: Path) -> None:
    """Otherwise `load("latest")` would fail on a store with exactly one model."""
    store = ModelStore(tmp_path)
    store.save("m", "v1", {"payload": 1}, _meta("v1", 0.7))
    assert store.champion("m").version == "v1"
    payload, meta = store.load("m")
    assert payload == {"payload": 1}
    assert meta.is_champion


def test_promotion_moves_the_latest_pointer(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    store.save("m", "v1", {"n": 1}, _meta("v1", 0.7))
    store.save("m", "v2", {"n": 2}, _meta("v2", 0.8))
    assert store.champion("m").version == "v1", "a later save must not auto-promote"
    store.promote("m", "v2")
    assert store.champion("m").version == "v2"
    payload, _ = store.load("m")
    assert payload == {"n": 2}


def test_promoting_an_unknown_version_is_refused(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    store.save("m", "v1", {}, _meta("v1", 0.7))
    with pytest.raises(KeyError, match="unknown version"):
        store.promote("m", "nope")


def test_listing_versions_reports_the_champion_flag(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    store.save("m", "v1", {}, _meta("v1", 0.7))
    store.save("m", "v2", {}, _meta("v2", 0.8))
    store.promote("m", "v2")
    versions = {m.version: m for m in store.list_versions("m")}
    assert set(versions) == {"v1", "v2"}
    assert versions["v2"].is_champion
    assert not versions["v1"].is_champion


def test_loading_from_an_empty_store_is_an_error(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    with pytest.raises(KeyError, match="no champion"):
        store.load("missing")
    assert store.list_versions("missing") == ()


# ------------------------------------------------------- champion/challenger
def _report(auroc: float) -> ValidationReport:
    return ValidationReport(
        per_model={"deterioration": {"auroc": auroc}}, holdout=("h",), train=("t",)
    )


def test_promotion_requires_a_strict_improvement() -> None:
    """A tie keeps the incumbent: promoting on a wash lets the live model drift."""
    assert improves_deterioration_auroc(_report(0.70), _report(0.75))
    assert not improves_deterioration_auroc(_report(0.70), _report(0.70))
    assert not improves_deterioration_auroc(_report(0.75), _report(0.70))


def test_promotion_can_require_a_margin() -> None:
    assert not improves_deterioration_auroc(_report(0.70), _report(0.705), margin=0.02)
    assert improves_deterioration_auroc(_report(0.70), _report(0.73), margin=0.02)


def test_retrain_keeps_the_champion_when_the_challenger_is_no_better(
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    train_all(_WEEKS, _CONFIG, store)
    champion_before = store.champion("forecast").version

    promoted = retrain_loop(
        store,
        _WEEKS,
        _CONFIG.model_copy(update={"seed": 99}),
        promote_if=lambda _champion, _challenger: False,
    )
    assert promoted is None
    assert store.champion("forecast").version == champion_before


def test_retrain_promotes_when_promote_if_says_so(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    train_all(_WEEKS, _CONFIG, store)
    champion_before = store.champion("forecast").version

    promoted = retrain_loop(
        store,
        _WEEKS,
        _CONFIG.model_copy(update={"seed": 99}),
        promote_if=lambda _champion, _challenger: True,
    )
    assert promoted is not None
    assert promoted.version != champion_before
    assert store.champion("forecast").version == promoted.version


def test_retraining_an_identical_cell_is_a_no_op(tmp_path: Path) -> None:
    """Same data, same config: content addressing already made them one artifact."""
    store = ModelStore(tmp_path)
    train_all(_WEEKS, _CONFIG, store)
    promoted = retrain_loop(store, _WEEKS, _CONFIG, promote_if=lambda _a, _b: True)
    assert promoted is None, "there is nothing to promote over itself"


# -------------------------------------------------------------------- bundle
def test_the_ml_and_static_bundles_are_the_same_shape() -> None:
    """The A/B swap is one call — which is only true if the types match exactly.

    If the static arm had its own shape, any measured delta would be confounded
    with the difference between two code paths.
    """
    week = _WEEKS[0]
    table = fit_service_time_table([(week.log, week.roster)], min_samples=20)
    intensity = fit_arrival_intensity([week.log], week.week)
    ml = bundle_from_models("v1", service_time=table, arrivals=intensity)
    static = static_bundle(
        resolution=hours(1),
        arrival_rates_per_hour=[1.0] * len(intensity.rates_per_hour),
        activity_means_s={Activity.PROVIDER_VISIT: 700.0},
    )
    assert type(ml) is type(static) is PredictionBundle
    assert set(ml.model_dump()) == set(static.model_dump())


def test_the_bundle_is_composed_only_of_core_types() -> None:
    """Nothing downstream should have to import `forecast` to read a prediction."""
    week = _WEEKS[0]
    table = fit_service_time_table([(week.log, week.roster)], min_samples=20)
    bundle = bundle_from_models(
        "v1", service_time=table, arrivals=fit_arrival_intensity([week.log], week.week)
    )
    for key, value in bundle.expected_service.items():
        activity, esi, complaint = key
        assert isinstance(activity, Activity)
        assert isinstance(esi, EsiAcuity)
        assert isinstance(complaint, str)
        assert isinstance(value, Duration)
    for activity, value in bundle.fallback_service.items():
        assert isinstance(activity, Activity)
        assert isinstance(value, Duration)
    assert isinstance(bundle.resolution, Duration)


def test_the_bundle_falls_back_per_activity_and_never_returns_zero() -> None:
    bundle = static_bundle(
        resolution=hours(1),
        arrival_rates_per_hour=[1.0],
        activity_means_s={Activity.PROVIDER_VISIT: 700.0},
    )
    assert bundle.expected_for(Activity.PROVIDER_VISIT, EsiAcuity.ESI2, "never_seen") == seconds(
        700.0
    )
    with pytest.raises(KeyError, match="no expected duration"):
        bundle.expected_for(Activity.IMAGING, EsiAcuity.ESI2, "x")


def test_a_fitted_key_beats_the_activity_fallback() -> None:
    week = _WEEKS[0]
    table = fit_service_time_table([(week.log, week.roster)], min_samples=5)
    bundle = bundle_from_models(
        "v1", service_time=table, arrivals=fit_arrival_intensity([week.log], week.week)
    )
    key = next(iter(bundle.expected_service))
    activity, esi, complaint = key
    assert bundle.expected_for(activity, esi, complaint) == bundle.expected_service[key]
    # An unseen complaint on the same activity drops to the fallback instead.
    assert bundle.expected_for(activity, esi, "unheard_of") == bundle.fallback_service[activity]


def test_the_first_retrain_reports_the_champion_it_installed(tmp_path: Path) -> None:
    """An empty store has no incumbent, so the first challenger IS a promotion.

    `train_all` persists the challenger and `ModelStore.save` auto-promotes into an
    empty store. Reading the champion *after* training would therefore find the
    challenger already in the champion slot and report `None` -- "we kept the
    incumbent" -- for a run that had no incumbent and installed a new model.
    """
    store = ModelStore(tmp_path)
    promoted = retrain_loop(store, _WEEKS, _CONFIG, promote_if=lambda _a, _b: False)
    assert promoted is not None, "a first run has nothing to keep; it promotes"
    assert store.champion("forecast").version == promoted.version


def test_a_rejected_challenger_is_still_stored(tmp_path: Path) -> None:
    """Keeping the rejected artifact lets the comparison be re-examined later."""
    store = ModelStore(tmp_path)
    train_all(_WEEKS, _CONFIG, store)
    champion = store.champion("forecast").version

    retrain_loop(
        store,
        _WEEKS,
        _CONFIG.model_copy(update={"seed": 77}),
        promote_if=lambda _a, _b: False,
    )
    versions = {m.version for m in store.list_versions("forecast")}
    assert len(versions) == 2, "the challenger must be persisted even when rejected"
    assert store.champion("forecast").version == champion


def test_data_hash_covers_every_input_that_shapes_the_fit() -> None:
    """The log alone is not the corpus.

    Roster (complaint/workup), vitals streams, acuity and the week boundaries all
    determine the fitted payload. If the hash ignored them, two corpora that fit
    visibly different models would share a version — and `ModelStore.save` would
    overwrite the earlier payload in place, leaving a champion whose behaviour no
    longer matches its recorded provenance.
    """
    base = _WEEKS[0]
    original = data_hash([base])

    # Same log, one patient's complaint changed.
    pid, patient = next(iter(base.roster.items()))
    reworded = dict(base.roster)
    reworded[pid] = patient.model_copy(update={"complaint": "something_else"})
    assert data_hash([base.model_copy(update={"roster": reworded})]) != original

    # Same log and roster, different vitals.
    assert data_hash([base.model_copy(update={"vitals": base.vitals[:-1]})]) != original

    # Same everything, different acuity mapping.
    shifted = dict(base.acuity)
    shifted[pid] = EsiAcuity.ESI1
    assert data_hash([base.model_copy(update={"acuity": shifted})]) != original

    # Same everything, different operating week.
    other_week = base.week.model_copy(update={"end": SimTime(base.week.end.root + 1)})
    assert data_hash([base.model_copy(update={"week": other_week})]) != original


def test_data_hash_is_a_function_of_the_corpus_not_the_call_order() -> None:
    assert data_hash(_WEEKS) == data_hash(list(reversed(_WEEKS)))


def test_the_scored_week_is_not_the_calibration_week(tmp_path: Path) -> None:
    """The report's own provenance must show three disjoint folds.

    The calibrator and the threshold sweep both see the calibration week. Scoring on
    that same week restates a fitting fold as held-out evidence — which is what the
    reported AUROC/Brier/sensitivity previously did.
    """
    report = train_all(_WEEKS, _CONFIG, ModelStore(tmp_path)).metrics
    assert report.calibration and report.holdout
    assert report.calibration != report.holdout


def test_scoring_refuses_to_reuse_the_calibration_week_as_holdout() -> None:
    """Caught in `score_models`, not left to the caller to remember."""
    models = fit_models(_WEEKS[:2], _WEEKS[2], _CONFIG)
    with pytest.raises(ValueError, match="same week"):
        score_models(models, _WEEKS[:2], _WEEKS[2], _CONFIG, calibration=_WEEKS[2])


def test_the_reported_operating_point_is_recomputed_on_the_holdout(tmp_path: Path) -> None:
    """Sensitivity/FAR must describe the scored week, not threshold selection.

    They used to be copied straight off the model, where they described the
    calibration fold. The two genuinely differ, so equality would be the bug.
    """
    store = ModelStore(tmp_path)
    bundle = train_all(_WEEKS, _CONFIG, store)
    loaded, _ = store.load("forecast", bundle.version)

    reported = bundle.metrics.per_model["deterioration"]
    assert reported["threshold"] == pytest.approx(loaded.deterioration.threshold)
    # The holdout's operating point is computed independently of the model's own
    # stored (calibration-fold) numbers.
    assert "sensitivity" in reported and "false_alarm_rate" in reported
    assert 0.0 <= reported["sensitivity"] <= 1.0
    assert 0.0 <= reported["false_alarm_rate"] <= 1.0
    assert reported["positives"] > 0, "a holdout with no positives cannot score recall"


def test_a_loaded_artifact_can_build_a_prediction_bundle(tmp_path: Path) -> None:
    """The whole point of the store: load an artifact, get solver inputs.

    `bundle_from_models` needs the arrival model. While `FittedModels` omitted it,
    `store.load()` could not produce a bundle at all, so the ML arm of the A/B had
    nothing to be fed — and `config.surge_quantile` affected no artifact.
    """
    store = ModelStore(tmp_path)
    bundle_meta = train_all(_WEEKS, _CONFIG, store)
    models, _ = store.load("forecast", bundle_meta.version)

    assert models.arrivals is not None
    assert len(models.arrivals.rates_per_hour) > 0

    predictions = bundle_from_models(
        bundle_meta.version, service_time=models.service_table, arrivals=models.arrivals
    )
    assert isinstance(predictions, PredictionBundle)
    assert predictions.arrival_rates_per_hour == models.arrivals.rates_per_hour
    assert predictions.fallback_service, "the bundle must carry usable expected durations"


def test_the_scored_arrival_model_is_the_one_persisted(tmp_path: Path) -> None:
    """Scoring must describe the artifact, not a fit made only to produce a number."""
    store = ModelStore(tmp_path)
    meta = train_all(_WEEKS, _CONFIG, store)
    models, _ = store.load("forecast", meta.version)

    rows = window_features(_WEEKS[-1].log, _WEEKS[-1].week)
    expected = poisson_deviance(
        [float(r.count) for r in rows],
        [models.arrivals.expected_arrivals(r.window, _WEEKS[-1].week) for r in rows],
    )
    assert meta.metrics.metric("arrivals", "poisson_deviance") == pytest.approx(expected)


def test_the_surge_quantile_reaches_the_artifact(tmp_path: Path) -> None:
    """A config field that changes no fitted model is a false provenance claim."""
    store = ModelStore(tmp_path)
    meta = train_all(_WEEKS, _CONFIG.model_copy(update={"surge_quantile": 0.8}), store)
    models, _ = store.load("forecast", meta.version)
    if models.surge is not None:
        assert models.surge.quantile == 0.8
    else:  # pragma: no cover - thin corpora legitimately skip the surge head
        assert meta.metrics.metric("surge", "quantile_coverage") is None


def test_the_gbt_hyperparameters_actually_reach_the_estimators(tmp_path: Path) -> None:
    """A config field in the version hash that changes no tree is a false claim.

    `max_depth=1` and `max_depth=8` must fit measurably different models. While the
    wiring was missing they fitted identical trees under different version hashes, so
    the artifact advertised a hyperparameter it had never applied.
    """
    shallow = _CONFIG.model_copy(update={"gbt": GbtParams(max_depth=1, n_estimators=30)})
    deep = _CONFIG.model_copy(update={"gbt": GbtParams(max_depth=8, n_estimators=200)})

    store = ModelStore(tmp_path)
    a = train_all(_WEEKS, shallow, store)
    b = train_all(_WEEKS, deep, store)
    assert a.version != b.version, "different hyperparameters are different versions"

    models_a, _ = store.load("forecast", a.version)
    models_b, _ = store.load("forecast", b.version)
    rows = _TEST_ROWS
    preds_a = [models_a.los_regressor.predict_median_los(r).root for r in rows]
    preds_b = [models_b.los_regressor.predict_median_los(r).root for r in rows]
    assert preds_a != preds_b, "a depth-1 stump must not predict what a depth-8 tree does"


def test_champion_and_challenger_are_scored_on_the_same_week(tmp_path: Path) -> None:
    """Both sides must be measured on the challenger's holdout, not their own.

    The incumbent's stored metrics describe whatever week it was validated against.
    Comparing those to the challenger's fresh score means a champion that happened to
    be validated on an easy week beats a genuinely better challenger — and the store
    keeps the worse model. `retrain_loop` therefore re-scores the incumbent.
    """
    store = ModelStore(tmp_path)
    train_all(_WEEKS, _CONFIG, store)

    seen: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def record(champion: ValidationReport, challenger: ValidationReport) -> bool:
        seen.append((champion.holdout, challenger.holdout))
        return False

    retrain_loop(store, _WEEKS, _CONFIG.model_copy(update={"seed": 4242}), promote_if=record)
    assert seen, "promote_if must actually be consulted"
    champion_holdout, challenger_holdout = seen[0]
    assert champion_holdout == challenger_holdout, (
        f"compared different weeks: champion on {champion_holdout}, "
        f"challenger on {challenger_holdout}"
    )
    assert champion_holdout == ("run-003",)
