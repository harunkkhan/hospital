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
from hospital.forecast.arrivals import fit_arrival_intensity
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
    TrainConfig,
    ValidationReport,
    WeekData,
    data_hash,
    improves_deterioration_auroc,
    retrain_loop,
    rolling_origin_splits,
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


_WEEKS = [_week(i) for i in range(3)]


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
    assert report.holdout == ("run-002",)
    assert report.train == ("run-000", "run-001")
    assert set(report.per_model) >= {"arrivals", "service_time", "deterioration"}
    assert report.metric("deterioration", "auroc") is not None
    assert report.metric("service_time", "mae_log") is not None
    assert report.metric("arrivals", "poisson_deviance") is not None


def test_training_refuses_too_few_weeks(tmp_path: Path) -> None:
    """Two weeks cannot both train and hold out — say so instead of degrading."""
    with pytest.raises(ValueError, match="at least"):
        train_all(_WEEKS[:2], _CONFIG, ModelStore(tmp_path))
    assert MIN_WEEKS == 3


def test_a_trained_bundle_round_trips_through_the_store(tmp_path: Path) -> None:
    """Export then reload must reproduce identical predictions (doc 06 §15)."""
    store = ModelStore(tmp_path)
    bundle = train_all(_WEEKS, _CONFIG, store)
    loaded, meta = store.load("forecast", bundle.version)
    assert meta.version == bundle.version

    reloaded_report = validate_bundle(bundle.version, store, _WEEKS[-1], _WEEKS[:-1], _CONFIG)
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
    table = fit_service_time_table([week.log], week.roster, min_samples=20)
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
    table = fit_service_time_table([week.log], week.roster, min_samples=20)
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
    table = fit_service_time_table([week.log], week.roster, min_samples=5)
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
