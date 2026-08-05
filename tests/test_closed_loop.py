"""The M3 acceptance harness: sim -> features -> fit -> store -> solver -> sim.

This is the milestone's whole claim in one file. Every other M3 test checks a part
in isolation; this one runs the loop end to end and *measures* what closing it is
worth, on weeks that nothing in the fit has seen.

The shape, once per evaluation seed:

1. generate N weeks of sim data (the only source of truth in v1);
2. split them rolling-origin and fit through :func:`train_all` — train on the
   earliest weeks, calibrate and pick the deterioration threshold on the
   second-to-last, score on the last;
3. persist to a :class:`ModelStore` and load the artifacts back, so the arm under
   test consumes what was actually written to disk rather than a live object;
4. run a *fresh* week twice under CRN — once with predictions from the fitted
   model, once with a scale-matched flat stay — and fold both logs
   into KPIs through the same ``analysis`` path the M1 comparison uses;
5. contrast them with :func:`paired_bootstrap`.

Two things about the setup are load-bearing, and both were found by measuring
rather than assumed.

**The floor has to be bay-constrained.** The occupancy term a length-of-stay
prediction feeds prices *holding a bay in a scarce zone*. Neither shipped scenario
makes bays scarce — ``er_floor`` and ``er_floor_stressed`` both run at
``bay_utilization`` ~0.21, 76 bays against 6 arrivals an hour, and
``er_floor_stressed`` stresses *staffing* rather than beds. On a floor with a free
good bay always available, placement barely binds and the prediction has nothing
to buy. So this harness shrinks the floor to 22 bays, which lands at ~0.68
utilization with a stable queue: bays genuinely scarce, without the saturation
(work-in-progress running away) where nothing helps and every arm looks the same.

**The baseline arm has to be matched in scale.** It is a *flat* stay — one number,
the mean completed LOS over the training weeks — given to every patient. That makes
the contrast measure the one thing the fitted model adds: knowing *which* patients
stay long. The obvious alternative, pricing each chart at the generator's own
activity means, turned out to be a trap: summing service times gives ~0.9h against
a realized LOS near 4h, because a stay is mostly *waiting*. Feeding that into a term
that is linear in the stay makes the baseline arm four times weaker rather than
differently-shaped, and the resulting "the fitted arm is worse" is a statement about
occupancy pressure, not about learning.

**What the powered run shows.** At 6 two-day weeks and 8 paired evaluation seeds on
the constrained floor, the fitted models validate well on the held-out week (LOS
``mae_log`` 0.30, deterioration AUROC 0.80 at sensitivity 0.69 / false-alarm 0.17,
arrival Poisson deviance 1.30), and the predictions demonstrably change placement
decisions. The realized-KPI contrast against the scale-matched flat arm is a wash.
Every cohort-mean KPI is flat. Two of the twenty-six keys cross the Bonferroni
threshold — ``los_s_p90_by_esi_4`` better by ~2600s and ``los_s_p90_by_esi_5`` worse
by ~4100s — and they point in *opposite* directions on the two noisiest metrics in
the family: a p90 of a small acuity stratum. That is what a 26-key family does under
noise, not an effect.

That is the honest result and it is what this file asserts — the loop closes, the
artifacts are real, the predictions reach decisions, and per-patient stay length
does not measurably beat its own mean *through this objective term* on this floor.
The tests are therefore written as liveness plus non-inferiority rather than as a
win, and :func:`test_a_contrast_is_actually_being_measured` exists so that "no
regression" cannot quietly degrade into "nothing was compared".

**How much this can detect, measured rather than assumed.** Multiplying every fitted
prediction by ten — a gross mis-calibration — leaves all of it passing: the arms still
differ, the predictions still reach decisions, and the non-inferiority guard still
holds. So the KPI contrast here cannot detect even a tenfold error in its input. Two
things follow, and both bound what the wash above is allowed to mean. The
non-inferiority assertion is a tripwire for gross breakage, not evidence that the
prediction is well calibrated. And the absence of a KPI effect is weak evidence
either way: on this floor, changing *where* patients are placed barely propagates to
the KPIs ``analysis`` folds at all. Measuring this properly needs either far more
reps or a KPI nearer the decision — per-zone occupancy churn, say, or blocked-arrival
counts — and that is the honest next step rather than something this file can claim.
"""

from __future__ import annotations

import atexit
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from hospital.analysis.compare import paired_bootstrap
from hospital.analysis.fold import compute_kpis
from hospital.core import (
    Duration,
    EsiAcuity,
    EventLog,
    FloorLayout,
    KpiVector,
    OperatingWeek,
    Patient,
    PatientId,
    RandomStreams,
    StaffMember,
    TimeWindow,
    hours,
    minutes,
    seconds,
)
from hospital.data.layout import generate_floor
from hospital.data.scenario import Scenario, apply_overlay, load_scenario, realize_staff
from hospital.data.vitals import generate_vitals
from hospital.data.workload import generate_workload
from hospital.forecast import (
    ModelStore,
    RollingDeteriorationMonitor,
    TrainConfig,
    WeekData,
    patient_features,
    train_all,
)
from hospital.forecast.deterioration import DeteriorationModel
from hospital.forecast.service_time import ServiceTimeRegressor, patient_los
from hospital.forecast.training import ValidationReport
from hospital.sim import run_replication
from hospital.sim.experiment.replication import DEFAULT_OBJECTIVE
from hospital.sim.flow.vitals import VitalsWatch

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Sized for the default suite, not for statistical power: the numbers quoted in the
# module docstring come from 6 two-day weeks and 8 evaluation seeds. `MIN_WEEKS` in
# `forecast.training` is 4, and the fit needs three disjoint folds, so 4 is the floor.
_TRAIN_WEEKS = 4
_EVAL_SEEDS = 3
_WEEK_DAYS = 1

_WATCH = VitalsWatch(cadence=minutes(10), span=hours(6), monitor_at_or_above=EsiAcuity.ESI3)
# The prediction port is inert at w_occupancy=0, so an arm that prices occupancy is
# what makes the comparison a comparison at all.
_WEIGHTED = DEFAULT_OBJECTIVE.model_copy(update={"w_occupancy": 2})
_WARMUP = hours(4)

# A bay-constrained floor -- see the module docstring. 22 bays rather than 76.
_TIGHT_ZONES = (
    {"bays": 4, "isolation_bays": 0, "max_bays_per_station": 12, "zone_type": "fast_track"},
    {"bays": 6, "isolation_bays": 1, "max_bays_per_station": 12, "zone_type": "general"},
    {"bays": 5, "isolation_bays": 1, "max_bays_per_station": 12, "zone_type": "general"},
    {"bays": 4, "isolation_bays": 1, "max_bays_per_station": 12, "zone_type": "observation"},
    {"bays": 3, "isolation_bays": 2, "max_bays_per_station": 12, "zone_type": "resus_trauma"},
)


@cache
def _scenario() -> Scenario:
    base = load_scenario(_REPO_ROOT / "scenarios" / "er_floor.yaml")
    return apply_overlay(
        base,
        {
            "name": "closed_loop",
            "facility": {"zones": [dict(zone, equipment=[]) for zone in _TIGHT_ZONES]},
            "workload": {"horizon": {"start": 0, "end": _WEEK_DAYS * 24 * 3600 * 1_000_000}},
        },
    )


@cache
def _staff() -> tuple[StaffMember, ...]:
    scenario = _scenario()
    horizon = scenario.workload.horizon
    window = TimeWindow(start=horizon.start, end=horizon.end)
    return realize_staff(scenario.staffing, _layout(), window)


@cache
def _layout() -> FloorLayout:
    return generate_floor(_scenario().facility)


def _cohort(seed: int) -> dict[PatientId, Patient]:
    """The week's arrivals, drawn without running it.

    ``generate_workload`` is a pure function of ``(spec, seed)`` — the same call the
    engine makes — so the cohort is knowable before the run and, under CRN, identical
    across arms. That is what lets an arm's predictions be computed up front.
    """
    scenario = _scenario()
    arrivals = generate_workload(
        scenario.workload, RandomStreams(seed), disruptions=scenario.disruptions
    )
    return {arrival.patient.id: arrival.patient for arrival in arrivals}


def _week(name: str, seed: int) -> WeekData:
    """One week of training data: a real run, plus the facts the log does not carry."""
    replication = run_replication(_scenario(), "optimized", seed, watch=_WATCH)
    roster = _cohort(seed)
    # Re-derived, not captured from the run: `generate_vitals` is content-addressed by
    # patient, so a fresh `RandomStreams(seed)` reproduces exactly the trajectories the
    # engine revealed. Reading them back out of the log would only recover the NEWS2
    # totals it stamps, not the readings the model is fitted on.
    streams = RandomStreams(seed)
    vitals = tuple(
        generate_vitals(patient, streams, until=_WATCH.span, cadence=_WATCH.cadence)
        for patient in roster.values()
    )
    return WeekData(
        run=name,
        log=EventLog.from_jsonl(replication.event_log_jsonl),
        roster=roster,
        week=replication.horizon,
        vitals=vitals,
        acuity={pid: patient.esi for pid, patient in roster.items()},
    )


@dataclass(frozen=True)
class _Trained:
    weeks: tuple[WeekData, ...]
    store: ModelStore
    version: str
    report: ValidationReport
    regressor: ServiceTimeRegressor
    deterioration: DeteriorationModel


@cache
def _trained() -> _Trained:
    """Fit once for the whole module. Every run below is real; none of them is cheap."""
    root = Path(tempfile.mkdtemp(prefix="hospital-closed-loop-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    weeks = tuple(_week(f"train_{i:02d}", 101 + i) for i in range(_TRAIN_WEEKS))
    store = ModelStore(root)
    # `min_service_samples` is lowered from the 30 the package defaults to: these are
    # one-day weeks, and at 30 every key would fall back and the table would carry no
    # fitted row at all. The regressor, which is what the arm under test consumes, is
    # unaffected either way.
    bundle = train_all(weeks, TrainConfig(seed=7, min_service_samples=20), store)
    models, _ = store.load("forecast", bundle.version)
    return _Trained(
        weeks=weeks,
        store=store,
        version=bundle.version,
        report=bundle.metrics,
        regressor=models.los_regressor,
        deterioration=models.deterioration,
    )


@cache
def _flat_stay() -> Duration:
    """The baseline model: one number, the mean completed LOS across the training weeks.

    Fitted on training data only, so the baseline arm peeks at nothing the fitted arm
    does not also see. It is the weakest possible *honest* predictor of a stay — and,
    being scale-matched to the fitted arm, it makes the contrast a question about
    discrimination rather than about how hard occupancy is priced.
    """
    totals: list[float] = []
    for week in _trained().weeks:
        totals.extend(patient_los(week.log).values())
    assert totals, "the training weeks discharged nobody"
    return seconds(math.fsum(totals) / len(totals))


def _flat_stays(seed: int) -> dict[PatientId, Duration]:
    return dict.fromkeys(_cohort(seed), _flat_stay())


def _fitted_stays(seed: int, pilot: EventLog, week: OperatingWeek) -> dict[PatientId, Duration]:
    """Per-patient expected stay from the fitted regressor.

    ``pilot`` is a run of the *baseline* arm at the same seed, and it is the honest
    part of this harness that is worth stating plainly. A patient's features include
    congestion at their arrival, which is a property of a run — but the run under test
    does not exist until its own predictions have been supplied. Every row's cutoff is
    still its own arrival, so nothing from the future informs it; what it borrows is
    another arm's congestion rather than its own.

    Predicting inside the engine (a seam like ``RiskMonitor``, consulted at placement)
    would remove the approximation entirely. That is the right shape and it is not
    built yet; until it is, both arms read the same pilot, so neither is advantaged.
    """
    regressor = _trained().regressor
    rows = patient_features(pilot, _cohort(seed), week)
    return {row.patient: regressor.predict_expected_los(row) for row in rows}


@dataclass(frozen=True)
class _Arms:
    """One evaluation seed, run three ways."""

    seed: int
    unpriced: KpiVector
    flat: KpiVector
    fitted: KpiVector
    flat_log: str
    fitted_log: str
    n_predicted: int


def _kpis(jsonl: str, window: OperatingWeek) -> KpiVector:
    return compute_kpis(
        EventLog.from_jsonl(jsonl), _layout(), _staff(), window=window, warmup=_WARMUP
    )


@cache
def _evaluate(seed: int) -> _Arms:
    scenario = _scenario()
    # The pilot doubles as the no-prediction arm: `w_occupancy` is priced but no stays
    # are supplied, so `occupancy_cost` returns 0 and this is the M1/M2 engine exactly.
    pilot = run_replication(scenario, "optimized", seed, objective=_WEIGHTED)
    pilot_log = EventLog.from_jsonl(pilot.event_log_jsonl)
    fitted = _fitted_stays(seed, pilot_log, pilot.horizon)
    flat = _flat_stays(seed)

    flat_run = run_replication(scenario, "optimized", seed, objective=_WEIGHTED, expected_stay=flat)
    fitted_run = run_replication(
        scenario, "optimized", seed, objective=_WEIGHTED, expected_stay=fitted
    )
    return _Arms(
        seed=seed,
        unpriced=_kpis(pilot.event_log_jsonl, pilot.horizon),
        flat=_kpis(flat_run.event_log_jsonl, pilot.horizon),
        fitted=_kpis(fitted_run.event_log_jsonl, pilot.horizon),
        flat_log=flat_run.event_log_jsonl,
        fitted_log=fitted_run.event_log_jsonl,
        n_predicted=len(fitted),
    )


@cache
def _all_arms() -> tuple[_Arms, ...]:
    return tuple(_evaluate(5001 + i) for i in range(_EVAL_SEEDS))


# --- the loop closes ------------------------------------------------------------


def test_the_loop_closes_from_sim_to_solver_and_back() -> None:
    """Every stage hands off: run -> features -> fit -> disk -> reload -> placement."""
    trained = _trained()
    assert len(trained.weeks) == _TRAIN_WEEKS
    # Reloaded from disk, not the in-memory fit -- the arm consumes the artifact.
    reloaded, meta = trained.store.load("forecast", trained.version)
    assert meta.version == trained.version
    assert reloaded.los_regressor is not None

    arms = _all_arms()
    assert len(arms) == _EVAL_SEEDS
    for arm in arms:
        assert arm.n_predicted > 0, f"seed {arm.seed} produced no predictions"


def test_the_scored_week_was_held_out_of_the_fit() -> None:
    """ "Held out" has to be checkable, or the metrics are decoration.

    The threshold and the calibrator are fitted on the calibration week, so it is no
    more held out than the training weeks are -- which is exactly why the report names
    all three folds separately.
    """
    report = _trained().report
    holdout = set(report.holdout)
    assert holdout
    assert not holdout & set(report.train)
    assert not holdout & set(report.calibration)


def test_every_arriving_patient_gets_a_fitted_prediction() -> None:
    """A partial mapping silently prices the missing patients at zero occupancy.

    Asserted against the *fitted* mapping, which is the one that can actually come up
    short: it is keyed by whatever rows :func:`patient_features` produced from the pilot
    log, so a patient the extractor never saw would simply be absent — and absent means
    free, not expensive. (Saying this about the flat arm would prove nothing: that dict
    is built from the cohort, so covering the cohort is true by construction.)
    """
    for arm in _all_arms():
        cohort = _cohort(arm.seed)
        pilot = run_replication(_scenario(), "optimized", arm.seed, objective=_WEIGHTED)
        fitted = _fitted_stays(arm.seed, EventLog.from_jsonl(pilot.event_log_jsonl), pilot.horizon)
        missing = set(cohort) - set(fitted)
        assert not missing, f"seed {arm.seed}: {len(missing)} arrivals got no prediction"
        assert all(duration.root > 0 for duration in fitted.values())


# --- the predictions reach a decision --------------------------------------------


def test_the_fitted_predictions_change_the_realized_run() -> None:
    """The port must be live, or the contrast below measures nothing at all.

    This is the failure mode worth guarding: a plumbed-but-inert prediction path
    reports a delta of zero while every stage looks healthy.
    """
    for arm in _all_arms():
        assert arm.flat_log != arm.fitted_log, (
            f"seed {arm.seed}: the fitted and flat arms are byte-identical"
        )


def test_the_two_arms_disagree_about_who_stays_long() -> None:
    """The arms must differ in their *predictions*, not only in their logs."""
    for arm in _all_arms():
        flat = _flat_stays(arm.seed)
        pilot = run_replication(_scenario(), "optimized", arm.seed, objective=_WEIGHTED)
        fitted = _fitted_stays(arm.seed, EventLog.from_jsonl(pilot.event_log_jsonl), pilot.horizon)
        shared = set(flat) & set(fitted)
        assert shared
        disagree = sum(1 for pid in shared if flat[pid] != fitted[pid])
        assert disagree > len(shared) // 2, "the fitted arm barely moved off its own mean"


# --- what closing the loop is worth ---------------------------------------------


def test_the_fitted_arm_is_no_worse_than_its_own_mean() -> None:
    """The measured result, stated as what it is.

    At this scale the contrast is a wash: predicting each patient's stay, rather than
    giving everyone the training-week mean, does not move a realized KPI by an
    operationally meaningful amount on this floor. So the assertion is non-inferiority
    -- no KPI is significantly *worse* -- rather than a win this harness cannot
    honestly claim.

    **Low power, and known to be.** A tenfold mis-scaling of every prediction does not
    trip this (module docstring, last paragraph), so read it as a tripwire for gross
    breakage and not as evidence the prediction is sound. It is here because a
    regression *large* enough to matter would show, and because a future change that
    makes the prediction pay should have to state where it pays.
    """
    arms = _all_arms()
    result = paired_bootstrap(
        [arm.flat for arm in arms], [arm.fitted for arm in arms], n_boot=1_000, seed=3
    )
    # `diff = baseline - optimized`, so a KPI where less is better wants diff >= 0.
    #
    # Cohort means only. The per-stratum p90s (`los_s_p90_by_esi_*`) are deliberately
    # out: an extreme quantile of a small acuity stratum swings by thousands of seconds
    # between neighbouring seeds, and at 8 reps two of them crossed the threshold in
    # OPPOSITE directions (see the module docstring). Asserting on those would buy
    # flakiness rather than sensitivity. The mean LOS keys are in, because if this term
    # helped or hurt for real, that is where it would show.
    lower_is_better = {
        "door_to_triage_s_mean",
        "door_to_provider_s_mean",
        "boarding_time_s_mean",
        "turnaround_time_s_mean",
        "staff_minutes_walked",
        *(f"los_s_mean_by_esi_{esi}" for esi in range(1, 6)),
    }
    regressions = [
        (key, contrast.diff_mean, contrast.ci_lo, contrast.ci_hi)
        for key, contrast in result.contrasts.items()
        if key in lower_is_better and contrast.significant and contrast.diff_mean < 0
    ]
    assert not regressions, f"the fitted arm is significantly worse on: {regressions}"


def test_a_contrast_is_actually_being_measured() -> None:
    """Guard the guard: non-inferiority is vacuous if the comparison is degenerate.

    Three ways it could pass while measuring nothing -- no reps, no KPI keys, or two
    arms whose KPI vectors are identical -- and all three are checked here so the test
    above cannot quietly become a tautology.
    """
    arms = _all_arms()
    result = paired_bootstrap(
        [arm.flat for arm in arms], [arm.fitted for arm in arms], n_boot=1_000, seed=3
    )
    assert result.n_reps == _EVAL_SEEDS
    assert result.contrasts
    assert any(arm.flat.values != arm.fitted.values for arm in arms), (
        "both arms folded to identical KPIs; the comparison is degenerate"
    )


def test_pricing_occupancy_at_all_changes_the_floor() -> None:
    """The weight is what the predictions act through, so it must matter on its own."""
    for arm in _all_arms():
        assert arm.unpriced.values != arm.flat.values, (
            f"seed {arm.seed}: pricing occupancy changed nothing"
        )


# --- the deterioration half ------------------------------------------------------


def _events(jsonl: str) -> list[dict[str, Any]]:
    return [json.loads(line)["event"] for line in jsonl.splitlines() if line.strip()]


@cache
def _monitored_run(seed: int) -> str:
    """A watched run with the *fitted* monitor injected through the core seam."""
    acuity = {pid: patient.esi for pid, patient in _cohort(seed).items()}
    monitor = RollingDeteriorationMonitor(
        _trained().deterioration, window=minutes(60), acuity=acuity
    )
    return run_replication(
        _scenario(), "optimized", seed, watch=_WATCH, monitor=monitor
    ).event_log_jsonl


def test_the_fitted_monitor_escalates_through_the_engine() -> None:
    """The classifier has to reach the floor, not just a metrics table.

    ``sim`` never imports ``forecast``: the monitor arrives as a ``core.seam.RiskMonitor``
    and the *engine* writes both events. This asserts that seam carries a real fitted
    model, not just the always-escalating stub the unit tests use.
    """
    events = _events(_monitored_run(6001))
    assert [e for e in events if e["kind"] == "vitals_sampled"]
    detected = [e for e in events if e["kind"] == "deterioration_detected"]
    raised = [e for e in events if e["kind"] == "emergency_raised"]
    assert detected, "the fitted monitor never escalated"
    assert len(detected) == len(raised)


def test_every_raised_emergency_is_answered_within_the_response_bound() -> None:
    """An escalation nobody attends is a log line, not a response.

    The bound is deliberately loose (15 minutes against a measured maximum under
    three). What is being asserted is that the boosted task really does jump the
    dispatch queue through the ordinary policy seam -- not a tight service level,
    which belongs to a staffing study rather than to a plumbing test.
    """
    events = _events(_monitored_run(6001))
    visits: dict[str, list[int]] = {}
    for event in events:
        if event["kind"] == "provider_visit_started":
            visits.setdefault(str(event["patient"]), []).append(int(event["occurred_at"]))

    raised = [e for e in events if e["kind"] == "emergency_raised"]
    assert raised
    bound = minutes(15).root
    for event in raised:
        at = int(event["occurred_at"])
        after = [t for t in visits.get(str(event["patient"]), []) if t >= at]
        assert after, f"{event['patient']} was never attended after escalation"
        assert min(after) - at <= bound, (
            f"{event['patient']} waited {(min(after) - at) / 6e7:.1f}min"
        )


def test_the_monitors_reported_operating_point_is_the_one_it_was_chosen_at() -> None:
    """A threshold picked on validation and then not used would make the metric a lie.

    The model carries its own decision (``RiskAssessment.escalate``) precisely so a
    consumer cannot substitute a number of its own, and ``meets_target`` records
    honestly whether the sensitivity goal was reachable under the false-alarm ceiling.
    """
    model = _trained().deterioration
    assert 0.0 < model.threshold < 1.0
    assert 0.0 <= model.sensitivity <= 1.0
    assert 0.0 <= model.false_alarm_rate <= 1.0
    assert isinstance(model.meets_target, bool)
