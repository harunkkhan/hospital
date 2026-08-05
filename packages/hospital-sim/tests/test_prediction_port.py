"""The prediction port through the engine: inert unless BOTH switches are on.

`run_replication` grew an `expected_stay` argument and `ObjectiveConfig` grew a
`w_occupancy` weight. Either alone must leave the realized run byte-identical to the
M1/M2 engine — that is what keeps every golden a check rather than a re-baseline —
and only both together may change a decision.

The last of those is the one that matters for the M3 acceptance test: a plumbed but
inert port would make a static-vs-ML comparison report a delta of zero while looking
perfectly healthy.
"""

from __future__ import annotations

import json

from _sim_fixtures import tiny_scenario

from hospital.core import Duration, PatientId, hours, minutes
from hospital.sim import run_replication
from hospital.sim.experiment.replication import DEFAULT_OBJECTIVE

_SCENARIO = tiny_scenario(horizon_hours=12, rate_per_hour=3.0)
_SEED = 9
_BASELINE_LOG = run_replication(_SCENARIO, "optimized", _SEED).event_log_jsonl
_WEIGHTED = DEFAULT_OBJECTIVE.model_copy(update={"w_occupancy": 2})


def _stays(log: str, per_patient: Duration) -> dict[PatientId, Duration]:
    """A flat predicted stay for everyone who arrived — enough to move a decision."""
    return {
        PatientId(json.loads(line)["event"]["patient"]): per_patient
        for line in log.splitlines()
        if json.loads(line)["event"]["kind"] == "patient_arrived"
    }


def test_predictions_without_a_weight_change_nothing() -> None:
    """Supplying a prediction to an arm that prices it at zero must cost nothing."""
    log = run_replication(
        _SCENARIO, "optimized", _SEED, expected_stay=_stays(_BASELINE_LOG, hours(4))
    ).event_log_jsonl
    assert log == _BASELINE_LOG


def test_a_weight_without_predictions_changes_nothing() -> None:
    """The other switch, independently: no prediction means no occupancy term."""
    log = run_replication(_SCENARIO, "optimized", _SEED, objective=_WEIGHTED).event_log_jsonl
    assert log == _BASELINE_LOG


def test_a_weight_and_predictions_together_change_the_run() -> None:
    """The decisive check — an inert port would silently make the A/B measure nothing."""
    log = run_replication(
        _SCENARIO,
        "optimized",
        _SEED,
        objective=_WEIGHTED,
        expected_stay=_stays(_BASELINE_LOG, hours(4)),
    ).event_log_jsonl
    assert log != _BASELINE_LOG, "the occupancy term reached no decision"


def test_the_baseline_arm_ignores_predictions_by_design() -> None:
    """That asymmetry IS the A/B: same realized week, only one arm is told the stay.

    If the baseline consumed them too, a static-vs-ML comparison would be comparing
    two ML arms and the measured delta would be meaningless.
    """
    plain = run_replication(_SCENARIO, "baseline", _SEED).event_log_jsonl
    fed = run_replication(
        _SCENARIO,
        "baseline",
        _SEED,
        objective=_WEIGHTED,
        expected_stay=_stays(plain, hours(4)),
    ).event_log_jsonl
    assert fed == plain


def test_the_term_reads_the_predicted_values_not_just_their_presence() -> None:
    """A different predicted *magnitude* must produce a different run.

    This is the test that caught a real calibration bug. Priced in raw seconds the
    occupancy term reached ~500 000 for a six-hour stay against a travel spread of
    ~240-1200, so it did not bias the choice, it dictated it — and once it swamps
    travel, ``argmin`` depends only on which zone is scarcest. Every realistic stay
    then picks the same bay: the port was live yet completely insensitive to the
    prediction, which is exactly how a static-vs-ML comparison reports a delta of
    zero while looking perfectly healthy. See `occupancy_cost` for the fix.
    """
    stays = _stays(_BASELINE_LOG, hours(6))
    brief = dict.fromkeys(stays, minutes(30))

    long_log = run_replication(
        _SCENARIO, "optimized", _SEED, objective=_WEIGHTED, expected_stay=stays
    ).event_log_jsonl
    brief_log = run_replication(
        _SCENARIO, "optimized", _SEED, objective=_WEIGHTED, expected_stay=brief
    ).event_log_jsonl
    assert long_log != brief_log, "the predicted magnitude reached no decision"


def test_a_per_patient_profile_differs_from_a_uniform_one() -> None:
    """Predictions that vary *between* patients must place them differently.

    A flat stay is what a degenerate model returns; the point of fitting one is that
    it separates the long stays from the short ones.
    """
    # PatientId is a RootModel and not orderable; sort on the wrapped string so the
    # split is deterministic.
    arrivals = sorted(_stays(_BASELINE_LOG, hours(6)), key=lambda pid: pid.root)
    varied = {
        pid: (hours(6) if index % 2 == 0 else minutes(30)) for index, pid in enumerate(arrivals)
    }
    uniform = dict.fromkeys(arrivals, hours(6))

    varied_log = run_replication(
        _SCENARIO, "optimized", _SEED, objective=_WEIGHTED, expected_stay=varied
    ).event_log_jsonl
    uniform_log = run_replication(
        _SCENARIO, "optimized", _SEED, objective=_WEIGHTED, expected_stay=uniform
    ).event_log_jsonl
    assert varied_log != uniform_log, "per-patient stay values reached no decision"


def test_the_port_is_deterministic() -> None:
    """Same seed, same predictions, same weight -> byte-identical run."""
    stays = _stays(_BASELINE_LOG, hours(4))
    a = run_replication(
        _SCENARIO, "optimized", _SEED, objective=_WEIGHTED, expected_stay=stays
    ).event_log_jsonl
    b = run_replication(
        _SCENARIO, "optimized", _SEED, objective=_WEIGHTED, expected_stay=stays
    ).event_log_jsonl
    assert a == b
