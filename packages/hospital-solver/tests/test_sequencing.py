"""sequencing: descending score, anti-starvation overtake, deterministic tie-break."""

from __future__ import annotations

from _solver_fixtures import decision_input, default_config, make_patient, waiting

from hospital.core import EsiAcuity
from hospital.solver.objective import acuity_urgency
from hospital.solver.sequencing import priority_score, sequence


def test_score_is_acuity_plus_starvation() -> None:
    config = default_config()
    from hospital.core.time import seconds

    esi = EsiAcuity.ESI3
    score = priority_score(esi, seconds(120), config=config, starvation_rate=2)
    assert score == acuity_urgency(config, esi) + 2 * 120


def test_higher_acuity_ranked_first_when_waits_equal() -> None:
    config = default_config()
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p-low", EsiAcuity.ESI5), 10),
            waiting(make_patient("p-high", EsiAcuity.ESI1), 10),
        )
    )
    ranked = sequence(di, config=config, starvation_rate=1)
    order = [item.patient.root for item in ranked if item.patient]
    assert order[0] == "p-high"
    assert [item.priority for item in ranked] == [0, 1]


def test_anti_starvation_esi5_overtakes_esi2() -> None:
    # ESI-5 overtakes ESI-2 once w5 > w2 + (u2 - u5)/alpha (doc 03 §4.4).
    config = default_config()
    alpha = 1
    u2 = acuity_urgency(config, EsiAcuity.ESI2)
    u5 = acuity_urgency(config, EsiAcuity.ESI5)
    w2 = 30
    threshold = w2 + (u2 - u5) // alpha
    long_wait = threshold + 5
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("p2", EsiAcuity.ESI2), w2),
            waiting(make_patient("p5", EsiAcuity.ESI5), long_wait),
        )
    )
    ranked = sequence(di, config=config, starvation_rate=alpha)
    order = [item.patient.root for item in ranked if item.patient]
    assert order[0] == "p5"  # long-waiting ESI-5 now outranks the ESI-2

    # Just below the threshold, the ESI-2 still leads.
    di_short = decision_input(
        waiting_patients=(
            waiting(make_patient("p2", EsiAcuity.ESI2), w2),
            waiting(make_patient("p5", EsiAcuity.ESI5), max(0, threshold - 5)),
        )
    )
    ranked_short = sequence(di_short, config=config, starvation_rate=alpha)
    order_short = [item.patient.root for item in ranked_short if item.patient]
    assert order_short[0] == "p2"


def test_deterministic_tiebreak_by_arrival_then_id() -> None:
    config = default_config()
    di = decision_input(
        waiting_patients=(
            waiting(make_patient("pb", EsiAcuity.ESI3, arrival_us=100), 0),
            waiting(make_patient("pa", EsiAcuity.ESI3, arrival_us=50), 0),
        )
    )
    ranked = sequence(di, config=config, starvation_rate=1)
    order = [item.patient.root for item in ranked if item.patient]
    assert order == ["pa", "pb"]  # earlier arrival first
