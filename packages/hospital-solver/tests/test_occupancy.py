"""The prediction port: it must be inert at weight 0 and decisive above it.

Two claims, and both matter for the M3 acceptance test:

* at ``w_occupancy=0`` the term vanishes and every existing weight is unchanged, so
  turning prediction on is opt-in and the M1 goldens stay a check rather than a
  re-baseline;
* above 0 it actually **changes which bay is chosen**. An inert port would make a
  static-vs-ML comparison measure nothing at all while appearing to run fine — which
  is the failure mode this file exists to rule out.
"""

from __future__ import annotations

from _solver_fixtures import decision_input, default_config, make_patient, waiting

from hospital.core import (
    Bay,
    BayId,
    Duration,
    EsiAcuity,
    NodeId,
    ZoneId,
    ZoneType,
    hours,
    minutes,
)
from hospital.solver.objective import ObjectiveConfig, assignment_coeffs
from hospital.solver.oracle import GraphRoutingOracle
from hospital.solver.placement import (
    occupancy_cost,
    residual_stays,
    travel_weight,
    zone_scarcity,
)

_SCARCE = ZoneId("scarce")
_ROOMY = ZoneId("roomy")


def _bay(name: str, zone: ZoneId) -> Bay:
    return Bay(
        id=BayId(name),
        zone=zone,
        zone_type=ZoneType.GENERAL,
        node=NodeId(f"n_{name}"),
        serving_station=NodeId("station"),
        isolation_capable=False,
        equipment=frozenset(),
    )


def test_scarcity_falls_as_a_zone_empties_and_never_rewards_filling() -> None:
    """Monotone in free bays, and never below 1 — occupancy is always a cost."""
    values = [zone_scarcity(free) for free in range(0, 15)]
    assert values == sorted(values, reverse=True), values
    assert min(values) >= 1
    assert zone_scarcity(0) > zone_scarcity(5) > zone_scarcity(20)


def test_the_term_is_exactly_zero_without_a_weight_or_without_a_prediction() -> None:
    """Both switches must independently make the port inert.

    This is what keeps `run_replication(predictions=None)` byte-identical to the
    M1/M2 engine: an arm that supplies neither a weight nor a stay pays nothing.
    """
    coeffs_off = assignment_coeffs(ObjectiveConfig(), EsiAcuity.ESI3)
    coeffs_on = assignment_coeffs(ObjectiveConfig(w_occupancy=3), EsiAcuity.ESI3)
    bay = _bay("b", _SCARCE)
    scarcity = {_SCARCE: zone_scarcity(0)}

    assert coeffs_off.occupancy_weight == 0
    assert occupancy_cost(bay, coeffs_off, hours(4), scarcity) == 0, "weight 0 -> no cost"
    assert occupancy_cost(bay, coeffs_on, None, scarcity) == 0, "no prediction -> no cost"
    assert occupancy_cost(bay, coeffs_on, hours(4), scarcity) > 0, "both present -> a real cost"


def test_a_longer_predicted_stay_costs_more_in_the_same_bay() -> None:
    coeffs = assignment_coeffs(ObjectiveConfig(w_occupancy=1), EsiAcuity.ESI3)
    bay = _bay("b", _SCARCE)
    scarcity = {_SCARCE: zone_scarcity(1)}
    short = occupancy_cost(bay, coeffs, minutes(40), scarcity)
    long = occupancy_cost(bay, coeffs, hours(4), scarcity)
    assert long > short


def test_the_same_stay_costs_more_in_a_scarcer_zone() -> None:
    """This is the interaction that makes the prediction usable at all.

    Travel is a static property of a bay, so a per-patient constant added to `w[p,b]`
    would cancel across bays and change nothing. The prediction only bites because it
    is multiplied by how scarce the bay's zone is.
    """
    coeffs = assignment_coeffs(ObjectiveConfig(w_occupancy=1), EsiAcuity.ESI3)
    scarcity = {_SCARCE: zone_scarcity(0), _ROOMY: zone_scarcity(12)}
    stay = hours(4)
    in_scarce = occupancy_cost(_bay("s", _SCARCE), coeffs, stay, scarcity)
    in_roomy = occupancy_cost(_bay("r", _ROOMY), coeffs, stay, scarcity)
    assert in_scarce > in_roomy


def test_the_term_changes_which_bay_wins() -> None:
    """The decisive claim: with two otherwise-equal bays, the long stay avoids scarcity.

    If this failed, the port would be plumbed but inert — and a static-vs-ML
    comparison built on it would report a delta of zero while looking healthy.
    """
    coeffs = assignment_coeffs(ObjectiveConfig(w_occupancy=1), EsiAcuity.ESI3)
    scarcity = {_SCARCE: zone_scarcity(0), _ROOMY: zone_scarcity(12)}
    scarce_bay, roomy_bay = _bay("s", _SCARCE), _bay("r", _ROOMY)

    def cheaper(stay: object) -> BayId:
        # Travel is identical by construction here (same serving station, and the
        # caller of `occupancy_cost` adds travel separately), so occupancy decides.
        costs = {
            b.id: occupancy_cost(b, coeffs, stay, scarcity)  # type: ignore[arg-type]
            for b in (scarce_bay, roomy_bay)
        }
        return min(costs, key=lambda k: (costs[k], k.root))

    assert cheaper(hours(4)) == roomy_bay.id, "a long stay must avoid the scarce zone"
    # With no prediction the term is 0 for both and the tie breaks on id alone --
    # i.e. exactly the pre-prediction behaviour.
    assert cheaper(None) == min((scarce_bay.id, roomy_bay.id), key=lambda b: b.root)


def test_acuity_scales_the_occupancy_cost_like_every_other_term() -> None:
    """`u(esi)` multiplies occupancy too, so the term speaks the objective's currency."""
    config = ObjectiveConfig(w_occupancy=1)
    bay = _bay("b", _SCARCE)
    scarcity = {_SCARCE: zone_scarcity(1)}
    sick = occupancy_cost(bay, assignment_coeffs(config, EsiAcuity.ESI1), hours(4), scarcity)
    well = occupancy_cost(bay, assignment_coeffs(config, EsiAcuity.ESI5), hours(4), scarcity)
    assert sick > well


def test_the_config_hash_moves_with_the_new_weight() -> None:
    """A weight that changed a decision but not the hash would be untraceable."""
    from hospital.solver.objective import config_hash

    assert config_hash(ObjectiveConfig()) != config_hash(ObjectiveConfig(w_occupancy=1))


def test_an_unknown_zone_defaults_to_the_mildest_scarcity() -> None:
    """A bay whose zone is absent from the snapshot must not blow up or dominate."""
    coeffs = assignment_coeffs(ObjectiveConfig(w_occupancy=1), EsiAcuity.ESI3)
    bay = _bay("b", ZoneId("unseen"))
    cost = occupancy_cost(bay, coeffs, hours(4), {})
    # Stated as behaviour rather than by re-deriving the formula: an absent zone is
    # priced exactly as the mildest scarcity, and strictly below the scarcest.
    assert cost == occupancy_cost(bay, coeffs, hours(4), {bay.zone: 1})
    assert cost < occupancy_cost(bay, coeffs, hours(4), {bay.zone: 12})


def test_the_term_stays_commensurate_with_the_travel_it_competes_with() -> None:
    """Calibration, not taste: the prediction must be able to lose to a travel saving.

    ``assignment_weight`` is ``travel + occupancy``, so their *relative* scale decides
    whether the prediction biases the choice or overrides it. Priced in raw seconds the
    occupancy term hit ~500 000 against a travel spread of ~1000, which made every
    realistic predicted stay pick the same bay — live port, zero sensitivity. These
    bounds fail if that regression comes back.
    """
    di = decision_input()
    oracle = GraphRoutingOracle(di.layout.graph)
    patient = make_patient("p", esi=EsiAcuity.ESI3)
    config = default_config(w_occupancy=2)
    coeffs = assignment_coeffs(config, EsiAcuity.ESI3)
    travel = [travel_weight(patient, bay, oracle, di.layout, coeffs) for bay in di.layout.bays]
    spread = max(travel) - min(travel)
    assert spread > 0, "the fixture must price bays differently for this to mean anything"

    bay = di.layout.bays[0]
    roomy = occupancy_cost(bay, coeffs, minutes(30), {bay.zone: 2})
    scarcest = occupancy_cost(bay, coeffs, hours(6), {bay.zone: 12})
    # A short stay in a half-empty zone must be a nudge travel can overrule...
    assert roomy < spread, (roomy, spread)
    # ...and even the worst case stays within a few travel spreads, not a thousand.
    assert scarcest < 5 * spread, (scarcest, spread)


def test_a_patient_who_already_waited_is_charged_for_less_bay_time() -> None:
    """The models predict arrival-to-discharge; a bay only holds what is left of it.

    Charging the whole predicted LOS gets the incentive backwards: of two identical
    patients, the one who has waited longer has *less* bay time ahead, yet the full
    duration would price them as the more expensive to admit.
    """
    fresh = make_patient("fresh", esi=EsiAcuity.ESI3)
    waited_long = make_patient("waited", esi=EsiAcuity.ESI3)
    now = hours(3)
    di = decision_input(
        waiting_patients=(waiting(fresh, 0), waiting(waited_long, hours(2).root / 1_000_000)),
        now_us=now.root,
    )
    # Same predicted stay for both; only the elapsed time since arrival differs.
    predicted = dict.fromkeys((fresh.id, waited_long.id), hours(6))
    residual = residual_stays(di, predicted)

    # `make_patient` stamps arrival_time=0, so at now=3h both have 3h elapsed and the
    # residual is 3h -- less than the 6h predicted, which is the point.
    assert residual[fresh.id] == hours(3)
    assert residual[waited_long.id] == hours(3)

    coeffs = assignment_coeffs(ObjectiveConfig(w_occupancy=2), EsiAcuity.ESI3)
    bay = _bay("b", _SCARCE)
    scarcity = {_SCARCE: 8}
    assert occupancy_cost(bay, coeffs, residual[fresh.id], scarcity) < occupancy_cost(
        bay, coeffs, hours(6), scarcity
    )


def test_a_stay_already_exceeded_costs_nothing_rather_than_less_than_nothing() -> None:
    """A stale prediction is a zero, never a negative -- which would reward scarcity."""
    patient = make_patient("overdue", esi=EsiAcuity.ESI3)
    di = decision_input(waiting_patients=(waiting(patient, 0),), now_us=hours(9).root)
    residual = residual_stays(di, {patient.id: hours(4)})
    assert residual[patient.id] == Duration(0)

    coeffs = assignment_coeffs(ObjectiveConfig(w_occupancy=2), EsiAcuity.ESI3)
    assert occupancy_cost(_bay("b", _SCARCE), coeffs, residual[patient.id], {_SCARCE: 12}) == 0


def test_no_predictions_means_no_residuals_to_compute() -> None:
    di = decision_input(waiting_patients=(waiting(make_patient("p"), 0),))
    assert residual_stays(di, None) == {}
    assert residual_stays(di, {}) == {}
