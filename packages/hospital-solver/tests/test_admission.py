"""Hospital-wide placement: which ward bed an admitted patient gets (M4 §3).

Through M4 §2 the answer was "the nearest free one", chosen by ``sim.flow.ward`` on a
poll — so the scarcest resource in a hospital-wide model was the one thing no policy
decided and no operator could override. Here it is an ordinary placement through the
seam, and these tests pin the three things that has to mean: the two care phases
cannot cross, ward preference outranks distance, and the greedy backend prices a bed
exactly as the exact one does.
"""

from __future__ import annotations

from _solver_fixtures import (
    all_free_ward_bays,
    bay_state,
    decision_input,
    default_config,
    make_patient,
    waiting,
    ward_compiled,
    ward_layout,
)

from hospital.core import (
    AWAITING_ADMISSION,
    WAITING_FOR_BAY,
    Bay,
    BayId,
    BayStatus,
    DecisionInput,
    EsiAcuity,
    PatientId,
    Plan,
)
from hospital.solver.heuristic import HeuristicPlacement
from hospital.solver.objective import AssignmentCoeffs, assignment_coeffs
from hospital.solver.oracle import GraphRoutingOracle
from hospital.solver.placement import (
    CpSatPlacement,
    admission_travel,
    admission_weight,
    admitted_patients,
    candidates,
    compat_pair,
    placement_weights,
    ward_rank,
    ward_rank_cost,
)
from hospital.solver.turnaround import unblock_value


def _oracle() -> GraphRoutingOracle:
    return GraphRoutingOracle(ward_layout().graph)


def _coeffs(esi: EsiAcuity = EsiAcuity.ESI3) -> AssignmentCoeffs:
    return assignment_coeffs(default_config(), esi)


def _bay(bay_id: str) -> Bay:
    return next(b for b in ward_layout().bays if b.id == BayId(bay_id))


def _di(
    *entries: tuple[str, EsiAcuity, str], occupied: tuple[tuple[str, str], ...] = ()
) -> DecisionInput:
    """A decision input over the ward layout: who is waiting, and who holds which bay."""
    holder = {BayId(bay): PatientId(who) for who, bay in occupied}
    bays = tuple(
        bay_state(state.bay.root, BayStatus.OCCUPIED, holder[state.bay].root)
        if state.bay in holder
        else state
        for state in all_free_ward_bays()
    )
    return decision_input(
        waiting_patients=tuple(
            waiting(make_patient(pid, esi=esi), 0.0, stage=stage) for pid, esi, stage in entries
        ),
        bays=bays,
        layout=ward_layout(),
    )


def _assignments(plan: Plan) -> dict[str, str]:
    return {
        item.patient.root: item.bay.root
        for item in plan.items
        if item.kind == "assign_bay" and item.patient is not None and item.bay is not None
    }


def test_a_patient_being_worked_up_gets_no_variable_for_a_ward_bed() -> None:
    """The phases cannot cross, and not merely because a ward bed scores badly.

    Expressibility, not cost: the pair has no decision variable at all, so no weight
    tuning and no degenerate objective can put a just-triaged patient in an ICU bed.
    """
    rules = ward_compiled()
    icu = _bay("icu-1")
    general = _bay("bay-1")
    ed_patient = make_patient("p-ed", esi=EsiAcuity.ESI2)

    assert compat_pair(ed_patient, general, rules, stage=WAITING_FOR_BAY)
    assert not compat_pair(ed_patient, icu, rules, stage=WAITING_FOR_BAY)
    # ...and symmetrically: an admitted patient is not sent back to a general bay.
    assert compat_pair(ed_patient, icu, rules, stage=AWAITING_ADMISSION)
    assert not compat_pair(ed_patient, general, rules, stage=AWAITING_ADMISSION)

    di = _di(
        ("p-ed", EsiAcuity.ESI2, WAITING_FOR_BAY),
        ("p-adm", EsiAcuity.ESI2, AWAITING_ADMISSION),
    )
    _patients, _bays, compat = candidates(di, rules)
    assert (PatientId("p-ed"), BayId("icu-1")) not in compat
    assert (PatientId("p-adm"), BayId("bay-1")) not in compat
    assert (PatientId("p-adm"), BayId("icu-1")) in compat


def test_both_phases_are_placed_in_one_solve() -> None:
    """One queue, one model: the ED placement and the admission are decided together."""
    di = _di(
        ("p-ed", EsiAcuity.ESI3, WAITING_FOR_BAY),
        ("p-adm", EsiAcuity.ESI1, AWAITING_ADMISSION),
    )
    result = CpSatPlacement().solve(di, _oracle(), config=default_config(), rules=ward_compiled())
    placed = _assignments(result.plan)
    assert placed["p-ed"] in {"bay-1", "bay-2"}
    assert placed["p-adm"] in {"icu-1", "icu-2"}


def test_ward_preference_outranks_a_nearer_bed() -> None:
    """An ESI-3 walks past two closer ICU beds to reach the med-surg one.

    The fixture is built so distance and preference disagree: ``icu-1`` is 20m from the
    hub and ``ms-1`` is 200m, but med-surg is where an ESI-3 belongs. Under the greedy
    claim this went the other way, which is precisely how first-available fills an ICU
    with patients who do not need one.
    """
    di = _di(("p-adm", EsiAcuity.ESI3, AWAITING_ADMISSION))
    oracle = _oracle()
    for backend in (CpSatPlacement(), HeuristicPlacement()):
        result = backend.solve(di, oracle, config=default_config(), rules=ward_compiled())
        assert _assignments(result.plan)["p-adm"] == "ms-1", backend.name


def test_a_scarce_ward_is_kept_for_the_acuity_that_needs_it() -> None:
    """With one bed of each kind free, the ICU goes to the ESI-1, not the ESI-3.

    The claim M4 §3 exists for. First-available would hand ``icu-1`` to whoever it
    scanned first; here the ESI-3's rank-2 penalty on the ICU is worth more than any
    distance saving, so the beds sort themselves by who needs them.
    """
    di = _di(
        ("p-low", EsiAcuity.ESI3, AWAITING_ADMISSION),
        ("p-crit", EsiAcuity.ESI1, AWAITING_ADMISSION),
        occupied=(("someone", "icu-2"),),
    )
    result = CpSatPlacement().solve(di, _oracle(), config=default_config(), rules=ward_compiled())
    placed = _assignments(result.plan)
    assert placed["p-crit"] == "icu-1"
    assert placed["p-low"] == "ms-1"


def test_an_esi3_still_takes_an_icu_bed_when_it_is_the_only_one() -> None:
    """Preference is a bias, never a refusal — place-first still dominates.

    The counterpart to the test above, and the reason the ward table is a heuristic on
    the solver's side of the seam rather than an ``AdmissionRule``: a hospital would
    rather have the patient in the wrong bed than in a corridor.
    """
    di = _di(("p-low", EsiAcuity.ESI3, AWAITING_ADMISSION), occupied=(("someone", "ms-1"),))
    result = CpSatPlacement().solve(di, _oracle(), config=default_config(), rules=ward_compiled())
    assert _assignments(result.plan)["p-low"] in {"icu-1", "icu-2"}


def test_the_escort_is_priced_from_the_bay_the_patient_is_boarding_in() -> None:
    """The trip that gets costed is the one the decision causes.

    A boarding patient holds an ED bay, so ``BayState.occupant`` locates them and the
    escort leg runs from there — no new seam field, and no pretending every admission
    starts at the entrance.
    """
    di = _di(("p-adm", EsiAcuity.ESI2, AWAITING_ADMISSION), occupied=(("p-adm", "bay-3"),))
    oracle = _oracle()
    patient = make_patient("p-adm", esi=EsiAcuity.ESI2)
    near = _bay("icu-1")
    far = _bay("ms-1")
    origin = _bay("bay-3").node

    assert admission_travel(patient, near, origin, oracle, _coeffs()) > 0
    assert admission_travel(patient, far, origin, oracle, _coeffs()) > admission_travel(
        patient, near, origin, oracle, _coeffs()
    )
    # And it reaches the model: the weight table prices the two beds apart.
    patients, bays, compat = candidates(di, ward_compiled())
    weights = placement_weights(di, patients, bays, compat, oracle, default_config())
    assert (
        weights[(PatientId("p-adm"), BayId("ms-1"))]
        != weights[(PatientId("p-adm"), BayId("icu-1"))]
    )


def test_rank_cost_strictly_dominates_every_escort_in_the_instance() -> None:
    """The lexicographic guarantee is derived, not tuned — so it holds on any floor."""
    assert ward_rank_cost(()) == 1
    assert ward_rank_cost((10, 900, 42)) == 901

    patient = make_patient("p", esi=EsiAcuity.ESI3)
    icu = _bay("icu-1")
    ms = _bay("ms-1")
    assert ward_rank(patient, ms) < ward_rank(patient, icu)

    oracle, coeffs = _oracle(), _coeffs()
    origin = _bay("bay-3").node
    cost = ward_rank_cost(admission_travel(patient, b, origin, oracle, coeffs) for b in (icu, ms))
    # Even though med-surg is much further, its better rank wins.
    assert admission_weight(patient, ms, origin, oracle, coeffs, rank_cost=cost) < admission_weight(
        patient, icu, origin, oracle, coeffs, rank_cost=cost
    )


def test_both_backends_price_an_admission_identically() -> None:
    """One weight table, two searches — the fallback must not be a second objective."""
    di = _di(
        ("p-adm", EsiAcuity.ESI2, AWAITING_ADMISSION),
        ("p-ed", EsiAcuity.ESI3, WAITING_FOR_BAY),
        occupied=(("p-adm", "bay-3"),),
    )
    oracle = _oracle()
    patients, bays, compat = candidates(di, ward_compiled())
    weights = placement_weights(di, patients, bays, compat, oracle, default_config())

    exact = CpSatPlacement().solve(di, oracle, config=default_config(), rules=ward_compiled())
    greedy = HeuristicPlacement().solve(di, oracle, config=default_config(), rules=ward_compiled())
    for plan in (exact.plan, greedy.plan):
        for pid, bid in _assignments(plan).items():
            assert (PatientId(pid), BayId(bid)) in weights


def test_admitted_patients_reads_the_queue_not_the_ward() -> None:
    """The solver's set is the queued restriction of the validator's whole-phase one."""
    di = _di(
        ("p-adm", EsiAcuity.ESI2, AWAITING_ADMISSION),
        ("p-ed", EsiAcuity.ESI3, WAITING_FOR_BAY),
        occupied=(("p-housed", "icu-2"),),
    )
    assert admitted_patients(di) == frozenset({PatientId("p-adm")})


def test_a_dirty_ward_bed_is_valued_by_the_admits_boarding_for_it() -> None:
    """Cleaning a ward bed unblocks two bays, and the lever has to be able to see it.

    Scored against the ED queue alone, a dirty ICU bed values at zero no matter how
    many admits are stacked up for it — so housekeeping deprioritizes the one clean
    that would free a ward bed *and* the ED bay its occupant is holding. The demand
    quantity is shared by turnaround, discharge, and dispatch's urgency, so this is
    one fix in one place for all three.
    """
    boarding = (
        waiting(make_patient("p-adm", esi=EsiAcuity.ESI1), 0.0, stage=AWAITING_ADMISSION),
        waiting(make_patient("p-adm2", esi=EsiAcuity.ESI2), 0.0, stage=AWAITING_ADMISSION),
    )
    ed_queue = (waiting(make_patient("p-ed", esi=EsiAcuity.ESI3), 0.0, stage=WAITING_FOR_BAY),)
    config, rules = default_config(), ward_compiled()

    icu, general = _bay("icu-1"), _bay("bay-1")
    assert unblock_value(icu, boarding, config=config, rules=rules) > 0
    # ...and the phases do not leak into each other's demand.
    assert unblock_value(icu, ed_queue, config=config, rules=rules) == 0
    assert unblock_value(general, boarding, config=config, rules=rules) == 0
    assert unblock_value(general, ed_queue, config=config, rules=rules) > 0
