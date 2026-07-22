"""discharge: rank by unblock value; documentation gated by floor load."""

from __future__ import annotations

from _solver_fixtures import (
    bay_state,
    decision_input,
    default_config,
    demo_compiled,
    make_patient,
    task,
    waiting,
)

from hospital.core import BayStatus, DecisionInput, EsiAcuity, PlanItem, StaffRole
from hospital.solver.discharge import FloorLoad, prioritize_discharge
from hospital.solver.oracle import GraphRoutingOracle

CONFIG = default_config()


def _oracle(di: DecisionInput) -> GraphRoutingOracle:
    return GraphRoutingOracle(di.layout.graph)


def _scenario(load: FloorLoad, with_doc: bool) -> tuple[PlanItem, ...]:
    # bay-1 (general) and bay-3 (resus) occupied by discharge-ready patients.
    bays = (
        bay_state("bay-1", BayStatus.OCCUPIED, occupant="occ-gen"),
        bay_state("bay-3", BayStatus.OCCUPIED, occupant="occ-resus"),
        bay_state("bay-2"),
        bay_state("bay-4"),
    )
    # Waiting demand: an ESI-1 (needs resus, unblocked by freeing bay-3) and an
    # ESI-3 (needs general, unblocked by freeing bay-1). Resus demand ranks higher.
    waiting_patients = (
        waiting(make_patient("w1", EsiAcuity.ESI1), 60),
        waiting(make_patient("w3", EsiAcuity.ESI3), 60),
    )
    tasks = [
        task("d-gen", "discharge", at="b1", role=StaffRole.PHYSICIAN, patient="occ-gen"),
        task("d-resus", "discharge", at="b3", role=StaffRole.PHYSICIAN, patient="occ-resus"),
    ]
    if with_doc:
        tasks.append(
            task("doc1", "documentation", at="gstat", role=StaffRole.PHYSICIAN, patient="occ-gen")
        )
    di = decision_input(waiting_patients=waiting_patients, bays=bays, tasks=tuple(tasks))
    return prioritize_discharge(di, _oracle(di), config=CONFIG, load=load, rules=demo_compiled())


def test_higher_unblock_value_ranked_first() -> None:
    items = _scenario(FloorLoad(), with_doc=False)
    discharges = {
        i.patient.root: i.priority
        for i in items
        if i.patient is not None and i.priority is not None
    }
    assert discharges["occ-resus"] < discharges["occ-gen"]  # lower rank = sooner


def test_all_items_are_discharge_kind_with_patient() -> None:
    items = _scenario(FloorLoad(provider_utilization=0.1), with_doc=True)
    assert all(i.kind == "discharge" for i in items)
    assert all(i.patient is not None for i in items)


def test_documentation_demoted_at_peak_promoted_off_peak() -> None:
    peak = _scenario(FloorLoad(provider_utilization=0.95), with_doc=True)
    off = _scenario(FloorLoad(provider_utilization=0.10), with_doc=True)
    doc_peak = next(i.priority for i in peak if i.stable_id == "documentation:doc1")
    doc_off = next(i.priority for i in off if i.stable_id == "documentation:doc1")
    assert doc_peak is not None and doc_off is not None
    assert doc_peak > doc_off  # peak load pushes documentation later
    # A demoted doc sits behind every discharge; a promoted one is far nearer.
    max_discharge_rank = max(
        i.priority for i in off if i.priority is not None and i.stable_id != "documentation:doc1"
    )
    assert doc_off > max_discharge_rank


def test_floorload_is_peak_threshold() -> None:
    assert FloorLoad(provider_utilization=0.9).is_peak()
    assert not FloorLoad(provider_utilization=0.5).is_peak()
    assert FloorLoad(nurse_utilization=0.95).is_peak()


def test_documentation_retains_task_identity() -> None:
    # Regression (review finding 3): documentation was emitted as
    # stable_id "discharge:<id>" with the task id dropped — paperwork was
    # indistinguishable from an actual discharge. PlanItemKind has no
    # documentation kind (core is off-limits here), so the discriminator is
    # the stable_id prefix plus the carried task id, which resolves to the
    # TaskSpec (kind="documentation") in DecisionInput.pending_tasks.
    items = _scenario(FloorLoad(provider_utilization=0.1), with_doc=True)
    by_id = {i.stable_id: i for i in items}
    doc = by_id["documentation:doc1"]
    assert doc.kind == "discharge"  # the only kind the core contract offers
    assert doc.task is not None and doc.task.root == "doc1"
    real = by_id["discharge:d-gen"]
    assert real.task is not None and real.task.root == "d-gen"
    assert "discharge:doc1" not in by_id  # paperwork no longer masquerades


def test_unblock_value_ignores_post_placement_waiters() -> None:
    # Regression (review finding 9): unblock demand summed ALL waiting
    # patients. An ESI-1 waiting for RESULTS (already placed) must not inflate
    # the resus discharge; only the general bay has genuine bay demand (the
    # needs_bay ESI-3), so occ-gen outranks occ-resus.
    bays = (
        bay_state("bay-1", BayStatus.OCCUPIED, occupant="occ-gen"),
        bay_state("bay-3", BayStatus.OCCUPIED, occupant="occ-resus"),
    )
    waiting_patients = (
        waiting(make_patient("w1", EsiAcuity.ESI1), 60, stage="awaiting_results"),
        waiting(make_patient("w3", EsiAcuity.ESI3), 60),  # needs_bay (default stage)
    )
    tasks = (
        task("d-gen", "discharge", at="b1", role=StaffRole.PHYSICIAN, patient="occ-gen"),
        task("d-resus", "discharge", at="b3", role=StaffRole.PHYSICIAN, patient="occ-resus"),
    )
    di = decision_input(waiting_patients=waiting_patients, bays=bays, tasks=tasks)
    items = prioritize_discharge(
        di, _oracle(di), config=CONFIG, load=FloorLoad(), rules=demo_compiled()
    )
    ranks = {
        i.patient.root: i.priority
        for i in items
        if i.patient is not None and i.priority is not None
    }
    assert ranks["occ-gen"] < ranks["occ-resus"]
