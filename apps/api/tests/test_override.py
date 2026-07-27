"""The override protocol: one validator, no repair, atomic (doc 07 §4).

Every operator action either applies with zero violations or is rejected with
the verbatim ``core.validation.Violation`` list and a byte-identical session —
and the API's verdict is the seam adapter's verdict, because both call the one
``validate()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from _api_fixtures import (
    create_run,
    make_app,
    make_patient,
    quiet_scenario,
    session_of,
    step,
    world_fingerprint,
)
from hospital.api.overrides import (
    BlockEdgeAction,
    BumpPriorityAction,
    CloseBayAction,
    ExpediteCleanAction,
    ExpediteDischargeAction,
    OverrideRequest,
    PinRegistry,
    ReassignAction,
    RerouteAction,
    apply_override,
    compile_plan_action,
)
from hospital.api.sessions import RunSession
from hospital.core import (
    Activity,
    BayId,
    BayStatus,
    EsiAcuity,
    InfeasiblePlan,
    NodeId,
    Patient,
    PatientId,
    Plan,
    PlanItem,
    RunId,
    StaffId,
    StaffRole,
    TaskId,
    ZoneType,
    seconds,
    validate,
)
from hospital.sim.seam_adapter import apply_plan, validation_context

if TYPE_CHECKING:
    from pathlib import Path

_QUIET = {"quiet": quiet_scenario()}


def _quiet_session() -> RunSession:
    """A session built directly (no HTTP) for unit/property tests."""
    return RunSession(RunId("t-quiet"), quiet_scenario(), "baseline", 7, pins=PinRegistry())


def _enqueue(session: RunSession, pid: str, esi: EsiAcuity) -> Patient:
    """Register a patient and put them in the bay wait queue, deterministically."""
    patient = make_patient(pid, esi=esi)
    session.world.register_patient(patient)
    session.world.set_patient_position(patient.id, session.layout.entrances[0])
    session.world.request_bay(patient, stage="bay_wait")
    return patient


def _bay_of(session: RunSession, zone: ZoneType, index: int = 0) -> BayId:
    bays = [b for b in session.layout.bays if b.zone_type == zone]
    return bays[index].id


def _post_override(
    client: TestClient, run_id: str, action: dict[str, Any], *, pin: bool = True
) -> Any:
    return client.post(f"/runs/{run_id}/override", json={"action": action, "pin": pin})


def _bay_assigned_events(session: RunSession) -> list[tuple[str, str, str]]:
    return [
        (e.event.patient.root, e.event.bay.root, e.event.by)
        for e in session.log
        if e.event.kind == "bay_assigned"
    ]


# ------------------------------------------------------------- valid actions
def test_valid_reassign_applies_with_operator_provenance(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        patient = _enqueue(session, "op_p1", EsiAcuity.ESI3)
        bay = _bay_of(session, ZoneType.GENERAL)

        response = _post_override(
            client, handle["run"], {"kind": "reassign", "patient": "op_p1", "bay": bay.root}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "applied"
        assert body["applied_at"] == 0
        assert body["plan"]["items"][0]["kind"] == "assign_bay"

        # The decision reached physics through the seam, stamped operator.
        assert session.world.occupant(bay) == patient.id
        assert _bay_assigned_events(session) == [("op_p1", bay.root, "operator")]

        # ... and is visible in the stream's next (snapshot) frame.
        with client.websocket_connect(f"/runs/{handle['run']}/stream") as ws:
            frame = cast("dict[str, Any]", ws.receive_json())
        occupied = {b["bay"]: b for b in frame["bays"] if b["occupant"] is not None}
        assert occupied[bay.root]["occupant"] == "op_p1"
        assert occupied[bay.root]["status"] == "occupied"


def test_bump_priority_reorders_the_queue(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        _enqueue(session, "first", EsiAcuity.ESI2)
        _enqueue(session, "second", EsiAcuity.ESI4)
        assert [w.patient.id.root for w in session.world.waiting_for_bay()][0] == "first"

        response = _post_override(
            client, handle["run"], {"kind": "bump_priority", "patient": "second"}
        )
        assert response.status_code == 200, response.text
        assert [w.patient.id.root for w in session.world.waiting_for_bay()][0] == "second"


def test_valid_reroute_dispatches_the_named_staff(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        patient = _enqueue(session, "rr_p", EsiAcuity.ESI3)
        physician = next(m for m in session.roster if m.role == StaffRole.PHYSICIAN)
        task = session.world.add_task(
            kind="provider_visit",
            patient=patient.id,
            at=session.layout.entrances[0],
            required_role=StaffRole.PHYSICIAN,
            activity=Activity.PROVIDER_VISIT,
            duration=seconds(120),
        )

        response = _post_override(
            client,
            handle["run"],
            {"kind": "reroute", "staff": physician.id.root, "task": task.spec.id.root},
        )
        assert response.status_code == 200, response.text
        assert session.world.staff_task(physician.id) == task.spec.id


def test_expedite_clean_and_discharge_boost_their_tasks(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        world = session.world
        node = session.layout.entrances[0]
        bay_a = _bay_of(session, ZoneType.GENERAL, 0)
        bay_b = _bay_of(session, ZoneType.GENERAL, 1)
        world.add_task(
            kind="cleaning", patient=None, at=node, required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING, duration=seconds(60), bay=bay_a,
        )
        clean_b = world.add_task(
            kind="cleaning", patient=None, at=node, required_role=StaffRole.HOUSEKEEPING,
            activity=Activity.CLEANING, duration=seconds(60), bay=bay_b,
        )
        patient = _enqueue(session, "doc_p", EsiAcuity.ESI3)
        doc = world.add_task(
            kind="documentation", patient=patient.id, at=node, required_role=StaffRole.NURSE,
            activity=Activity.DOCUMENTATION, duration=seconds(60),
        )

        response = _post_override(
            client, handle["run"], {"kind": "expedite_clean", "bay": bay_b.root}
        )
        assert response.status_code == 200, response.text
        assert world.pending_tasks()[0].id == clean_b.spec.id

        response = _post_override(
            client, handle["run"], {"kind": "expedite_discharge", "patient": "doc_p"}
        )
        assert response.status_code == 200, response.text
        assert world.pending_tasks()[0].id == doc.spec.id


def test_close_free_bay_and_block_edge_apply(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        bay = _bay_of(session, ZoneType.FAST_TRACK)

        response = _post_override(client, handle["run"], {"kind": "close_bay", "bay": bay.root})
        assert response.status_code == 200, response.text
        assert session.world.bay_status(bay) is BayStatus.CLOSED

        # A closed bay now rejects placement — the context delta reached the validator.
        _enqueue(session, "late_p", EsiAcuity.ESI4)
        rejected = _post_override(
            client, handle["run"], {"kind": "reassign", "patient": "late_p", "bay": bay.root}
        )
        assert rejected.status_code == 422
        kinds = {v["kind"] for v in rejected.json()["violations"]}
        assert "bay_incompatible" in kinds

        edge = session.layout.graph.edges[0]
        response = _post_override(
            client, handle["run"], {"kind": "block_edge", "edge": [edge.a.root, edge.b.root]}
        )
        assert response.status_code == 200, response.text
        assert (edge.a, edge.b) in session.world.blocked_edges


# ---------------------------------------------------------- rejected actions
def test_reassign_into_occupied_bay_is_rejected_unchanged(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        prior = make_patient("prior", esi=EsiAcuity.ESI3)
        session.world.register_patient(prior)
        bay = _bay_of(session, ZoneType.GENERAL)
        session.world.assign_bay(bay, prior.id)
        _enqueue(session, "hopeful", EsiAcuity.ESI3)

        before = world_fingerprint(session)
        response = _post_override(
            client, handle["run"], {"kind": "reassign", "patient": "hopeful", "bay": bay.root}
        )
        assert response.status_code == 422
        violations = response.json()["violations"]
        assert violations, "a rejection must carry the concrete violations"
        assert {v["kind"] for v in violations} <= {"bay_incompatible", "double_booked"}
        assert world_fingerprint(session) == before, "rejection must be atomic"


def test_esi1_into_fast_track_is_bay_incompatible(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        _enqueue(session, "crit", EsiAcuity.ESI1)
        bay = _bay_of(session, ZoneType.FAST_TRACK)

        response = _post_override(
            client, handle["run"], {"kind": "reassign", "patient": "crit", "bay": bay.root}
        )
        assert response.status_code == 422
        assert any(v["kind"] == "bay_incompatible" for v in response.json()["violations"])


def test_skill_less_reroute_is_staff_lacks_skill(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        patient = _enqueue(session, "sk_p", EsiAcuity.ESI3)
        porter = next(m for m in session.roster if m.role == StaffRole.PORTER)
        task = session.world.add_task(
            kind="provider_visit",
            patient=patient.id,
            at=session.layout.entrances[0],
            required_role=StaffRole.PHYSICIAN,
            activity=Activity.PROVIDER_VISIT,
            duration=seconds(120),
        )

        response = _post_override(
            client,
            handle["run"],
            {"kind": "reroute", "staff": porter.id.root, "task": task.spec.id.root},
        )
        assert response.status_code == 422
        assert any(v["kind"] == "staff_lacks_skill" for v in response.json()["violations"])


def test_unknown_entities_are_rejected(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        for action in (
            {"kind": "reroute", "staff": "ghost", "task": "task_999999"},
            {"kind": "bump_priority", "patient": "ghost"},
            {"kind": "close_bay", "bay": "ghost"},
            {"kind": "block_edge", "edge": ["nowhere_a", "nowhere_b"]},
            {"kind": "expedite_clean", "bay": "ghost"},
        ):
            response = _post_override(client, handle["run"], action)
            assert response.status_code == 422, action
            assert any(
                v["kind"] == "unknown_entity" for v in response.json()["violations"]
            ), action


def test_close_bay_with_occupant_does_not_evict(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        occupant = make_patient("occ", esi=EsiAcuity.ESI3)
        session.world.register_patient(occupant)
        bay = _bay_of(session, ZoneType.GENERAL)
        session.world.assign_bay(bay, occupant.id)

        before = world_fingerprint(session)
        response = _post_override(client, handle["run"], {"kind": "close_bay", "bay": bay.root})
        assert response.status_code == 422
        violations = response.json()["violations"]
        # The standing assign_bay item is stranded by the context delta.
        assert any(
            v["kind"] == "bay_incompatible" and v["entity"] == bay.root for v in violations
        )
        assert session.world.bay_status(bay) is BayStatus.OCCUPIED
        assert world_fingerprint(session) == before


# ------------------------------------------------- one validator, two checks
def test_api_verdict_equals_seam_adapter_verdict(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        _enqueue(session, "crit", EsiAcuity.ESI1)
        bay = _bay_of(session, ZoneType.FAST_TRACK)
        action = ReassignAction(patient=PatientId("crit"), bay=bay)

        plan = compile_plan_action(action, session.world)
        assert isinstance(plan, Plan)
        ctx = validation_context(session.world, session.rules)
        direct = validate(plan, ctx)
        assert direct, "the crafted action must be infeasible"

        # The same function, called by the adapter, produces the same verdict...
        with pytest.raises(InfeasiblePlan) as excinfo:
            apply_plan(
                session.world, plan, ctx, session.executor, session.log, origin="operator"
            )
        assert excinfo.value.violations == direct

        # ... and the HTTP surface returns it verbatim.
        response = _post_override(
            client, handle["run"], {"kind": "reassign", "patient": "crit", "bay": bay.root}
        )
        assert response.status_code == 422
        assert response.json()["violations"] == [v.model_dump(mode="json") for v in direct]


# ------------------------------------------------------------------- pins
def test_pinned_reassign_survives_ticks_without_churn(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        _enqueue(session, "pin_p", EsiAcuity.ESI3)
        bay = _bay_of(session, ZoneType.GENERAL)

        response = _post_override(
            client, handle["run"], {"kind": "reassign", "patient": "pin_p", "bay": bay.root}
        )
        assert response.status_code == 200
        assert len(session.pins) == 1

        # Force decision ticks: the pinned item is re-affirmed as a no-op, never churn.
        session.world.request_decision()
        step(client, handle["run"], granularity="tick")
        session.world.request_decision()
        step(client, handle["run"], granularity="tick")
        assert _bay_assigned_events(session) == [("pin_p", bay.root, "operator")]


def test_pin_registry_merge_holds_decisions_against_a_resolve() -> None:
    session = _quiet_session()
    world = session.world
    pinned = _enqueue(session, "keep_me", EsiAcuity.ESI3)
    _enqueue(session, "other", EsiAcuity.ESI2)
    bay = _bay_of(session, ZoneType.GENERAL)

    pins = PinRegistry()
    assign_plan = compile_plan_action(ReassignAction(patient=pinned.id, bay=bay), world)
    assert isinstance(assign_plan, Plan)
    pins.record(ReassignAction(patient=pinned.id, bay=bay), assign_plan)
    pins.record(BumpPriorityAction(patient=pinned.id), Plan(items=()))

    # A re-solve that would overwrite both pinned decisions...
    solver_plan = Plan(
        items=(
            PlanItem(
                stable_id="assign:other", kind="assign_bay", patient=PatientId("other"), bay=bay
            ),
            PlanItem(
                stable_id="seq:waiting", kind="sequence", order=("other", "keep_me")
            ),
        )
    )
    merged = pins.merge(solver_plan, world)
    assert merged is not None
    by_kind = {item.kind: item for item in merged.items}
    # ... yields to the pins: the conflicting assignment is dropped, the pinned
    # item is carried, and the pinned patient leads the sequence.
    assert by_kind["assign_bay"].patient == pinned.id
    assert by_kind["sequence"].order is not None
    assert by_kind["sequence"].order[0] == "keep_me"

    # Empty registry is an exact pass-through (determinism guarantee).
    empty = PinRegistry()
    assert empty.merge(solver_plan, world) is solver_plan


def test_unpinned_override_records_nothing(tmp_path: Path) -> None:
    app = make_app(tmp_path, _QUIET)
    with TestClient(app) as client:
        handle = create_run(client, scenario_id="quiet")
        session = session_of(app, handle["run"])
        _enqueue(session, "np_p", EsiAcuity.ESI3)
        bay = _bay_of(session, ZoneType.GENERAL)
        response = _post_override(
            client,
            handle["run"],
            {"kind": "reassign", "patient": "np_p", "bay": bay.root},
            pin=False,
        )
        assert response.status_code == 200
        assert len(session.pins) == 0


# ------------------------------------------------------- the atomicity property
_TEMPLATE = _quiet_session()
_PATIENT_IDS = st.sampled_from((PatientId("p1"), PatientId("p2"), PatientId("ghost")))
_BAY_IDS = st.sampled_from((*(b.id for b in _TEMPLATE.layout.bays), BayId("ghost")))
_STAFF_IDS = st.sampled_from((*(m.id for m in _TEMPLATE.roster), StaffId("ghost")))
_TASK_IDS = st.sampled_from((TaskId("task_000000"), TaskId("task_999999")))
_NODE_IDS = st.sampled_from((*(n.id for n in _TEMPLATE.layout.graph.nodes[:4]), NodeId("ghost")))

_ACTIONS = st.one_of(
    st.builds(ReassignAction, patient=_PATIENT_IDS, bay=_BAY_IDS),
    st.builds(BumpPriorityAction, patient=_PATIENT_IDS),
    st.builds(RerouteAction, staff=_STAFF_IDS, task=_TASK_IDS),
    st.builds(ExpediteCleanAction, bay=_BAY_IDS),
    st.builds(ExpediteDischargeAction, patient=_PATIENT_IDS),
    st.builds(CloseBayAction, bay=_BAY_IDS),
    st.builds(BlockEdgeAction, edge=st.tuples(_NODE_IDS, _NODE_IDS)),
)


def _arranged_session() -> RunSession:
    """A fresh session with accept-able and reject-able states for every action kind."""
    session = _quiet_session()
    world = session.world
    _enqueue(session, "p1", EsiAcuity.ESI3)
    p2 = make_patient("p2", esi=EsiAcuity.ESI3)
    world.register_patient(p2)
    world.assign_bay(_bay_of(session, ZoneType.GENERAL, 1), p2.id)
    world.add_task(
        kind="provider_visit",
        patient=PatientId("p1"),
        at=session.layout.entrances[0],
        required_role=StaffRole.PHYSICIAN,
        activity=Activity.PROVIDER_VISIT,
        duration=seconds(60),
    )
    return session


@settings(max_examples=30, deadline=None)
@given(action=_ACTIONS, pin=st.booleans())
def test_any_action_applies_cleanly_or_rejects_atomically(
    action: ReassignAction
    | BumpPriorityAction
    | RerouteAction
    | ExpediteCleanAction
    | ExpediteDischargeAction
    | CloseBayAction
    | BlockEdgeAction,
    pin: bool,
) -> None:
    session = _arranged_session()
    before = world_fingerprint(session)
    result = apply_override(session, OverrideRequest(action=action, pin=pin))
    if result.status == "rejected":
        assert len(result.violations) >= 1
        assert world_fingerprint(session) == before, f"non-atomic rejection for {action!r}"
    else:
        assert result.status == "applied"


def test_override_on_unknown_run_is_404(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, _QUIET)) as client:
        response = client.post(
            "/runs/nope/override",
            json={"action": {"kind": "bump_priority", "patient": "p"}},
        )
        assert response.status_code == 404
