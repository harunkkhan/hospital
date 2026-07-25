"""Bed cleaning as a value-of-unblocking assignment (doc 03 §4.6).

Cleaning is valued not by "a dirty bay" but by *how much acuity-weighted demand
it frees*: ``value(b) = Σ_{p waiting, compat[p,b]} u(esi(p))``. Idle housekeepers
are matched to dirty bays to maximize freed high-demand capacity net of response
travel: ``max Σ (w_time·value(b) - w_travel·distance(h, b))·y``. ``u``, ``w_time``,
and ``w_travel`` all come from the one ``ObjectiveConfig``, so "unblocking a
critical-patient bay" is priced in the same currency as everything else -- the
``w_time`` factor puts the acuity-demand term on the same scale as travel seconds
(the value/travel balance is the objective-scaling knob, assumption 6).

The valuation is myopic (currently-waiting compatible patients only; no arrivals
lookahead) and ``compat``-coupled — a bay no waiting patient is compatible with
scores ``0`` and is deprioritized. Acceptable under M1; recomputed every re-solve.
Only patients actually waiting *for a bay* (``NEEDS_BAY_STAGES``) count as
demand — a placed patient waiting for a provider/lab/documentation is not
unblocked by a clean. Housekeeper candidates are filtered against
``rules.skills_for("cleaning")`` (e.g. hazmat), the same union the validator's
SkillRule check applies, in addition to the HOUSEKEEPING role.
"""

from __future__ import annotations

from hospital.core import (
    MICROS_PER_SEC,
    Bay,
    BayId,
    BayStatus,
    CompiledRules,
    DecisionInput,
    PlanItem,
    StaffId,
    StaffMember,
    StaffRole,
    StaffState,
    WaitingPatient,
)
from hospital.solver.objective import ObjectiveConfig, acuity_urgency
from hospital.solver.placement import NEEDS_BAY_STAGES, compat_pair
from hospital.solver.protocol import RoutingOracle


def unblock_value(
    bay: Bay,
    waiting: tuple[WaitingPatient, ...],
    *,
    config: ObjectiveConfig,
    rules: CompiledRules,
) -> int:
    """``value(b) = Σ_{p waiting for a bay, compat[p,b]} u(esi(p))`` (doc 03 §4.6).

    The ONE value-of-unblocking quantity: how much acuity-weighted demand
    freeing ``bay`` would unblock. Shared by turnaround (a clean frees the
    bay), discharge (a discharge frees it identically), and dispatch's
    priority-augmented urgency — never re-derived per lever. Only patients
    actually waiting FOR A BAY (``NEEDS_BAY_STAGES``) count as demand; a
    placed patient waiting on providers/labs/documentation is not unblocked.
    """
    return sum(
        acuity_urgency(config, wp.patient.esi)
        for wp in waiting
        if wp.stage in NEEDS_BAY_STAGES and compat_pair(wp.patient, bay, rules)
    )


def prioritize_cleaning(
    di: DecisionInput,
    oracle: RoutingOracle,
    *,
    config: ObjectiveConfig,
    rules: CompiledRules,
    staff_members: tuple[StaffMember, ...] = (),
) -> tuple[PlanItem, ...]:
    """Assign idle housekeepers to dirty bays; emit ``kind="clean"`` items."""
    static_bays = {b.id: b for b in di.layout.bays}
    dirty: list[Bay] = sorted(
        (
            static_bays[bs.bay]
            for bs in di.bays
            if bs.status == BayStatus.CLEANING and bs.bay in static_bays
        ),
        key=lambda b: b.id.root,
    )
    members = {m.id: m for m in staff_members}
    # Qualification = HOUSEKEEPING role AND the compiled cleaning skills (the
    # same rules.skills_for union the validator applies to a cleaning task).
    cleaning_skills = rules.skills_for("cleaning")
    housekeepers: list[StaffState] = sorted(
        (
            ss
            for ss in di.staff
            if ss.current_task is None
            and ss.busy_until is None
            and ss.staff in members
            and members[ss.staff].role == StaffRole.HOUSEKEEPING
            and cleaning_skills <= members[ss.staff].skills
        ),
        key=lambda ss: ss.staff.root,
    )
    if not dirty or not housekeepers:
        return ()

    # Demand a clean unblocks = patients waiting FOR A BAY only; post-placement
    # stages (awaiting provider/labs/documentation) are not freed by a clean.
    value = {bay.id: unblock_value(bay, di.waiting, config=config, rules=rules) for bay in dirty}
    matched = _solve_cleaning(oracle, config, dirty, housekeepers, value)
    return tuple(
        PlanItem(stable_id=f"clean:{bid.root}", kind="clean", bay=bid, staff=sid)
        for (sid, bid) in matched
    )


def _solve_cleaning(
    oracle: RoutingOracle,
    config: ObjectiveConfig,
    dirty: list[Bay],
    housekeepers: list[StaffState],
    value: dict[BayId, int],
) -> list[tuple[StaffId, BayId]]:
    """Max-weight housekeeper→bay matching; keep only net-beneficial cleans."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    y: dict[tuple[StaffId, BayId], cp_model.IntVar] = {}
    weight: dict[tuple[StaffId, BayId], int] = {}
    for ss in housekeepers:
        for bay in dirty:
            resp_s = oracle.distance(ss.at, bay.node).root // MICROS_PER_SEC
            # w_time puts the acuity-demand term on the same currency as w_travel*seconds
            # (the value/travel balance is the objective-scaling knob, assumption 6).
            weight[(ss.staff, bay.id)] = config.w_time * value[bay.id] - config.w_travel * resp_s
            y[(ss.staff, bay.id)] = model.new_bool_var(f"y_{ss.staff.root}_{bay.id.root}")
    for ss in housekeepers:
        model.add(sum([y[k] for k in y if k[0] == ss.staff]) <= 1)
    for bay in dirty:
        model.add(sum([y[k] for k in y if k[1] == bay.id]) <= 1)
    model.maximize(sum([weight[k] * y[k] for k in y]))
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    solver.parameters.num_search_workers = 1
    solver.solve(model)
    matched = [k for k in y if solver.value(y[k]) == 1 and weight[k] > 0]
    return sorted(matched, key=lambda k: k[1].root)


__all__ = ["prioritize_cleaning", "unblock_value"]
