"""CP-SAT bay/zone assignment — the exact placement backend (doc 03 §4.3, PLAN §5.1).

``x[p,b] ∈ {0,1}`` minimizes a linear restriction of ``weighted_total`` plus an
unplaced-wait penalty; warm-started, time-capped, and **self-validated** before
return (doc 00 §5 rule 5). Key disciplines:

* ``compat[p,b]`` is evaluated by the *same* rule predicates
  :func:`hospital.core.validation.validate` uses, so the model and the validator
  cannot drift; only compatible vars are created (``(C3)`` by omission).
* The model is **always feasible** — the all-unplaced solution costs
  ``Σ_p reward[p]``. Scarcity surfaces as unplaced patients (penalized, ranked
  by the ONE sequencing score ``u(esi) + alpha·waited``), never a crash. The
  place-first big-M is **derived from the instance's own cost bounds** (see
  the formulation note), so the objective is *provably* lexicographic:
  place-first, then the sequencing score, then minimize travel — for every
  instance and every config, not just tuned ones.
* Determinism: fixed ``random_seed``, ``num_search_workers = 1``, sorted
  model-build order, and a **deterministic-time** search budget (never a
  wall-clock cap, whose interrupt point depends on OS scheduling) →
  byte-reproducible solves for CRN / golden traces. If the budget expires with
  no incumbent, the greedy backend supplies a deterministic ``HEURISTIC``
  fallback — a result is never labeled ``FEASIBLE`` without an incumbent.

``ortools`` is imported lazily *inside* ``solve`` so importing this module (e.g.
for the shared helpers the heuristic reuses) never pulls OR-Tools.

Formulation note — the instance-derived place-first big-M. The unplaced penalty
is ``reward[p] = (wait_penalty[p] + 1) · B`` with ``B = Σ_p max_b w[p,b] + 1``
and ``wait_penalty[p] = w_time·unplaced_wait_penalty·score(p)``, where
``score(p) = u(esi) + alpha·waited_s`` is the ONE ``sequencing.priority_score``
(at the shared ``DEFAULT_STARVATION_RATE``). Since any solution's total travel
is at most ``B - 1 < B``: (1) placing one more patient always improves the
objective by at least ``reward[p] - w[p,b] ≥ B - (B - 1) = 1``, so capacity is
never left idle to save travel; (2) whenever two placement sets differ in
``Σ wait_penalty`` (integers, so by ≥ 1), the reward gap ``≥ B`` dominates any
travel gap ``< B``, so who gets placed under scarcity follows the sequencing
score exactly — the same anti-starvation ranking the ``sequence`` lever enacts
on the bay queue, so a long-waiting low-acuity patient genuinely overtakes
(the earlier ``u·(waited+1)`` pricing disagreed with that enacted order,
making the sequencing override a placement no-op); (3) travel only breaks the
remaining ties. A just-arrived patient's priority is ``u(esi)`` — acuity-ranked
from second zero — and the ``+1`` reward base enforces place-first even under
a degenerate config (``w_time`` or ``unplaced_wait_penalty`` of ``0``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

from hospital.core import (
    MICROS_PER_SEC,
    Bay,
    BayId,
    BayStatus,
    CompiledRules,
    DecisionInput,
    Duration,
    InfeasiblePlan,
    NodeId,
    Patient,
    PatientId,
    Plan,
    PlanItem,
    ValidationContext,
    ZoneType,
    seconds,
    validate,
)
from hospital.solver.objective import AssignmentCoeffs, ObjectiveConfig, assignment_coeffs
from hospital.solver.protocol import RoutingOracle, SolveResult, SolverStatus
from hospital.solver.sequencing import DEFAULT_STARVATION_RATE, priority_score

if TYPE_CHECKING:
    from collections.abc import Container

    from hospital.core import FloorLayout, ZoneId

# Stages of ``WaitingPatient.stage`` that mean "awaiting a bay". The exact stage
# vocabulary is not fixed in ``core.seam`` yet; this is the placement convention.
NEEDS_BAY_STAGES: frozenset[str] = frozenset({"needs_bay", "awaiting_bay", "waiting_for_bay"})

# Default real-time solve budget (doc 03 §4.3 example); small so tests stay fast.
# Interpreted as CP-SAT *deterministic time* (reproducible work units calibrated
# to ~1 s of single-threaded search each), not wall-clock — see ``solve``.
DEFAULT_TIME_CAP: Duration = seconds(0.05)

_CandidateSet = tuple[list[Patient], list[Bay], set[tuple[PatientId, BayId]]]


def _needs_bay(stage: str) -> bool:
    return stage in NEEDS_BAY_STAGES


def compat_pair(patient: Patient, bay: Bay, rules: CompiledRules) -> bool:
    """``compat[p,b]`` — the same predicates the validator applies (doc 03 §4.3)."""
    if bay.zone_type not in rules.zone_types_for(patient.esi):
        return False
    if rules.equipment_for(patient.esi) - bay.equipment:
        return False
    return not (
        rules.isolation_enforced and patient.isolation_required and not bay.isolation_capable
    )


def candidates(di: DecisionInput, rules: CompiledRules) -> _CandidateSet:
    """Patients needing a bay, FREE bays, and the sparse compatible ``(p,b)`` set.

    Both lists are sorted by id so the CP-SAT model build is byte-stable.
    """
    static_bays = {b.id: b for b in di.layout.bays}
    patients = sorted(
        (wp.patient for wp in di.waiting if _needs_bay(wp.stage)), key=lambda p: p.id.root
    )
    free_bays = sorted(
        (
            static_bays[bs.bay]
            for bs in di.bays
            if bs.status == BayStatus.FREE and bs.bay in static_bays
        ),
        key=lambda b: b.id.root,
    )
    compat = {(p.id, b.id) for p in patients for b in free_bays if compat_pair(p, b, rules)}
    return patients, free_bays, compat


def _seconds(duration: Duration) -> int:
    """Floor a µs :class:`Duration` to whole seconds (doc 03 §4.1)."""
    return duration.root // MICROS_PER_SEC


def _nearest(src: NodeId, nodes: tuple[NodeId, ...], oracle: RoutingOracle) -> NodeId | None:
    """The node in ``nodes`` minimizing oracle distance from ``src`` (or ``None``)."""
    if not nodes:
        return None
    return min(nodes, key=lambda n: oracle.distance(src, n).root)


def travel_weight(
    patient: Patient, bay: Bay, oracle: RoutingOracle, layout: FloorLayout, coeffs: AssignmentCoeffs
) -> int:
    """``w[p,b]`` — expected downstream travel via the oracle (doc 03 §4.3).

    ``arrival + caregiver + imaging`` seconds, scaled by the acuity travel
    weight. This is a placement *proxy*, not a physics prediction — doc 03
    flags every term as a 🟡-tunable heuristic (§4.3/§7) — and it only *biases*
    which valid bay is chosen; it can never make a plan infeasible. The
    composition is fitted to what the twin's physics actually walks, with three
    documented deviations from the doc's literal formula, each toward truth:

    * **arrival** (added): the round trip between the floor's arrival end
      (``entrances[0]`` — triage sits beside it) and the bay. Both directions
      are physically walked in-system: the patient (plus, for the
      non-ambulatory, a porter escort) covers entrance->bay on placement, and
      the discharge exit retraces bay->entrance at the end of the stay.
      Omitting it prices a bay across the floor identically to one beside
      triage — placements drift to the far end and the arrival/exit walks
      swamp the caregiver savings.
    * **caregiver is one-way per visit** (the doc's ``2x`` round trip): staff
      idle in place where they finish and dispatch re-targets them from
      wherever they stand, so the return leg mostly never happens — the doc
      itself flags the round trip as an over-count. ``labs`` are bedside
      draws (the patient never moves; the *sample* travels off-graph), so a
      lab counts as one more caregiver visit, not the doc's bay<->analyzer
      round trip that no actor walks.
    * **imaging is doubled** (patient + escort): every imaging transport is
      escorted, so two actors walk both legs of every bay<->modality trip.
    """

    def one_way(a: NodeId, b: NodeId) -> int:
        return _seconds(oracle.distance(a, b))

    def round_trip(a: NodeId, b: NodeId) -> int:
        return one_way(a, b) + one_way(b, a)

    node = bay.node
    arrival = round_trip(layout.entrances[0], node) if layout.entrances else 0
    visits = patient.workup.provider_visits + patient.workup.nurse_visits + patient.workup.labs
    caregiver = visits * one_way(bay.serving_station, node)
    imaging = 0
    for _modality in patient.workup.imaging:
        target = _nearest(node, layout.imaging_nodes, oracle)
        if target is not None:
            imaging += 2 * round_trip(node, target)
    return coeffs.travel_weight * (arrival + caregiver + imaging)


def occupied_by_zone_type(di: DecisionInput) -> dict[ZoneType, int]:
    """Count of currently-OCCUPIED bays per zone type (matches the validator's count)."""
    static_bays = {b.id: b for b in di.layout.bays}
    counts: dict[ZoneType, int] = {}
    for bs in di.bays:
        bay = static_bays.get(bs.bay)
        if bay is not None and bs.status == BayStatus.OCCUPIED:
            counts[bay.zone_type] = counts.get(bay.zone_type, 0) + 1
    return counts


def zone_remaining(di: DecisionInput) -> dict[ZoneId, int]:
    """Per-``ZoneId`` remaining ``= Zone.capacity - |OCCUPIED or CLEANING|`` (doc 03 §4.3).

    Never negative: an already-over-capacity zone clamps to ``0`` rather than
    making the model infeasible.
    """
    static_bays = {b.id: b for b in di.layout.bays}
    used: dict[ZoneId, int] = {}
    for bs in di.bays:
        bay = static_bays.get(bs.bay)
        if bay is not None and bs.status in (BayStatus.OCCUPIED, BayStatus.CLEANING):
            used[bay.zone] = used.get(bay.zone, 0) + 1
    return {zone.id: max(0, zone.capacity - used.get(zone.id, 0)) for zone in di.layout.zones}


def _waited(di: DecisionInput) -> dict[PatientId, Duration]:
    return {wp.patient.id: wp.waited for wp in di.waiting}


class CpSatPlacement:
    """The CP-SAT bay/zone assignment backend (implements ``Solver``)."""

    name: ClassVar[str] = "placement_cpsat"
    version: ClassVar[str] = "1.0.0"

    def solve(
        self,
        di: DecisionInput,
        oracle: RoutingOracle,
        *,
        config: ObjectiveConfig,
        rules: CompiledRules,
        time_cap: Duration | None = None,
        warm_start: Plan | None = None,
    ) -> SolveResult:
        from ortools.sat.python import cp_model

        started_ns = time.perf_counter_ns()
        patients, bays, compat = candidates(di, rules)
        layout = di.layout
        patient_by_id = {p.id: p for p in patients}
        bay_by_id = {b.id: b for b in bays}
        waited = _waited(di)
        coeffs = {esi: assignment_coeffs(config, esi) for esi in {p.esi for p in patients}}

        model = cp_model.CpModel()
        x: dict[tuple[PatientId, BayId], cp_model.IntVar] = {
            (p.id, b.id): model.new_bool_var(f"x_{p.id.root}_{b.id.root}")
            for p in patients
            for b in bays
            if (p.id, b.id) in compat
        }
        weight = {
            (pid, bid): travel_weight(
                patient_by_id[pid], bay_by_id[bid], oracle, layout, coeffs[patient_by_id[pid].esi]
            )
            for (pid, bid) in x
        }
        # Scarcity priority = the ONE sequencing score u(esi) + alpha·waited
        # (anti-starvation), scaled onto the objective's wait-penalty currency —
        # so who wins a scarce bay agrees with the enacted queue order.
        wait_penalty = {
            p.id: config.w_time
            * config.unplaced_wait_penalty
            * priority_score(
                p.esi,
                waited.get(p.id, Duration(0)),
                config=config,
                starvation_rate=DEFAULT_STARVATION_RATE,
            )
            for p in patients
        }
        # Place-first big-M derived from THIS instance's cost bounds (module
        # docstring): B strictly exceeds any solution's total travel (each
        # patient occupies at most one bay, so travel ≤ Σ_p max_b w[p,b]).
        max_weight_by_patient: dict[PatientId, int] = {}
        for (pid, _bid), w in weight.items():
            max_weight_by_patient[pid] = max(w, max_weight_by_patient.get(pid, 0))
        big_b = sum(max_weight_by_patient.values()) + 1
        reward = {p.id: (wait_penalty[p.id] + 1) * big_b for p in patients}

        # Objective: Σ w[p,b]·x + Σ reward[p]·(1 - Σ_b x[p,b]) — lexicographic
        # place-first, then acuity-weighted priority, then travel.
        objective: list[cp_model.LinearExprT] = [weight[k] * x[k] for k in x]
        for p in patients:
            placed = [x[(p.id, b.id)] for b in bays if (p.id, b.id) in x]
            objective.append(reward[p.id])
            objective.extend(-reward[p.id] * var for var in placed)
        model.minimize(sum(objective))

        # (C1) each patient at most one bay.
        for p in patients:
            vars_p = [x[(p.id, b.id)] for b in bays if (p.id, b.id) in x]
            if vars_p:
                model.add(sum(vars_p) <= 1)
        # (C2) each bay at most one patient.
        for b in bays:
            vars_b = [x[(p.id, b.id)] for p in patients if (p.id, b.id) in x]
            if vars_b:
                model.add(sum(vars_b) <= 1)
        # (C4a) per-zone-type rule capacity (matches the validator's OCCUPIED count).
        occupied_zt = occupied_by_zone_type(di)
        for zone_type in {b.zone_type for b in bays}:
            cap = rules.capacity_for(zone_type)
            if cap is None:
                continue
            vars_zt = [
                x[(p.id, b.id)] for (p, b) in _pairs(patients, bays, x) if b.zone_type == zone_type
            ]
            if vars_zt:
                model.add(sum(vars_zt) <= max(0, cap - occupied_zt.get(zone_type, 0)))
        # (C4b) per-zone entity capacity (staffing-limited zones, doc 03 §4.3).
        remaining = zone_remaining(di)
        for zone in layout.zones:
            vars_z = [x[(p.id, b.id)] for (p, b) in _pairs(patients, bays, x) if b.zone == zone.id]
            if vars_z:
                model.add(sum(vars_z) <= remaining.get(zone.id, zone.capacity))

        # Warm start: hint the still-valid x[p,b] toward the prior assignment.
        if warm_start is not None:
            for item in warm_start.items:
                if item.kind == "assign_bay" and item.patient is not None and item.bay is not None:
                    key = (item.patient, item.bay)
                    if key in x:
                        model.add_hint(x[key], 1)

        solver = cp_model.CpSolver()
        cap_dur = time_cap if time_cap is not None else DEFAULT_TIME_CAP
        # Deterministic-time budget, NEVER max_time_in_seconds: a wall-clock cap
        # lets OS scheduling pick the incumbent, so the same DecisionInput could
        # yield different plans (breaking CRN / byte-identical golden traces).
        # Deterministic time is CP-SAT's reproducible work counter, calibrated
        # to roughly a second of single-threaded search per unit.
        solver.parameters.max_deterministic_time = cap_dur.root / MICROS_PER_SEC
        solver.parameters.random_seed = 0
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # No incumbent (UNKNOWN under a tiny budget; INFEASIBLE/MODEL_INVALID
            # cannot occur — the all-unplaced solution is always feasible — so
            # those are defensive). Never report FEASIBLE without an incumbent:
            # delegate to the deterministic greedy backend, labeled honestly.
            from hospital.solver.heuristic import HeuristicPlacement

            fallback = HeuristicPlacement().solve(di, oracle, config=config, rules=rules)
            elapsed_us = (time.perf_counter_ns() - started_ns) // 1000
            return SolveResult(
                plan=fallback.plan,
                status=SolverStatus.HEURISTIC,
                objective_value=None,
                solve_wall_us=elapsed_us,
                backend=fallback.backend,
            )

        assignments = [k for k, var in x.items() if solver.value(var) == 1]
        objective_value: int | None = int(solver.objective_value)
        solver_status = (
            SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE
        )

        assignments.sort(key=lambda pb: (pb[0].root, pb[1].root))
        plan = Plan(
            items=tuple(
                PlanItem(stable_id=f"assign:{pid.root}", kind="assign_bay", patient=pid, bay=bid)
                for pid, bid in assignments
            )
        )
        self_validate(plan, di, rules)
        elapsed_us = (time.perf_counter_ns() - started_ns) // 1000
        return SolveResult(
            plan=plan,
            status=solver_status,
            objective_value=objective_value,
            solve_wall_us=elapsed_us,
            backend=self.name,
        )


def _pairs(
    patients: list[Patient],
    bays: list[Bay],
    keys: Container[tuple[PatientId, BayId]],
) -> list[tuple[Patient, Bay]]:
    """The ``(patient, bay)`` pairs that have a decision variable (compatible only)."""
    return [(p, b) for p in patients for b in bays if (p.id, b.id) in keys]


def self_validate(plan: Plan, di: DecisionInput, rules: CompiledRules) -> None:
    """Independent cross-check (doc 00 §5 rule 5) — never repair, raise instead."""
    ctx = ValidationContext(
        layout=di.layout,
        bays=di.bays,
        staff=di.staff,
        rules=rules,
        patients=tuple(wp.patient for wp in di.waiting),
    )
    violations = validate(plan, ctx)
    if violations:
        raise InfeasiblePlan(violations)


__all__ = [
    "DEFAULT_TIME_CAP",
    "NEEDS_BAY_STAGES",
    "CpSatPlacement",
    "candidates",
    "compat_pair",
    "occupied_by_zone_type",
    "self_validate",
    "travel_weight",
    "zone_remaining",
]
