"""Nearest-qualified assignment + batched-round routing (doc 03 §4.5).

Two oracle-backed levers; this module contains **no** pathfinding of its own
(rule 3) — every spatial cost flows through ``GraphRoutingOracle.distance``.

* ``assign_staff`` — idle staff ↔ pending tasks. A single task is the trivial
  ``argmin`` ("nearest qualified"); multiple tasks/staff solve a global
  assignment (CP-SAT) so a myopic nearest choice can't strand another task with
  no qualified staff nearby. Skills/role are **hard exclusions** (no variable for
  a disallowed pair), so the SkillRule can never be violated.
* ``route_visits`` — an open-path stop sequence for one staff member: exact
  Held-Karp for ``≤ exact_max`` stops, nearest-neighbour + 2-opt above. Open
  path (no return-to-start term); 2-opt is first-improvement with a fixed scan
  order → a deterministic local optimum.

Assignment objective (doc 03 §4.5's ``u(t)``): serve-first (max cardinality,
via the same instance-derived big-M construction as placement), then ONE
weighted cost consistent with ``hospital.solver.objective``'s single scalar —

    ``cost(s,t) = w_time · u(t) · (waited_s(t) + travel_s(s,t))
                  + w_travel · travel_s(s,t)``

where ``waited_s + travel_s`` is the wait the task will have accumulated when
service starts (staff must arrive first), ``u`` is the one ``acuity_urgency``
curve read from ``TaskSpec.esi`` (patient-less work — cleaning — prices at the
curve's floor), and a task left UNSERVED under scarcity is priced as if it
waits ``config.unplaced_wait_penalty`` more seconds (the same unserved-demand
constant placement prices with). Two prior failure shapes this replaces: with
travel-only costs the matched subset under scarcity starved the far resus zone
(ESI-1 clinical inversion); with strict acuity tiers + FIFO-within-tier the
matching degenerated to the baseline service order and the travel savings
vanished. The weighted form trades them continuously: urgency buys travel in
proportion to ``w_time·Δu·unplaced_wait_penalty``, and between equal acuities
travel alone decides (sunk wait cancels between serving and deferring).

Staff role/skills are not in ``DecisionInput``, so ``staff_members`` is
supplied explicitly. ``assign_staff`` also takes the compiled ``rules``:
qualification is judged on ``task.required_skills | rules.skills_for(task.kind)``
— the SAME union :func:`hospital.core.validation.validate` applies — so
dispatch can never emit a plan the one validator rejects.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from hospital.core import (
    MICROS_PER_SEC,
    CompiledRules,
    DecisionInput,
    NodeId,
    PlanItem,
    StaffId,
    StaffMember,
    TaskId,
    TaskSpec,
)
from hospital.solver.objective import ObjectiveConfig, acuity_urgency
from hospital.solver.oracle import EMPTY_MASK, RouteMask
from hospital.solver.protocol import RoutingOracle

if TYPE_CHECKING:
    from hospital.core import StaffState

_DistFn = Callable[[NodeId, NodeId], int]


def assign_staff(
    di: DecisionInput,
    oracle: RoutingOracle,
    *,
    config: ObjectiveConfig,
    rules: CompiledRules,
    tasks: tuple[TaskSpec, ...] | None = None,
    staff_members: tuple[StaffMember, ...] = (),
) -> tuple[PlanItem, ...]:
    """Assign idle qualified staff to pending tasks; emit ``kind="dispatch"`` items."""
    task_list = sorted(tasks if tasks is not None else di.pending_tasks, key=lambda t: t.id.root)
    idle = sorted(
        (ss for ss in di.staff if ss.current_task is None and ss.busy_until is None),
        key=lambda ss: ss.staff.root,
    )
    members = {m.id: m for m in staff_members}
    urgency = _task_urgency(task_list, config)
    waited = {t.id: max(0, (di.now.root - t.ready_at.root) // MICROS_PER_SEC) for t in task_list}
    cost: dict[tuple[StaffId, TaskId], int] = {}
    for ss in idle:
        member = members.get(ss.staff)
        if member is None:
            continue
        for task in task_list:
            # The SAME union the validator's SkillRule check applies (rule 5):
            # per-task skills plus the compiled skills for the task kind.
            required = task.required_skills | rules.skills_for(task.kind)
            if member.role == task.required_role and required <= member.skills:
                travel_s = oracle.distance(ss.at, task.at).root // MICROS_PER_SEC
                expected_wait_s = waited[task.id] + travel_s
                cost[(ss.staff, task.id)] = (
                    config.w_time * urgency[task.id] * expected_wait_s + config.w_travel * travel_s
                )
    if not cost:
        return ()
    deferral = {
        t.id: config.w_time * urgency[t.id] * (waited[t.id] + config.unplaced_wait_penalty)
        for t in task_list
    }
    matched = _solve_assignment(cost, idle, task_list, deferral)
    return tuple(
        PlanItem(stable_id=f"dispatch:{tid.root}", kind="dispatch", staff=sid, task=tid)
        for (sid, tid) in sorted(matched, key=lambda st: st[1].root)
    )


def _task_urgency(tasks: list[TaskSpec], config: ObjectiveConfig) -> dict[TaskId, int]:
    """``u(t)`` — the one ``acuity_urgency`` curve; patient-less work at the floor.

    A task with no acuity (cleaning) prices at the curve's minimum urgency so it
    is neutral: it never outranks any patient's urgency, and among themselves
    such tasks are matched on travel alone.
    """
    floor_urgency = min(u for _, u in config.acuity_urgency)
    return {
        t.id: acuity_urgency(config, t.esi) if t.esi is not None else floor_urgency for t in tasks
    }


def _solve_assignment(
    cost: dict[tuple[StaffId, TaskId], int],
    idle: list[StaffState],
    tasks: list[TaskSpec],
    deferral: dict[TaskId, int],
) -> list[tuple[StaffId, TaskId]]:
    """Serve-first, then min weighted cost (single task = cheapest qualified staff).

    Minimizes ``Σ_served cost[s,t] + Σ_unserved deferral[t]`` subject to
    serve-first: with ``B = Σ_t (max_s cost[s,t] + deferral[t]) + 1`` and
    ``reward[t] = deferral[t] + B``, minimizing ``Σ (cost - reward)·y`` makes
    any matching that serves one more task strictly better (a (k+1)-matching's
    objective is at most ``-(k+1)B + Σ_t max_s cost``, a k-matching's at least
    ``-kB - Σ_t deferral``, and ``B`` exceeds their gap by 1) — capacity is
    never left idle. Among equal-cardinality matchings the residual
    ``Σ_served (cost - deferral)`` is exactly the weighted objective above with
    the constant ``Σ_all deferral`` dropped: per served pair it reduces to
    ``(w_time·u + w_travel)·travel_s - w_time·u·unplaced_wait_penalty`` (sunk
    wait cancels), so urgency and travel trade continuously — no tiers.
    """
    if len(tasks) == 1:
        task = tasks[0]
        candidates = [(c, sid) for (sid, tid), c in cost.items() if tid == task.id]
        if not candidates:
            return []
        best = min(candidates, key=lambda cs: (cs[0], cs[1].root))
        return [(best[1], task.id)]

    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    y = {k: model.new_bool_var(f"y_{k[0].root}_{k[1].root}") for k in cost}
    max_cost_by_task: dict[TaskId, int] = {}
    for (_sid, tid), c in cost.items():
        max_cost_by_task[tid] = max(c, max_cost_by_task.get(tid, 0))
    big_b = sum(c + deferral[tid] for tid, c in max_cost_by_task.items()) + 1
    reward = {tid: deferral[tid] + big_b for tid in max_cost_by_task}
    for ss in idle:
        staff_vars = [y[k] for k in cost if k[0] == ss.staff]
        if staff_vars:
            model.add(sum(staff_vars) <= 1)
    for task in tasks:
        task_vars = [y[k] for k in cost if k[1] == task.id]
        if task_vars:
            model.add(sum(task_vars) <= 1)
    # Minimize Σ cost·y + Σ reward·(unmatched) — the constant Σ reward is
    # dropped, leaving: minimize Σ (cost[s,t] - reward[t])·y[s,t].
    model.minimize(sum([(cost[k] - reward[k[1]]) * y[k] for k in cost]))
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    solver.parameters.num_search_workers = 1
    solver.solve(model)
    return [k for k in cost if solver.value(y[k]) == 1]


def route_visits(
    start: NodeId,
    stops: Sequence[NodeId],
    oracle: RoutingOracle,
    *,
    mask: RouteMask = EMPTY_MASK,
    exact_max: int = 10,
) -> tuple[NodeId, ...]:
    """Order ``stops`` to minimize open-path travel from ``start``."""
    unique = list(dict.fromkeys(stops))
    if not unique:
        return ()

    def distance(a: NodeId, b: NodeId) -> int:
        return oracle.distance(a, b, mask=mask).root

    if len(unique) <= exact_max:
        return _held_karp(start, unique, distance)
    return _nn_two_opt(start, unique, distance)


def _held_karp(start: NodeId, stops: list[NodeId], d: _DistFn) -> tuple[NodeId, ...]:
    """Exact open-path TSP (no return-to-start term). ``O(n²·2ⁿ)`` (doc 03 §4.5)."""
    n = len(stops)
    if n == 1:
        return (stops[0],)
    full = 1 << n
    g: list[list[float]] = [[math.inf] * n for _ in range(full)]
    parent: list[list[int]] = [[-1] * n for _ in range(full)]
    for j in range(n):
        g[1 << j][j] = d(start, stops[j])
    for mask in range(full):
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            cur = g[mask][j]
            if math.isinf(cur):
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nxt = mask | (1 << k)
                cand = cur + d(stops[j], stops[k])
                if cand < g[nxt][k]:
                    g[nxt][k] = cand
                    parent[nxt][k] = j
    end = full - 1
    best_j = min(range(n), key=lambda j: (g[end][j], stops[j].root))
    order: list[NodeId] = []
    mask, j = end, best_j
    while j != -1:
        order.append(stops[j])
        prev_j = parent[mask][j]
        mask ^= 1 << j
        j = prev_j
    order.reverse()
    return tuple(order)


def _nn_two_opt(start: NodeId, stops: list[NodeId], d: _DistFn) -> tuple[NodeId, ...]:
    """Nearest-neighbour tour + first-improvement 2-opt → deterministic local optimum."""
    remaining = list(stops)
    tour: list[NodeId] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda s: (d(cur, s), s.root))
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    def path_len(seq: list[NodeId]) -> int:
        total = 0
        prev = start
        for stop in seq:
            total += d(prev, stop)
            prev = stop
        return total

    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for k in range(i + 1, len(tour)):
                candidate = tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]
                if path_len(candidate) < path_len(tour):
                    tour = candidate
                    improved = True
                    break
            if improved:
                break
    return tuple(tour)


__all__ = ["assign_staff", "route_visits"]
