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

Deviation note: ``core.seam.TaskSpec`` carries no acuity/urgency field (the doc
§7 #13 assumption was superseded by core), so task urgency ``u(t)`` is uniform
here and the assignment cost is ``w_travel · distance`` — still "nearest
qualified". Staff role/skills are not in ``DecisionInput`` either, so
``staff_members`` is supplied explicitly. ``assign_staff`` also takes the
compiled ``rules``: qualification is judged on
``task.required_skills | rules.skills_for(task.kind)`` — the SAME union
:func:`hospital.core.validation.validate` applies — so dispatch can never emit
a plan the one validator rejects.
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
from hospital.solver.objective import ObjectiveConfig
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
                dist_s = oracle.distance(ss.at, task.at).root // MICROS_PER_SEC
                cost[(ss.staff, task.id)] = config.w_travel * dist_s
    if not cost:
        return ()
    matched = _solve_assignment(cost, idle, task_list)
    return tuple(
        PlanItem(stable_id=f"dispatch:{tid.root}", kind="dispatch", staff=sid, task=tid)
        for (sid, tid) in sorted(matched, key=lambda st: st[1].root)
    )


def _solve_assignment(
    cost: dict[tuple[StaffId, TaskId], int],
    idle: list[StaffState],
    tasks: list[TaskSpec],
) -> list[tuple[StaffId, TaskId]]:
    """Max-cardinality, min-cost matching (single task = nearest qualified)."""
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
    # big must dominate the total cost spread so cardinality is strictly
    # lexicographic: adding ANY match improves the objective by at least 1
    # (max(cost)+1 would let one cheap match beat two expensive ones,
    # stranding a coverable task — the myopia this solver exists to avoid).
    big = sum(cost.values()) + 1
    for ss in idle:
        staff_vars = [y[k] for k in cost if k[0] == ss.staff]
        if staff_vars:
            model.add(sum(staff_vars) <= 1)
    for task in tasks:
        task_vars = [y[k] for k in cost if k[1] == task.id]
        if task_vars:
            model.add(sum(task_vars) <= 1)
    # Maximize matches first (big-M), then minimize travel: minimize Σ(cost-big)·y.
    model.minimize(sum([(cost[k] - big) * y[k] for k in cost]))
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
