"""Staff scheduling — the input adapter and the covering MIP (doc 03 §4.8).

**``load_roster``.** Staffing is a scenario *input* (assumption 19): the sim
supplies per-staff shift windows as **core types** and this adapter validates
they cover the operating week and emits ``kind="staffing"`` plan items. It
deliberately does **not** import ``data.StaffingSpec`` — that would be a forbidden
sideways ``solver → data`` import. *You set staffing; we measure.*

Each item **carries its shift ``TimeWindow``** — rosters with different shift
boundaries must produce different payloads. ``core.seam.PlanItem`` has no time
fields, so the window rides in the ``order`` payload channel as canonical
integer-µs strings ``(str(start), str(end))``; :func:`staffing_window` is the
one decoder, so no consumer parses by hand.

Coverage gaps are *surfaced*, not filled (never-repair applied to inputs): a week
the supplied windows do not span raises ``ValueError``.

**``solve_coverage``** is the other half, and the last operational lever to stop being
an input: a covering MIP that *chooses* the roster from forecast demand instead of
accepting one. Demand arrives as a plain ``Mapping`` so ``solver`` stays a leaf — the
arrival intensity that produces it lives in ``hospital.forecast``, which by contract
cannot import this module, so a composition root threads the two.

The two entry points meet at the same ``kind="staffing"`` contract: ``solve_coverage``
answers *what the roster should be* and ``load_roster`` packages *a* roster into plan
items, so a solved roster is scheduled by handing its assignments to the latter.
"""

from __future__ import annotations

from collections.abc import Mapping

from hospital.core import (
    FrozenModel,
    OperatingWeek,
    PlanItem,
    SimTime,
    StaffId,
    StaffMember,
    StaffRole,
    TimeWindow,
)


class ShiftAssignment(FrozenModel):
    """A staff member on shift for a window (the covering-MIP output shape)."""

    staff: StaffId
    role: StaffRole
    window: TimeWindow


def _covers(week: OperatingWeek, windows: list[TimeWindow]) -> bool:
    """Whether the union of ``windows`` covers the half-open ``[week.start, week.end)``."""
    intervals = sorted(((w.start.root, w.end.root) for w in windows), key=lambda iv: iv[0])
    cursor = week.start.root
    end = week.end.root
    if cursor >= end:
        return True
    for start, stop in intervals:
        if start > cursor:
            return False  # a gap opened before this window starts
        cursor = max(cursor, stop)
        if cursor >= end:
            return True
    return cursor >= end


def load_roster(
    staff: tuple[StaffMember, ...],
    windows: Mapping[StaffId, tuple[TimeWindow, ...]],
    week: OperatingWeek,
) -> tuple[PlanItem, ...]:
    """Package supplied shift windows into ``kind="staffing"`` items (M1 adapter)."""
    all_windows = [w for wins in windows.values() for w in wins]
    if not _covers(week, all_windows):
        raise ValueError(
            f"roster does not cover the operating week [{week.start.root}, {week.end.root})"
        )
    items: list[PlanItem] = []
    for member in sorted(staff, key=lambda m: m.id.root):
        for index, window in enumerate(windows.get(member.id, ())):
            items.append(
                PlanItem(
                    stable_id=f"staffing:{member.id.root}:{index}",
                    kind="staffing",
                    staff=member.id,
                    # The shift window, in the order payload channel (module
                    # docstring); decode with staffing_window().
                    order=(str(window.start.root), str(window.end.root)),
                )
            )
    return tuple(items)


def staffing_window(item: PlanItem) -> TimeWindow:
    """Recover the shift :class:`TimeWindow` a ``staffing`` item carries.

    The single decoder for the ``order``-channel encoding ``load_roster`` emits;
    a non-staffing item or one without a window payload is a caller error.
    """
    if item.kind != "staffing" or item.order is None or len(item.order) != 2:
        raise ValueError(f"plan item {item.stable_id!r} carries no staffing window")
    start, end = item.order
    return TimeWindow(start=SimTime(int(start)), end=SimTime(int(end)))


def covers(shift: TimeWindow, block: TimeWindow) -> bool:
    """Whether working ``shift`` puts someone on the floor for the whole of ``block``.

    Containment, not overlap. A shift that covers half a block staffs half of it, and
    counting it as coverage would let the solver meet a demand of three with three people
    who are each present for twenty minutes of the hour. Partial credit is expressible
    (make the blocks finer) but must not be silent.
    """
    return shift.start <= block.start and block.end <= shift.end


def _shift_hours(shift: TimeWindow) -> int:
    """A shift's length in whole hours, floored, minimum 1.

    The objective's unit is staff-hours (doc: "cover demand at minimum staff-hours"), and
    CP-SAT needs integers. Flooring a 90-minute shift to one hour under-prices it, which
    is why the floor is documented rather than hidden: shift grids in this model are whole
    hours, so the rounding is exact for every input the scenario schema can produce.
    """
    return max(1, (shift.end.root - shift.start.root) // (3_600 * 1_000_000))


def solve_coverage(
    demand: Mapping[tuple[StaffRole, int], int],
    shifts: tuple[TimeWindow, ...],
    *,
    role_cost: Mapping[StaffRole, int],
    blocks: tuple[TimeWindow, ...],
) -> tuple[ShiftAssignment, ...]:
    """The covering MIP: cheapest roster that meets per-role demand in every block.

    The last lever in the operational set to stop being an input. Through M4 staffing was
    "you set it, we measure"; this chooses it — how many of each role start each candidate
    shift — to cover forecast demand at minimum cost-weighted staff-hours.

    ``demand[(role, b)]`` is the headcount of ``role`` needed throughout ``blocks[b]``;
    the ``blocks`` grid is passed explicitly rather than implied by the integer key,
    because a bare index cannot say what span it refers to and two callers would
    eventually disagree about it. ``demand`` stays a plain ``Mapping`` so ``solver``
    remains a leaf: the arrival forecast that produces it lives in ``hospital.forecast``,
    which by contract cannot import this module, and a composition root threads the two.

    Formulation. One integer variable ``n[role, s]`` per role and candidate shift, so the
    solver picks *counts* rather than assigning named people — headcount is the decision a
    scheduler actually makes, and naming individuals here would invent a second roster
    identity to reconcile with the scenario's. Coverage is
    ``Σ_{s covers b} n[role, s] >= demand[role, b]`` for every ``(role, b)``, and the
    objective minimizes ``Σ role_cost[role] · hours(s) · n[role, s]``.

    **Infeasibility is reported, never absorbed.** A block with demand that no candidate
    shift contains has no feasible roster, and the same is true of a role with demand and
    no cost. Both raise :class:`ValueError` naming the culprit rather than returning a
    roster that quietly under-covers — the input equivalent of validate-never-repair,
    already the rule ``load_roster`` follows for coverage gaps.

    Determinism, as everywhere else a solver runs here: sorted build order, fixed seed,
    a single search worker, and synthesized staff ids derived from ``(role, shift, k)``
    so the same demand always yields the same roster byte-for-byte.
    """
    from ortools.sat.python import cp_model

    roles = sorted({role for role, _ in demand}, key=lambda r: r.value)
    missing_cost = [r.value for r in roles if r not in role_cost]
    if missing_cost:
        raise ValueError(f"no role_cost for demanded role(s): {sorted(missing_cost)}")

    for role in roles:
        for index, block in enumerate(blocks):
            if demand.get((role, index), 0) > 0 and not any(covers(s, block) for s in shifts):
                raise ValueError(
                    f"no candidate shift covers block {index} "
                    f"[{block.start.root}, {block.end.root}) demanded by {role.value}"
                )

    model = cp_model.CpModel()
    order = sorted(range(len(shifts)), key=lambda i: (shifts[i].start.root, shifts[i].end.root, i))
    n: dict[tuple[StaffRole, int], cp_model.IntVar] = {}
    for role in roles:
        peak = max((demand.get((role, b), 0) for b in range(len(blocks))), default=0)
        for s in order:
            # Bounded by the peak: no cheapest roster ever needs more of one role on one
            # shift than the largest single-block demand, and an unbounded integer var
            # would leave CP-SAT searching a domain it can never use.
            n[(role, s)] = model.new_int_var(0, peak, f"n_{role.value}_{s}")

    for role in roles:
        for index, block in enumerate(blocks):
            need = demand.get((role, index), 0)
            if need <= 0:
                continue
            model.add(sum(n[(role, s)] for s in order if covers(shifts[s], block)) >= need)

    model.minimize(
        sum(
            role_cost[role] * _shift_hours(shifts[s]) * n[(role, s)]
            for role in roles
            for s in order
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError("no feasible roster covers the supplied demand")

    out: list[ShiftAssignment] = []
    for role in roles:
        for s in order:
            for k in range(int(solver.value(n[(role, s)]))):
                out.append(
                    ShiftAssignment(
                        staff=StaffId(f"{role.value}_{s:02d}_{k:02d}"),
                        role=role,
                        window=shifts[s],
                    )
                )
    return tuple(out)


__all__ = ["ShiftAssignment", "covers", "load_roster", "solve_coverage", "staffing_window"]
