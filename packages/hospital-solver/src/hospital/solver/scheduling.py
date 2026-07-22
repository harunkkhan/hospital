"""Staff scheduling — M1 input adapter; covering MIP deferred (doc 03 §4.8).

**M1 — ``load_roster``.** Staffing is a scenario *input* (assumption 19): the sim
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

**Later (M3+) — ``solve_coverage``.** A covering MIP over forecast demand (passed
in as a plain ``Mapping`` so ``solver`` stays a leaf). Deferred; the stub fixes
the interface so the MIP is a drop-in behind the same ``staffing`` contract.
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


def solve_coverage(
    demand: Mapping[tuple[StaffRole, int], int],
    shifts: tuple[TimeWindow, ...],
    *,
    role_cost: Mapping[StaffRole, int],
) -> tuple[ShiftAssignment, ...]:
    """Covering MIP over forecast demand (M3+). Deferred — interface fixed here."""
    raise NotImplementedError(
        "solve_coverage (covering MIP) is deferred to M3+; M1 uses load_roster"
    )


__all__ = ["ShiftAssignment", "load_roster", "solve_coverage", "staffing_window"]
