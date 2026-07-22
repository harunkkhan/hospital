"""Staff scheduling — M1 input adapter; covering MIP deferred (doc 03 §4.8).

**M1 — ``load_roster``.** Staffing is a scenario *input* (assumption 19): the sim
supplies per-staff shift windows as **core types** and this adapter validates
they cover the operating week and emits ``kind="staffing"`` plan items. It
deliberately does **not** import ``data.StaffingSpec`` — that would be a forbidden
sideways ``solver → data`` import. *You set staffing; we measure.*

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
        for index, _window in enumerate(windows.get(member.id, ())):
            items.append(
                PlanItem(
                    stable_id=f"staffing:{member.id.root}:{index}",
                    kind="staffing",
                    staff=member.id,
                )
            )
    return tuple(items)


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


__all__ = ["ShiftAssignment", "load_roster", "solve_coverage"]
