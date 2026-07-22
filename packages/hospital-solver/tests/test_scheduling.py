"""scheduling: the M1 load_roster adapter — coverage-checked, deterministic, core-typed."""

from __future__ import annotations

import subprocess
import sys

import pytest
from _solver_fixtures import demo_compiled, staff_member, tiny_layout

from hospital.core import (
    OperatingWeek,
    Plan,
    SimTime,
    StaffId,
    StaffMember,
    StaffRole,
    TimeWindow,
    ValidationContext,
    hours,
    validate,
)
from hospital.solver.scheduling import ShiftAssignment, load_roster, solve_coverage


def _week() -> OperatingWeek:
    return OperatingWeek.one_week()


def _window(start_h: float, end_h: float) -> TimeWindow:
    return TimeWindow(start=SimTime(hours(start_h).root), end=SimTime(hours(end_h).root))


def _roster() -> tuple[tuple[StaffMember, ...], dict[StaffId, tuple[TimeWindow, ...]]]:
    """Two staff whose windows jointly cover the week (a StaffingSpec-like input)."""
    md = staff_member("md-1", StaffRole.PHYSICIAN, skills=frozenset({"md"}))
    rn = staff_member("rn-1", StaffRole.NURSE)
    windows = {
        md.id: (_window(0, 84),),
        rn.id: (_window(84, 168), _window(0, 12)),  # overlap with md is fine
    }
    return (md, rn), windows


def test_roster_round_trips_into_staffing_items() -> None:
    staff, windows = _roster()
    items = load_roster(staff, windows, _week())
    assert all(item.kind == "staffing" for item in items)
    # One item per (staff, window), keyed by a stable per-window id.
    assert [item.stable_id for item in items] == [
        "staffing:md-1:0",
        "staffing:rn-1:0",
        "staffing:rn-1:1",
    ]
    assert [item.staff.root for item in items if item.staff is not None] == [
        "md-1",
        "rn-1",
        "rn-1",
    ]


def test_deterministic_regardless_of_input_order() -> None:
    staff, windows = _roster()
    reordered = dict(reversed(list(windows.items())))
    assert load_roster(staff, windows, _week()) == load_roster(
        tuple(reversed(staff)), reordered, _week()
    )


def test_coverage_gap_raises_never_repairs() -> None:
    # Windows span only the first half of the week -> a scenario error, surfaced.
    md = staff_member("md-1", StaffRole.PHYSICIAN)
    with pytest.raises(ValueError, match="does not cover"):
        load_roster((md,), {md.id: (_window(0, 84),)}, _week())


def test_coverage_is_union_across_staff() -> None:
    # Neither staff member covers the week alone; together they do.
    staff, windows = _roster()
    items = load_roster(staff, windows, _week())
    assert len(items) == 3


def test_staff_without_windows_emit_no_items() -> None:
    staff, windows = _roster()
    idle = staff_member("hk-9", StaffRole.HOUSEKEEPING)
    items = load_roster((*staff, idle), windows, _week())
    assert all(item.staff is not None and item.staff.root != "hk-9" for item in items)


def test_roster_items_pass_the_one_validator() -> None:
    staff, windows = _roster()
    plan = Plan(items=load_roster(staff, windows, _week()))
    ctx = ValidationContext(
        layout=tiny_layout(),
        bays=(),
        staff=(),
        rules=demo_compiled(),
        staff_members=staff,
    )
    assert validate(plan, ctx) == ()


def test_no_data_import_solver_stays_a_leaf() -> None:
    # Must run in a fresh interpreter so this process's imports don't pollute it.
    code = (
        "import sys; import hospital.solver.scheduling; "
        "assert not any(m == 'hospital.data' or m.startswith('hospital.data.') "
        "for m in sys.modules), 'scheduling imported hospital.data'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_solve_coverage_is_a_deferred_stub() -> None:
    # The interface is fixed now (drop-in later); calling it is an explicit error.
    with pytest.raises(NotImplementedError, match="M3"):
        solve_coverage({}, (), role_cost={})
    # The output shape is constructible today, so the M3 swap changes no consumer.
    shift = ShiftAssignment(staff=StaffId("md-1"), role=StaffRole.PHYSICIAN, window=_window(0, 8))
    assert shift.window.duration().root == hours(8).root
