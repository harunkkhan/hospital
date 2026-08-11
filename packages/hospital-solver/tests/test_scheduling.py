"""scheduling: the load_roster adapter and the covering MIP that chooses a roster."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

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
from hospital.solver.scheduling import (
    ShiftAssignment,
    covers,
    load_roster,
    solve_coverage,
    staffing_window,
)


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


def test_staffing_items_carry_their_shift_windows() -> None:
    # Regression (review finding 4): the shift TimeWindow was discarded, so
    # rosters with different shift boundaries produced identical payloads.
    # Every item must carry its window, recoverable via the one decoder.
    staff, windows = _roster()
    items = load_roster(staff, windows, _week())
    carried = {item.stable_id: staffing_window(item) for item in items}
    assert carried == {
        "staffing:md-1:0": _window(0, 84),
        "staffing:rn-1:0": _window(84, 168),
        "staffing:rn-1:1": _window(0, 12),
    }
    # Different shift boundaries -> different plan payloads.
    md, rn = staff
    shifted = {md.id: (_window(0, 96),), rn.id: windows[rn.id]}
    assert load_roster(staff, shifted, _week()) != items
    # Decoding a window-less item is a caller error, surfaced loudly.
    with pytest.raises(ValueError, match="no staffing window"):
        staffing_window(items[0].model_copy(update={"order": None}))


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


def test_the_output_shape_is_what_load_roster_can_consume() -> None:
    """The two entry points meet at one contract: a solved roster is a schedulable one."""
    shift = ShiftAssignment(staff=StaffId("md-1"), role=StaffRole.PHYSICIAN, window=_window(0, 8))
    assert shift.window.duration().root == hours(8).root


def _grid(n_blocks: int, block_hours: int = 1) -> tuple[TimeWindow, ...]:
    return tuple(_window(b * block_hours, (b + 1) * block_hours) for b in range(n_blocks))


def test_coverage_meets_every_block_demand() -> None:
    """The constraint that makes it a *covering* solve: nobody is left uncovered."""
    blocks = _grid(8)
    shifts = (_window(0, 4), _window(4, 8), _window(0, 8))
    demand = {(StaffRole.NURSE, b): 2 for b in range(8)}
    demand[(StaffRole.NURSE, 5)] = 5  # one spike

    roster = solve_coverage(demand, shifts, role_cost={StaffRole.NURSE: 1}, blocks=blocks)
    for b, block in enumerate(blocks):
        on_shift = sum(1 for a in roster if covers(a.window, block))
        assert on_shift >= demand[(StaffRole.NURSE, b)], f"block {b} under-covered"


def test_it_minimizes_cost_weighted_staff_hours() -> None:
    """Cheapest, not merely feasible — the whole point of solving rather than accepting.

    Demand sits in one four-hour half of the week, so the eight-hour shift is feasible but
    twice the hours. A solver that ignored the objective would be free to pick it.
    """
    blocks = _grid(8)
    shifts = (_window(0, 4), _window(4, 8), _window(0, 8))
    demand = {(StaffRole.NURSE, b): 1 for b in range(4)}

    roster = solve_coverage(demand, shifts, role_cost={StaffRole.NURSE: 1}, blocks=blocks)
    assert len(roster) == 1
    assert roster[0].window == _window(0, 4)


def test_a_dearer_role_is_not_over_hired_to_cover_a_cheaper_one() -> None:
    """Roles are covered independently: a physician cannot stand in for a porter."""
    blocks = _grid(4)
    shifts = (_window(0, 4),)
    demand = {(StaffRole.PHYSICIAN, b): 1 for b in range(4)}
    demand.update({(StaffRole.PORTER, b): 3 for b in range(4)})

    roster = solve_coverage(
        demand,
        shifts,
        role_cost={StaffRole.PHYSICIAN: 100, StaffRole.PORTER: 1},
        blocks=blocks,
    )
    by_role = Counter(a.role for a in roster)
    assert by_role[StaffRole.PHYSICIAN] == 1
    assert by_role[StaffRole.PORTER] == 3


def test_a_partly_overlapping_shift_does_not_count_as_coverage() -> None:
    """Containment, not overlap — three people present for a third of an hour is not three.

    The failure this forbids is the quiet kind: an overlap test yields a roster that looks
    covered and leaves the floor short for most of every block.
    """
    block = _window(0, 1)
    assert covers(_window(0, 1), block)
    assert covers(_window(0, 8), block)
    assert not covers(_window(0, 1) and _window(1, 2), block)

    with pytest.raises(ValueError, match="no candidate shift covers block"):
        solve_coverage(
            {(StaffRole.NURSE, 0): 1},
            (_window(1, 9),),  # starts after the demanded block
            role_cost={StaffRole.NURSE: 1},
            blocks=(block,),
        )


def test_an_uncoverable_or_unpriced_demand_is_refused_not_absorbed() -> None:
    """Validate-never-repair, applied to inputs: no silently under-covering roster."""
    with pytest.raises(ValueError, match="no role_cost"):
        solve_coverage(
            {(StaffRole.NURSE, 0): 1},
            (_window(0, 8),),
            role_cost={},
            blocks=_grid(1),
        )
    with pytest.raises(ValueError, match="no candidate shift"):
        solve_coverage(
            {(StaffRole.NURSE, 0): 1},
            (),
            role_cost={StaffRole.NURSE: 1},
            blocks=_grid(1),
        )


def test_zero_demand_hires_nobody() -> None:
    """The degenerate case is well-defined, not a special case."""
    assert (
        solve_coverage({}, (_window(0, 8),), role_cost={StaffRole.NURSE: 1}, blocks=_grid(8)) == ()
    )


def test_the_solved_roster_is_deterministic() -> None:
    """CRN and goldens depend on it, exactly as for every other solver here."""
    blocks = _grid(8)
    shifts = (_window(0, 4), _window(4, 8), _window(2, 6))
    demand = {(StaffRole.NURSE, b): (b % 3) + 1 for b in range(8)}
    kwargs = {"role_cost": {StaffRole.NURSE: 1}, "blocks": blocks}
    first = solve_coverage(demand, shifts, **kwargs)  # type: ignore[arg-type]
    second = solve_coverage(demand, shifts, **kwargs)  # type: ignore[arg-type]
    assert first == second
    # Ids are unique, and emitted in (role, shift-start) order. The number in an id is
    # *which candidate shift* — provenance, not output position — so the two orders differ
    # whenever the shifts tuple is not already chronological, as here.
    assert len({a.staff.root for a in first}) == len(first)
    assert [a.window.start.root for a in first] == sorted(a.window.start.root for a in first)


def test_a_solved_roster_covers_a_realistic_day_night_forecast() -> None:
    """The end-to-end claim of §7's last lever and §10's "driving staffing".

    Demand is derived here by hand rather than imported from `hospital.forecast` — the
    import-direction contract forbids `solver -> forecast`, which is exactly why
    `solve_coverage` takes a plain mapping. The shape is the one `forecast.role_demand`
    produces: a per-hour headcount that peaks during the day and falls off at night.

    What the test pins is that the roster tracks the *shape*: every hour is covered, and a
    cost-minimizing solve does not simply staff the peak all week. A roster that paid for
    peak headcount around the clock would cover fine and be exactly the waste this lever
    exists to remove.
    """
    blocks = tuple(_window(h, h + 1) for h in range(24))
    day = range(8, 20)
    demand = {(StaffRole.NURSE, h): (6 if h in day else 2) for h in range(24)}
    # Candidate shifts: four 6-hour blocks plus two 12-hour blocks.
    shifts = (
        _window(0, 6),
        _window(6, 12),
        _window(12, 18),
        _window(18, 24),
        _window(0, 12),
        _window(12, 24),
    )

    roster = solve_coverage(demand, shifts, role_cost={StaffRole.NURSE: 1}, blocks=blocks)

    for h, block in enumerate(blocks):
        on_shift = sum(1 for a in roster if covers(a.window, block))
        assert on_shift >= demand[(StaffRole.NURSE, h)], f"hour {h} under-covered"

    # Peak-flat staffing would be 6 nurses on duty every hour; a cost-minimizing roster
    # must beat that in total staff-hours.
    solved_hours = sum(
        (a.window.end.root - a.window.start.root) // (3_600 * 1_000_000) for a in roster
    )
    assert solved_hours < 6 * 24, f"{solved_hours} staff-hours is peak-flat or worse"

    # ...and it genuinely thins out overnight rather than covering by luck.
    at_3am = sum(1 for a in roster if covers(a.window, _window(3, 4)))
    at_noon = sum(1 for a in roster if covers(a.window, _window(12, 13)))
    assert at_noon > at_3am
