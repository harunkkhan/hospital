"""CLI argument contract — the parser rejects unrunnable inputs at parse time."""

from __future__ import annotations

import pytest

from hospital.sim_runner.cli import build_parser


def test_reps_zero_is_a_parse_error() -> None:
    # Regression: `--reps 0` used to "succeed" with an empty report (no
    # replications, empty contrasts). argparse must reject it with a usage
    # error (exit code 2) before any simulation work starts.
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["run", "--scenario", "scenarios/er_floor.yaml", "--reps", "0"])
    assert excinfo.value.code == 2


def test_reps_negative_is_a_parse_error() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["run", "--scenario", "scenarios/er_floor.yaml", "--reps", "-3"])
    assert excinfo.value.code == 2


def test_reps_non_integer_is_a_parse_error() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["run", "--scenario", "scenarios/er_floor.yaml", "--reps", "two"])
    assert excinfo.value.code == 2


def test_reps_positive_parses_and_defaults_hold() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--scenario", "scenarios/er_floor.yaml", "--reps", "2"])
    assert args.reps == 2
    defaulted = parser.parse_args(["run", "--scenario", "scenarios/er_floor.yaml"])
    assert defaulted.reps == 5
