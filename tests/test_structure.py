"""Structural smoke test for the Phase 0 scaffold.

Layout-level only — real modules arrive in later milestones. Verifies the three
invariants that make the ``hospital.*`` namespace span distributions correctly:

  * every distribution ships ``src/hospital/<subpkg>/__init__.py`` + ``py.typed``;
  * ``hospital`` stays an *implicit* namespace package (PEP 420) — there is no
    ``src/hospital/__init__.py`` anywhere, since a stray one would turn
    ``hospital`` into a regular package and silently break cross-distribution
    imports;
  * each ``hospital.<subpkg>`` actually imports from the editable-installed venv.
"""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (distribution directory relative to the repo root, subpackage name)
DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("packages/hospital-core", "core"),
    ("packages/hospital-data", "data"),
    ("packages/hospital-solver", "solver"),
    ("packages/hospital-analysis", "analysis"),
    ("packages/hospital-sim", "sim"),
    ("packages/hospital-forecast", "forecast"),
    ("apps/sim-runner", "sim_runner"),
)

# Subpackages expected to be importable at Phase 0 (sim_runner is an app CLI
# whose importability is exercised once it has real modules).
IMPORTABLE_SUBPACKAGES: tuple[str, ...] = (
    "core",
    "data",
    "solver",
    "analysis",
    "sim",
    "forecast",
)


def test_each_distribution_ships_subpackage_and_py_typed() -> None:
    for dist_dir, subpkg in DISTRIBUTIONS:
        subpkg_dir = REPO_ROOT / dist_dir / "src" / "hospital" / subpkg
        assert (subpkg_dir / "__init__.py").is_file(), (
            f"{dist_dir}: missing src/hospital/{subpkg}/__init__.py"
        )
        assert (subpkg_dir / "py.typed").is_file(), (
            f"{dist_dir}: missing src/hospital/{subpkg}/py.typed"
        )


def test_hospital_namespace_has_no_top_level_init() -> None:
    strays = sorted(str(p) for p in REPO_ROOT.glob("*/*/src/hospital/__init__.py"))
    assert strays == [], (
        "hospital must be an implicit namespace package (PEP 420); "
        f"found stray top-level __init__.py: {strays}"
    )


def test_subpackages_import() -> None:
    for subpkg in IMPORTABLE_SUBPACKAGES:
        module = importlib.import_module(f"hospital.{subpkg}")
        assert module.__name__ == f"hospital.{subpkg}"
