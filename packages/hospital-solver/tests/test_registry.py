"""registry: lazy import + caching; typed miss; ortools-free package import."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hospital.core import UnknownEntity
from hospital.solver import available_backends, get_backend


def test_available_backends_sorted() -> None:
    names = available_backends()
    assert names == tuple(sorted(names))
    assert set(names) == {"placement_cpsat", "placement_greedy"}


def test_get_backend_constructs_and_caches() -> None:
    a = get_backend("placement_greedy")
    b = get_backend("placement_greedy")
    assert a is b  # cached instance, reused across calls
    assert a.name == "placement_greedy"
    assert a.version == "1.0.0"


def test_get_backend_cpsat() -> None:
    backend = get_backend("placement_cpsat")
    assert backend.name == "placement_cpsat"


def test_unknown_backend_raises_typed_error() -> None:
    with pytest.raises(UnknownEntity):
        get_backend("does_not_exist")


def test_importing_solver_does_not_import_ortools() -> None:
    # Must run in a fresh interpreter: other tests import ortools in-process.
    code = (
        "import sys; import hospital.solver; "
        "assert 'ortools' not in sys.modules, 'ortools imported on package import'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_greedy_backend_usable_without_ortools() -> None:
    # HeuristicPlacement must import and be usable even if ortools is absent.
    code = (
        "import sys, builtins; _real = builtins.__import__\n"
        "def _blocked(name, *a, **k):\n"
        "    if name.split('.')[0] == 'ortools':\n"
        "        raise ImportError('ortools blocked for test')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _blocked\n"
        "from hospital.solver.heuristic import HeuristicPlacement\n"
        "assert HeuristicPlacement().name == 'placement_greedy'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
