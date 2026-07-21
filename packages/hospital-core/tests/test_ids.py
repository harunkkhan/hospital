"""Typed ids: distinct classes, so cross-type misuse is a *type* error.

The ``reportUnnecessaryTypeIgnoreComment`` directive makes the static assertion
real: if a ``BayId`` were accepted where a ``PatientId`` is required, the
``# type: ignore`` below would be *unnecessary* and pyright (strict, over tests)
would fail — so the test genuinely enforces cross-type rejection at type-check
time, not just at runtime.
"""
# pyright: reportUnnecessaryTypeIgnoreComment=true

from __future__ import annotations

from typing import assert_type

from hospital.core import BayId, PatientId


def _route_to_patient(patient: PatientId) -> str:
    return patient.root


def test_ids_are_distinct_at_runtime() -> None:
    assert PatientId("x") != BayId("x")
    assert hash(PatientId("x")) != hash(BayId("x"))
    assert_type(PatientId("x"), PatientId)


def test_same_type_equal_and_hashable() -> None:
    assert PatientId("p1") == PatientId("p1")
    lookup: dict[PatientId, int] = {PatientId("p1"): 7}
    assert lookup[PatientId("p1")] == 7


def test_cross_type_use_is_a_type_error() -> None:
    # Statically rejected: a BayId is not a PatientId. The ignore is required.
    assert _route_to_patient(BayId("b"))  # type: ignore[arg-type]
