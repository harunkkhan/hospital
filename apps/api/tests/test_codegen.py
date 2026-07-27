"""The TS-contract artifact: byte-stable regeneration, one document, tagged unions.

The drift gate (doc 07 §6.2): any pydantic contract change not regenerated into
``apps/api/schema/openapi.json`` fails here — the mechanical guarantee that the
browser's contract equals the backend's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from _api_fixtures import make_app
from hospital.api.codegen import SCHEMA_PATH, build_schema_document, render_schema

if TYPE_CHECKING:
    from pathlib import Path


def test_committed_schema_artifact_is_current() -> None:
    """Regenerating must be byte-stable against the committed artifact.

    On failure: `uv run python -m hospital.api.codegen dump-schema --out
    apps/api/schema/openapi.json` and review the contract change.
    """
    assert SCHEMA_PATH.is_file(), f"missing schema artifact: {SCHEMA_PATH}"
    assert render_schema() == SCHEMA_PATH.read_text()


def test_regeneration_is_deterministic() -> None:
    assert render_schema() == render_schema()


def test_pure_wire_models_are_registered() -> None:
    """StreamFrame is not an HTTP body — wire.py must register it explicitly."""
    schemas: dict[str, Any] = build_schema_document()["components"]["schemas"]
    for name in (
        "StreamFrame",
        "StaffKinematic",
        "BayFrame",
        "QueueFrame",
        "PatientChip",
        "EventEnvelope",
        "Plan",
        "PlanItem",
        "Violation",
        "KpiVector",
    ):
        assert name in schemas, name


def test_operator_action_survives_as_a_tagged_union() -> None:
    """`Field(discriminator="kind")` must reach the schema so TS narrows on kind."""
    schemas: dict[str, Any] = build_schema_document()["components"]["schemas"]
    action: dict[str, Any] = schemas["OverrideRequest"]["properties"]["action"]
    assert action["discriminator"]["propertyName"] == "kind"
    mapped = set(action["discriminator"]["mapping"])
    assert mapped == {
        "reassign",
        "bump_priority",
        "reroute",
        "expedite_clean",
        "expedite_discharge",
        "close_bay",
        "block_edge",
    }
    # The event union inside the frame tail is tagged the same way.
    event: dict[str, Any] = schemas["EventEnvelope"]["properties"]["event"]
    assert event["discriminator"]["propertyName"] == "kind"


def test_served_openapi_equals_the_codegen_dump(tmp_path: Path) -> None:
    """One schema, two exits (doc 07 nuances 7.1)."""
    with TestClient(make_app(tmp_path)) as client:
        served = client.get("/openapi.json")
        assert served.status_code == 200
        assert served.json() == build_schema_document()
