"""The one wire-contract manifest (doc 07 §6 / nuances 7.6).

FastAPI derives ``components/schemas`` for every request/response model, and
those transitively pull the reused contract types (``core.seam.Plan``,
``core.validation.Violation``, ``core.kpi.KpiVector``, the ``OperatorAction``
union, ...). The **streamed** ``StreamFrame`` is not an HTTP body, so it — and
everything only it references, such as ``core.events.EventEnvelope`` — would be
silently MISSING from the document. :data:`WIRE_MODELS` lists exactly those
pure-wire models; :func:`merge_wire_schemas` folds their JSON schemas into the
OpenAPI document so the TS codegen sees the complete contract in one manifest.

This module also re-exports the full wire surface (imported, never redefined)
for programmatic consumers.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from hospital.api.overrides import (
    BlockEdgeAction,
    BumpPriorityAction,
    CloseBayAction,
    ExpediteCleanAction,
    ExpediteDischargeAction,
    OperatorAction,
    OverrideAccepted,
    OverrideRejected,
    OverrideRequest,
    ReassignAction,
    RerouteAction,
)
from hospital.api.runs import (
    CompareResponse,
    KpiContrast,
    RunHandle,
    RunRequest,
    ScenarioCreated,
    ScenarioInline,
    ScenarioRef,
    ScenarioSummary,
)
from hospital.api.sessions import ControlCommand, SessionState
from hospital.api.stream import (
    BayFrame,
    PatientChip,
    PendingTask,
    QueueFrame,
    StaffKinematic,
    StreamFrame,
)
from hospital.core import EventEnvelope, KpiVector, Plan, PlanItem, Violation
from hospital.data.scenario import Scenario

# The pure-wire models no HTTP body carries: streamed only, hence explicitly
# registered (everything else already reaches the OpenAPI document through the
# endpoints' request/response models).
WIRE_MODELS: tuple[type[BaseModel], ...] = (
    StreamFrame,
    StaffKinematic,
    BayFrame,
    QueueFrame,
    PatientChip,
    PendingTask,
)


def wire_schemas() -> dict[str, Any]:
    """JSON schemas (serialization mode — server -> client) for the pure-wire models."""
    _, top = models_json_schema(
        [(model, "serialization") for model in WIRE_MODELS],
        ref_template="#/components/schemas/{model}",
    )
    return dict(cast("dict[str, Any]", top.get("$defs", {})))


def merge_wire_schemas(document: dict[str, Any]) -> dict[str, Any]:
    """Fold the pure-wire schemas into an OpenAPI document's components.

    Endpoint-derived schema names win on collision (``setdefault``): both sides
    are generated from the same pydantic models, so a collision is the same
    contract arriving twice, and keeping FastAPI's copy preserves its refs.
    """
    components = cast("dict[str, Any]", document.setdefault("components", {}))
    schemas = cast("dict[str, Any]", components.setdefault("schemas", {}))
    for name, schema in sorted(wire_schemas().items()):
        schemas.setdefault(name, schema)
    return document


__all__ = [
    "WIRE_MODELS",
    "BayFrame",
    "BlockEdgeAction",
    "BumpPriorityAction",
    "CloseBayAction",
    "CompareResponse",
    "ControlCommand",
    "EventEnvelope",
    "ExpediteCleanAction",
    "ExpediteDischargeAction",
    "KpiContrast",
    "KpiVector",
    "OperatorAction",
    "OverrideAccepted",
    "OverrideRejected",
    "OverrideRequest",
    "PatientChip",
    "PendingTask",
    "Plan",
    "PlanItem",
    "QueueFrame",
    "ReassignAction",
    "RerouteAction",
    "RunHandle",
    "RunRequest",
    "Scenario",
    "ScenarioCreated",
    "ScenarioInline",
    "ScenarioRef",
    "ScenarioSummary",
    "SessionState",
    "StaffKinematic",
    "StreamFrame",
    "Violation",
    "merge_wire_schemas",
    "wire_schemas",
]
