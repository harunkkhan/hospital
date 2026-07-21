"""The one event schema + append-only log — the single source of truth.

Design constraints (nuance 1.9):

* **No side channels.** Adding an event *kind* is the only way to record a new
  fact, so ``analysis`` can reconstruct everything from the log. The union below
  is therefore **fully enumerated** — a missing kind is an un-observable fact and
  a silent hole in every KPI. (The three vitals/deterioration/emergency events
  are declared now for M3.)
* **Ordering.** µs collisions are common, so ``EventEnvelope.sequence`` is the
  monotonic within-timestamp tiebreak. Everything downstream orders by
  ``(occurred_at, sequence)``.
* **Byte-stable JSONL.** Each line is canonical JSON with **sorted keys** and
  **no floats** (durations are int µs), so golden-trace hashes never flap across
  platforms.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from hospital.core.enums import Activity, ArrivalMode, DispositionKind, EsiAcuity
from hospital.core.ids import BayId, NodeId, PatientId, StaffId
from hospital.core.models import FrozenModel
from hospital.core.time import Duration, SimTime


class _Ev(FrozenModel):
    """Base for every event: the instant it occurred (µs)."""

    occurred_at: SimTime


class PatientArrived(_Ev):
    kind: Literal["patient_arrived"] = "patient_arrived"
    patient: PatientId
    mode: ArrivalMode


class TriageStarted(_Ev):
    kind: Literal["triage_started"] = "triage_started"
    patient: PatientId
    staff: StaffId


class TriageCompleted(_Ev):
    kind: Literal["triage_completed"] = "triage_completed"
    patient: PatientId
    esi: EsiAcuity


class BayRequested(_Ev):
    kind: Literal["bay_requested"] = "bay_requested"
    patient: PatientId


class BayAssigned(_Ev):
    kind: Literal["bay_assigned"] = "bay_assigned"
    patient: PatientId
    bay: BayId
    by: Literal["solver", "operator", "baseline"]


class PatientMoved(_Ev):
    kind: Literal["patient_moved"] = "patient_moved"
    patient: PatientId
    edge: tuple[NodeId, NodeId]
    seconds: Duration


class ProviderVisitStarted(_Ev):
    kind: Literal["provider_visit_started"] = "provider_visit_started"
    patient: PatientId
    staff: StaffId


class ProviderVisitCompleted(_Ev):
    kind: Literal["provider_visit_completed"] = "provider_visit_completed"
    patient: PatientId
    staff: StaffId


class NurseVisitStarted(_Ev):
    kind: Literal["nurse_visit_started"] = "nurse_visit_started"
    patient: PatientId
    staff: StaffId


class NurseVisitCompleted(_Ev):
    kind: Literal["nurse_visit_completed"] = "nurse_visit_completed"
    patient: PatientId
    staff: StaffId


class TestOrdered(_Ev):
    kind: Literal["test_ordered"] = "test_ordered"
    patient: PatientId
    activity: Activity


class TestResulted(_Ev):
    kind: Literal["test_resulted"] = "test_resulted"
    patient: PatientId
    activity: Activity


class DocumentationStarted(_Ev):
    kind: Literal["documentation_started"] = "documentation_started"
    patient: PatientId
    staff: StaffId


class DocumentationCompleted(_Ev):
    kind: Literal["documentation_completed"] = "documentation_completed"
    patient: PatientId
    staff: StaffId


class DispositionDecided(_Ev):
    kind: Literal["disposition_decided"] = "disposition_decided"
    patient: PatientId
    disposition: DispositionKind


class DischargeStarted(_Ev):
    kind: Literal["discharge_started"] = "discharge_started"
    patient: PatientId


class DischargeCompleted(_Ev):
    kind: Literal["discharge_completed"] = "discharge_completed"
    patient: PatientId


class BayCleaningStarted(_Ev):
    kind: Literal["bay_cleaning_started"] = "bay_cleaning_started"
    bay: BayId
    staff: StaffId


class BayCleaningCompleted(_Ev):
    kind: Literal["bay_cleaning_completed"] = "bay_cleaning_completed"
    bay: BayId
    staff: StaffId


class StaffMoved(_Ev):
    kind: Literal["staff_moved"] = "staff_moved"
    staff: StaffId
    edge: tuple[NodeId, NodeId]
    seconds: Duration


class StaffIdle(_Ev):
    kind: Literal["staff_idle"] = "staff_idle"
    staff: StaffId
    at: NodeId


class DisruptionInjected(_Ev):
    kind: Literal["disruption_injected"] = "disruption_injected"
    disruption: str
    detail: str = ""


class VitalsSampled(_Ev):
    kind: Literal["vitals_sampled"] = "vitals_sampled"
    patient: PatientId
    news2: int


class DeteriorationDetected(_Ev):
    kind: Literal["deterioration_detected"] = "deterioration_detected"
    patient: PatientId
    news2: int


class EmergencyRaised(_Ev):
    kind: Literal["emergency_raised"] = "emergency_raised"
    patient: PatientId


Event = Annotated[
    PatientArrived
    | TriageStarted
    | TriageCompleted
    | BayRequested
    | BayAssigned
    | PatientMoved
    | ProviderVisitStarted
    | ProviderVisitCompleted
    | NurseVisitStarted
    | NurseVisitCompleted
    | TestOrdered
    | TestResulted
    | DocumentationStarted
    | DocumentationCompleted
    | DispositionDecided
    | DischargeStarted
    | DischargeCompleted
    | BayCleaningStarted
    | BayCleaningCompleted
    | StaffMoved
    | StaffIdle
    | DisruptionInjected
    | VitalsSampled
    | DeteriorationDetected
    | EmergencyRaised,
    Field(discriminator="kind"),
]


class EventEnvelope(FrozenModel):
    """An event plus its global sequence and optional causal parent sequence."""

    event: Event
    sequence: int
    caused_by: int | None = None


_ENVELOPE_ADAPTER = TypeAdapter(EventEnvelope)


class EventLog:
    """An append-only event log. ``sim`` is the only writer; everyone else reads."""

    def __init__(self) -> None:
        self._envelopes: list[EventEnvelope] = []

    def append(self, e: Event, *, caused_by: int | None = None) -> int:
        """Append ``e`` and return its assigned global sequence number."""
        sequence = len(self._envelopes)
        self._envelopes.append(EventEnvelope(event=e, sequence=sequence, caused_by=caused_by))
        return sequence

    def __iter__(self) -> Iterator[EventEnvelope]:
        return iter(self._envelopes)

    def __len__(self) -> int:
        return len(self._envelopes)

    def ordered(self) -> tuple[EventEnvelope, ...]:
        """Envelopes in canonical ``(occurred_at, sequence)`` order."""
        return tuple(
            sorted(self._envelopes, key=lambda env: (env.event.occurred_at.root, env.sequence))
        )

    def to_jsonl(self) -> str:
        """Serialize to canonical JSONL (sorted keys, no floats), one line per envelope."""
        lines = [
            json.dumps(env.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            for env in self._envelopes
        ]
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, s: str) -> EventLog:
        """Parse canonical JSONL back into an :class:`EventLog` (sequences preserved)."""
        log = cls()
        for line in s.splitlines():
            if not line.strip():
                continue
            log._envelopes.append(_ENVELOPE_ADAPTER.validate_python(json.loads(line)))
        return log


__all__ = [
    "BayAssigned",
    "BayCleaningCompleted",
    "BayCleaningStarted",
    "BayRequested",
    "DeteriorationDetected",
    "DischargeCompleted",
    "DischargeStarted",
    "DispositionDecided",
    "DisruptionInjected",
    "DocumentationCompleted",
    "DocumentationStarted",
    "EmergencyRaised",
    "Event",
    "EventEnvelope",
    "EventLog",
    "NurseVisitCompleted",
    "NurseVisitStarted",
    "PatientArrived",
    "PatientMoved",
    "ProviderVisitCompleted",
    "ProviderVisitStarted",
    "StaffIdle",
    "StaffMoved",
    "TestOrdered",
    "TestResulted",
    "TriageCompleted",
    "TriageStarted",
    "VitalsSampled",
]
