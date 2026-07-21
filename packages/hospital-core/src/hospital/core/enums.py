"""Closed vocabularies shared across every package.

Two deliberate design points (nuance 1.5):

* **Acuity direction is inverted.** ``EsiAcuity`` is an ``IntEnum`` where
  **1 = most critical** and 5 = least. Any sequencing/placement *weight* must
  therefore be a **decreasing** function of the enum value (higher weight for a
  *lower* ESI number). This is the classic sign trap — code that gets it
  backwards still "runs" but prioritizes exactly wrong. Compute weights via
  :func:`EsiAcuity.priority_weight` (or otherwise invert explicitly) and never
  use the raw value as a weight.
* ``Activity`` is the **join key** between ``sim.physics.service_times`` and
  ``forecast.service_time``; keeping it here is what stops the two service-time
  vocabularies from drifting. Adding an activity is a versioned contract change
  touching both.

Acuity is an ``IntEnum`` because it needs ordering; the rest are ``StrEnum`` so
they serialize to stable, human-readable strings in the JSONL (no floats).
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class EsiAcuity(IntEnum):
    """Emergency Severity Index. **1 = most critical**, 5 = least critical."""

    ESI1 = 1
    ESI2 = 2
    ESI3 = 3
    ESI4 = 4
    ESI5 = 5

    def priority_weight(self) -> int:
        """Sequencing weight — **inverse** of the ESI value (ESI1 -> 5, ESI5 -> 1).

        Use this instead of the raw enum value anywhere a "more critical =>
        higher priority" ordering is needed, so the inversion is explicit.
        """
        return 6 - int(self)


class ArrivalMode(StrEnum):
    """How a patient reached the ED."""

    WALK_IN = "walk_in"
    AMBULANCE = "ambulance"


class ZoneType(StrEnum):
    """Functional type of a care zone."""

    TRIAGE = "triage"
    GENERAL = "general"
    RESUS_TRAUMA = "resus_trauma"
    FAST_TRACK = "fast_track"
    OBSERVATION = "observation"
    IMAGING = "imaging"
    LAB = "lab"


class BayStatus(StrEnum):
    """Dynamic bay status (lives in ``sim.physics.world``, never in ``core.Bay``)."""

    FREE = "free"
    OCCUPIED = "occupied"
    CLEANING = "cleaning"
    CLOSED = "closed"


class StaffRole(StrEnum):
    """A staff member's role."""

    PHYSICIAN = "physician"
    NURSE = "nurse"
    TECH = "tech"
    PORTER = "porter"
    HOUSEKEEPING = "housekeeping"


class DispositionKind(StrEnum):
    """Terminal disposition of a patient."""

    DISCHARGE = "discharge"
    ADMIT = "admit"
    TRANSFER = "transfer"


class Activity(StrEnum):
    """Unit of work — the service-time/forecast join key. Adding one is a contract change."""

    TRIAGE = "triage"
    PROVIDER_VISIT = "provider_visit"
    NURSE_VISIT = "nurse_visit"
    IMAGING = "imaging"
    LAB = "lab"
    DOCUMENTATION = "documentation"
    DISCHARGE = "discharge"
    CLEANING = "cleaning"
    TRANSPORT = "transport"


__all__ = [
    "Activity",
    "ArrivalMode",
    "BayStatus",
    "DispositionKind",
    "EsiAcuity",
    "StaffRole",
    "ZoneType",
]
