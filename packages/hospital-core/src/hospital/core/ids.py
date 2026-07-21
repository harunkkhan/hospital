"""Typed string identities that cannot be confused for one another.

Each concrete id is a **distinct** ``RootModel[str]`` subclass, so passing a
``BayId`` where a ``PatientId`` is required is a *type* error (pyright rejects
it), not a runtime surprise. The custom ``__hash__`` supports hot-path dict
lookups (``bay_by_id``, ``staff_by_id``) and set membership.

Ids are minted deterministically by the data generator (``p0001``, ``bay-03``,
…), never via ``uuid4`` — RNG substream keys embed the patient id, so a random
id would break replay and CRN (nuance 1.4).

Implementation note: ``__hash__`` is (re)declared on **every** concrete subclass,
not just ``TypedId``. pyright models pydantic classes via ``dataclass_transform``
and re-synthesizes ``__hash__ = None`` on each model subclass, so an inherited
``__hash__`` is not seen as making a *subclass* hashable — it must be defined on
the leaf. The hash mixes in the class name so ids of different types never
collide even with equal string payloads.
"""

from __future__ import annotations

from pydantic import RootModel


def _id_hash(obj: TypedId) -> int:
    return hash((type(obj).__name__, obj.root))


class TypedId(RootModel[str]):
    """Base for every typed string id. Frozen and hashable."""

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return _id_hash(self)

    def __str__(self) -> str:
        return self.root


class PatientId(TypedId):
    """Identifies a patient."""

    def __hash__(self) -> int:
        return _id_hash(self)


class BayId(TypedId):
    """Identifies a bay."""

    def __hash__(self) -> int:
        return _id_hash(self)


class ZoneId(TypedId):
    """Identifies a zone."""

    def __hash__(self) -> int:
        return _id_hash(self)


class NodeId(TypedId):
    """Identifies a vertex in the ``RouteGraph``."""

    def __hash__(self) -> int:
        return _id_hash(self)


class StaffId(TypedId):
    """Identifies a staff member."""

    def __hash__(self) -> int:
        return _id_hash(self)


class TaskId(TypedId):
    """Identifies a unit of pending work."""

    def __hash__(self) -> int:
        return _id_hash(self)


class RunId(TypedId):
    """Identifies a single simulation run/replication."""

    def __hash__(self) -> int:
        return _id_hash(self)


__all__ = [
    "BayId",
    "NodeId",
    "PatientId",
    "RunId",
    "StaffId",
    "TaskId",
    "TypedId",
    "ZoneId",
]
