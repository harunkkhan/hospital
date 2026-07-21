"""The frozen, validated base every boundary type subclasses.

``FrozenModel`` is the deliberate middle ground between plain dataclasses (no
validation) and mutable pydantic models (no hashing/determinism): it is
immutable, validated, and structurally hashable, which is exactly what the
determinism/provenance machinery leans on.

Invariants baked in here (see nuance 1.1):

* ``frozen=True`` + ``extra="forbid"`` means the declared field set *is* the
  contract — adding a field is a deliberate, reviewable change.
* ``validate_default=True`` validates defaults so a bad default can't slip in.
* ``revalidate_instances="subclass-instances"`` re-validates a subclass passed
  where a base is expected (safety over a small validation cost).

Because a frozen model is hashable **only if every field is hashable**, boundary
collections must be ``tuple``/``frozenset`` — never ``list``/``dict`` (a ``list``
default silently makes the model unhashable and breaks its use as a dict key).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable, validated, hashable base for every value that crosses a seam."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="subclass-instances",
    )


__all__ = ["FrozenModel"]
