"""Integer-scaled physical units so distance/time math is exact.

Distances are integer **centimetres**; speeds are integer **cm/s**. Floats
appear only at the display edge (``to_metres``) and are never fed back into
logic — this is what stops accumulated float drift from making two identical
runs diverge.

``walk_duration`` reuses :func:`hospital.core.time.seconds` for its float→µs
step, so it shares the one banker's-rounding rule with the rest of the repo (a
per-edge rounding difference would compound into a false ``staff_minutes_walked``
delta — nuance 1.2).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, RootModel

from hospital.core.time import Duration, seconds


class Distance(RootModel[Annotated[int, Field(ge=0)]]):
    """A non-negative distance in centimetres."""

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.root)

    def to_metres(self) -> float:
        """Display-only float metres — never fed back into logic."""
        return self.root / 100.0


class WalkSpeed(RootModel[Annotated[int, Field(gt=0)]]):
    """A strictly-positive walking speed in centimetres per second."""

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.root)


def walk_duration(distance: Distance, speed: WalkSpeed) -> Duration:
    """Time to traverse ``distance`` at ``speed``, banker's-rounded to µs.

    Uses the same rounding as :func:`hospital.core.time.seconds`, so edge
    traversal times never drift relative to service times.
    """
    return seconds(distance.root / speed.root)


__all__ = ["Distance", "WalkSpeed", "walk_duration"]
