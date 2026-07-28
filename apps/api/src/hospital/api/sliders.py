"""The console's slider vocabulary -> real ``Scenario`` edits (doc 07 §3.7/§7.5).

``ScenarioControls`` posts five named knobs (``apps/web`` ``SUGGESTED_KEYS``)::

    workload.arrival_rate_multiplier   staffing.nurse_count
    workload.ambulance_share           staffing.physician_count
                                       facility.fast_track_bays

Not one of them is a literal path into the ``Scenario`` document. One is
*relative* (a multiplier over the base rate, so it cannot be a leaf value at
all); one is a plain rename; two address a headcount that lives in **both**
``staffing.blocks`` and ``staffing.default_counts``; one addresses a single entry
of the ``facility.zones`` tuple. Handed to ``apply_overlay`` verbatim they are
unknown fields, and ``extra="forbid"`` turns every slider the console has into a
422 — the whole panel is unreachable from the real client.

So the aliases are compiled here, against the base scenario, into the nested
overlay ``data.scenario.apply_overlay`` already validates. A key that is not an
alias stays the literal dotted path :class:`ScenarioInline` documents, so the two
vocabularies coexist and a power user can still address a real field directly.

**No validation lives here.** Every compiled edit goes through
``Scenario.model_validate`` inside ``apply_overlay``, so an out-of-range slider
is a data-layer rejection surfaced as 422, never an API-invented rule. What this
module owns is only the *translation* — the console's names, and what they mean.

(A file beyond doc 07 §2's list for ``apps/api``: the translation is substantial
and independently testable, and folding it into ``runs.py`` would bury the
console's vocabulary inside the run resource.)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from hospital.core import StaffRole, ZoneType
from hospital.data.scenario import Scenario, apply_overlay

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# One alias -> the leaf paths its compiled edit writes. Used to reject a request
# that also addresses one of those paths literally: "multiply the base rate" and
# "set the base rate" cannot both be honored, and picking one silently would make
# the result depend on dict ordering.
_ALIAS_TARGETS: Mapping[str, tuple[str, ...]] = {
    "workload.arrival_rate_multiplier": ("workload.base_rate_per_hour",),
    "workload.ambulance_share": ("workload.ambulance_fraction",),
    "staffing.nurse_count": ("staffing.blocks", "staffing.default_counts"),
    "staffing.physician_count": ("staffing.blocks", "staffing.default_counts"),
    "facility.fast_track_bays": ("facility.zones",),
}


def _as_count(value: float, key: str) -> int:
    """A headcount/bay count: whole and finite, or rejected.

    Sliders arrive as JSON numbers, so ``4`` and ``4.0`` are the same wire value
    and both mean four. ``4.5`` is rejected rather than truncated — silently
    realizing four nurses for a request that said four-and-a-half would make the
    scenario disagree with the request that produced it. Non-finite values fail
    the same check (``inf.is_integer()`` is False), so they never reach ``int()``.
    """
    if not math.isfinite(value) or not float(value).is_integer():
        raise ValueError(f"{key} must be a whole, finite count (got {value!r})")
    return int(value)


def _arrival_rate_multiplier(base: Scenario, value: float) -> dict[str, object]:
    """Scale the base arrival rate. Relative by definition, hence not a leaf path."""
    return {"workload": {"base_rate_per_hour": base.workload.base_rate_per_hour * value}}


def _ambulance_share(base: Scenario, value: float) -> dict[str, object]:
    """The console's name for ``workload.ambulance_fraction`` — a pure rename."""
    del base
    return {"workload": {"ambulance_fraction": value}}


def _headcount(role: StaffRole) -> Callable[[Scenario, float], dict[str, object]]:
    """Set one role's headcount to ``value`` for the whole run.

    ``realize_staff`` reads a role's count as the max over the ``ShiftBlock``s
    overlapping the run window, and falls back to ``default_counts`` **only** for
    roles no overlapping block supplies. So a slider that wrote just
    ``default_counts`` would be a no-op on any scenario with explicit shifts (the
    real ones: ``scenarios/er_floor*.yaml`` schedule all five roles) — the number
    has to land in every block as well as the default.

    Writing every block flattens whatever per-shift variation the base had for
    this role, which is exactly what a single-number slider asks for; per-shift
    staffing needs a scenario file, not a slider.
    """
    key = f"staffing.{role.value}_count"

    def compile_headcount(base: Scenario, value: float) -> dict[str, object]:
        count = _as_count(value, key)
        blocks: list[object] = []
        for block in base.staffing.blocks:
            raw = cast("dict[str, object]", block.model_dump(mode="json"))
            counts = cast("dict[str, object]", raw["role_counts"])
            counts[role.value] = count
            blocks.append(raw)
        defaults: dict[str, object] = {r.value: c for r, c in base.staffing.default_counts.items()}
        defaults[role.value] = count
        return {"staffing": {"blocks": blocks, "default_counts": defaults}}

    return compile_headcount


def _fast_track_bays(base: Scenario, value: float) -> dict[str, object]:
    """Resize the fast-track zone (adding the zone if the base has none).

    ``facility.zones`` is a tuple, which ``_deep_merge`` replaces wholesale, so
    the full list is rebuilt from the base with one entry retargeted.
    """
    count = _as_count(value, "facility.fast_track_bays")
    zones: list[object] = []
    resized = False
    for quota in base.facility.zones:
        raw = cast("dict[str, object]", quota.model_dump(mode="json"))
        if quota.zone_type is ZoneType.FAST_TRACK:
            resized = True
            raw["bays"] = count
            # `isolation_bays <= bays` is a ZoneQuota invariant. Shrinking the zone
            # past its isolation allocation shrinks that subset with it: the slider
            # states the zone's size, and an isolation subset is a subset of it.
            # Rejecting instead would be a dead end the operator cannot act on.
            raw["isolation_bays"] = min(quota.isolation_bays, count)
        zones.append(raw)
    if not resized:
        zones.append({"zone_type": ZoneType.FAST_TRACK.value, "bays": count})
    return {"facility": {"zones": zones}}


_ALIASES: Mapping[str, Callable[[Scenario, float], dict[str, object]]] = {
    "workload.arrival_rate_multiplier": _arrival_rate_multiplier,
    "workload.ambulance_share": _ambulance_share,
    "staffing.nurse_count": _headcount(StaffRole.NURSE),
    "staffing.physician_count": _headcount(StaffRole.PHYSICIAN),
    "facility.fast_track_bays": _fast_track_bays,
}

SLIDER_KEYS: tuple[str, ...] = tuple(sorted(_ALIASES))


def nested_overlay(overrides: Mapping[str, float]) -> dict[str, object]:
    """Compile ``{"a.b.c": v}`` literal paths into the nested overlay mapping."""
    out: dict[str, object] = {}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor = out
        for part in parts[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ValueError(f"conflicting override paths at {part!r} in {dotted!r}")
            cursor = cast("dict[str, object]", nxt)
        cursor[parts[-1]] = value
    return out


def _reject_alias_conflicts(aliases: Mapping[str, float], literals: Mapping[str, float]) -> None:
    for alias in sorted(aliases):
        for target in _ALIAS_TARGETS[alias]:
            clashing = sorted(
                key for key in literals if key == target or key.startswith(f"{target}.")
            )
            if clashing:
                raise ValueError(
                    f"override {alias!r} already sets {target!r}; "
                    f"remove it or {clashing!r}, not both"
                )


def compile_overrides(base: Scenario, overrides: Mapping[str, float]) -> Scenario:
    """Apply console sliders and literal dotted paths to ``base``.

    Aliases are applied **one at a time**, each recompiled against the scenario
    produced by the previous one. That sequencing is load-bearing, not tidiness:
    both headcount aliases rebuild the whole ``staffing.blocks`` tuple, and a
    tuple replaces wholesale under ``_deep_merge`` — merged as independent
    fragments, setting nurses *and* physicians would silently drop one of them.

    Literal paths are applied last, as one overlay. Raises ``ValueError`` (or a
    pydantic ``ValidationError``, itself a ``ValueError``) when the data layer
    rejects an intermediate result, so callers surface one 422 either way.
    """
    aliases = {key: value for key, value in overrides.items() if key in _ALIASES}
    literals = {key: value for key, value in overrides.items() if key not in _ALIASES}
    _reject_alias_conflicts(aliases, literals)

    scenario = base
    # Sorted so the derived scenario is a function of the override *set*, not of
    # the JSON object's key order.
    for key in sorted(aliases):
        scenario = apply_overlay(scenario, _ALIASES[key](scenario, aliases[key]))
    if literals:
        scenario = apply_overlay(scenario, nested_overlay(literals))
    return scenario


__all__ = ["SLIDER_KEYS", "compile_overrides", "nested_overlay"]
