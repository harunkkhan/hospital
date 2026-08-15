"""The console's slider vocabulary -> real ``Scenario`` edits (doc 07 §3.7/§7.5).

The console posts named knobs::

    workload.arrival_rate_multiplier   staffing.physician_count  staffing.porter_count
    workload.ambulance_share           staffing.nurse_count      staffing.housekeeping_count
    workload.isolation_share           staffing.tech_count
    facility.{fast_track,general,observation,resus}_bays

Not one of them is a literal path into the ``Scenario`` document. One is
*relative* (a multiplier over the base rate, so it cannot be a leaf value at
all); two are plain renames; five address a headcount that lives in **both**
``staffing.blocks`` and ``staffing.default_counts``; four address entries of the
``facility.zones`` tuple. Handed to ``apply_overlay`` verbatim they are unknown
fields, and ``extra="forbid"`` turns every slider the console has into a 422 —
the whole panel is unreachable from the real client.

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

# Which console stem names each resizable zone type. Explicit rather than derived
# from the enum value because the console says "resus", not "resus_trauma" — and
# because the list is a deliberate subset: these four are the ED *care* zones
# `data.layout.generate_floor` lays out from a `ZoneQuota`. Triage capacity is
# `facility.triage_rooms` and imaging/lab are their own scalars (all three are
# literal paths already), and the ward types (ICU, surgery, ...) belong to a
# floor spec, not to a slider on an emergency department.
_BAY_KEY_STEM: Mapping[ZoneType, str] = {
    ZoneType.FAST_TRACK: "fast_track",
    ZoneType.GENERAL: "general",
    ZoneType.OBSERVATION: "observation",
    ZoneType.RESUS_TRAUMA: "resus",
}

# One alias -> the leaf paths its compiled edit writes. Used to reject a request
# that also addresses one of those paths literally: "multiply the base rate" and
# "set the base rate" cannot both be honored, and picking one silently would make
# the result depend on dict ordering.
_ALIAS_TARGETS: Mapping[str, tuple[str, ...]] = {
    "workload.arrival_rate_multiplier": ("workload.base_rate_per_hour",),
    "workload.ambulance_share": ("workload.ambulance_fraction",),
    "workload.isolation_share": ("workload.isolation_fraction",),
    **{
        f"staffing.{role.value}_count": ("staffing.blocks", "staffing.default_counts")
        # Every role, from the core vocabulary rather than a list restated here:
        # a headcount knob is exactly "a role ``realize_staff`` can put on the
        # floor", so adding a role to ``core.StaffRole`` should not need an edit
        # in the console's translation layer to become adjustable. The two
        # non-clinical roles matter most: housekeeping turns bays over and
        # porters move people, which is what actually gates bed availability.
        for role in StaffRole
    },
    **{f"facility.{stem}_bays": ("facility.zones",) for stem in _BAY_KEY_STEM.values()},
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


def _isolation_share(base: Scenario, value: float) -> dict[str, object]:
    """The console's name for ``workload.isolation_fraction`` — a pure rename.

    Demand-side, not capacity-side, even though it reads like a facility knob: it
    changes the *mix that arrives*, and the isolation-capable bays it then
    competes for are a separate ``ZoneQuota`` field the bay sliders already
    carry. Pairing the two is the point — raising the share without raising the
    isolation allocation is precisely the squeeze an operator wants to see.
    """
    del base
    return {"workload": {"isolation_fraction": value}}


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


def _apportion(total: int, weights: list[int]) -> list[int]:
    """Split ``total`` across ``weights`` in proportion, by largest remainder.

    Chosen for one property: when ``total`` equals the sum of ``weights``, every
    share comes back **exactly** as it went in (each ``total * w // pool`` is
    ``w`` with zero remainder). So re-stating a zone type's current size rewrites
    nothing — a knob at rest cannot quietly re-cut a floor's zones, which is what
    lets the panel show a slider sitting on its base value.

    An all-zero pool has no proportions to preserve, so it splits evenly, with
    the leftover to the earliest zones — arbitrary, but deterministic, and it
    only arises for a zone type that currently has no bays at all.
    """
    n = len(weights)
    pool = sum(weights)
    if pool <= 0:
        shares = [total // n] * n
        for i in range(total - sum(shares)):
            shares[i] += 1
        return shares
    shares = [total * w // pool for w in weights]
    remainders = [(total * w) % pool for w in weights]
    order = sorted(range(n), key=lambda i: (-remainders[i], i))
    for i in order[: total - sum(shares)]:
        shares[i] += 1
    return shares


def _zone_bays(zone_type: ZoneType) -> Callable[[Scenario, float], dict[str, object]]:
    """Resize one zone type's bay allocation across the whole floor.

    The slider states the **total** number of bays of this type, not a per-zone
    figure, because a floor may allocate a type more than once — the committed
    ``scenarios/er_floor.yaml`` has two ``general`` zones — and "general bays" is
    a fact about the ED, not about which of its two general wings you meant.
    Writing the same number into each matching zone (what the fast-track-only
    version did, harmlessly, because fast track appears once) would have doubled
    the floor on the first drag of a two-zone type.

    The total is apportioned back over the matching zones in proportion to what
    they have now (:func:`_apportion`), so the relative shape of the floor is
    preserved and the base total is a fixed point.

    ``facility.zones`` is a tuple, which ``_deep_merge`` replaces wholesale, so
    the full list is rebuilt from the base with the matching entries retargeted.
    """
    key = f"facility.{_BAY_KEY_STEM[zone_type]}_bays"

    def compile_bays(base: Scenario, value: float) -> dict[str, object]:
        total = _as_count(value, key)
        quotas = base.facility.zones
        zones = [cast("dict[str, object]", quota.model_dump(mode="json")) for quota in quotas]
        matching = [i for i, quota in enumerate(quotas) if quota.zone_type is zone_type]
        if not matching:
            # The "no such zone yet" branch, revisited now that the alias spans
            # four zone types. Opening one is still the right reading for each of
            # them: general / observation / resus_trauma / fast_track are all ED
            # care zones `generate_floor` lays out from a quota, and a floor with
            # none of a type is a floor where "give me N of them" can only mean
            # "open the zone". It would NOT be right for the vocabulary's other
            # capacity knobs, which is why `_BAY_KEY_STEM` stops where it does.
            # A zero request adds nothing: "no bays of a type the floor does not
            # have" is already the state being asked for, and an empty quota
            # would only litter the tuple.
            if total > 0:
                zones.append({"zone_type": zone_type.value, "bays": total})
            return {"facility": {"zones": zones}}
        for index, share in zip(
            matching, _apportion(total, [quotas[i].bays for i in matching]), strict=True
        ):
            zones[index]["bays"] = share
            # `isolation_bays <= bays` is a ZoneQuota invariant. Shrinking the zone
            # past its isolation allocation shrinks that subset with it: the slider
            # states the zone's size, and an isolation subset is a subset of it.
            # Rejecting instead would be a dead end the operator cannot act on.
            zones[index]["isolation_bays"] = min(quotas[index].isolation_bays, share)
        return {"facility": {"zones": zones}}

    return compile_bays


_ALIASES: Mapping[str, Callable[[Scenario, float], dict[str, object]]] = {
    "workload.arrival_rate_multiplier": _arrival_rate_multiplier,
    "workload.ambulance_share": _ambulance_share,
    "workload.isolation_share": _isolation_share,
    **{f"staffing.{role.value}_count": _headcount(role) for role in StaffRole},
    **{f"facility.{stem}_bays": _zone_bays(zone) for zone, stem in _BAY_KEY_STEM.items()},
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
