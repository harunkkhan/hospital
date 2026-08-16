"""The console's slider vocabulary -> real ``Scenario`` edits (doc 07 §3.7/§7.5).

The Scenario Lab pushes an ED the only two ways an OR model can be pushed —
**demand** (how much work arrives) and **supply** (how much of it can be met) —
and this module is where those names become edits::

    demand     workload.arrival_rate_multiplier  workload.ambulance_share
               workload.isolation_share
    staffing   staffing.physician_count  staffing.nurse_count  staffing.tech_count
    capacity   facility.{fast_track,general,observation,resus}_bays
               facility.triage_rooms  facility.imaging_suites  facility.lab_stations
               staffing.porter_count  staffing.housekeeping_count

**Supply here is capacity, not consumables.** Nothing in this simulator tracks
gloves, saline or kits — there is no inventory state, no consumption, no
reorder — so there is no consumable knob to publish and inventing one would be a
``core``/``sim`` fiction dressed as a setting. What actually gates a patient is a
bay to put them in, a room/suite/station to work them up in, and the labour that
turns a bay over: housekeeping cleans it, porters move people. Those two roles
therefore sit in the *capacity* group rather than with the clinicians — a bay
nobody has cleaned is not capacity you have.

Most of these names are not literal paths into the ``Scenario`` document. One is
*relative* (a multiplier over the base rate, so it cannot be a leaf value at
all); two are plain renames; five address a headcount that lives in **both**
``staffing.blocks`` and ``staffing.default_counts``; four address entries of the
``facility.zones`` tuple. Handed to ``apply_overlay`` verbatim they are unknown
fields, and ``extra="forbid"`` turns every one of them into a 422 — the whole
panel unreachable from the real client.

So the aliases are compiled here, against the base scenario, into the nested
overlay ``data.scenario.apply_overlay`` already validates. A key that is not an
alias stays the literal dotted path :class:`ScenarioInline` documents, so the two
vocabularies coexist and a power user can still address a real field directly.
``facility.imaging_suites``/``lab_stations``/``triage_rooms`` are exactly that
case and deliberately have **no** alias: each is a single scalar leaf whose
console name and document path already coincide, so an alias would be a second
name for one field and a second thing to keep in sync. They are still in
:data:`_CATALOGUE`, because the catalogue is the console's *vocabulary*, not the
alias table — it must carry every knob the panel draws, translated or not.

Every knob also carries an **inverse** (``read``): what it is worth in a given
base scenario. A slider that cannot say where the base already sits has to open
somewhere invented, and the first drag then rewrites a field the operator never
meant to touch.

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
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, HTTPException, Request

from hospital.core import FrozenModel, StaffRole, TimeWindow, ZoneType
from hospital.data.layout import generate_floor
from hospital.data.scenario import Scenario, apply_overlay, realize_staff

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import FastAPI

    # Type-only, and it must stay that way: ``runs`` imports ``compile_overrides``
    # from here at runtime, so a runtime import back would close the cycle.
    from hospital.api.runs import ScenarioStore

router = APIRouter()

SliderGroup = Literal["demand", "staffing", "capacity"]


class SliderSpec(FrozenModel):
    """One knob of the console's parameter vocabulary, read against a base scenario.

    ``min``/``max``/``step`` are **affordances**, not validation: they live here,
    next to the alias they bound, so the panel cannot drift from what the data
    layer accepts, but the server still judges every submitted value through
    ``apply_overlay`` (see :func:`compile_overrides`). Widening a range here can
    therefore only widen what the UI *offers* — never what the model accepts.

    ``value`` is what this knob is worth in the base the catalogue was read
    against, so the panel opens on the truth. For the relative knob it is
    always 1.0 (see :func:`_read_multiplier`).
    """

    key: str
    label: str
    group: SliderGroup
    min: float
    max: float
    step: float
    unit: str
    value: float


class SliderCatalogue(FrozenModel):
    """The knobs the console may draw, and where ``scenario`` currently sits on each."""

    scenario: str
    knobs: tuple[SliderSpec, ...]


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


# --------------------------------------------------------------- compile (name -> edit)
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

    Chosen for one property the alias depends on: when ``total`` equals the sum
    of ``weights``, every share comes back **exactly** as it went in (each
    ``total * w // pool`` is ``w`` with zero remainder). That is what makes the
    catalogue's ``read`` a true inverse — dropping a slider back onto the value
    the catalogue reported rewrites nothing, so an untouched knob cannot quietly
    re-cut a floor's zones.

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


# ------------------------------------------------------------------ read (edit -> name)
@dataclass(frozen=True)
class _BaseReading:
    """A base scenario plus the one derived reading that is expensive to take.

    ``realized_staff`` is computed once per catalogue rather than once per knob:
    it costs a ``generate_floor``, and five headcount knobs asking independently
    would pay for five floors to answer five questions about one roster.
    """

    scenario: Scenario
    realized_staff: Mapping[StaffRole, int]


def _read_base(scenario: Scenario) -> _BaseReading:
    """Read the derived facts the knobs need — by ASKING ``data``, not restating it.

    The headcount a scenario really puts on the floor is not
    ``staffing.default_counts``: ``realize_staff`` takes the max over the
    ``ShiftBlock``s overlapping the run window and falls back to the defaults
    only for roles no block supplies. Re-deriving that rule here would be a
    second copy of a ``data`` rule, free to drift from the one the run actually
    uses — so this calls ``realize_staff`` itself and counts the roster it
    returns. The floor generation exists only because ``realize_staff`` homes
    staff to stations; the ground floor's stations are enough to count, and the
    count does not depend on which station anybody homed to.
    """
    horizon = scenario.workload.horizon
    window = TimeWindow(start=horizon.start, end=horizon.end)
    roster = realize_staff(scenario.staffing, generate_floor(scenario.facility), window)
    return _BaseReading(scenario=scenario, realized_staff=Counter(m.role for m in roster))


def _read_multiplier(reading: _BaseReading) -> float:
    """A relative knob reads 1.0 against its own base — always, by construction.

    ``workload.arrival_rate_multiplier`` scales whatever the base rate is, so
    "the multiplier this scenario currently has" is not a fact the document
    stores; the scaling is already baked into ``base_rate_per_hour``. Read
    against the scenario a panel is editing, the honest answer is "you have not
    scaled anything yet" — 1.0. (Read against a scenario *derived* by a 2.0 drag,
    it is 1.0 again, and correctly so: relative to the thing now in front of you,
    nothing is scaled. The panel keeps its own drag position; the catalogue only
    tells it where the base sits.)
    """
    del reading
    return 1.0


def _read_ambulance_share(reading: _BaseReading) -> float:
    return reading.scenario.workload.ambulance_fraction


def _read_isolation_share(reading: _BaseReading) -> float:
    return reading.scenario.workload.isolation_fraction


def _read_headcount(role: StaffRole) -> Callable[[_BaseReading], float]:
    def read(reading: _BaseReading) -> float:
        return float(reading.realized_staff.get(role, 0))

    return read


def _read_zone_bays(zone_type: ZoneType) -> Callable[[_BaseReading], float]:
    def read(reading: _BaseReading) -> float:
        return float(
            sum(q.bays for q in reading.scenario.facility.zones if q.zone_type is zone_type)
        )

    return read


def _read_literal(dotted: str) -> Callable[[_BaseReading], float]:
    """Read a knob that needs no alias straight off the document it names."""

    def read(reading: _BaseReading) -> float:
        cursor: object = reading.scenario.model_dump(mode="json")
        for part in dotted.split("."):
            if not isinstance(cursor, Mapping):
                raise ValueError(f"{dotted!r} does not address a Scenario leaf")
            cursor = cast("Mapping[str, object]", cursor)[part]
        if isinstance(cursor, bool) or not isinstance(cursor, int | float):
            raise ValueError(f"{dotted!r} is not a numeric leaf (got {cursor!r})")
        return float(cursor)

    return read


# ------------------------------------------------------------------------- the vocabulary
@dataclass(frozen=True)
class _Alias:
    """A console name that is not a leaf path: how to write it, and how to read it back.

    ``targets`` are the leaf paths the compiled edit writes, used to reject a
    request that also addresses one of them literally: "multiply the base rate"
    and "set the base rate" cannot both be honored, and picking one silently
    would make the result depend on dict ordering.
    """

    compile: Callable[[Scenario, float], dict[str, object]]
    read: Callable[[_BaseReading], float]
    targets: tuple[str, ...]


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

_STAFFING_TARGETS = ("staffing.blocks", "staffing.default_counts")


def _headcount_alias(role: StaffRole) -> _Alias:
    return _Alias(compile=_headcount(role), read=_read_headcount(role), targets=_STAFFING_TARGETS)


def _zone_bays_alias(zone_type: ZoneType) -> _Alias:
    return _Alias(
        compile=_zone_bays(zone_type),
        read=_read_zone_bays(zone_type),
        targets=("facility.zones",),
    )


_ALIASES: Mapping[str, _Alias] = {
    "workload.arrival_rate_multiplier": _Alias(
        compile=_arrival_rate_multiplier,
        read=_read_multiplier,
        targets=("workload.base_rate_per_hour",),
    ),
    "workload.ambulance_share": _Alias(
        compile=_ambulance_share,
        read=_read_ambulance_share,
        targets=("workload.ambulance_fraction",),
    ),
    "workload.isolation_share": _Alias(
        compile=_isolation_share,
        read=_read_isolation_share,
        targets=("workload.isolation_fraction",),
    ),
    # Every role, from the core vocabulary rather than a list restated here: a
    # headcount knob is exactly "a role ``realize_staff`` can put on the floor",
    # so adding a role to ``core.StaffRole`` should not need an edit in the
    # console's translation layer to become adjustable. (It does still need a
    # `_CATALOGUE` entry to become *visible* — a label and a range are editorial,
    # and the coverage test below fails loudly until someone writes them.)
    **{f"staffing.{role.value}_count": _headcount_alias(role) for role in StaffRole},
    **{
        f"facility.{stem}_bays": _zone_bays_alias(zone_type)
        for zone_type, stem in _BAY_KEY_STEM.items()
    },
}

SLIDER_KEYS: tuple[str, ...] = tuple(sorted(_ALIASES))


@dataclass(frozen=True)
class _Knob:
    """A catalogue entry: what the panel draws, minus the value it is read at."""

    key: str
    label: str
    group: SliderGroup
    minimum: float
    maximum: float
    step: float
    unit: str


# The published vocabulary, in the order the panel draws it. Ranges are DATA and
# they live here, beside the alias each one bounds, so a range cannot drift from
# the translation it constrains — the console holds no parameter list of its own.
# They are affordances only: the server still validates every submitted value
# through ``apply_overlay`` (a share above 1.0 is rejected by
# ``WorkloadSpec.ambulance_fraction``, not by anything here), and it deliberately
# does NOT reject a value merely for falling outside a range below. Enforcing the
# affordance would invent a rule the data layer does not have — and a caller can
# always address the same field by its literal path anyway.
_CATALOGUE: tuple[_Knob, ...] = (
    # -- demand: how much work walks (or rolls) through the door -------------
    _Knob(
        key="workload.arrival_rate_multiplier",
        label="Arrival rate",
        group="demand",
        minimum=0.25,
        maximum=3.0,
        step=0.05,
        # A multiplication sign, deliberately: this knob's unit IS "times the
        # base rate", and RUF001 only flags it for being confusable with "x".
        unit="× base",  # noqa: RUF001
    ),
    _Knob(
        key="workload.ambulance_share",
        label="Ambulance arrivals",
        group="demand",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        unit="share",
    ),
    _Knob(
        key="workload.isolation_share",
        label="Isolation required",
        group="demand",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        unit="share",
    ),
    # -- staffing: the clinicians who see patients ---------------------------
    _Knob(
        key="staffing.physician_count",
        label="Physicians",
        group="staffing",
        minimum=0,
        maximum=24,
        step=1,
        unit="on duty",
    ),
    _Knob(
        key="staffing.nurse_count",
        label="Nurses",
        group="staffing",
        minimum=0,
        maximum=40,
        step=1,
        unit="on duty",
    ),
    _Knob(
        key="staffing.tech_count",
        label="Techs",
        group="staffing",
        minimum=0,
        maximum=24,
        step=1,
        unit="on duty",
    ),
    # -- capacity: bays, rooms, and the labour that turns them over ----------
    _Knob(
        key="facility.general_bays",
        label="General bays",
        group="capacity",
        minimum=0,
        maximum=80,
        step=1,
        unit="bays",
    ),
    _Knob(
        key="facility.observation_bays",
        label="Observation bays",
        group="capacity",
        minimum=0,
        maximum=60,
        step=1,
        unit="bays",
    ),
    _Knob(
        key="facility.resus_bays",
        label="Resus / trauma bays",
        group="capacity",
        minimum=0,
        maximum=30,
        step=1,
        unit="bays",
    ),
    _Knob(
        key="facility.fast_track_bays",
        label="Fast-track bays",
        group="capacity",
        minimum=0,
        maximum=40,
        step=1,
        unit="bays",
    ),
    _Knob(
        key="facility.triage_rooms",
        label="Triage rooms",
        group="capacity",
        minimum=0,
        maximum=20,
        step=1,
        unit="rooms",
    ),
    _Knob(
        key="facility.imaging_suites",
        label="Imaging suites",
        group="capacity",
        minimum=0,
        maximum=12,
        step=1,
        unit="suites",
    ),
    _Knob(
        key="facility.lab_stations",
        label="Lab stations",
        group="capacity",
        minimum=0,
        maximum=12,
        step=1,
        unit="stations",
    ),
    _Knob(
        key="staffing.porter_count",
        label="Porters",
        group="capacity",
        minimum=0,
        maximum=20,
        step=1,
        unit="on duty",
    ),
    _Knob(
        key="staffing.housekeeping_count",
        label="Housekeeping",
        group="capacity",
        minimum=0,
        maximum=20,
        step=1,
        unit="on duty",
    ),
)


def catalogue(base: Scenario) -> tuple[SliderSpec, ...]:
    """The published knobs, each read against ``base``.

    One ``_read_base`` for the whole catalogue (it generates a floor), then a
    pure read per knob: an alias uses its own inverse, and a knob with no alias
    is read straight off the document path it already names.
    """
    reading = _read_base(base)
    return tuple(
        SliderSpec(
            key=knob.key,
            label=knob.label,
            group=knob.group,
            min=knob.minimum,
            max=knob.maximum,
            step=knob.step,
            unit=knob.unit,
            value=alias.read(reading) if alias is not None else _read_literal(knob.key)(reading),
        )
        for knob in _CATALOGUE
        for alias in (_ALIASES.get(knob.key),)
    )


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
        for target in _ALIASES[alias].targets:
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
    every headcount alias rebuilds the whole ``staffing.blocks`` tuple and every
    bay alias rebuilds the whole ``facility.zones`` tuple, and a tuple replaces
    wholesale under ``_deep_merge`` — merged as independent fragments, setting
    nurses *and* physicians (or general *and* resus bays) would silently drop one
    of them.

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
        scenario = apply_overlay(scenario, _ALIASES[key].compile(scenario, aliases[key]))
    if literals:
        scenario = apply_overlay(scenario, nested_overlay(literals))
    return scenario


@router.get("/scenarios/{scenario_id}/sliders", response_model=SliderCatalogue)
async def get_slider_catalogue(scenario_id: str, request: Request) -> SliderCatalogue:
    """The console's parameter vocabulary, read against one stored scenario.

    A **sub-resource of the scenario**, not a flat ``/sliders``, because half of
    what a knob needs is a fact about the base it will edit: its current value.
    A base-less catalogue could publish names and ranges but would have to answer
    "where is this now?" with a default — and a panel that opens on a default is
    a panel whose first drag silently rewrites a field nobody touched. Hanging it
    off ``/scenarios/{id}`` also means the unknown-base 404 is the same lookup
    (and the same answer) the rest of the scenario resource gives.

    Read-only and derived: it stores nothing, and re-reading it after a
    ``POST /scenarios`` against the *derived* id shows where that variant now
    sits.
    """
    store = cast("ScenarioStore", cast("FastAPI", request.app).state.scenarios)
    base = store.get(scenario_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {scenario_id}")
    return SliderCatalogue(scenario=scenario_id, knobs=catalogue(base))


__all__ = [
    "SLIDER_KEYS",
    "SliderCatalogue",
    "SliderSpec",
    "catalogue",
    "compile_overrides",
    "nested_overlay",
    "router",
]
