"""Declarative, frozen constraint rules — compiled once into the validator kernel.

The rule *vocabulary* lives here exactly once; ``solver`` encodes these same
rules while searching and :func:`hospital.core.validation.validate` re-checks
them on plan acceptance — one rule source, two enforcers (defense in depth, not
duplication; nuance 1.11).

The ``CompatibilityRule`` is the **single source** of the acuity->zone_type /
isolation / equipment mapping, so the placement solver and the validator cannot
drift. ``rules_hash`` (canonical sorted-JSON sha256) stamps a plan's provenance
to the exact rule config that judged it.

Placement is judged against **one of two** whitelists, selected by the patient's
care phase: ``CompatibilityRule`` for a patient being worked up in the ED, and
``AdmissionRule`` for one whose disposition was ADMIT and who now needs an
inpatient bed. :meth:`CompiledRules.zone_types_for_stage` is where that choice is
made, once, for every enforcer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, cast

from pydantic import Field, TypeAdapter

from hospital.core.enums import Activity, EsiAcuity, ZoneType
from hospital.core.models import FrozenModel
from hospital.core.seam import AWAITING_ADMISSION


class CompatibilityRule(FrozenModel):
    """The single acuity->zone_type / isolation / equipment mapping."""

    kind: Literal["compat"] = "compat"
    # An ESI acuity may be placed in these zone types.
    allowed_zone_types: frozenset[tuple[EsiAcuity, ZoneType]] = frozenset()
    # If set, an isolation-required patient must go to an isolation-capable bay.
    isolation_enforced: bool = True
    # A patient of this acuity requires the assigned bay to carry this equipment.
    required_equipment: frozenset[tuple[EsiAcuity, str]] = frozenset()


class AdmissionRule(FrozenModel):
    """The single acuity->ward zone_type mapping for an ADMITTED patient (M4).

    Deliberately *not* more entries in :class:`CompatibilityRule`, which the validator
    and the placement solver both read keyed on acuity alone. The two rules answer
    different questions: compatibility says where a patient may be **worked up**,
    admission says where they may be **admitted**. Folding wards into the first would
    make an ESI-2 eligible for a resus bay thereby eligible for an ICU bed the moment
    they finish triage — a placement no rule could then refuse, because nothing in the
    key distinguishes the two moments. The patient's care phase
    (:data:`~hospital.core.seam.AWAITING_ADMISSION`) selects which of the two applies.

    Additive by construction: a scenario that declares no ``AdmissionRule`` admits
    nobody anywhere, which is exactly right for an ED-only floor plan and is why every
    rule set written before wards existed hashes and behaves as it did.

    Isolation and equipment are *not* restated here. Those constraints are properties
    of the patient and the bay, not of the phase, so :class:`CompiledRules` applies the
    compatibility rule's ``isolation_enforced`` / ``required_equipment`` to both kinds
    of placement — one source, asked twice.
    """

    kind: Literal["admission"] = "admission"
    # An ESI acuity may be admitted into these (ward) zone types.
    allowed_zone_types: frozenset[tuple[EsiAcuity, ZoneType]] = frozenset()


class CapacityRule(FrozenModel):
    """A per-zone-type concurrent-occupancy cap."""

    kind: Literal["capacity"] = "capacity"
    zone_type: ZoneType
    max_occupancy: int


class SkillRule(FrozenModel):
    """Skills a task-kind requires of the staff performing it."""

    kind: Literal["skill"] = "skill"
    task_kind: str
    required_skills: frozenset[str] = frozenset()


class PrecedenceRule(FrozenModel):
    """Activity ``before`` must precede activity ``after`` in a patient's order."""

    kind: Literal["precedence"] = "precedence"
    before: Activity
    after: Activity


Rule = Annotated[
    CompatibilityRule | AdmissionRule | CapacityRule | SkillRule | PrecedenceRule,
    Field(discriminator="kind"),
]


class CompiledRules(FrozenModel):
    """Indexed, frozen constraint kernel produced by :func:`compile_rules`."""

    allowed_zone_types: frozenset[tuple[EsiAcuity, ZoneType]] = frozenset()
    admission_zone_types: frozenset[tuple[EsiAcuity, ZoneType]] = frozenset()
    isolation_enforced: bool = True
    required_equipment: frozenset[tuple[EsiAcuity, str]] = frozenset()
    capacities: frozenset[tuple[ZoneType, int]] = frozenset()
    skills: frozenset[tuple[str, frozenset[str]]] = frozenset()
    precedences: frozenset[tuple[Activity, Activity]] = frozenset()
    rules_hash: str = ""

    def zone_types_for(self, acuity: EsiAcuity) -> frozenset[ZoneType]:
        """Zone types an ``acuity`` patient may be placed in to be worked up."""
        return frozenset(zt for (a, zt) in self.allowed_zone_types if a == acuity)

    def ward_zone_types_for(self, acuity: EsiAcuity) -> frozenset[ZoneType]:
        """Ward zone types an ``acuity`` patient may be ADMITTED into (M4).

        Whitelist semantics, exactly like :meth:`zone_types_for`: empty means "may be
        admitted nowhere", never "anywhere". An ED-only floor plan has no ward beds to
        offer either way, so the two readings only diverge for a scenario that built a
        ward and then forgot to say who may go in it — which should hold the patient in
        the ED rather than silently admit them.
        """
        return frozenset(zt for (a, zt) in self.admission_zone_types if a == acuity)

    def zone_types_for_stage(self, acuity: EsiAcuity, stage: str) -> frozenset[ZoneType]:
        """The whitelist that judges a placement, chosen by the patient's care phase.

        The one place the phase->whitelist mapping is written. Both enforcement points
        — :func:`hospital.core.validation.validate` and the placement solver's
        ``compat`` derivation — route through here, so they cannot drift on the
        question of which rule applies any more than they can on what it says.
        """
        if stage == AWAITING_ADMISSION:
            return self.ward_zone_types_for(acuity)
        return self.zone_types_for(acuity)

    def equipment_for(self, acuity: EsiAcuity) -> frozenset[str]:
        """Equipment a bay must carry to receive an ``acuity`` patient."""
        return frozenset(eq for (a, eq) in self.required_equipment if a == acuity)

    def capacity_for(self, zone_type: ZoneType) -> int | None:
        """Most-restrictive concurrent-occupancy cap for ``zone_type`` (or ``None``)."""
        caps = [cap for (zt, cap) in self.capacities if zt == zone_type]
        return min(caps) if caps else None

    def skills_for(self, task_kind: str) -> frozenset[str]:
        """Union of required skills across all skill rules for ``task_kind``."""
        out: set[str] = set()
        for tk, skills in self.skills:
            if tk == task_kind:
                out |= set(skills)
        return frozenset(out)


_RULE_ADAPTER: TypeAdapter[Rule] = TypeAdapter(Rule)


def _canonicalize(obj: object) -> object:
    """Recursively canonicalize parsed JSON so the encoding is content-addressable.

    ``frozenset``/``tuple`` rule fields serialize to JSON *arrays* whose element
    order follows process-dependent set iteration (string/enum hashing is salted
    by ``PYTHONHASHSEED``). ``json.dumps(..., sort_keys=True)`` sorts object keys
    but **not** array elements, so identical rules would otherwise hash
    differently across processes. Sorting every array by its element's canonical
    JSON makes the hash a pure function of rule *content*.
    """
    if isinstance(obj, dict):
        d = cast("dict[str, object]", obj)
        return {k: _canonicalize(v) for k, v in d.items()}
    if isinstance(obj, list):
        items = [_canonicalize(v) for v in cast("list[object]", obj)]
        return sorted(items, key=lambda e: json.dumps(e, sort_keys=True, separators=(",", ":")))
    return obj


def rules_hash(rules: tuple[Rule, ...]) -> str:
    """Canonical, order-independent sha256 over the rule set (sorted-JSON)."""
    encoded = sorted(
        json.dumps(
            _canonicalize(_RULE_ADAPTER.dump_python(r, mode="json")),
            sort_keys=True,
            separators=(",", ":"),
        )
        for r in rules
    )
    payload = json.dumps(encoded, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_rules(rules: tuple[Rule, ...]) -> CompiledRules:
    """Compile a rule tuple into the indexed :class:`CompiledRules` kernel."""
    allowed: set[tuple[EsiAcuity, ZoneType]] = set()
    admission: set[tuple[EsiAcuity, ZoneType]] = set()
    equipment: set[tuple[EsiAcuity, str]] = set()
    capacities: set[tuple[ZoneType, int]] = set()
    skills: set[tuple[str, frozenset[str]]] = set()
    precedences: set[tuple[Activity, Activity]] = set()
    compat_seen = False
    isolation_enforced = True

    for rule in rules:
        if isinstance(rule, CompatibilityRule):
            if not compat_seen:
                isolation_enforced = rule.isolation_enforced
                compat_seen = True
            else:
                isolation_enforced = isolation_enforced or rule.isolation_enforced
            allowed |= set(rule.allowed_zone_types)
            equipment |= set(rule.required_equipment)
        elif isinstance(rule, AdmissionRule):
            admission |= set(rule.allowed_zone_types)
        elif isinstance(rule, CapacityRule):
            capacities.add((rule.zone_type, rule.max_occupancy))
        elif isinstance(rule, SkillRule):
            skills.add((rule.task_kind, rule.required_skills))
        else:
            precedences.add((rule.before, rule.after))

    return CompiledRules(
        allowed_zone_types=frozenset(allowed),
        admission_zone_types=frozenset(admission),
        isolation_enforced=isolation_enforced,
        required_equipment=frozenset(equipment),
        capacities=frozenset(capacities),
        skills=frozenset(skills),
        precedences=frozenset(precedences),
        rules_hash=rules_hash(rules),
    )


__all__ = [
    "AdmissionRule",
    "CapacityRule",
    "CompatibilityRule",
    "CompiledRules",
    "PrecedenceRule",
    "Rule",
    "SkillRule",
    "compile_rules",
    "rules_hash",
]
