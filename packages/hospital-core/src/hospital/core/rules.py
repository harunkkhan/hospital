"""Declarative, frozen constraint rules — compiled once into the validator kernel.

The rule *vocabulary* lives here exactly once; ``solver`` encodes these same
rules while searching and :func:`hospital.core.validation.validate` re-checks
them on plan acceptance — one rule source, two enforcers (defense in depth, not
duplication; nuance 1.11).

The ``CompatibilityRule`` is the **single source** of the acuity->zone_type /
isolation / equipment mapping, so the placement solver and the validator cannot
drift. ``rules_hash`` (canonical sorted-JSON sha256) stamps a plan's provenance
to the exact rule config that judged it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, cast

from pydantic import Field, TypeAdapter

from hospital.core.enums import Activity, EsiAcuity, ZoneType
from hospital.core.models import FrozenModel


class CompatibilityRule(FrozenModel):
    """The single acuity->zone_type / isolation / equipment mapping."""

    kind: Literal["compat"] = "compat"
    # An ESI acuity may be placed in these zone types.
    allowed_zone_types: frozenset[tuple[EsiAcuity, ZoneType]] = frozenset()
    # If set, an isolation-required patient must go to an isolation-capable bay.
    isolation_enforced: bool = True
    # A patient of this acuity requires the assigned bay to carry this equipment.
    required_equipment: frozenset[tuple[EsiAcuity, str]] = frozenset()


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
    CompatibilityRule | CapacityRule | SkillRule | PrecedenceRule,
    Field(discriminator="kind"),
]


class CompiledRules(FrozenModel):
    """Indexed, frozen constraint kernel produced by :func:`compile_rules`."""

    allowed_zone_types: frozenset[tuple[EsiAcuity, ZoneType]] = frozenset()
    isolation_enforced: bool = True
    required_equipment: frozenset[tuple[EsiAcuity, str]] = frozenset()
    capacities: frozenset[tuple[ZoneType, int]] = frozenset()
    skills: frozenset[tuple[str, frozenset[str]]] = frozenset()
    precedences: frozenset[tuple[Activity, Activity]] = frozenset()
    rules_hash: str = ""

    def zone_types_for(self, acuity: EsiAcuity) -> frozenset[ZoneType]:
        """Zone types an ``acuity`` patient may be placed in."""
        return frozenset(zt for (a, zt) in self.allowed_zone_types if a == acuity)

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
        elif isinstance(rule, CapacityRule):
            capacities.add((rule.zone_type, rule.max_occupancy))
        elif isinstance(rule, SkillRule):
            skills.add((rule.task_kind, rule.required_skills))
        else:
            precedences.add((rule.before, rule.after))

    return CompiledRules(
        allowed_zone_types=frozenset(allowed),
        isolation_enforced=isolation_enforced,
        required_equipment=frozenset(equipment),
        capacities=frozenset(capacities),
        skills=frozenset(skills),
        precedences=frozenset(precedences),
        rules_hash=rules_hash(rules),
    )


__all__ = [
    "CapacityRule",
    "CompatibilityRule",
    "CompiledRules",
    "PrecedenceRule",
    "Rule",
    "SkillRule",
    "compile_rules",
    "rules_hash",
]
