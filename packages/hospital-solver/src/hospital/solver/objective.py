"""The single scalar objective for the whole repo (doc 00 §4, doc 03 §4.1).

``weighted_total`` is the *sole* aggregator that ranks candidate plans and folds
scorecards; every lever minimizes a **linear restriction** of it whose
coefficients come from :func:`assignment_coeffs` / :func:`acuity_urgency` — never
a second, hand-tuned weight set (doc 00 §5 rule 6).

Two invariants baked in here:

* **Everything is integer.** ``T_a`` and ``S`` are integer seconds, ``u``/``w``/
  penalties are ints, ``scale`` is a fixed-point factor. CP-SAT is an integer
  solver, and integer math keeps BASELINE/OPTIMIZED byte-identical (no float
  drift flapping a golden-trace hash).
* **The acuity-weight sign lives in one place.** The urgency multiplier
  ``u(esi)`` defaults to :meth:`hospital.core.enums.EsiAcuity.priority_weight`
  (ESI-1 highest), so the classic sign trap — higher weight for the *lower* ESI
  number — is never re-derived inline (DECISIONS D8). The map is stored
  explicitly so it stays *data* (tunable) rather than a computed ``1/esi``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import Field

from hospital.core import EsiAcuity, FrozenModel

# An immutable empty mapping — a safe default that avoids a mutable ``{}`` default.
_EMPTY_PENALTIES: Mapping[str, int] = MappingProxyType({})


def default_acuity_urgency() -> dict[EsiAcuity, int]:
    """The canonical urgency curve ``u(esi)`` — derived from ``priority_weight``.

    Building it from :meth:`EsiAcuity.priority_weight` is what keeps the acuity
    sign in exactly one place (DECISIONS D8); the result is stored as data so a
    scenario may override the curve without re-deriving the inversion.
    """
    return {esi: esi.priority_weight() for esi in EsiAcuity}


class ObjectiveConfig(FrozenModel):
    """The frozen weight set — the one config every lever's coefficients derive from."""

    version: str = "1.0.0"
    scale: int = 1
    w_time: int = 1
    w_travel: int = 1
    acuity_urgency: Mapping[EsiAcuity, int] = Field(default_factory=default_acuity_urgency)
    unplaced_wait_penalty: int = 1000


class AssignmentCoeffs(FrozenModel):
    """Per-patient linear coefficients drawn from the one :class:`ObjectiveConfig`."""

    travel_weight: int  # w_travel * u(esi) — per-second travel cost for this patient
    wait_penalty: int  # w_time * u(esi) * unplaced_wait_penalty — per-second unplaced cost


def acuity_urgency(config: ObjectiveConfig, esi: EsiAcuity) -> int:
    """The urgency multiplier ``u(esi)`` — the same curve every lever reads."""
    return config.acuity_urgency[esi]


def assignment_coeffs(config: ObjectiveConfig, esi: EsiAcuity) -> AssignmentCoeffs:
    """The travel/wait coefficients for an ``esi`` patient (doc 03 §4.1)."""
    u = acuity_urgency(config, esi)
    return AssignmentCoeffs(
        travel_weight=config.w_travel * u,
        wait_penalty=config.w_time * u * config.unplaced_wait_penalty,
    )


def weighted_total(
    *,
    patient_time_s: Mapping[EsiAcuity, int],
    staff_travel_s: int,
    penalties: Mapping[str, int] = _EMPTY_PENALTIES,
    config: ObjectiveConfig,
) -> int:
    """The SINGLE scalar cost, integer-scaled (doc 03 §4.1).

    ``weighted_total = scale * [ w_time * Σ_a u(a)·T_a + w_travel·S + Σ_k p_k ]``.
    """
    acuity_time = sum(acuity_urgency(config, a) * t for a, t in patient_time_s.items())
    unscaled = (
        config.w_time * acuity_time + config.w_travel * staff_travel_s + sum(penalties.values())
    )
    return config.scale * unscaled


def config_hash(config: ObjectiveConfig) -> str:
    """Canonical sha256 over the config (sorted-JSON, ``acuity_urgency`` sorted by key).

    Byte-different configs hash differently; identical configs in any field order
    collide by construction. This is the value :func:`hospital.solver.stamping.stamp`
    attaches so a plan on the wire is traceable to its exact weight set.
    """
    payload = {
        "version": config.version,
        "scale": config.scale,
        "w_time": config.w_time,
        "w_travel": config.w_travel,
        "unplaced_wait_penalty": config.unplaced_wait_penalty,
        "acuity_urgency": {
            str(int(a)): config.acuity_urgency[a] for a in sorted(config.acuity_urgency, key=int)
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "AssignmentCoeffs",
    "ObjectiveConfig",
    "acuity_urgency",
    "assignment_coeffs",
    "config_hash",
    "default_acuity_urgency",
    "weighted_total",
]
