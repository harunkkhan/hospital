"""The closed, versioned KPI contract (27 keys).

``analysis.fold`` produces a :class:`KpiVector`; ``sim.experiment.scorecard`` and
``api.metrics`` consume it. A new KPI is a new key here (versioned), so downstream
never guesses (nuance 1.12).

Two conventions decided here and applied everywhere:

* **Empty strata -> NaN, never omitted.** ``los_s_*_by_esi_k`` for an ESI ``k``
  with zero patients is reported as ``float('nan')``. The vector therefore always
  carries the *complete* key set, which is what lets the contract be genuinely
  *closed*: :class:`KpiVector` requires **exactly** ``KPI_KEYS`` — an extra *or*
  missing key is a :class:`~hospital.core.errors.KpiContractError`. (Consumers
  such as ``paired_bootstrap`` must NaN-skip empty strata.)
* **``staff_frac_*`` sum to 1.0 within eps.** They are constructed with ``idle``
  as the residual so they sum exactly; the check uses a tolerance, not ``== 1.0``,
  because of float accumulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import field_serializer, field_validator, model_validator

from hospital.core.errors import KpiContractError
from hospital.core.models import FrozenModel

_EPS: Final[float] = 1e-9

_LOS_KEYS: Final[tuple[str, ...]] = tuple(
    f"los_s_{stat}_by_esi_{k}" for stat in ("mean", "p90") for k in (1, 2, 3, 4, 5)
)

KPI_KEYS: Final[tuple[str, ...]] = (
    "completions_per_week",
    "wip_end_of_week",
    "door_to_triage_s_mean",
    "door_to_triage_s_p90",
    "door_to_provider_s_mean",
    "door_to_provider_s_p90",
    *_LOS_KEYS,
    "staff_minutes_walked",
    "bay_utilization",
    "turnaround_time_s_mean",
    "boarding_time_s_mean",
    "provider_util",
    "nurse_util",
    "staff_frac_walk",
    "staff_frac_direct_care",
    "staff_frac_cleaning",
    "staff_frac_documentation",
    "staff_frac_idle",
)

STAFF_FRAC_KEYS: Final[tuple[str, ...]] = (
    "staff_frac_walk",
    "staff_frac_direct_care",
    "staff_frac_cleaning",
    "staff_frac_documentation",
    "staff_frac_idle",
)

# Every proportion KPI is bounded to [0, 1]. ``staff_frac_*`` additionally sum to
# 1.0; the utilization proportions are independently bounded.
PROPORTION_KEYS: Final[tuple[str, ...]] = (
    *STAFF_FRAC_KEYS,
    "bay_utilization",
    "provider_util",
    "nurse_util",
)

_KPI_KEY_SET: Final[frozenset[str]] = frozenset(KPI_KEYS)


class KpiVector(FrozenModel):
    """A complete KPI reading — keys are exactly ``KPI_KEYS`` (closed contract)."""

    values: Mapping[str, float]

    @field_validator("values", mode="after")
    @classmethod
    def _freeze_values(cls, values: Mapping[str, float]) -> Mapping[str, float]:
        """Store a read-only copy so the validated contract can't be mutated post hoc."""
        return MappingProxyType(dict(values))

    @field_serializer("values")
    def _serialize_values(self, values: Mapping[str, float]) -> dict[str, float]:
        return dict(values)

    @model_validator(mode="after")
    def _check_contract(self) -> KpiVector:
        keys = set(self.values.keys())
        extra = keys - _KPI_KEY_SET
        if extra:
            raise KpiContractError(f"unknown KPI key(s): {sorted(extra)}")
        missing = _KPI_KEY_SET - keys
        if missing:
            raise KpiContractError(f"missing KPI key(s): {sorted(missing)}")
        fractions = [self.values[k] for k in STAFF_FRAC_KEYS]
        if any(math.isnan(f) for f in fractions):
            raise KpiContractError("staff_frac_* must be finite")
        # Every proportion KPI must lie in [0, 1] — a fraction outside the range is
        # invalid even when the fractions happen to sum to 1.0.
        for key in PROPORTION_KEYS:
            value = self.values[key]
            if not math.isnan(value) and not 0.0 <= value <= 1.0:
                raise KpiContractError(f"proportion {key}={value} out of range [0, 1]")
        total = math.fsum(fractions)
        if abs(total - 1.0) > _EPS:
            raise KpiContractError(f"staff_frac_* sum to {total}, expected 1.0 +/- {_EPS}")
        return self


__all__ = ["KPI_KEYS", "STAFF_FRAC_KEYS", "KpiVector"]
