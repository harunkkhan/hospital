"""Cost model interface — **deferred**, no numbers in M1 (nuance 1.13).

Cost is a pure function of a :class:`~hospital.core.kpi.KpiVector`, which keeps
money entirely out of ``sim``/``solver`` (clean separation, and why cost can
wait until analytics are observable, post-M4). Only the ``Protocol`` and the
``Money`` value type live here; no rates, tiers, or concrete implementation.

Risk to watch: some cost drivers (overtime, penalty tiers) may need signals not
in the current ``KPI_KEYS`` — landing cost later may require *extending* the KPI
contract, which is a versioned change, not a silent one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import RootModel

from hospital.core.kpi import KpiVector


class Money(RootModel[int]):
    """A monetary amount in integer minor units (e.g. cents). No rates defined yet."""

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.root)


@runtime_checkable
class CostModel(Protocol):
    """Prices a KPI reading into :class:`Money`. Implemented after M4."""

    def price(self, kpis: KpiVector) -> Money: ...


__all__ = ["CostModel", "Money"]
