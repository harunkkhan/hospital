"""Acuity-weighted priority + anti-starvation (doc 03 §4.4). Pure function, no registry.

``priority_score(esi, waited) = u(esi) + alpha·waited_seconds`` reuses the *same*
``acuity_urgency`` curve placement and ``weighted_total`` use, so acuity is
valued identically everywhere (the sign trap is inherited, not re-risked).

Anti-starvation guarantee: with ``alpha = starvation_rate > 0`` every patient's
score grows without bound, so a lower-acuity patient eventually overtakes a
higher one — an ESI-5 overtakes an ESI-2 waiting ``w₂`` once
``w₅ > w₂ + (u₂ - u₅)/alpha``. No patient waits unboundedly ("never turn away").
"""

from __future__ import annotations

from hospital.core import MICROS_PER_SEC, DecisionInput, Duration, EsiAcuity, PlanItem
from hospital.solver.objective import ObjectiveConfig, acuity_urgency


def priority_score(
    esi: EsiAcuity, waited: Duration, *, config: ObjectiveConfig, starvation_rate: int
) -> int:
    """``u(esi) + alpha·waited_seconds`` — higher is served sooner (µs floored to seconds)."""
    waited_seconds = waited.root // MICROS_PER_SEC
    return acuity_urgency(config, esi) + starvation_rate * waited_seconds


def sequence(
    di: DecisionInput, *, config: ObjectiveConfig, starvation_rate: int
) -> tuple[PlanItem, ...]:
    """Rank ``di.waiting`` descending by score; emit ``kind="sequence"`` items.

    Deterministic total-order tie-break ``(-score, arrival_time, patient_id)`` so
    equal-score orderings are stable and only genuine threshold crossings reorder.
    """
    ranked = sorted(
        di.waiting,
        key=lambda wp: (
            -priority_score(
                wp.patient.esi, wp.waited, config=config, starvation_rate=starvation_rate
            ),
            wp.patient.arrival_time.root,
            wp.patient.id.root,
        ),
    )
    return tuple(
        PlanItem(
            stable_id=f"seq:{wp.patient.id.root}",
            kind="sequence",
            patient=wp.patient.id,
            priority=rank,
        )
        for rank, wp in enumerate(ranked)
    )


__all__ = ["priority_score", "sequence"]
