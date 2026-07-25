"""Discharge expedite + documentation load gate (doc 03 §4.7). A v1 priority policy.

Two coupled levers:

* **Expedite discharges that free scarce bays.** ``value(d)`` reuses the same
  unblock-demand quantity as turnaround (a discharge frees bay ``b`` exactly as a
  clean does): ``value(d) = Σ_{q waiting, compat[q,b]} u(esi(q))``. Discharge
  tasks are ranked by value descending.
* **Documentation load gate.** Documentation is deferrable; running it at peak
  steals provider/nurse capacity. It is *demoted* above a utilization threshold
  and *promoted* in low-load windows — a **soft** priority multiplier, never a
  hard ban (which could defer documentation indefinitely).

``FloorLoad`` utilizations are plain fractions used only for the soft gate
comparison — they never enter the integer objective, so they cannot flap a
golden hash. The ``gamma·boarding_seconds`` term of the doc is omitted: ``DecisionInput``
carries no boarding clock, so ``value(d)`` is the unblock-demand term alone (M1).
Only patients waiting *for a bay* (``NEEDS_BAY_STAGES``) count toward the
unblock value — a placed patient waiting for providers/labs/documentation is
not freed by a discharge.

Representation note: ``core.seam.PlanItemKind`` has no ``documentation`` kind,
so documentation items are emitted as ``kind="discharge"`` — but they retain
their task identity: ``task=TaskSpec.id`` is carried on every item, and the
``stable_id`` prefix discriminates (``discharge:<task>`` vs
``documentation:<task>``). A consumer distinguishes paperwork from an actual
discharge by the prefix, or by resolving ``item.task`` to its ``TaskSpec.kind``
in ``DecisionInput.pending_tasks``. This is the strongest representation the
existing core contract allows without a core change (forbidden from here).
"""

from __future__ import annotations

from hospital.core import (
    Bay,
    CompiledRules,
    DecisionInput,
    PatientId,
    PlanItem,
    TaskSpec,
)
from hospital.core.models import FrozenModel
from hospital.solver.objective import ObjectiveConfig
from hospital.solver.protocol import RoutingOracle
from hospital.solver.turnaround import unblock_value

# Above this provider/nurse utilization, documentation is demoted (doc 03 §4.7).
DEFAULT_LOAD_THRESHOLD: float = 0.8

# Priority band offsets: lower number = served sooner.
_DOC_PROMOTE_BAND = 1_000
_DOC_DEMOTE_BAND = 1_000_000


class FloorLoad(FrozenModel):
    """Current provider/nurse utilization (fractions in ``[0, 1]``) — the gate input."""

    provider_utilization: float = 0.0
    nurse_utilization: float = 0.0

    def is_peak(self, threshold: float = DEFAULT_LOAD_THRESHOLD) -> bool:
        """Whether either pool is above the documentation-gate threshold."""
        return self.provider_utilization > threshold or self.nurse_utilization > threshold


def prioritize_discharge(
    di: DecisionInput,
    oracle: RoutingOracle,
    *,
    config: ObjectiveConfig,
    load: FloorLoad,
    rules: CompiledRules,
) -> tuple[PlanItem, ...]:
    """Rank discharges by unblock value; gate documentation by floor load."""
    del oracle  # v1 priority policy: travel-aware clerk assignment is deferred
    static_bays = {b.id: b for b in di.layout.bays}
    bay_by_occupant: dict[PatientId, Bay] = {}
    for bs in di.bays:
        bay = static_bays.get(bs.bay)
        if bay is not None and bs.occupant is not None:
            bay_by_occupant[bs.occupant] = bay

    def task_unblock_value(task: TaskSpec) -> int:
        if task.patient is None:
            return 0
        bay = bay_by_occupant.get(task.patient)
        if bay is None:
            return 0
        # A discharge frees a BAY exactly as a clean does: the one shared
        # value-of-unblocking quantity (solver.turnaround.unblock_value).
        return unblock_value(bay, di.waiting, config=config, rules=rules)

    discharge_tasks = [
        t for t in di.pending_tasks if t.kind == "discharge" and t.patient is not None
    ]
    doc_tasks = [t for t in di.pending_tasks if t.kind == "documentation" and t.patient is not None]

    ranked = sorted(discharge_tasks, key=lambda t: (-task_unblock_value(t), t.id.root))
    items = [
        PlanItem(
            stable_id=f"discharge:{t.id.root}",
            kind="discharge",
            patient=t.patient,
            task=t.id,
            priority=rank,
        )
        for rank, t in enumerate(ranked)
    ]

    # Documentation keeps its task identity: kind must be "discharge" (no
    # documentation PlanItemKind in core), but the stable_id prefix and the
    # carried task id make paperwork distinguishable from an actual discharge.
    doc_band = _DOC_DEMOTE_BAND if load.is_peak() else _DOC_PROMOTE_BAND
    for offset, t in enumerate(sorted(doc_tasks, key=lambda t: t.id.root)):
        items.append(
            PlanItem(
                stable_id=f"documentation:{t.id.root}",
                kind="discharge",
                patient=t.patient,
                task=t.id,
                priority=doc_band + offset,
            )
        )
    return tuple(items)


__all__ = ["DEFAULT_LOAD_THRESHOLD", "FloorLoad", "prioritize_discharge"]
