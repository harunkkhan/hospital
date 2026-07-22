"""``detect_bottleneck`` — binding-constraint detection + staff work-concentration.

``share_of_cycle(R) = total_wait(R) / total_cycle_s`` reads as "fraction of all
observed patient-time spent waiting in R's queue" (doc 05 §4.3 / nuance 5.5);
the binding constraint is the resource class with the largest share.
``resources`` is returned fully ranked (not just the winner) so a near-tie
second place — a co-binding partner — stays visible; read a narrow argmax gap
as "these two co-bind", never as "this one resource is it".

``gini()`` measures staff work-concentration (0 = perfectly even, higher = more
concentrated), normalized to ``[0, 1]`` via the ``n/(n-1)`` sample correction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from hospital.analysis._index import EventIndex, build_index
from hospital.analysis._stats import (
    DEFAULT_WARMUP,
    DEFAULT_WINDOW,
    clip_seconds,
    measurement_window,
)
from hospital.analysis.utilization import utilization_report
from hospital.core import (
    Activity,
    Duration,
    EventLog,
    FloorLayout,
    FrozenModel,
    OperatingWeek,
    SimTime,
    StaffMember,
    TimeWindow,
)

__all__ = ["BottleneckReport", "ResourceWait", "detect_bottleneck", "gini"]

_FIXED_RESOURCES: tuple[str, ...] = (
    "triage",
    "provider",
    "nurse",
    "imaging",
    "lab",
    "housekeeping",
)


class ResourceWait(FrozenModel):
    resource: str
    total_wait_s: float
    n_requests: int
    mean_wait_s: float
    share_of_cycle: float


class BottleneckReport(FrozenModel):
    binding: str
    resources: tuple[ResourceWait, ...]
    total_cycle_s: float
    gini_by_role: Mapping[str, float]
    gini_overall: float


def gini(values: Sequence[float]) -> float:
    """Sample-normalized Gini coefficient of ``values``, mapped onto ``[0, 1]``.

    ``n < 2`` (concentration is undefined for a singleton) or ``sum(values) ==
    0`` (nothing to distribute) both return ``0.0`` by convention (doc 05
    nuance 5.5) rather than propagate a ``0/0``.
    """
    xs = list(values)
    n = len(xs)
    if n < 2:
        return 0.0
    total = math.fsum(xs)
    if total <= 0.0:
        return 0.0
    sorted_xs = sorted(xs)
    weighted_sum = math.fsum((i + 1) * x for i, x in enumerate(sorted_xs))
    g_raw = (2.0 * weighted_sum) / (n * total) - (n + 1) / n
    return g_raw * n / (n - 1)


def _resource_wait(
    resource: str, pairs: list[tuple[SimTime, SimTime]], m: TimeWindow
) -> ResourceWait:
    total_wait = math.fsum(clip_seconds(req, grant, m) for req, grant in pairs)
    n_requests = sum(1 for req, _grant in pairs if m.contains(req))
    mean_wait = total_wait / n_requests if n_requests > 0 else float("nan")
    return ResourceWait(
        resource=resource,
        total_wait_s=total_wait,
        n_requests=n_requests,
        mean_wait_s=mean_wait,
        share_of_cycle=float("nan"),  # filled in by the caller once total_cycle_s is known
    )


def detect_bottleneck(
    log: EventLog,
    layout: FloorLayout,
    roster: tuple[StaffMember, ...],
    *,
    window: OperatingWeek = DEFAULT_WINDOW,
    warmup: Duration = DEFAULT_WARMUP,
    index: EventIndex | None = None,
) -> BottleneckReport:
    idx = index if index is not None else build_index(log, layout, roster)
    m = measurement_window(window, warmup)
    bay_zone_type = {b.id: b.zone_type for b in layout.bays}

    pairs_by_resource: dict[str, list[tuple[SimTime, SimTime]]] = {
        name: [] for name in _FIXED_RESOURCES
    }
    for zone_type in {b.zone_type for b in layout.bays}:
        pairs_by_resource.setdefault(f"bay:{zone_type.value}", [])

    for p in idx.patients.values():
        if p.bay_requested_at is not None and p.bay_ready is not None and p.bay is not None:
            zone_type = bay_zone_type.get(p.bay)
            if zone_type is not None:
                pairs_by_resource.setdefault(f"bay:{zone_type.value}", []).append(
                    (p.bay_requested_at, p.bay_ready)
                )
        if p.triage_start is not None:
            pairs_by_resource["triage"].append((p.arrival, p.triage_start))
        if p.bay_arrival is not None and p.provider_start is not None:
            pairs_by_resource["provider"].append((p.bay_arrival, p.provider_start))
        if p.bay_arrival is not None and p.nurse_start is not None:
            pairs_by_resource["nurse"].append((p.bay_arrival, p.nurse_start))
        for test in p.test_intervals:
            resource = "imaging" if test.activity == Activity.IMAGING else "lab"
            pairs_by_resource.setdefault(resource, []).append((test.start, test.end))

    for cycles in idx.bays.values():
        for cyc in cycles:
            if cyc.exit is not None and cyc.clean_start is not None:
                pairs_by_resource["housekeeping"].append((cyc.exit, cyc.clean_start))

    total_cycle_s = math.fsum(
        clip_seconds(p.arrival, p.exit if p.exit is not None else window.end, m)
        for p in idx.patients.values()
    )

    resources: list[ResourceWait] = []
    for name, pairs in pairs_by_resource.items():
        rw = _resource_wait(name, pairs, m)
        share = rw.total_wait_s / total_cycle_s if total_cycle_s > 0.0 else float("nan")
        resources.append(rw.model_copy(update={"share_of_cycle": share}))

    def _sort_key(rw: ResourceWait) -> tuple[float, str]:
        share = rw.share_of_cycle
        # NaN shares (no observed patient-time) rank LAST in the table.
        return (-share if not math.isnan(share) else float("inf"), rw.resource)

    resources.sort(key=_sort_key)
    # The binding constraint must be backed by a finite share: with no
    # patient-time in the window every share is NaN and there is NO binding
    # constraint (empty), not an arbitrary alphabetically-first resource.
    binding = next((rw.resource for rw in resources if not math.isnan(rw.share_of_cycle)), "")

    util = utilization_report(log, roster, window=window, warmup=warmup, index=idx)
    loads_by_role: dict[str, list[float]] = {}
    all_loads: list[float] = []
    for b in util.per_staff:
        load = b.walk_s + b.direct_care_s + b.cleaning_s + b.documentation_s
        loads_by_role.setdefault(b.role.value, []).append(load)
        all_loads.append(load)
    gini_by_role = {role: gini(loads) for role, loads in loads_by_role.items()}
    gini_overall = gini(all_loads)

    return BottleneckReport(
        binding=binding,
        resources=tuple(resources),
        total_cycle_s=total_cycle_s,
        gini_by_role=gini_by_role,
        gini_overall=gini_overall,
    )
