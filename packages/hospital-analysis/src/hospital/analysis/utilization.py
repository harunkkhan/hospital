"""Partition every staff-second into ``{walk, direct_care, cleaning, documentation, idle}``.

``idle`` is the RESIDUAL (doc 05 §4.4 / nuance 5.6): ``walk + direct_care +
cleaning + documentation + idle == on_shift_s`` by construction — which is
exactly why the pooled ``staff_frac_*`` sum to 1.0 with no float drift beyond
the one shared denominator. This is the ONE utilization computation in the
repo; ``fold.compute_kpis`` (keys 21-27) reads these numbers verbatim rather
than re-deriving them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Final

from hospital.analysis._index import EventIndex, ServiceInterval, build_index
from hospital.analysis._stats import (
    DEFAULT_WARMUP,
    DEFAULT_WINDOW,
    clip_seconds,
    measurement_window,
)
from hospital.core import (
    Duration,
    EventLog,
    FloorLayout,
    FrozenModel,
    OperatingWeek,
    RouteGraph,
    StaffId,
    StaffMember,
    StaffRole,
    TimeWindow,
)
from hospital.core.kpi import STAFF_FRAC_KEYS

__all__ = [
    "StaffSecondBudget",
    "UtilizationReport",
    "classify_staff_seconds",
    "utilization_report",
]

# A staff second going negative-idle by more than this (float boundary-clipping
# noise) indicates the walk/service disjointness assumption actually broke —
# a real bug, not rounding, so it is not silently clamped away (nuance 5.6).
_IDLE_TOLERANCE_S: Final[float] = 1e-6

# utilization_report's public signature (doc 05 §3) carries no `layout` — it
# never reads bay-cycle data, only patient service intervals and staff walk/
# cleaning intervals. When called without a shared `index=`, it builds one
# against this trivial empty floor; harmless for utilization's own output.
# Callers that need bay-consistent output across fold/waits/bottleneck/
# utilization in one pass build the shared index themselves (as report.fold_arm
# does) and pass `index=` here too.
_EMPTY_LAYOUT: Final[FloorLayout] = FloorLayout(
    graph=RouteGraph(nodes=(), edges=()),
    zones=(),
    bays=(),
    stations=(),
    entrances=(),
    imaging_nodes=(),
    lab_nodes=(),
)

# ESI sign-trap has no analogue here, but the physician/nurse KPI naming does:
# `provider_util`/`nurse_util` are the KPI names for the PHYSICIAN/NURSE roles.
_ROLE_UTIL_KEY: Final[dict[StaffRole, str]] = {
    StaffRole.PHYSICIAN: "provider_util",
    StaffRole.NURSE: "nurse_util",
}


class StaffSecondBudget(FrozenModel):
    """One staff member's disjoint partition of ``on_shift_s`` (idle is residual)."""

    staff: StaffId
    role: StaffRole
    on_shift_s: float
    walk_s: float
    direct_care_s: float
    cleaning_s: float
    documentation_s: float
    idle_s: float


class UtilizationReport(FrozenModel):
    per_staff: tuple[StaffSecondBudget, ...]
    fractions: Mapping[str, float]  # staff_frac_* pooled over ALL staff, sums to 1.0
    util_by_role: Mapping[str, float]  # "provider_util", "nurse_util", ...
    # The shared denominator behind every fraction above: total paid staff-seconds over
    # the measurement window. Exposed because a *fraction* cannot be priced — turning
    # "6% of staff time was spent walking" into money needs the hours that 6% is of
    # (M4b). It is the one quantity that makes the whole budget extensive.
    on_shift_s: float = 0.0


def _service_seconds(iv: ServiceInterval, m: TimeWindow) -> float:
    """Clipped seconds of a service interval within ``m``.

    An OPEN interval (``end=None`` — service still in progress at the end of
    the log) is measured through to the measurement horizon ``m.end``: staff
    mid-service at a censored run end are working, not idle.
    """
    end = iv.end if iv.end is not None else m.end
    return clip_seconds(iv.start, end, m)


def classify_staff_seconds(
    index: EventIndex,
    roster: tuple[StaffMember, ...],
    *,
    window: OperatingWeek,
    warmup: Duration,
) -> tuple[StaffSecondBudget, ...]:
    """Per-staff second budget: the disjoint partition that ``UtilizationReport`` pools."""
    m = measurement_window(window, warmup)
    # M1 staffing is a fixed scenario input: on_shift = the whole measurement
    # window minus absence. `DisruptionInjected` carries only a free-text
    # `disruption`/`detail` string in the current core schema — no structured
    # staff-id/time-range payload to subtract — so absence is not modeled here
    # (judgment call; revisit once disruptions carry a structured target).
    on_shift_total = m.duration().to_seconds()

    direct_care: dict[StaffId, float] = defaultdict(float)
    documentation: dict[StaffId, float] = defaultdict(float)
    for trace in index.patients.values():
        triage_iv = trace.triage_interval
        if triage_iv is not None and triage_iv.staff is not None:
            direct_care[triage_iv.staff] += _service_seconds(triage_iv, m)
        for iv in (*trace.provider_intervals, *trace.nurse_intervals):
            if iv.staff is not None:
                direct_care[iv.staff] += _service_seconds(iv, m)
        for iv in trace.documentation_intervals:
            if iv.staff is not None:
                documentation[iv.staff] += _service_seconds(iv, m)
        # NOTE: DischargeStarted/DischargeCompleted carry no `staff` field in
        # the current core.events schema, so discharge paperwork cannot be
        # staff-attributed here; only `Documentation*` feeds documentation_s
        # (deviation from doc 05 §4.4's "Documentation* AND Discharge*", forced
        # by the actual event schema — see final report).

    budgets: list[StaffSecondBudget] = []
    for member in roster:
        trace = index.staff.get(member.id)
        walk_s = sum(clip_seconds(s, e, m) for s, e in (trace.walk_intervals if trace else ()))
        cleaning_s = sum(
            clip_seconds(s, e, m) for s, e in (trace.cleaning_intervals if trace else ())
        )
        dc = direct_care.get(member.id, 0.0)
        doc = documentation.get(member.id, 0.0)
        idle_s = on_shift_total - walk_s - dc - cleaning_s - doc
        if idle_s < -_IDLE_TOLERANCE_S:
            raise ValueError(
                f"staff {member.id}: idle_s={idle_s:.6f}s — walk/service intervals "
                "overlap (the one-thing-at-a-time disjointness assumption broke)"
            )
        idle_s = max(idle_s, 0.0)
        budgets.append(
            StaffSecondBudget(
                staff=member.id,
                role=member.role,
                on_shift_s=on_shift_total,
                walk_s=walk_s,
                direct_care_s=dc,
                cleaning_s=cleaning_s,
                documentation_s=doc,
                idle_s=idle_s,
            )
        )
    return tuple(budgets)


def utilization_report(
    log: EventLog,
    roster: tuple[StaffMember, ...],
    *,
    window: OperatingWeek = DEFAULT_WINDOW,
    warmup: Duration = DEFAULT_WARMUP,
    index: EventIndex | None = None,
) -> UtilizationReport:
    idx = index if index is not None else build_index(log, _EMPTY_LAYOUT, roster)
    per_staff = classify_staff_seconds(idx, roster, window=window, warmup=warmup)

    totals = dict.fromkeys(("walk", "direct_care", "cleaning", "documentation", "idle"), 0.0)
    on_shift_total = 0.0
    role_idle: dict[StaffRole, float] = defaultdict(float)
    role_shift: dict[StaffRole, float] = defaultdict(float)
    for b in per_staff:
        totals["walk"] += b.walk_s
        totals["direct_care"] += b.direct_care_s
        totals["cleaning"] += b.cleaning_s
        totals["documentation"] += b.documentation_s
        totals["idle"] += b.idle_s
        on_shift_total += b.on_shift_s
        role_idle[b.role] += b.idle_s
        role_shift[b.role] += b.on_shift_s

    if on_shift_total > 0.0:
        fractions = {
            "staff_frac_walk": totals["walk"] / on_shift_total,
            "staff_frac_direct_care": totals["direct_care"] / on_shift_total,
            "staff_frac_cleaning": totals["cleaning"] / on_shift_total,
            "staff_frac_documentation": totals["documentation"] / on_shift_total,
            "staff_frac_idle": totals["idle"] / on_shift_total,
        }
    else:
        # No roster / zero on-shift time: fall back to all-idle so the closed
        # KPI contract (staff_frac_* finite, summing to 1.0) still holds.
        fractions = dict.fromkeys(STAFF_FRAC_KEYS, 0.0)
        fractions["staff_frac_idle"] = 1.0

    util_by_role: dict[str, float] = {"provider_util": float("nan"), "nurse_util": float("nan")}
    seen_roles: set[StaffRole] = set()
    for role, shift_s in role_shift.items():
        key = _ROLE_UTIL_KEY.get(role, f"{role.value}_util")
        util_by_role[key] = 1.0 - role_idle[role] / shift_s if shift_s > 0.0 else float("nan")
        seen_roles.add(role)

    return UtilizationReport(
        per_staff=per_staff,
        fractions=fractions,
        util_by_role=util_by_role,
        on_shift_s=on_shift_total,
    )
