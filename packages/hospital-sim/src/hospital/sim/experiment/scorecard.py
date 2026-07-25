"""Fold a ``Replication`` to a ``Scorecard``; rank lexicographically (doc 04 §3.14).

Anti-duplication is the whole point of this module: the KPI numbers come from
``analysis.fold.compute_kpis`` (the ONE fold over the ``EventLog``) and the
scalar comes from ``solver.objective.weighted_total`` (the ONE cost
aggregator) — ``sim`` re-implements neither, and ``completions``/``wip`` are
read from the folded vector, never recomputed, so the scorecard can't disagree
with ``analysis`` about the headline counts.

``weighted_total`` takes physical quantities (integer acuity-seconds and
travel-seconds), not a ``KpiVector`` — so :func:`objective_inputs` assembles
those inputs from the log (a reduction to the objective's *arguments*, not a
second KPI fold; the aggregation itself stays in ``solver``).

Ranking is **lexicographic, never the scalar alone** (PLAN §6.8, principle 6):
``(wip asc, door-to-provider asc, staff-minutes-walked asc, run_id)``. A single
weighted scalar can be gamed — a low ``weighted_total`` can quietly hide
un-completed patients; no cleverness on a lower key can buy back a worse WIP.
The ``run_id`` tail makes the order total (a platform-stable sort).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hospital.analysis import compute_kpis
from hospital.core import (
    DischargeCompleted,
    Duration,
    EsiAcuity,
    EventLog,
    FrozenModel,
    KpiVector,
    OperatingWeek,
    PatientArrived,
    PatientId,
    RunId,
    SimTime,
    StaffMoved,
    TimeWindow,
    TriageCompleted,
    hours,
)
from hospital.data.layout import generate_floor
from hospital.data.scenario import realize_staff
from hospital.solver import ObjectiveConfig, SolverStatus, weighted_total

if TYPE_CHECKING:
    from hospital.sim.experiment.replication import Replication

_MICROS_PER_SEC = 1_000_000


class Scorecard(FrozenModel):
    """One run's headline reading: the folded KPIs plus the one objective scalar."""

    run_id: RunId
    arm: str
    seed: int
    kpis: KpiVector
    objective_total: int
    completions: int
    wip: int
    # The optimized arm's WORST-observed placement solve claim over the run
    # (None for baseline): a fallback week is distinguishable from a proven one.
    status: SolverStatus | None = None


def objective_inputs(log: EventLog, horizon: OperatingWeek) -> tuple[dict[EsiAcuity, int], int]:
    """The physical arguments of ``weighted_total``: acuity-seconds and travel-seconds.

    Per-patient time-in-system is ``min(exit, horizon.end) - arrival`` (WIP is
    clipped at the horizon, so backlog *costs* — an arm cannot pocket a low
    scalar by stranding patients). Acuity comes from ``TriageCompleted`` — a
    patient un-triaged at the cutoff has no observable acuity in the log and is
    excluded (their time is seconds spent walking to/queueing for triage).
    Travel is the sum of every ``StaffMoved`` edge over the full horizon.
    """
    arrivals: dict[PatientId, SimTime] = {}
    acuities: dict[PatientId, EsiAcuity] = {}
    exits: dict[PatientId, SimTime] = {}
    travel_us = 0
    for envelope in log.ordered():
        event = envelope.event
        if isinstance(event, PatientArrived):
            arrivals[event.patient] = event.occurred_at
        elif isinstance(event, TriageCompleted):
            acuities[event.patient] = event.esi
        elif isinstance(event, DischargeCompleted):
            exits[event.patient] = event.occurred_at
        elif isinstance(event, StaffMoved):
            travel_us += event.seconds.root

    patient_time_s: dict[EsiAcuity, int] = {}
    end = horizon.end
    for pid, arrived_at in arrivals.items():
        acuity = acuities.get(pid)
        if acuity is None:
            continue
        left_at = exits.get(pid, end)
        clipped = min(left_at.root, end.root)
        patient_time_s[acuity] = (
            patient_time_s.get(acuity, 0) + (clipped - arrived_at.root) // _MICROS_PER_SEC
        )
    return patient_time_s, travel_us // _MICROS_PER_SEC


def _default_warmup(horizon: OperatingWeek) -> Duration:
    """24h for a real week; a quarter of the horizon for short test runs."""
    span = horizon.end.root - horizon.start.root
    return Duration(min(hours(24).root, span // 4))


def fold_scorecard(
    rep: Replication, objective: ObjectiveConfig, *, warmup: Duration | None = None
) -> Scorecard:
    """Fold the persisted bytes: re-parse the JSONL, one KPI fold, one scalar."""
    log = EventLog.from_jsonl(rep.event_log_jsonl)
    layout = generate_floor(rep.scenario.facility)
    window = TimeWindow(start=rep.horizon.start, end=rep.horizon.end)
    roster = realize_staff(rep.scenario.staffing, layout, window)
    effective_warmup = warmup if warmup is not None else _default_warmup(rep.horizon)
    kpis = compute_kpis(log, layout, roster, window=rep.horizon, warmup=effective_warmup)
    patient_time_s, staff_travel_s = objective_inputs(log, rep.horizon)
    total = weighted_total(
        patient_time_s=patient_time_s, staff_travel_s=staff_travel_s, config=objective
    )
    return Scorecard(
        run_id=rep.run_id,
        arm=rep.arm,
        seed=rep.seed,
        kpis=kpis,
        objective_total=total,
        completions=int(kpis.values["completions_per_week"]),
        wip=int(kpis.values["wip_end_of_week"]),
        status=rep.solver_status,
    )


def _sortable(value: float) -> float:
    """NaN (an empty stratum) ranks last, never poisons the sort's total order."""
    return float("inf") if math.isnan(value) else value


def rank_candidates(cards: tuple[Scorecard, ...]) -> tuple[Scorecard, ...]:
    """LEXICOGRAPHIC rank — the scalar is a headline, never the selector."""
    return tuple(
        sorted(
            cards,
            key=lambda c: (
                c.wip,
                _sortable(c.kpis.values["door_to_provider_s_mean"]),
                _sortable(c.kpis.values["staff_minutes_walked"]),
                c.run_id.root,
            ),
        )
    )


__all__ = ["Scorecard", "fold_scorecard", "objective_inputs", "rank_candidates"]
