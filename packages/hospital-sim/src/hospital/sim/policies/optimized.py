"""The OPTIMIZED arm — thin adapters over ``hospital.solver`` backends (doc 04 §3.8).

Every policy below is a *marshaller*, never an optimizer (anti-dup rule 3): it
hands the immutable ``DecisionInput`` (+ the read-only ``RoutingOracle``) to the
canonical ``hospital.solver`` lever and returns that lever's ``PlanItem``s.
No optimization math, pathfinding, RNG, event formatting, or validation lives
here — the placement backend self-checks with the one ``validate()`` before
returning, and the seam adapter re-checks on apply (one implementation, two
enforcement points). The policies draw no randomness at all; the reserved
``substream("policy", ...)`` domain stays untouched (CP-SAT is deterministic
given its input, doc 04 §4.5).

Adapter-shape notes (judgment calls, recorded in the build report):

* **Placement** goes through the registry backend (``placement_cpsat`` by
  default), is warm-started from the previous stamped plan (doc 03 §4.3's
  rolling re-solve), and every result passes ``solver.stamping.stamp`` — the
  provenance choke point — before its items are returned.
* **Sequencing** reuses ``solver.sequencing.sequence`` (the one scoring
  function) and re-shapes its ranked per-patient items into the single
  ``order``-payload item the seam enacts (``World.resequence_waiting`` reads
  ``PlanItem.order``); the ranking itself stays in ``solver``.
* **Turnaround/discharge** hold the oracle from construction: their lever
  Protocols pass only the ``DecisionInput`` (doc 04 §3.6), while the solver
  functions price response travel through the oracle — a read-only query
  surface, threaded once by the factory. Their PRIORITY outputs reach physics
  through the dispatch cost (``SolverDispatch`` folds
  ``solver.dispatch.priority_urgencies`` into ``u(t)``): the enacted
  ``clean``/``discharge`` items only boost a FIFO the global matching never
  reads, so a priority that stayed out of the cost was a no-op under scarcity.
* **Discharge** feeds the load gate the neutral ``FloorLoad()`` default:
  ``DecisionInput`` deliberately carries no utilization signal (no hidden
  fields), so in v1 documentation is always in the promoted band.
* **Staffing** is input-only in v1 (🟡 A7): the roster comes from the
  scenario, so the optimized arm reuses the baseline ``InputStaffing`` no-op
  lever rather than duplicating it.

Each adapter early-returns ``()`` when its lever's candidate set is empty (no
waiting patient, no FREE bay, no pending task, no CLEANING bay). That guard is
pure marshalling — the lever would return an empty plan anyway — and keeps the
per-tick cost proportional to what there is to decide.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hospital.core import (
    BayStatus,
    CompiledRules,
    DecisionInput,
    Duration,
    PatientId,
    PlanItem,
    StaffMember,
)
from hospital.sim.policies.baseline import InputStaffing
from hospital.sim.policies.protocols import PolicySet
from hospital.solver import (
    ObjectiveConfig,
    RoutingOracle,
    Solver,
    SolverStatus,
    assign_staff,
    get_backend,
    prioritize_cleaning,
    prioritize_discharge,
    priority_urgencies,
    stamp,
)
from hospital.solver.discharge import FloorLoad
from hospital.solver.sequencing import DEFAULT_STARVATION_RATE
from hospital.solver.sequencing import sequence as score_sequence

if TYPE_CHECKING:
    from hospital.core import Plan

# The registry name of the default placement backend (doc 03 §3.2).
PLACEMENT_BACKEND = "placement_cpsat"


# Ordering of solve claims by strength: OPTIMAL is a proof, FEASIBLE a
# cap-truncated incumbent, HEURISTIC a constructive fallback with no claim.
# The "worst observed" over a run is the weakest claim any tick relied on.
_STATUS_SEVERITY: dict[SolverStatus, int] = {
    SolverStatus.OPTIMAL: 0,
    SolverStatus.FEASIBLE: 1,
    SolverStatus.HEURISTIC: 2,
}


@dataclass
class SolverPlacement:
    """``PlacementPolicy`` — marshal to the registry placement backend + stamp.

    Holds the previous stamped plan as the next solve's warm start (the rolling
    re-solve of doc 03 §4.3). ``last_status`` exposes the most recent solve's
    ``SolverStatus`` claim and ``worst_status`` the weakest claim observed over
    the whole run — recorded, never hidden (PLAN §5): a week that ever fell
    back to the greedy heuristic must not present itself as proven-optimal.
    """

    backend: Solver
    objective: ObjectiveConfig
    rules: CompiledRules
    # Per-patient predicted length of stay, or empty. Held here rather than read from
    # `DecisionInput` because a prediction is not floor state (see `Solver.solve`).
    expected_stay: Mapping[PatientId, Duration] = field(default_factory=dict[PatientId, Duration])
    _warm: Plan | None = field(default=None, init=False, repr=False)

    @property
    def last_status(self) -> str | None:
        return self._last_status

    @property
    def worst_status(self) -> SolverStatus | None:
        """The weakest solve claim of the run (None until the first solve)."""
        return self._worst_status

    _last_status: str | None = field(default=None, init=False, repr=False)
    _worst_status: SolverStatus | None = field(default=None, init=False, repr=False)

    def place(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]:
        if not di.waiting or not any(bs.status is BayStatus.FREE for bs in di.bays):
            return ()  # nothing to place / nowhere to place — the empty solve
        result = self.backend.solve(
            di,
            oracle,
            config=self.objective,
            rules=self.rules,
            warm_start=self._warm,
            expected_stay=self.expected_stay or None,
        )
        stamped = stamp(
            result, self.objective, rules_hash=self.rules.rules_hash or None, now=di.now
        )
        self._warm = stamped.plan
        self._last_status = stamped.status.value
        if self._worst_status is None or (
            _STATUS_SEVERITY[stamped.status] > _STATUS_SEVERITY[self._worst_status]
        ):
            self._worst_status = stamped.status
        return stamped.plan.items


@dataclass(frozen=True)
class SolverSequencing:
    """``SequencingPolicy`` — the one acuity+anti-starvation scoring, re-shaped.

    ``solver.sequencing.sequence`` emits one ranked item per patient; the seam
    enacts a queue order through the single ``order`` payload
    (``World.resequence_waiting``), so the ranked items are marshalled into one
    ``sequence`` item whose ``order`` is the scored service order.
    """

    objective: ObjectiveConfig
    starvation_rate: int = DEFAULT_STARVATION_RATE

    def sequence(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        if not di.waiting:
            return ()
        ranked = score_sequence(di, config=self.objective, starvation_rate=self.starvation_rate)
        order = tuple(item.patient.root for item in ranked if item.patient is not None)
        if not order:
            return ()
        return (PlanItem(stable_id="seq:waiting", kind="sequence", order=order),)


@dataclass(frozen=True)
class SolverDispatch:
    """``DispatchPolicy`` — the global assignment (CP-SAT matching) lever.

    ``solver.dispatch.assign_staff`` is serve-first (max-cardinality), then
    min weighted cost ``w_time·u(t)·(waited + travel) + w_travel·travel``
    (urgency and travel trade continuously; deferral under scarcity is priced
    at ``unplaced_wait_penalty``) over idle qualified staff x pending tasks,
    judged on the same skill union the validator applies. A new task triggers
    a same-instant decision tick (``World.add_task`` callers request one), so
    a high-urgency arrival is reconsidered immediately.

    ``u(t)`` is priority-augmented via ``solver.dispatch.priority_urgencies``:
    the turnaround/discharge levers' value-of-unblocking and the documentation
    load gate enter the SAME weighted cost the matching minimizes — dispatch
    is the actuator (it hands tasks to staff), so a priority that never
    reaches its cost is a no-op under scarcity. The neutral ``FloorLoad()``
    default mirrors ``SolverDischarge`` (v1: no utilization signal).
    """

    objective: ObjectiveConfig
    rules: CompiledRules
    roster: tuple[StaffMember, ...]

    def dispatch(self, di: DecisionInput, oracle: RoutingOracle) -> tuple[PlanItem, ...]:
        if not di.pending_tasks:
            return ()
        overrides = priority_urgencies(di, config=self.objective, rules=self.rules)
        return assign_staff(
            di,
            oracle,
            config=self.objective,
            rules=self.rules,
            staff_members=self.roster,
            urgency_override=overrides,
        )


@dataclass(frozen=True)
class SolverTurnaround:
    """``TurnaroundPolicy`` — cleaning as value-of-unblocking assignment."""

    oracle: RoutingOracle
    objective: ObjectiveConfig
    rules: CompiledRules
    roster: tuple[StaffMember, ...]

    def turnaround(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        if not any(bs.status is BayStatus.CLEANING for bs in di.bays):
            return ()
        return prioritize_cleaning(
            di, self.oracle, config=self.objective, rules=self.rules, staff_members=self.roster
        )


@dataclass(frozen=True)
class SolverDischarge:
    """``DischargePolicy`` — discharge expedite + documentation load gate."""

    oracle: RoutingOracle
    objective: ObjectiveConfig
    rules: CompiledRules

    def discharge(self, di: DecisionInput) -> tuple[PlanItem, ...]:
        if not any(t.kind in ("discharge", "documentation") for t in di.pending_tasks):
            return ()
        return prioritize_discharge(
            di, self.oracle, config=self.objective, load=FloorLoad(), rules=self.rules
        )


def make_optimized_policies(
    *,
    oracle: RoutingOracle,
    objective: ObjectiveConfig,
    rules: CompiledRules,
    roster: tuple[StaffMember, ...],
    placement_backend: str = PLACEMENT_BACKEND,
    expected_stay: Mapping[PatientId, Duration] | None = None,
) -> PolicySet:
    """Wire the solver-backed ``PolicySet`` (origin ``"solver"``)."""
    return PolicySet(
        placement=SolverPlacement(
            backend=get_backend(placement_backend),
            objective=objective,
            rules=rules,
            expected_stay=dict(expected_stay or {}),
        ),
        sequencing=SolverSequencing(objective=objective),
        dispatch=SolverDispatch(objective=objective, rules=rules, roster=roster),
        turnaround=SolverTurnaround(oracle=oracle, objective=objective, rules=rules, roster=roster),
        discharge=SolverDischarge(oracle=oracle, objective=objective, rules=rules),
        staffing=InputStaffing(),
        origin="solver",
    )


__all__ = [
    "DEFAULT_STARVATION_RATE",
    "PLACEMENT_BACKEND",
    "SolverDischarge",
    "SolverDispatch",
    "SolverPlacement",
    "SolverSequencing",
    "SolverTurnaround",
    "make_optimized_policies",
]
