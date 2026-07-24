"""Contended SimPy pools + staff mailboxes (doc 04 §3.4).

Three deliberately different resource semantics:

* **imaging / lab / triage — capacity-N ``PriorityResource``.** The genuine
  contention points ("the CT is busy"). Requests carry
  ``priority = -esi.priority_weight()`` — SimPy serves ascending, so ESI-1
  (weight 5 -> priority -5) jumps the queue; the sign inversion lives in the one
  core helper, never re-derived here (DECISIONS D8). Each imaging suite / lab
  station is its own capacity-1 resource at its own graph node.
* **bays — no SimPy resource at all** (🟡 A6 exercised). Bay allocation is
  *decided* by the placement policy, never FIFO-raced, so a cap-1 ``Resource``
  would be a second representation of the bay lifecycle that
  ``World.bay_status`` already owns — two representations drift; one is dropped.
  The occupied→cleaning hold is the ``CLEANING`` status itself.
* **staff — agents, not counters.** One ``Store`` mailbox per staff member;
  dispatch *names* a staff (``mailbox[staff].put(task)``) and the agent blocks
  on ``get()``. There is no ``Resource(capacity=n_nurses)`` — a counted pool
  would erase position and with it the walking cost this project measures.

Triage rooms have no dedicated ``FloorLayout`` field, so they are discovered by
node label (``data.layout`` labels them ``"triage"``) — the one place a label is
load-bearing (judgment call, flagged in the build report).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import simpy

from hospital.core import FloorLayout, LayoutError, NodeId, StaffId, StaffMember


@dataclass(frozen=True)
class ResourcePool:
    """A frozen container of mutable SimPy handles (mappings fixed at build)."""

    imaging: Mapping[NodeId, simpy.PriorityResource]
    lab: Mapping[NodeId, simpy.PriorityResource]
    triage: simpy.PriorityResource
    triage_nodes: tuple[NodeId, ...]
    mailboxes: Mapping[StaffId, simpy.Store]


def build_resources(
    env: simpy.Environment, layout: FloorLayout, staff: tuple[StaffMember, ...]
) -> ResourcePool:
    """Build the pools for ``layout`` and one mailbox per realized staff member.

    Takes the *realized* roster (not the ``StaffingSpec``): mailboxes are keyed
    by concrete ``StaffId``s, which only exist after ``data.realize_staff`` —
    the composition root realizes first, then builds (deviation from the doc's
    ``staffing: StaffingSpec`` parameter, noted in the build report).
    """
    imaging = {n: simpy.PriorityResource(env, capacity=1) for n in layout.imaging_nodes}
    lab = {n: simpy.PriorityResource(env, capacity=1) for n in layout.lab_nodes}
    triage_nodes = tuple(
        sorted((n.id for n in layout.graph.nodes if n.label == "triage"), key=lambda i: i.root)
    )
    if not triage_nodes:
        raise LayoutError("layout has no triage rooms (no nodes labelled 'triage')")
    triage = simpy.PriorityResource(env, capacity=len(triage_nodes))
    mailboxes = {m.id: simpy.Store(env) for m in staff}
    return ResourcePool(
        imaging=imaging,
        lab=lab,
        triage=triage,
        triage_nodes=triage_nodes,
        mailboxes=mailboxes,
    )


__all__ = ["ResourcePool", "build_resources"]
