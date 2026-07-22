"""``export_movement_traces`` — a pure fold from ``EventLog`` to per-edge rows.

No RNG, one pass, O(events) (doc 02 §2.5). Reads via ``core.events`` iteration —
the only event-reader path (anti-duplication rule 4) — and emits *raw* rows,
deliberately not aggregates: KPI aggregation is ``analysis.fold``'s job, and
duplicating it here would fork that logic.

**Deviation from the doc-02 §4.5 one-line signature:** ``MovementRow.distance``
is a required ``Distance``, but ``StaffMoved``/``PatientMoved`` events carry
only ``edge`` and ``seconds`` — not ``distance``. Doc 02 §5.5 itself says
``distance`` is "looked up from the run's ``RouteGraph`` (passed alongside the
log ...)", which the one-arg signature can't do. This implementation therefore
takes the run's ``graph`` as a required second parameter; an edge absent from
it is a contract violation (raises :class:`~hospital.core.LayoutError`), never
a silently-zeroed distance.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Literal

from hospital.core import (
    Distance,
    Duration,
    EventLog,
    FrozenModel,
    LayoutError,
    NodeId,
    PatientMoved,
    RouteGraph,
    SimTime,
    StaffMoved,
    hours,
)

_US_PER_HOUR = hours(1).root
_US_PER_DAY = _US_PER_HOUR * 24

_CSV_COLUMNS: tuple[str, ...] = (
    "entity",
    "entity_kind",
    "a",
    "b",
    "distance_cm",
    "seconds_us",
    "occurred_at_us",
    "sim_day",
    "hour_of_day",
)


class MovementRow(FrozenModel):
    """One staff/patient traversal of one graph edge."""

    entity: str
    entity_kind: Literal["staff", "patient"]
    a: NodeId
    b: NodeId
    distance: Distance
    seconds: Duration
    occurred_at: SimTime
    sim_day: int
    hour_of_day: int


class MovementTable(FrozenModel):
    """The folded movement rows, plus columnar/CSV export helpers."""

    rows: tuple[MovementRow, ...]

    def to_columns(self) -> Mapping[str, tuple[object, ...]]:
        """Dict-of-tuples (columnar) — lets ``forecast.features`` build a frame without pandas."""
        return {
            "entity": tuple(r.entity for r in self.rows),
            "entity_kind": tuple(r.entity_kind for r in self.rows),
            "a": tuple(r.a.root for r in self.rows),
            "b": tuple(r.b.root for r in self.rows),
            "distance_cm": tuple(r.distance.root for r in self.rows),
            "seconds_us": tuple(r.seconds.root for r in self.rows),
            "occurred_at_us": tuple(r.occurred_at.root for r in self.rows),
            "sim_day": tuple(r.sim_day for r in self.rows),
            "hour_of_day": tuple(r.hour_of_day for r in self.rows),
        }

    def to_csv(self) -> str:
        """The golden-fixture CSV form: header + one row per movement, fixed column order.

        Written through the ``csv`` module (RFC 4180 quoting, ``\\n`` line
        terminator) so an id containing a comma, quote, or newline is escaped
        rather than corrupting the table.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(_CSV_COLUMNS)
        for r in self.rows:
            writer.writerow(
                (
                    r.entity,
                    r.entity_kind,
                    r.a.root,
                    r.b.root,
                    r.distance.root,
                    r.seconds.root,
                    r.occurred_at.root,
                    r.sim_day,
                    r.hour_of_day,
                )
            )
        return buffer.getvalue()


def _sim_day(occurred_at: SimTime) -> int:
    return occurred_at.root // _US_PER_DAY


def _hour_of_day(occurred_at: SimTime) -> int:
    return (occurred_at.root // _US_PER_HOUR) % 24


def _edge_distance_lookup(graph: RouteGraph) -> dict[tuple[str, str], Distance]:
    lookup: dict[tuple[str, str], Distance] = {}
    for e in graph.edges:
        lookup[(e.a.root, e.b.root)] = e.distance
        if e.bidirectional:
            lookup[(e.b.root, e.a.root)] = e.distance
    return lookup


def _row(
    entity: str,
    kind: Literal["staff", "patient"],
    edge: tuple[NodeId, NodeId],
    seconds: Duration,
    occurred_at: SimTime,
    lookup: Mapping[tuple[str, str], Distance],
) -> MovementRow:
    a, b = edge
    key = (a.root, b.root)
    if key not in lookup:
        raise LayoutError(f"movement edge {a.root}->{b.root} is absent from the run's RouteGraph")
    return MovementRow(
        entity=entity,
        entity_kind=kind,
        a=a,
        b=b,
        distance=lookup[key],
        seconds=seconds,
        occurred_at=occurred_at,
        sim_day=_sim_day(occurred_at),
        hour_of_day=_hour_of_day(occurred_at),
    )


def export_movement_traces(log: EventLog, graph: RouteGraph) -> MovementTable:
    """Fold every ``staff_moved``/``patient_moved`` event into a raw ``MovementRow``.

    Iterates ``log.ordered()`` (canonical ``(occurred_at, sequence)`` order, per
    the core determinism contract) so rows come out time-ordered for as-of
    slicing. Pure fold: no ``RandomStreams``, and running it twice on the same
    ``(log, graph)`` is byte-identical.
    """
    lookup = _edge_distance_lookup(graph)
    rows: list[MovementRow] = []
    for env in log.ordered():
        event = env.event
        if isinstance(event, StaffMoved):
            rows.append(
                _row(
                    str(event.staff), "staff", event.edge, event.seconds, event.occurred_at, lookup
                )
            )
        elif isinstance(event, PatientMoved):
            rows.append(
                _row(
                    str(event.patient),
                    "patient",
                    event.edge,
                    event.seconds,
                    event.occurred_at,
                    lookup,
                )
            )
    return MovementTable(rows=tuple(rows))


__all__ = ["MovementRow", "MovementTable", "export_movement_traces"]
