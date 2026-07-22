"""``export_movement_traces``: fold correctness, no RNG, deterministic replay.

A cross-reader agreement check against ``analysis.fold``'s
``staff_minutes_walked`` (doc 02 §8) is deferred: ``hospital-analysis`` is still
an empty scaffold in this worktree (built by a separate package), so there is
nothing to cross-check against yet.
"""

from __future__ import annotations

import csv
import io

from hospital.core import (
    Distance,
    EventLog,
    NodeId,
    PatientId,
    PatientMoved,
    RouteEdge,
    RouteGraph,
    RouteNode,
    SimTime,
    StaffId,
    StaffMoved,
    WalkSpeed,
    walk_duration,
)
from hospital.data.movement import export_movement_traces

_SPEED = WalkSpeed(100)
_HOUR_US = 3_600_000_000
_DAY_US = 24 * _HOUR_US


def _tiny_graph() -> RouteGraph:
    nodes = (
        RouteNode(id=NodeId("a"), label="x", x_cm=0, y_cm=0),
        RouteNode(id=NodeId("b"), label="x", x_cm=500, y_cm=0),
    )
    d = Distance(500)
    edge = RouteEdge(a=NodeId("a"), b=NodeId("b"), distance=d, seconds=walk_duration(d, _SPEED))
    return RouteGraph(nodes=nodes, edges=(edge,))


def _build_log(graph: RouteGraph) -> EventLog:
    edge = graph.edges[0]
    log = EventLog()
    log.append(
        StaffMoved(
            occurred_at=SimTime(_HOUR_US * 2),
            staff=StaffId("s1"),
            edge=(edge.a, edge.b),
            seconds=edge.seconds,
        )
    )
    log.append(
        PatientMoved(
            occurred_at=SimTime(_DAY_US + _HOUR_US * 5),
            patient=PatientId("p1"),
            edge=(edge.b, edge.a),
            seconds=edge.seconds,
        )
    )
    return log


def test_fold_produces_one_row_per_movement_event() -> None:
    graph = _tiny_graph()
    log = _build_log(graph)
    table = export_movement_traces(log, graph)
    assert len(table.rows) == 2


def test_row_fields_match_the_source_events() -> None:
    graph = _tiny_graph()
    log = _build_log(graph)
    table = export_movement_traces(log, graph)

    staff_row = next(r for r in table.rows if r.entity_kind == "staff")
    assert staff_row.entity == "s1"
    assert staff_row.a == NodeId("a")
    assert staff_row.b == NodeId("b")
    assert staff_row.distance == Distance(500)
    assert staff_row.sim_day == 0
    assert staff_row.hour_of_day == 2

    patient_row = next(r for r in table.rows if r.entity_kind == "patient")
    assert patient_row.entity == "p1"
    assert patient_row.distance == Distance(500)  # recovered via the bidirectional edge
    assert patient_row.sim_day == 1
    assert patient_row.hour_of_day == 5


def test_export_takes_no_rng_and_is_byte_identical_on_repeat() -> None:
    graph = _tiny_graph()
    log = _build_log(graph)
    first = export_movement_traces(log, graph)
    second = export_movement_traces(log, graph)
    assert first == second
    assert first.to_csv() == second.to_csv()


def test_to_columns_is_columnar_and_aligned_with_rows() -> None:
    graph = _tiny_graph()
    log = _build_log(graph)
    table = export_movement_traces(log, graph)
    columns = table.to_columns()
    assert len(columns["entity"]) == len(table.rows)
    for i, row in enumerate(table.rows):
        assert columns["entity"][i] == row.entity
        assert columns["distance_cm"][i] == row.distance.root
        assert columns["sim_day"][i] == row.sim_day


def test_to_csv_has_header_and_one_line_per_row() -> None:
    graph = _tiny_graph()
    log = _build_log(graph)
    table = export_movement_traces(log, graph)
    lines = table.to_csv().splitlines()
    assert (
        lines[0]
        == "entity,entity_kind,a,b,distance_cm,seconds_us,occurred_at_us,sim_day,hour_of_day"
    )
    assert len(lines) == 1 + len(table.rows)


def test_empty_log_yields_empty_table() -> None:
    graph = _tiny_graph()
    table = export_movement_traces(EventLog(), graph)
    assert table.rows == ()


# Finding 14: a raw string join corrupts the table when an id contains a comma,
# quote, or newline — the CSV must escape per RFC 4180 and parse back exactly.
def test_to_csv_escapes_ids_containing_delimiters_quotes_and_newlines() -> None:
    graph = _tiny_graph()
    edge = graph.edges[0]
    tricky = 'nurse,7 "the fast"\nnight shift'
    log = EventLog()
    log.append(
        StaffMoved(
            occurred_at=SimTime(_HOUR_US),
            staff=StaffId(tricky),
            edge=(edge.a, edge.b),
            seconds=edge.seconds,
        )
    )
    table = export_movement_traces(log, graph)
    parsed = list(csv.reader(io.StringIO(table.to_csv())))
    assert len(parsed) == 2  # header + one data row, despite the embedded newline
    header, row = parsed
    assert header == [
        "entity",
        "entity_kind",
        "a",
        "b",
        "distance_cm",
        "seconds_us",
        "occurred_at_us",
        "sim_day",
        "hour_of_day",
    ]
    assert row[0] == tricky  # round-trips exactly
    assert row[1] == "staff"
    assert row[4] == "500"
