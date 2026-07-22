"""``build_resources`` — pool shapes, triage discovery, directed mailboxes."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import simpy
from _sim_fixtures import build_physics, tiny_facility

from hospital.core import LayoutError
from hospital.data.layout import generate_floor
from hospital.sim.physics.resources import build_resources


def test_pools_match_layout() -> None:
    h = build_physics()
    assert set(h.resources.imaging) == set(h.layout.imaging_nodes)
    assert set(h.resources.lab) == set(h.layout.lab_nodes)
    assert all(res.capacity == 1 for res in h.resources.imaging.values())
    assert h.resources.triage.capacity == len(h.resources.triage_nodes) == 2


def test_one_mailbox_per_staff() -> None:
    h = build_physics()
    assert set(h.resources.mailboxes) == {m.id for m in h.roster}


def test_mailbox_dispatch_is_directed_not_a_race() -> None:
    h = build_physics()
    a, b = h.roster[0], h.roster[1]
    h.resources.mailboxes[a.id].put("task-for-a")
    got: dict[str, object] = {}

    def listen(name: str, store: simpy.Store) -> Generator[simpy.Event, object]:
        item = yield store.get()
        got[name] = item

    h.env.process(listen("a", h.resources.mailboxes[a.id]))
    h.env.process(listen("b", h.resources.mailboxes[b.id]))
    h.env.run(until=1)
    assert got == {"a": "task-for-a"}


def test_layout_without_triage_rooms_is_refused() -> None:
    layout = generate_floor(tiny_facility().model_copy(update={"triage_rooms": 0}))
    env = simpy.Environment()
    with pytest.raises(LayoutError, match="no triage rooms"):
        build_resources(env, layout, ())
