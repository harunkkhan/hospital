"""The vitals reading value type — one shape, shared by the generator and the seam.

``core.events.VitalsSampled`` deliberately carries only ``(patient, news2)``: the
event log is a published artifact (the web console types it verbatim), and a full
physiological trace on every tick would bloat it for the sake of one consumer.
But a risk monitor needs the actual numbers, not just their NEWS2 aggregate.

So the reading is a value type owned here, and travels **alongside** the event
into :meth:`hospital.core.seam.RiskMonitor.observe`. ``data.vitals.VitalsSample``
extends it with a timestamp rather than restating the fields, so the generator's
output and the seam's input are the same six numbers by construction — there is
no second declaration to drift.

Units are integer-scaled for exact reproducibility: temperature is tenths of a
degree Celsius (``temp_c_x10``), everything else a whole unit.
"""

from __future__ import annotations

from hospital.core.models import FrozenModel


class VitalsReading(FrozenModel):
    """One set of observed vitals, without a time. Integer-scaled throughout."""

    hr: int
    spo2: int
    sbp: int
    dbp: int
    temp_c_x10: int
    rr: int

    @property
    def temp_c(self) -> float:
        """Temperature in degrees Celsius — the scale the NEWS2 rubric is written in."""
        return self.temp_c_x10 / 10.0


__all__ = ["VitalsReading"]
