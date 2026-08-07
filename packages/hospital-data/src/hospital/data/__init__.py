"""hospital.data — deterministic ER floor-layout and patient/workload generators.

Re-exports only the package's public surface (doc 00 §3 / doc 02 §2); everything
else (``FacilitySpec``, ``WorkloadSpec``, ``apply_overlay``, ``realize_staff``,
``MovementRow``/``MovementTable``, the M3 ``VitalsSample``/``VitalsStream``, …)
is reachable by importing the owning submodule directly
(``hospital.data.scenario``, ``hospital.data.movement``, ``hospital.data.vitals``).
"""

from __future__ import annotations

from hospital.data.hospital import generate_hospital
from hospital.data.layout import generate_floor
from hospital.data.movement import export_movement_traces
from hospital.data.scenario import (
    ElevatorSpec,
    FloorSpec,
    HospitalSpec,
    Scenario,
    dump_scenario,
    load_arm,
    load_scenario,
)
from hospital.data.vitals import generate_vitals
from hospital.data.workload import PatientArrival, generate_workload

__all__ = [
    "ElevatorSpec",
    "FloorSpec",
    "HospitalSpec",
    "PatientArrival",
    "Scenario",
    "dump_scenario",
    "export_movement_traces",
    "generate_floor",
    "generate_hospital",
    "generate_vitals",
    "generate_workload",
    "load_arm",
    "load_scenario",
]
