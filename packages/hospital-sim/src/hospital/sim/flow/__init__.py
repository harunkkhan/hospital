"""``hospital.sim.flow`` — the SimPy processes: patients and staff agents.

One process per patient (the 9-step ER flow) and one per staff member
(idle -> travel -> serve -> idle). Import the submodules directly — nothing is
re-exported here (doc 00 §3).
"""

from __future__ import annotations
