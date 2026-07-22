"""``hospital.sim.physics`` — the single mutable island of the repo.

``world`` owns every fact that changes during a run; ``executor`` is the sole
advancer of ``env.now``; ``service_times`` is the one physics sampler wiring;
``resources`` builds the contended SimPy pools. Import the submodules directly
(``hospital.sim.physics.world``, …) — nothing is re-exported here (doc 00 §3).
"""

from __future__ import annotations
