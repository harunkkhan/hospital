"""``hospital.sim.experiment`` — composition root, disruptions, scorecard, comparison.

``replication.run_replication`` is the only place components are constructed
and wired; ``disruptions`` injects exogenous stress identically across arms;
``scorecard``/``comparison`` fold and compare runs by delegating to
``hospital.analysis``/``hospital.solver``. Import the submodules directly —
nothing is re-exported here (doc 00 §3).
"""

from __future__ import annotations
