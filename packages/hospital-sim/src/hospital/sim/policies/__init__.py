"""``hospital.sim.policies`` — the decision layer behind one Protocol set.

``protocols`` defines the six lever protocols and ``PolicySet``; ``baseline``
is the myopic null arm; ``optimized`` (next phase) adapts ``hospital.solver``
backends; ``factory.make_policies`` is the single spot an arm is chosen.
Import the submodules directly — nothing is re-exported here (doc 00 §3).
"""

from __future__ import annotations
