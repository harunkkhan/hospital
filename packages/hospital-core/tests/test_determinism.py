"""Process-order-independence of the rule provenance hash (finding #1).

``rules_hash`` is computed under several ``PYTHONHASHSEED`` values in fresh
subprocesses. ``frozenset`` iteration order over ``str``/``StrEnum`` elements is
salted by the hash seed, so a hash that leaks set-iteration order differs per
process. A content-addressable provenance hash must not.
"""

from __future__ import annotations

import os
import subprocess
import sys

_TESTS_DIR = os.path.dirname(__file__)


def _run(script: str, seed: int) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONPATH"] = _TESTS_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


_HASH_SCRIPT = """
from hospital.core import (
    CapacityRule, CompatibilityRule, EsiAcuity, PrecedenceRule, SkillRule, ZoneType,
)
from hospital.core.enums import Activity
from hospital.core.rules import rules_hash

rules = (
    CompatibilityRule(
        allowed_zone_types=frozenset({
            (EsiAcuity.ESI1, ZoneType.RESUS_TRAUMA),
            (EsiAcuity.ESI1, ZoneType.GENERAL),
            (EsiAcuity.ESI3, ZoneType.GENERAL),
            (EsiAcuity.ESI3, ZoneType.FAST_TRACK),
            (EsiAcuity.ESI5, ZoneType.FAST_TRACK),
            (EsiAcuity.ESI2, ZoneType.OBSERVATION),
            (EsiAcuity.ESI4, ZoneType.IMAGING),
        }),
        required_equipment=frozenset({
            (EsiAcuity.ESI1, "monitor"), (EsiAcuity.ESI2, "vent"), (EsiAcuity.ESI3, "iv"),
        }),
    ),
    CapacityRule(zone_type=ZoneType.GENERAL, max_occupancy=2),
    SkillRule(task_kind="provider_visit", required_skills=frozenset({"md", "acls", "picc"})),
    PrecedenceRule(before=Activity.TRIAGE, after=Activity.PROVIDER_VISIT),
)
print(rules_hash(rules))
"""


def test_rules_hash_is_stable_across_hash_seeds() -> None:
    hashes = {_run(_HASH_SCRIPT, seed) for seed in range(6)}
    assert len(hashes) == 1, f"rules_hash varied across hash seeds: {hashes}"
