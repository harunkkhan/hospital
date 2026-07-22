"""``vitals.py`` is an M3 stub — asserts the documented signature raises cleanly.

Full M3 vitals-generation tests (deterministic per-patient trajectories,
append-only lengthening, monotone post-onset drift, acuity-conditioned
deterioration frequency — doc 02 §8) are deferred along with the feature
itself; this only pins the stub's contract so downstream milestones can
depend on the shape of the API today.
"""

from __future__ import annotations

import pytest

from hospital.core import (
    ArrivalMode,
    EsiAcuity,
    Patient,
    PatientId,
    RandomStreams,
    SimTime,
    WorkupNeeds,
    hours,
)
from hospital.data.vitals import generate_vitals


def _patient() -> Patient:
    return Patient(
        id=PatientId("p1"),
        arrival_time=SimTime(0),
        arrival_mode=ArrivalMode.WALK_IN,
        esi=EsiAcuity.ESI3,
        complaint="chest_pain",
        isolation_required=False,
        workup=WorkupNeeds(provider_visits=1, nurse_visits=1, imaging=(), labs=0, procedures=0),
    )


def test_generate_vitals_raises_not_implemented_for_m3() -> None:
    with pytest.raises(NotImplementedError, match="M3"):
        generate_vitals(_patient(), RandomStreams(1), until=hours(4), cadence=hours(1))
