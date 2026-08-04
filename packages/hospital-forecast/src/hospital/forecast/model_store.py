"""Versioned artifacts and the prediction -> solver-input bridge (doc 06 §9).

Two responsibilities:

* :class:`ModelStore` — versioned payloads on disk at
  ``<root>/<name>/<version>/{payload.joblib, meta.json}``, with a champion
  pointer. Artifacts are content-addressed by ``(data_hash, config_hash)``, so
  the same data and config reproduce the same version string.
* :class:`PredictionBundle` / :class:`PredictionAdapter` — the seam. ``forecast``
  cannot be imported by ``solver`` or ``sim`` (the graph runs downward), so
  predictions are *published as data*: a bundle of **core-typed values** that a
  composition root reads and passes into the solver's existing input slots.

**Deviation from doc 06 §9, on purpose.** That sketch types the bundle's fields
as ``forecast`` classes (``ServiceTimeTable``, ``ArrivalIntensityModel``), which
would defeat the stated goal in the same paragraph — "composed **only** of core
types ... so the composition root can pass its fields to the solver **without any
downstream package importing forecast**". The fields here are therefore plain
mappings of ``core`` types. The bundle is a *message*, not a model handle.

``static_bundle`` produces the identically-shaped bundle from scenario constants.
That symmetry is the whole point: the A/B harness swaps one call and changes
nothing else, so a measured delta is attributable to the predictions rather than
to two different code paths.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from hospital.core import (
    Activity,
    Duration,
    EsiAcuity,
    FrozenModel,
    PatientId,
    StaffRole,
    seconds,
)
from hospital.forecast import _estimators

if TYPE_CHECKING:
    from hospital.forecast.arrivals import ArrivalIntensityModel, SurgeForecast
    from hospital.forecast.service_time import ServiceTimeTable

_META_FILE: Final[str] = "meta.json"
_PAYLOAD_FILE: Final[str] = "payload.joblib"
_CHAMPION_FILE: Final[str] = "champion.txt"
LATEST: Final[str] = "latest"

# The key the solver's expected-duration lookups are indexed by. A tuple of core
# enums plus the complaint string — no forecast type crosses the seam.
ServiceKey = tuple[Activity, EsiAcuity, str]


def canonical_hash(payload: object) -> str:
    """Canonical sorted-JSON sha256 (doc 06 §13-7).

    Forecast-local: ``core.rules_hash`` and ``solver.objective.config_hash`` each
    canonicalize one specific model rather than exposing a reusable helper, so
    there is nothing to import. The *convention* is shared — sorted keys, tight
    separators, sha256 — so hashes from all three read alike.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ArtifactMeta(FrozenModel):
    """What a stored version is, and what it scored."""

    name: str
    version: str
    data_hash: str
    config_hash: str
    trained_at: str
    metrics: Mapping[str, float]
    is_champion: bool = False


class PredictionBundle(FrozenModel):
    """Solver inputs, as core-typed data. No ``forecast`` import needed to read it.

    Every field mirrors an input the solver already accepts, so wiring a bundle in
    is a substitution at the call site rather than a new code path:

    * ``expected_service`` / ``fallback_service`` -> the expected durations behind
      ``solver.placement``'s ``w[p,b]`` and the ``solver.objective`` time term;
    * ``per_patient_los`` -> turnaround value and discharge estimates;
    * ``arrival_rates_per_hour`` / ``surge_upper`` -> ``solver.scheduling``'s
      covering demand;
    * ``staffing_hint`` -> the optional headcount suggestion (doc 06 §13-10).
    """

    version: str
    resolution: Duration
    arrival_rates_per_hour: tuple[float, ...]
    expected_service: Mapping[ServiceKey, Duration]
    fallback_service: Mapping[Activity, Duration]
    per_patient_los: Mapping[PatientId, Duration] = {}
    surge_upper: tuple[float, ...] | None = None
    staffing_hint: Mapping[tuple[StaffRole, int], int] | None = None

    def expected_for(self, activity: Activity, esi: EsiAcuity, complaint: str) -> Duration:
        """The best available expected duration, falling back per activity.

        Raises rather than return zero for a wholly unknown activity: telling the
        solver that a task takes no time is worse than telling it nothing.
        """
        found = self.expected_service.get((activity, esi, complaint))
        if found is not None:
            return found
        fallback = self.fallback_service.get(activity)
        if fallback is None:
            raise KeyError(f"no expected duration for {activity}")
        return fallback


class ModelStore:
    """Versioned artifacts on disk, with a champion pointer per model name."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def save(self, name: str, version: str, payload: object, meta: ArtifactMeta) -> None:
        """Write a payload and its metadata. The first version saved is champion.

        A store whose only version is not champion would make ``load("latest")``
        fail on a fresh install, so the first save promotes itself.
        """
        target = self._dir(name, version)
        target.mkdir(parents=True, exist_ok=True)
        _estimators.dump(payload, target / _PAYLOAD_FILE)
        (target / _META_FILE).write_text(meta.model_dump_json(indent=2))
        if self._champion_pointer(name) is None:
            self.promote(name, version)

    def _champion_pointer(self, name: str) -> str | None:
        pointer = self.root / name / _CHAMPION_FILE
        if not pointer.is_file():
            return None
        return pointer.read_text().strip() or None

    def promote(self, name: str, version: str) -> None:
        """Make ``version`` the champion. Refuses a version that was never saved."""
        if not self._dir(name, version).is_dir():
            raise KeyError(f"cannot promote unknown version {name}/{version}")
        pointer = self.root / name / _CHAMPION_FILE
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(version)

    def champion(self, name: str) -> ArtifactMeta:
        version = self._champion_pointer(name)
        if version is None:
            raise KeyError(f"no champion registered for {name}")
        return self._read_meta(name, version)

    def _read_meta(self, name: str, version: str) -> ArtifactMeta:
        path = self._dir(name, version) / _META_FILE
        if not path.is_file():
            raise KeyError(f"unknown artifact {name}/{version}")
        meta = ArtifactMeta.model_validate_json(path.read_text())
        return meta.model_copy(update={"is_champion": version == self._champion_pointer(name)})

    def resolve(self, name: str, version: str = LATEST) -> str:
        """Resolve ``"latest"`` to the champion; any other value is taken literally."""
        if version != LATEST:
            return version
        return self._champion_pointer(name) or ""

    def load(self, name: str, version: str = LATEST) -> tuple[Any, ArtifactMeta]:
        """Load a payload plus its metadata. ``"latest"`` resolves to the champion."""
        resolved = self.resolve(name, version)
        if not resolved:
            raise KeyError(f"no champion registered for {name}")
        meta = self._read_meta(name, resolved)
        payload = _estimators.load(self._dir(name, resolved) / _PAYLOAD_FILE)
        return payload, meta

    def list_versions(self, name: str) -> tuple[ArtifactMeta, ...]:
        """Every stored version, oldest name first (versions sort lexically)."""
        directory = self.root / name
        if not directory.is_dir():
            return ()
        versions = sorted(p.name for p in directory.iterdir() if p.is_dir())
        return tuple(self._read_meta(name, v) for v in versions)


def bundle_from_models(
    version: str,
    *,
    service_time: ServiceTimeTable,
    arrivals: ArrivalIntensityModel,
    per_patient_los: Mapping[PatientId, Duration] | None = None,
    surge: SurgeForecast | None = None,
    staffing_hint: Mapping[tuple[StaffRole, int], int] | None = None,
) -> PredictionBundle:
    """Project fitted models onto the core-typed bundle the solver consumes."""
    expected: dict[ServiceKey, Duration] = {
        (key.activity, key.esi, key.complaint): seconds(params.mean_s)
        for key, params in service_time.params.items()
    }
    fallbacks: dict[Activity, Duration] = {
        activity: seconds(params.mean_s) for activity, params in service_time.fallbacks.items()
    }
    return PredictionBundle(
        version=version,
        resolution=arrivals.resolution,
        arrival_rates_per_hour=arrivals.rates_per_hour,
        expected_service=expected,
        fallback_service=fallbacks,
        per_patient_los=dict(per_patient_los or {}),
        surge_upper=surge.upper_q if surge is not None else None,
        staffing_hint=dict(staffing_hint) if staffing_hint is not None else None,
    )


def static_bundle(
    *,
    resolution: Duration,
    arrival_rates_per_hour: Sequence[float],
    activity_means_s: Mapping[Activity, float],
    version: str = "static",
) -> PredictionBundle:
    """The A/B baseline arm: the same shape, from scenario constants.

    Deliberately identical in type to :func:`bundle_from_models` so the harness
    swaps one call. If the static arm had its own shape, the comparison would be
    measuring two code paths rather than two sets of numbers.
    """
    return PredictionBundle(
        version=version,
        resolution=resolution,
        arrival_rates_per_hour=tuple(arrival_rates_per_hour),
        expected_service={},
        fallback_service={activity: seconds(mean) for activity, mean in activity_means_s.items()},
    )


__all__ = [
    "LATEST",
    "ArtifactMeta",
    "ModelStore",
    "PredictionBundle",
    "ServiceKey",
    "bundle_from_models",
    "canonical_hash",
    "static_bundle",
]
