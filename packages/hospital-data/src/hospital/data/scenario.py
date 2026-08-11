"""The ``Scenario`` schema, YAML codec, arm overlays, and ``realize_staff``.

Pure construction throughout (doc 02 §2.1): ``Scenario.model_validate``,
``load_scenario``, ``apply_overlay``, ``load_arm``, and ``realize_staff`` take no
``RandomStreams`` and draw nothing. The same YAML bytes always yield the
byte-identical ``Scenario`` — this is the construction half of the
construction/sampling split that makes baseline vs optimized comparable.

Cross-field validators **reject** a malformed file rather than silently
normalizing it (same "reject, not fix" philosophy as the core plan validator):
a mis-summed ``esi_mix`` or an out-of-range fraction is a load-time error, not a
quietly-renormalized copy that would make the on-disk file disagree with the
in-memory model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, cast

import yaml
from pydantic import Field, field_serializer, field_validator, model_validator

from hospital.core import (
    CostRates,
    Duration,
    EsiAcuity,
    FloorLayout,
    FrozenModel,
    NodeId,
    OperatingWeek,
    Rule,
    SimTime,
    StaffId,
    StaffMember,
    StaffRole,
    TimeWindow,
    ZoneType,
)

_MIX_TOLERANCE = 1e-6

# Every rate, profile multiplier, and categorical weight must be finite and
# non-negative *at load* — an ``.inf`` arrival rate would make the Poisson
# sampler's exponential scale zero and spin ``generate_workload`` forever, and
# a NaN weight would poison a mix's total silently.
_Weight = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
# A probability proper: finite and within [0, 1].
_Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
# A staffing headcount: ``range(-n)`` is silently empty, so negatives must be
# rejected at load rather than realizing zero staff.
_HeadCount = Annotated[int, Field(ge=0)]

_DEFAULT_SKILLS: Mapping[StaffRole, frozenset[str]] = {
    StaffRole.PHYSICIAN: frozenset({"md"}),
    StaffRole.NURSE: frozenset({"rn"}),
    StaffRole.TECH: frozenset({"tech"}),
    StaffRole.PORTER: frozenset(),
    StaffRole.HOUSEKEEPING: frozenset(),
}


def _check_mix_sums_to_one[K](mix: Mapping[K, float], *, field_name: str) -> None:
    total = sum(mix.values())
    if abs(total - 1.0) > _MIX_TOLERANCE:
        raise ValueError(f"{field_name} must sum to ~1.0 (got {total})")


class ZoneQuota(FrozenModel):
    """One acute-care zone's bay allocation (a ``facility.zones`` entry)."""

    zone_type: ZoneType
    bays: int = Field(ge=0)
    isolation_bays: int = Field(default=0, ge=0)
    equipment: frozenset[str] = frozenset()
    max_bays_per_station: int = Field(default=12, gt=0)

    @field_serializer("equipment")
    def _serialize_equipment(self, value: frozenset[str]) -> list[str]:
        """Emit sorted order — a ``frozenset`` iterates in hash-table order, which
        varies with ``PYTHONHASHSEED`` and would break byte-stable YAML dumps."""
        return sorted(value)

    @model_validator(mode="after")
    def _check_isolation_subset(self) -> ZoneQuota:
        if self.isolation_bays > self.bays:
            raise ValueError(
                f"isolation_bays ({self.isolation_bays}) must be <= bays ({self.bays})"
            )
        return self


class FacilitySpec(FrozenModel):
    """The ``scale`` argument to ``generate_floor`` — a pure geometry spec."""

    target_area_sqft: int = Field(default=100_000, gt=0)
    aspect_ratio: float = Field(default=1.55, gt=0, allow_inf_nan=False)
    walk_speed_cm_s: int = Field(default=120, gt=0)
    corridor_margin_cm: int = Field(default=600, gt=0)
    room_depth_cm: int = Field(default=420, gt=0)
    zones: tuple[ZoneQuota, ...] = Field(default_factory=tuple)
    imaging_suites: int = Field(default=3, ge=0)
    lab_stations: int = Field(default=2, ge=0)
    triage_rooms: int = Field(default=6, ge=0)

    @model_validator(mode="after")
    def _check_at_least_one_bay(self) -> FacilitySpec:
        if sum(q.bays for q in self.zones) < 1:
            raise ValueError("facility.zones must allocate at least one bay in total")
        return self


class FloorSpec(FrozenModel):
    """One floor: a name, and the geometry spec that fills it."""

    name: str = Field(min_length=1)
    facility: FacilitySpec


class HospitalSpec(FrozenModel):
    """A stack of floors and the elevators joining them.

    ``floors[0]`` is the ground floor and the only one with entrances: ambulances and
    walk-ins arrive at the emergency department, and everything above is reached through
    the shafts.
    """

    floors: tuple[FloorSpec, ...] = Field(min_length=1)
    elevator_shafts: int = Field(default=2, ge=1)
    # Time for the car to move one floor, and the fixed cost of a boarding/exit cycle.
    # `dwell` is charged once per shaft edge traversed, which is what makes a two-floor
    # trip cheaper than two one-floor trips through a lobby.
    seconds_per_floor: float = Field(default=12.0, gt=0.0, allow_inf_nan=False)
    dwell_seconds: float = Field(default=20.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _floor_names_are_unique(self) -> HospitalSpec:
        names = [floor.name for floor in self.floors]
        if len(names) != len(set(names)):
            raise ValueError("hospital.floors have duplicate names")
        return self


class ElevatorSpec(FrozenModel):
    """The shafts joining a scenario's floors. Ignored when there is only one."""

    shafts: int = Field(default=2, ge=1)
    seconds_per_floor: float = Field(default=12.0, gt=0.0, allow_inf_nan=False)
    dwell_seconds: float = Field(default=20.0, ge=0.0, allow_inf_nan=False)


class WorkupProfile(FrozenModel):
    """A complaint-keyed profile driving ``distributions.sample_workup``."""

    provider_visits_mean: float = Field(ge=1.0, allow_inf_nan=False)
    nurse_visits_mean: float = Field(ge=0.0, allow_inf_nan=False)
    imaging_prob: Mapping[ZoneType, _Probability] = {}
    labs_mean: float = Field(ge=0.0, allow_inf_nan=False)
    procedure_prob: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("imaging_prob", mode="after")
    @classmethod
    def _freeze_imaging_prob(cls, value: Mapping[ZoneType, float]) -> Mapping[ZoneType, float]:
        """Store a read-only copy — a frozen model must not expose a mutable dict."""
        return MappingProxyType(dict(value))

    @field_serializer("imaging_prob")
    def _serialize_imaging_prob(self, value: Mapping[ZoneType, float]) -> dict[ZoneType, float]:
        return dict(value)


class WorkloadSpec(FrozenModel):
    """The one-week arrival + attribute-mix spec consumed by ``generate_workload``."""

    horizon: OperatingWeek = Field(default_factory=OperatingWeek.one_week)
    base_rate_per_hour: float = Field(ge=0.0, allow_inf_nan=False)
    hourly_profile: tuple[_Weight, ...]
    dow_profile: tuple[_Weight, ...]
    esi_mix: Mapping[EsiAcuity, _Weight]
    complaint_mix: Mapping[str, _Weight]
    ambulance_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    isolation_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    workups: Mapping[str, WorkupProfile]
    esi_workup_scale: Mapping[EsiAcuity, _Weight] = {}

    @field_validator("esi_mix", "complaint_mix", "workups", "esi_workup_scale", mode="after")
    @classmethod
    def _freeze_mappings(cls, value: Mapping[object, object]) -> Mapping[object, object]:
        """Store read-only copies — a frozen model must not expose mutable dicts."""
        return MappingProxyType(dict(value))

    @field_serializer("esi_mix", "complaint_mix", "workups", "esi_workup_scale")
    def _serialize_mappings(self, value: Mapping[object, object]) -> dict[object, object]:
        return dict(value)

    @model_validator(mode="after")
    def _check_profiles(self) -> WorkloadSpec:
        if len(self.hourly_profile) != 24:
            raise ValueError(f"hourly_profile must have length 24 (got {len(self.hourly_profile)})")
        if len(self.dow_profile) != 7:
            raise ValueError(f"dow_profile must have length 7 (got {len(self.dow_profile)})")
        _check_mix_sums_to_one(self.esi_mix, field_name="esi_mix")
        _check_mix_sums_to_one(self.complaint_mix, field_name="complaint_mix")
        unknown = set(self.complaint_mix.keys()) - set(self.workups.keys())
        if unknown:
            raise ValueError(f"complaint_mix has no workup profile for: {sorted(unknown)}")
        return self


class ShiftBlock(FrozenModel):
    """A staffing window with its role headcounts (M1: a static input, not solved)."""

    window: TimeWindow
    role_counts: Mapping[StaffRole, _HeadCount]

    @field_validator("role_counts", mode="after")
    @classmethod
    def _freeze_role_counts(cls, value: Mapping[StaffRole, int]) -> Mapping[StaffRole, int]:
        """Store a read-only copy — a frozen model must not expose a mutable dict."""
        return MappingProxyType(dict(value))

    @field_serializer("role_counts")
    def _serialize_role_counts(self, value: Mapping[StaffRole, int]) -> dict[StaffRole, int]:
        return dict(value)


class StaffingSpec(FrozenModel):
    """Static staffing input: explicit shift blocks plus a default coverage level."""

    blocks: tuple[ShiftBlock, ...] = ()
    default_counts: Mapping[StaffRole, _HeadCount] = {}

    @field_validator("default_counts", mode="after")
    @classmethod
    def _freeze_default_counts(cls, value: Mapping[StaffRole, int]) -> Mapping[StaffRole, int]:
        """Store a read-only copy — a frozen model must not expose a mutable dict."""
        return MappingProxyType(dict(value))

    @field_serializer("default_counts")
    def _serialize_default_counts(self, value: Mapping[StaffRole, int]) -> dict[StaffRole, int]:
        return dict(value)


class DisruptionEvent(FrozenModel):
    """A scheduled disruption. ``data`` owns the schedule; ``sim`` owns the mechanism."""

    kind: Literal["surge", "staff_absence", "zone_closure", "imaging_outage"]
    at: SimTime
    duration: Duration
    # A surge λ-multiplier is a rate: it must be finite (an ``.inf`` magnitude
    # would spin the surge sampler forever, exactly like an infinite base rate).
    magnitude: float | None = Field(default=None, allow_inf_nan=False)
    target: str | None = None


class DisruptionSpec(FrozenModel):
    """The disruption schedule carried by a ``Scenario`` (empty by default)."""

    events: tuple[DisruptionEvent, ...] = ()


# The rate vocabulary is ``core.CostRates``, not a spec type mirroring it here (M4b).
# Same reasoning as ``rules: tuple[Rule, ...]``: the frozen value lives in the lowest
# package that needs it and the scenario simply holds one, so there is no second
# definition to drift. The M1 placeholder ``CostSpec`` was empty precisely so that
# whatever landed later could be the real thing rather than a translation of it.
CostSpec = CostRates


class Scenario(FrozenModel):
    """The complete, versioned scenario: facility + workload + staffing + schedule."""

    name: str
    seed: int
    facility: FacilitySpec
    workload: WorkloadSpec
    staffing: StaffingSpec
    disruptions: DisruptionSpec = DisruptionSpec()
    rules: tuple[Rule, ...] = ()
    # Dollar rates, or None to report time only — which every committed scenario does.
    # `core.CostRates` has no default rates on purpose (see its module docstring): what
    # an hour of nursing costs is a fact about a hospital, not about a simulator.
    cost: CostRates | None = None
    # The floors above the emergency department. Empty is the ED-only hospital every
    # scenario described before M4 — `facility` is the ground floor either way, so the
    # building is never stated twice.
    upper_floors: tuple[FloorSpec, ...] = ()
    elevators: ElevatorSpec = ElevatorSpec()

    def hospital(self) -> HospitalSpec:
        """The whole building: the ED on the ground, ``upper_floors`` above it."""
        return HospitalSpec(
            floors=(FloorSpec(name="ground", facility=self.facility), *self.upper_floors),
            elevator_shafts=self.elevators.shafts,
            seconds_per_floor=self.elevators.seconds_per_floor,
            dwell_seconds=self.elevators.dwell_seconds,
        )


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a ``Scenario`` from a YAML file (``safe_load`` only)."""
    raw = yaml.safe_load(Path(path).read_text())
    return Scenario.model_validate(raw)


def dump_scenario(scenario: Scenario, path: str | Path) -> None:
    """Write ``scenario`` as canonical, diff-stable YAML (sorted keys, ``safe_dump``)."""
    data = scenario.model_dump(mode="json")
    Path(path).write_text(yaml.safe_dump(data, sort_keys=True, default_flow_style=False))


def _deep_merge(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    """Recursively merge mapping keys; non-mapping values (incl. sequences) replace wholesale."""
    merged: dict[str, object] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            existing_map = cast("Mapping[str, object]", existing)
            value_map = cast("Mapping[str, object]", value)
            merged[key] = _deep_merge(existing_map, value_map)
        else:
            merged[key] = value
    return merged


def apply_overlay(base: Scenario, overlay: Mapping[str, object]) -> Scenario:
    """Deep-merge ``overlay`` onto ``base`` and re-validate as a fresh ``Scenario``."""
    base_dict = cast("Mapping[str, object]", base.model_dump(mode="json"))
    merged = _deep_merge(base_dict, overlay)
    return Scenario.model_validate(merged)


def load_arm(base_path: str | Path, overlay_path: str | Path) -> Scenario:
    """Load ``base_path`` and apply the overlay YAML at ``overlay_path`` onto it.

    Only a ``null`` root (an empty file) or a mapping root is a valid overlay —
    an empty mapping means "no changes". Any other root type (a list, a scalar)
    is a malformed arm file and raises rather than silently running the
    baseline in place of the requested arm.
    """
    base = load_scenario(base_path)
    raw = yaml.safe_load(Path(overlay_path).read_text())
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError(
            f"arm overlay root must be a mapping or empty (got {type(raw).__name__} "
            f"in {overlay_path})"
        )
    return apply_overlay(base, cast("Mapping[str, object]", raw))


def _windows_overlap(a: TimeWindow, b: TimeWindow) -> bool:
    """Half-open interval overlap: ``a`` and ``b`` share at least one instant."""
    return a.start < b.end and b.start < a.end


def realize_staff(
    spec: StaffingSpec, layout: FloorLayout, window: TimeWindow
) -> tuple[StaffMember, ...]:
    """Materialize concrete ``StaffMember``s on duty during ``window``.

    Role headcounts come from any ``ShiftBlock`` whose window overlaps
    ``window`` (the max per role across overlapping blocks); ``default_counts``
    fills in **only** the roles that no overlapping block supplies — a block
    that explicitly schedules 2 nurses realizes 2 nurses, never a higher
    default. Home stations are assigned by deterministic round-robin over
    ``layout.stations`` — the k-th staff member of a role homes to
    ``stations[k % len(stations)]``. No randomness: staff placement is
    construction, not sampling.
    """
    if not layout.stations:
        raise ValueError("layout has no stations to home staff to")

    role_counts: dict[StaffRole, int] = {}
    for block in spec.blocks:
        if _windows_overlap(block.window, window):
            for role, count in block.role_counts.items():
                role_counts[role] = max(role_counts.get(role, 0), count)
    for role, count in spec.default_counts.items():
        role_counts.setdefault(role, count)

    stations = layout.stations
    staff: list[StaffMember] = []
    for role in sorted(role_counts, key=lambda r: r.value):
        count = role_counts[role]
        skills = _DEFAULT_SKILLS.get(role, frozenset())
        for k in range(count):
            staff.append(
                StaffMember(
                    id=StaffId(f"staff_{role.value}_{k:03d}"),
                    role=role,
                    home_station=NodeId(stations[k % len(stations)].root),
                    skills=skills,
                )
            )
    return tuple(staff)


__all__ = [
    "CostSpec",
    "DisruptionEvent",
    "DisruptionSpec",
    "FacilitySpec",
    "Scenario",
    "ShiftBlock",
    "StaffingSpec",
    "WorkloadSpec",
    "WorkupProfile",
    "ZoneQuota",
    "apply_overlay",
    "dump_scenario",
    "load_arm",
    "load_scenario",
    "realize_staff",
]
