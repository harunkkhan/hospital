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
from typing import Literal, cast

import yaml
from pydantic import Field, model_validator

from hospital.core import (
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
    aspect_ratio: float = Field(default=1.55, gt=0)
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


class WorkupProfile(FrozenModel):
    """A complaint-keyed profile driving ``distributions.sample_workup``."""

    provider_visits_mean: float = Field(ge=1.0)
    nurse_visits_mean: float = Field(ge=0.0)
    imaging_prob: Mapping[ZoneType, float] = {}
    labs_mean: float = Field(ge=0.0)
    procedure_prob: float = Field(ge=0.0, le=1.0)


class WorkloadSpec(FrozenModel):
    """The one-week arrival + attribute-mix spec consumed by ``generate_workload``."""

    horizon: OperatingWeek = Field(default_factory=OperatingWeek.one_week)
    base_rate_per_hour: float = Field(ge=0.0)
    hourly_profile: tuple[float, ...]
    dow_profile: tuple[float, ...]
    esi_mix: Mapping[EsiAcuity, float]
    complaint_mix: Mapping[str, float]
    ambulance_fraction: float = Field(ge=0.0, le=1.0)
    isolation_fraction: float = Field(ge=0.0, le=1.0)
    workups: Mapping[str, WorkupProfile]
    esi_workup_scale: Mapping[EsiAcuity, float] = {}

    @model_validator(mode="after")
    def _check_profiles(self) -> WorkloadSpec:
        if len(self.hourly_profile) != 24:
            raise ValueError(f"hourly_profile must have length 24 (got {len(self.hourly_profile)})")
        if len(self.dow_profile) != 7:
            raise ValueError(f"dow_profile must have length 7 (got {len(self.dow_profile)})")
        if any(v < 0.0 for v in self.hourly_profile):
            raise ValueError("hourly_profile values must be >= 0")
        if any(v < 0.0 for v in self.dow_profile):
            raise ValueError("dow_profile values must be >= 0")
        _check_mix_sums_to_one(self.esi_mix, field_name="esi_mix")
        _check_mix_sums_to_one(self.complaint_mix, field_name="complaint_mix")
        unknown = set(self.complaint_mix.keys()) - set(self.workups.keys())
        if unknown:
            raise ValueError(f"complaint_mix has no workup profile for: {sorted(unknown)}")
        return self


class ShiftBlock(FrozenModel):
    """A staffing window with its role headcounts (M1: a static input, not solved)."""

    window: TimeWindow
    role_counts: Mapping[StaffRole, int]


class StaffingSpec(FrozenModel):
    """Static staffing input: explicit shift blocks plus a default coverage level."""

    blocks: tuple[ShiftBlock, ...] = ()
    default_counts: Mapping[StaffRole, int] = {}


class DisruptionEvent(FrozenModel):
    """A scheduled disruption. ``data`` owns the schedule; ``sim`` owns the mechanism."""

    kind: Literal["surge", "staff_absence", "zone_closure", "imaging_outage"]
    at: SimTime
    duration: Duration
    magnitude: float | None = None
    target: str | None = None


class DisruptionSpec(FrozenModel):
    """The disruption schedule carried by a ``Scenario`` (empty by default)."""

    events: tuple[DisruptionEvent, ...] = ()


class CostSpec(FrozenModel):
    """DEFERRED — mirrors ``core.cost``. No dollar rates in M1.

    Empty now, but ``extra="forbid"`` (inherited from ``FrozenModel``) means a
    future stray cost key is validated the day it is added, not silently accepted.
    """


class Scenario(FrozenModel):
    """The complete, versioned scenario: facility + workload + staffing + schedule."""

    name: str
    seed: int
    facility: FacilitySpec
    workload: WorkloadSpec
    staffing: StaffingSpec
    disruptions: DisruptionSpec = DisruptionSpec()
    rules: tuple[Rule, ...] = ()
    cost: CostSpec | None = None


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
    """Load ``base_path`` and apply the overlay YAML at ``overlay_path`` onto it."""
    base = load_scenario(base_path)
    raw = yaml.safe_load(Path(overlay_path).read_text())
    overlay: Mapping[str, object] = (
        cast("Mapping[str, object]", raw) if isinstance(raw, dict) else {}
    )
    return apply_overlay(base, overlay)


def _windows_overlap(a: TimeWindow, b: TimeWindow) -> bool:
    """Half-open interval overlap: ``a`` and ``b`` share at least one instant."""
    return a.start < b.end and b.start < a.end


def realize_staff(
    spec: StaffingSpec, layout: FloorLayout, window: TimeWindow
) -> tuple[StaffMember, ...]:
    """Materialize concrete ``StaffMember``s on duty during ``window``.

    Role headcounts come from any ``ShiftBlock`` whose window overlaps
    ``window`` (the max per role across overlapping blocks); a role with no
    overlapping block falls back to ``default_counts``. Home stations are
    assigned by deterministic round-robin over ``layout.stations`` — the
    k-th staff member of a role homes to ``stations[k % len(stations)]``.
    No randomness: staff placement is construction, not sampling.
    """
    if not layout.stations:
        raise ValueError("layout has no stations to home staff to")

    role_counts: dict[StaffRole, int] = dict(spec.default_counts)
    for block in spec.blocks:
        if _windows_overlap(block.window, window):
            for role, count in block.role_counts.items():
                role_counts[role] = max(role_counts.get(role, 0), count)

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
