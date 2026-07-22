"""objective: the single scalar cost, its coefficients, and the canonical hash."""

from __future__ import annotations

import pytest
from _solver_fixtures import default_config
from pydantic import ValidationError

from hospital.core import EsiAcuity
from hospital.solver.objective import (
    ObjectiveConfig,
    acuity_urgency,
    assignment_coeffs,
    config_hash,
    default_acuity_urgency,
    weighted_total,
)


def test_default_urgency_uses_priority_weight_sign() -> None:
    # ESI-1 (most critical) must weigh MORE than ESI-5 (the sign trap).
    urg = default_acuity_urgency()
    assert urg[EsiAcuity.ESI1] == EsiAcuity.ESI1.priority_weight()
    assert urg[EsiAcuity.ESI1] > urg[EsiAcuity.ESI5]
    config = default_config()
    assert acuity_urgency(config, EsiAcuity.ESI1) > acuity_urgency(config, EsiAcuity.ESI5)


def test_assignment_coeffs_formula() -> None:
    config = default_config(w_time=3, w_travel=5, unplaced_wait_penalty=7)
    for esi in EsiAcuity:
        u = acuity_urgency(config, esi)
        coeffs = assignment_coeffs(config, esi)
        assert coeffs.travel_weight == 5 * u
        assert coeffs.wait_penalty == 3 * u * 7


def test_weighted_total_is_integer_and_scorecard_recomputes() -> None:
    config = default_config(w_time=2, w_travel=3, scale=10)
    patient_time = {EsiAcuity.ESI1: 100, EsiAcuity.ESI3: 200}
    penalties = {"boarding": 50, "unplaced": 5}
    total = weighted_total(
        patient_time_s=patient_time, staff_travel_s=40, penalties=penalties, config=config
    )
    assert isinstance(total, int)
    # Scorecard-style recomputation from parts must equal the one aggregator.
    acuity_time = sum(acuity_urgency(config, a) * t for a, t in patient_time.items())
    expected = config.scale * (config.w_time * acuity_time + config.w_travel * 40 + 55)
    assert total == expected


def test_weighted_total_monotone_in_each_term() -> None:
    config = default_config()
    base = weighted_total(patient_time_s={EsiAcuity.ESI3: 100}, staff_travel_s=10, config=config)
    more_time = weighted_total(
        patient_time_s={EsiAcuity.ESI3: 200}, staff_travel_s=10, config=config
    )
    more_travel = weighted_total(
        patient_time_s={EsiAcuity.ESI3: 100}, staff_travel_s=20, config=config
    )
    more_pen = weighted_total(
        patient_time_s={EsiAcuity.ESI3: 100},
        staff_travel_s=10,
        penalties={"x": 1},
        config=config,
    )
    assert more_time > base
    assert more_travel > base
    assert more_pen > base


def test_config_hash_stable_under_field_reorder() -> None:
    a = ObjectiveConfig(w_time=1, w_travel=2, unplaced_wait_penalty=3)
    b = ObjectiveConfig(unplaced_wait_penalty=3, w_travel=2, w_time=1)
    assert config_hash(a) == config_hash(b)


def test_config_hash_distinct_under_any_value_change() -> None:
    base = default_config()
    assert config_hash(base) != config_hash(default_config(w_travel=base.w_travel + 1))
    assert config_hash(base) != config_hash(default_config(scale=base.scale + 1))
    assert config_hash(base) != config_hash(default_config(version="other"))
    custom = default_config(acuity_urgency={e: 1 for e in EsiAcuity})
    assert config_hash(base) != config_hash(custom)


def test_config_hash_is_hex_sha256() -> None:
    digest = config_hash(default_config())
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_is_hashable_with_immutable_urgency_pairs() -> None:
    # Regression (review finding 7): acuity_urgency materialized as a mutable
    # dict inside FrozenModel — mutable and unhashable, so the provenance
    # surface could drift between solve and stamp. The curve must be a
    # genuinely immutable, canonically sorted tuple of pairs.
    config = default_config()
    assert isinstance(hash(config), int)  # raised TypeError with a dict field
    assert isinstance(config.acuity_urgency, tuple)
    assert all(isinstance(pair, tuple) for pair in config.acuity_urgency)
    # Pair input in any order canonicalizes to the same sorted pairs (a mapping
    # is also accepted; see test_config_hash_distinct_under_any_value_change).
    reordered = ObjectiveConfig(
        acuity_urgency=tuple(reversed(list(default_acuity_urgency().items())))
    )
    assert reordered.acuity_urgency == config.acuity_urgency
    assert hash(reordered) == hash(config)
    assert config_hash(reordered) == config_hash(config)
    # Duplicate acuities would make the lookup ambiguous — rejected outright.
    with pytest.raises(ValidationError):
        ObjectiveConfig(acuity_urgency=((EsiAcuity.ESI1, 5), (EsiAcuity.ESI1, 4)))
