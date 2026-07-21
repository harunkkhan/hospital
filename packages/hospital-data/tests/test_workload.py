"""``generate_workload``: determinism, CRN isolation, rate, ids, surge additivity."""

from __future__ import annotations

import math

from _data_fixtures import small_workload
from hypothesis import given, settings
from hypothesis import strategies as st

from hospital.core import Duration, RandomStreams, SimTime
from hospital.data.scenario import DisruptionEvent, DisruptionSpec
from hospital.data.workload import generate_workload


def test_generate_workload_is_deterministic_given_a_seed() -> None:
    spec = small_workload()
    a = generate_workload(spec, RandomStreams(99))
    b = generate_workload(spec, RandomStreams(99))
    assert a == b


def test_independent_random_streams_with_same_seed_agree() -> None:
    spec = small_workload()
    a = generate_workload(spec, RandomStreams(7))
    b = generate_workload(spec, RandomStreams(7))
    assert a == b


def test_crn_perturbing_complaint_mix_leaves_other_draws_identical() -> None:
    base = small_workload(complaint_mix={"chest_pain": 0.6, "laceration": 0.4})
    perturbed = small_workload(complaint_mix={"chest_pain": 0.1, "laceration": 0.9})

    base_arrivals = generate_workload(base, RandomStreams(11))
    perturbed_arrivals = generate_workload(perturbed, RandomStreams(11))

    by_key = {(a.hour_index, a.within_hour): a.patient for a in base_arrivals}
    by_key_p = {(a.hour_index, a.within_hour): a.patient for a in perturbed_arrivals}
    assert set(by_key) == set(by_key_p)

    complaint_differs = 0
    for key, patient in by_key.items():
        other = by_key_p[key]
        assert patient.arrival_time == other.arrival_time
        assert patient.esi == other.esi
        assert patient.arrival_mode == other.arrival_mode
        assert patient.isolation_required == other.isolation_required
        assert patient.id == other.id
        if patient.complaint != other.complaint:
            complaint_differs += 1
    # The whole point of per-attribute keying: the perturbed field actually
    # moves for at least some patients (otherwise this test would be vacuous).
    assert complaint_differs > 0


def test_patient_ids_are_deterministic_and_unique() -> None:
    spec = small_workload()
    arrivals = generate_workload(spec, RandomStreams(3))
    ids = [a.patient.id.root for a in arrivals]
    assert len(ids) == len(set(ids))
    assert all(pid.startswith("p_") for pid in ids)
    again = generate_workload(spec, RandomStreams(3))
    assert [a.patient.id.root for a in again] == ids


def test_arrivals_sorted_with_dense_sequence() -> None:
    spec = small_workload()
    arrivals = generate_workload(spec, RandomStreams(5))
    times = [a.patient.arrival_time.root for a in arrivals]
    assert times == sorted(times)
    assert [a.sequence for a in arrivals] == list(range(len(arrivals)))


@settings(max_examples=15, deadline=None)
@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_arrival_count_tracks_poisson_intensity(seed: int) -> None:
    # Flat hourly/dow profiles at base_rate_per_hour=4.0 over a 168h week ->
    # expected total = 4.0 * 168 = 672; Poisson sd ~ sqrt(672) ~= 26.
    spec = small_workload()
    arrivals = generate_workload(spec, RandomStreams(seed))
    expected = 4.0 * 168
    sd = math.sqrt(expected)
    assert abs(len(arrivals) - expected) < 8 * sd


def test_surge_is_purely_additive_over_the_base_week() -> None:
    spec = small_workload()
    seed = 42
    base = generate_workload(spec, RandomStreams(seed))
    disruptions = DisruptionSpec(
        events=(
            DisruptionEvent(
                kind="surge",
                at=SimTime(10 * 3_600_000_000),
                duration=Duration(4 * 3_600_000_000),
                magnitude=3.0,
            ),
        )
    )
    with_surge = generate_workload(spec, RandomStreams(seed), disruptions=disruptions)

    base_by_id = {a.patient.id.root: a.patient for a in base}
    surge_base_by_id = {
        a.patient.id.root: a.patient for a in with_surge if not a.patient.id.root.startswith("s")
    }
    assert base_by_id.keys() == surge_base_by_id.keys()
    assert all(base_by_id[pid] == surge_base_by_id[pid] for pid in base_by_id)
    assert len(with_surge) > len(base)

    surge_ids = [a.patient.id.root for a in with_surge if a.patient.id.root.startswith("s")]
    assert len(surge_ids) == len(set(surge_ids))


def test_surge_magnitude_at_or_below_one_adds_nothing() -> None:
    spec = small_workload()
    seed = 13
    base = generate_workload(spec, RandomStreams(seed))
    disruptions = DisruptionSpec(
        events=(
            DisruptionEvent(
                kind="surge", at=SimTime(0), duration=Duration(3_600_000_000), magnitude=1.0
            ),
        )
    )
    same = generate_workload(spec, RandomStreams(seed), disruptions=disruptions)
    assert same == base


def test_zero_rate_hour_draws_no_arrivals() -> None:
    zeroed = tuple(0.0 for _ in range(24))
    spec = small_workload(hourly_profile=zeroed)
    arrivals = generate_workload(spec, RandomStreams(1))
    assert arrivals == ()
