"""``ServiceTimes`` — CRN key isolation, determinism, hard-KeyError rows."""

from __future__ import annotations

import pytest
from _sim_fixtures import make_patient

from hospital.core import Activity, EsiAcuity, PatientId, RandomStreams
from hospital.sim.physics.service_times import (
    ServiceTable,
    ServiceTimes,
    default_service_table,
    sample_boarding_delay,
    sample_disposition,
)


def _service_times(seed: int = 7) -> ServiceTimes:
    return ServiceTimes(RandomStreams(seed), default_service_table())


class TestCrn:
    def test_same_key_same_draw_regardless_of_call_order(self) -> None:
        a = _service_times()
        b = _service_times()
        # consume unrelated draws first on `b` — must not perturb the keyed draw
        for i in range(5):
            b.sample(
                Activity.NURSE_VISIT,
                EsiAcuity.ESI4,
                "chest_pain",
                patient=PatientId("other"),
                index=i,
            )
        key_draw_a = a.sample(
            Activity.PROVIDER_VISIT, EsiAcuity.ESI2, "chest_pain", patient=PatientId("p1")
        )
        key_draw_b = b.sample(
            Activity.PROVIDER_VISIT, EsiAcuity.ESI2, "chest_pain", patient=PatientId("p1")
        )
        assert key_draw_a == key_draw_b

    def test_perturbing_one_key_leaves_others_byte_identical(self) -> None:
        st = _service_times()
        others_before = [
            st.sample(
                Activity.NURSE_VISIT, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p2"), index=i
            )
            for i in range(3)
        ]
        # an extra draw on a different (patient, activity, index) key...
        st.sample(Activity.IMAGING, EsiAcuity.ESI1, "chest_pain", patient=PatientId("p9"))
        others_after = [
            st.sample(
                Activity.NURSE_VISIT, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p2"), index=i
            )
            for i in range(3)
        ]
        assert others_before == others_after

    def test_index_disambiguates_repeat_visits(self) -> None:
        st = _service_times()
        first = st.sample(
            Activity.NURSE_VISIT, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p1"), index=0
        )
        second = st.sample(
            Activity.NURSE_VISIT, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p1"), index=1
        )
        assert first != second

    def test_different_seeds_differ(self) -> None:
        a = _service_times(seed=1).sample(
            Activity.TRIAGE, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p1")
        )
        b = _service_times(seed=2).sample(
            Activity.TRIAGE, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p1")
        )
        assert a != b


class TestTable:
    def test_missing_row_is_a_hard_key_error(self) -> None:
        table = ServiceTable(rows={(Activity.TRIAGE, EsiAcuity.ESI3): (300.0, 0.3)})
        st = ServiceTimes(RandomStreams(1), table)
        with pytest.raises(KeyError, match="no service-time row"):
            st.sample(Activity.PROVIDER_VISIT, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p"))

    def test_cv_zero_degenerates_to_exact_mean(self) -> None:
        table = ServiceTable(rows={(Activity.TRIAGE, EsiAcuity.ESI3): (300.0, 0.0)})
        st = ServiceTimes(RandomStreams(1), table)
        d = st.sample(Activity.TRIAGE, EsiAcuity.ESI3, "chest_pain", patient=PatientId("p"))
        assert d.root == 300_000_000

    def test_empirical_mean_tracks_table_mean(self) -> None:
        st = _service_times()
        mean_s, _cv = default_service_table().lookup(
            Activity.PROVIDER_VISIT, EsiAcuity.ESI3, "chest_pain"
        )
        draws = [
            st.sample(
                Activity.PROVIDER_VISIT,
                EsiAcuity.ESI3,
                "chest_pain",
                patient=PatientId(f"p{i}"),
            ).root
            / 1_000_000
            for i in range(400)
        ]
        empirical = sum(draws) / len(draws)
        assert abs(empirical - mean_s) / mean_s < 0.15

    def test_result_delay_unknown_activity_raises(self) -> None:
        st = _service_times()
        with pytest.raises(KeyError, match="no result-delay row"):
            st.result_delay(Activity.TRIAGE, patient=PatientId("p"))


class TestDispositionAndBoarding:
    def test_disposition_is_content_addressed_on_the_patient(self) -> None:
        streams = RandomStreams(7)
        p = make_patient("p1", esi=EsiAcuity.ESI2)
        first = sample_disposition(streams, p)
        # unrelated draws in between must not change the patient's fate
        for i in range(10):
            sample_disposition(streams, make_patient(f"q{i}"))
        assert sample_disposition(streams, p) == first

    def test_boarding_delay_positive_and_deterministic(self) -> None:
        streams = RandomStreams(7)
        p = make_patient("p1")
        d1 = sample_boarding_delay(streams, p)
        d2 = sample_boarding_delay(streams, p)
        assert d1 == d2
        assert d1.root > 0
