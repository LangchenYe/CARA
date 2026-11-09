"""Coverage, bounded outputs and honest arithmetic diagnostics for SQL context."""
from types import SimpleNamespace

import numpy as np

import benchmark_family_sql as study


def test_complete_candidate_stream_is_chunked_without_drops_or_large_load_tensors():
    chunks = study.chunk_slices(1020, 30, 121, 8)
    assert [index for start, stop in chunks for index in range(start, stop)] == list(range(1020))
    assert all((stop - start) * 30 * 121 * 8 * 8 <= study.MAX_RESULT_BYTES for start, stop in chunks)
    assert study.chunk_slices(0, 30, 121, 8) == ()


def test_native_sql_arithmetic_differences_are_not_called_exact():
    reference_loads = np.array([[[[1., 1. + 1e-12]]]])
    observed_loads = np.nextafter(reference_loads, np.inf)
    reference_profiles = np.zeros((1, 1, 6))
    observed_profiles = reference_profiles.copy()
    observed_profiles[0, 0, 3] = np.nextafter(0., 1.)
    result = study.arithmetic_differences(observed_loads, observed_profiles,
                                         reference_loads, reference_profiles, 1.)
    assert result["different_load_bits"] == 2
    assert result["different_profile_bits"] == 1
    assert result["strict_threshold_flips"] == result["tolerant_threshold_flips"] == 1
    aggregate = study.aggregate_diagnostics([result])
    assert aggregate["canonical_load_bitwise_equal"] is False
    assert aggregate["canonical_profile_bitwise_equal"] is False
    assert aggregate["different_profile_bits_by_metric"]["raw"] == 1


def test_zero_query_case_remains_in_all_three_rotations_without_invoking_an_engine(monkeypatch):
    monkeypatch.setattr(study, "prepare_case", lambda data, origin: (
        np.zeros((121, 4, 8)), (0, 0, 1, 1), (),
        {"family_requests": [{"pair": (0, 1), "mutable": (0, 2), "candidate_start": 0, "candidate_stop": 0}],
         "common_input_cpu_seconds": 0., "candidate_digest": "empty"}))

    def forbidden(*args, **kwargs):
        raise AssertionError("A zero-query setting must not call an execution engine")

    for name in ("full_queries", "shared_queries", "sql_queries"):
        monkeypatch.setattr(study, name, forbidden)
    result = study.run_case(SimpleNamespace(name="toy", capacity=1.), 288, 0)
    assert result["status"] == "zero_query" and result["B"] == 121
    assert len(result["trials"]) == 3
    assert all(result["median_seconds"][plan]["cpu_seconds"] == 0 for plan in study.PLANS)


def test_resident_sql_streams_one_global_group_relation_in_complete_order():
    panel = np.random.default_rng(7163).random((3, 8, 8))
    assignments = ((0, 0, 0, 0, 1, 1, 1, 1), (0, 1, 0, 0, 1, 1, 1, 0))
    _, whole_loads, whole_profiles = study.sql_queries(panel, assignments, 1., True)
    visited = []

    def verify(start, stop, loads, profiles):
        visited.append((start, stop))
        assert loads.tobytes() == whole_loads[start:stop].tobytes()
        assert profiles.tobytes() == whole_profiles[start:stop].tobytes()
        return study.arithmetic_differences(loads, profiles, whole_loads[start:stop], whole_profiles[start:stop], 1.)

    costs = study.resident_sql(panel, assignments, 1., ((0, 1), (1, 2)), verify)
    assert costs["demand_materializations"] == 1
    assert costs["distinct_resident_groups"] == 4
    assert visited == [(0, 1), (1, 2)]
    assert costs["arithmetic"]["canonical_profile_bitwise_equal"] is True


def test_resident_canonical_reader_computes_profiles_once_and_bounds_load_inspection(monkeypatch):
    panel = np.random.default_rng(7164).random((3, 8, 8))
    assignments = ((0, 0, 0, 0, 1, 1, 1, 1), (0, 1, 0, 0, 1, 1, 1, 0))
    calls = {"constructors": 0, "profiles": [], "loads": []}
    original = study.SharedQueryReplay

    class Tracked(original):
        def __init__(self, *args, **kwargs):
            calls["constructors"] += 1
            super().__init__(*args, **kwargs)

        def profiles(self, values):
            calls["profiles"].append(len(values))
            return super().profiles(values)

        def loads_many(self, values):
            calls["loads"].append(len(values))
            return super().loads_many(values)

    monkeypatch.setattr(study, "SharedQueryReplay", Tracked)

    def verify(start, stop, loads, profiles):
        _, expected_loads, expected_profiles = study.full_queries(panel, assignments[start:stop], 1.)
        return study.arithmetic_differences(loads, profiles, expected_loads, expected_profiles, 1.)

    costs = study.resident_shared(panel, assignments[0], assignments, 1., ((0, 1), (1, 2)), verify)
    assert calls == {"constructors": 1, "profiles": [2], "loads": [1, 1]}
    assert costs["arithmetic"]["canonical_load_bitwise_equal"] is True
    assert costs["arithmetic"]["canonical_profile_bitwise_equal"] is True
