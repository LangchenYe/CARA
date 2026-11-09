"""Canonical semantics and chunk-independent work for compiled family plans."""
import numpy as np
import pytest

from cara.data import canonical_assignment, host_memberships
from cara.query_engine import SharedQueryReplay
from cara.query_plan import compiled_profiles
from cara.refinement import panel_profile


def query_inputs():
    rng = np.random.default_rng(68013)
    seed = canonical_assignment(np.arange(36) % 6)
    panel = rng.lognormal(-3., 2., (5, 36, 9))
    panel[0, :, 0] = rng.choice([0., .1, np.nextafter(.1, 0.),
        np.nextafter(.1, 1.), 1., 2. ** -53, np.nextafter(0., 1.)], 36)
    assignments = [seed]
    for _ in range(17):
        labels = np.asarray(seed).copy()
        left, right = rng.choice(len(seed), 2, replace=False)
        labels[left], labels[right] = labels[right], labels[left]
        assignments.append(canonical_assignment(labels))
    assignments.extend((assignments[1], seed, assignments[1]))
    return panel, seed, assignments


def test_compiled_plan_preserves_boundaries_order_and_evicted_views():
    panel, seed, assignments = query_inputs()
    expected = np.stack([panel_profile(panel, assignment, 1.) for assignment in assignments])
    for max_bytes in (1, 1000, 1 << 20):
        engine = SharedQueryReplay(panel, seed, 1., max_views=4, max_bytes=max_bytes)
        for cells in (1, 67, 1 << 20):
            diagnostics = {}
            actual = compiled_profiles(engine, assignments, cells, diagnostics=diagnostics)
            assert actual.tobytes() == expected.tobytes()
            assert engine.cached_view_bytes <= max_bytes
            assert diagnostics["output_chunks"] >= 1
            assert diagnostics["result_array_bytes"] == actual.nbytes


def test_output_chunking_does_not_repeat_membership_construction():
    panel, seed, assignments = query_inputs()
    results = []
    for cells in (1, 1 << 20):
        engine = SharedQueryReplay(panel, seed, 1., max_views=2)
        stats = {}
        actual = compiled_profiles(engine, assignments, cells, diagnostics=stats)
        assert engine.unique_groups == stats["distinct_memberships"]
        results.append((actual.tobytes(), engine.unique_groups, engine.materialized_values,
                        stats["projection_table_bytes"], stats["candidate_membership_index_bytes"]))
    assert results[0] == results[1]
    ordinary = SharedQueryReplay(panel, seed, 1., max_views=2)
    reference = ordinary.profiles(assignments)
    assert reference.tobytes() == actual.tobytes()
    for field in ("hits", "projection_hits", "group_queries", "full_values", "unique_groups"):
        assert getattr(ordinary, field) == getattr(engine, field)


def test_compact_group_input_preserves_bytes_without_assignment_scans(monkeypatch):
    import cara.query_plan as plan
    panel, seed, assignments = query_inputs()
    expected = np.stack([panel_profile(panel, assignment, 1.) for assignment in assignments])
    groups = [host_memberships(assignment) for assignment in assignments]
    engine = SharedQueryReplay(panel, seed, 1., max_views=2)
    monkeypatch.setattr(plan, "host_memberships", lambda _: pytest.fail("Compact input rebuilt full labels"))
    actual = compiled_profiles(engine, memberships=groups, profile_cells=1)
    assert actual.tobytes() == expected.tobytes()
    with pytest.raises(ValueError, match="but not both"):
        compiled_profiles(engine, assignments, memberships=groups)


def test_empty_invalid_and_all_idle_queries():
    panel = np.zeros((2, 6, 3))
    first, second = (0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 1, 1)
    engine = SharedQueryReplay(panel, first, 1.)
    assert compiled_profiles(engine, []).shape == (0, 2, 6)
    expected = engine.profiles([second, first, second])
    assert compiled_profiles(engine, [second, first, second], 1).tobytes() == expected.tobytes()
    with pytest.raises(ValueError, match="equal host count"):
        compiled_profiles(engine, [first, (0,) * 6])
    with pytest.raises(ValueError, match="equal host count"):
        compiled_profiles(engine, [(0,) * 5])
    with pytest.raises(ValueError, match="positive integer"):
        compiled_profiles(engine, [first], 0)
