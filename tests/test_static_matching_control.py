"""Static reduction against independent assignment and complete-pool oracles."""
from itertools import permutations, product
from time import process_time

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import benchmark_family_execution as execution
import benchmark_family_queries as queries
from benchmark_static_matching import FamilyMatching, _entries, _static_maximum, matching_control
from cara import compiled_matching
from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import ResidualFamily
from cara.refinement import migration_profile


def independent_maximum(matrix):
    rows, columns = matrix.shape
    size = max(rows, columns)
    padded = np.zeros((size, size), dtype=np.int64)
    padded[:rows, :columns] = matrix
    if size <= 5:
        return max(sum(int(padded[row, column]) for row, column in enumerate(order))
                   for order in permutations(range(size)))
    left, right = linear_sum_assignment(padded, maximize=True)
    return int(padded[left, right].sum())


def test_exhaustive_binary_three_by_three_reduction():
    for values in product((0, 1), repeat=9):
        matrix = np.array(values, dtype=np.int64).reshape(3, 3)
        stats = {}
        result = _static_maximum(3, _entries(matrix), stats, process_time())
        assert result == independent_maximum(matrix)
        assert stats["matching_static_reductions"] == 1
        assert stats["matching_static_locked_edges"] + stats["matching_static_core_rows_sum"] == 3


def test_sparse_dense_rectangular_and_tied_matrices():
    rng = np.random.default_rng(73129)
    for shape in ((1, 5), (5, 1), (2, 3), (3, 2), (4, 5), (5, 4), (8, 9), (9, 8)):
        for dense in (False, True):
            for _ in range(4):
                matrix = rng.integers(0, 8, size=shape, dtype=np.int64)
                if not dense:
                    matrix[rng.random(shape) < .7] = 0
                assert _static_maximum(max(shape), _entries(matrix), {}, process_time()) == independent_maximum(matrix)


def test_component_sharing_and_outer_count_cache():
    incumbent = canonical_assignment(np.arange(40 * 8) % 40)
    groups = host_memberships(incumbent)
    mutable = tuple(sorted(groups[0][:2] + groups[1][:2]))
    with matching_control(True):
        family = ResidualFamily(incumbent, incumbent, (0, 1), mutable, matching_backend="compiled")
    assert isinstance(family._factor, FamilyMatching)
    assert family.stats["matching_unchanged_components"] == 38
    for mask in family.nonempty_masks():
        assert family.movement_count(mask) == migration_profile(incumbent, family.assignment(mask))["count"]
    assert family.stats["matching_static_signature_reductions"] == 9
    assert family.stats["matching_static_input_rows_sum"] == 2 * 9
    old = family.stats["matching_static_reductions"]
    for mask in family.nonempty_masks():
        family.movement_count(mask)
    assert family.stats["matching_static_reductions"] == old
    assert family.seed_movement_count() == 0
    assert family.stats["matching_static_seed_reductions"] == 1


def test_random_mixed_families_and_relaxed_bounds():
    rng = np.random.default_rng(80223)
    for hosts in (2, 3, 5, 7):
        for _ in range(4):
            incumbent = canonical_assignment(np.arange(hosts * 4) % hosts)
            seed = canonical_assignment(rng.permutation(incumbent))
            pair = tuple(sorted(rng.choice(hosts, 2, replace=False).tolist()))
            groups = host_memberships(seed)
            mutable = tuple(sorted(groups[pair[0]][:2] + groups[pair[1]][:2]))
            with matching_control(True):
                family = ResidualFamily(incumbent, seed, pair, mutable, matching_backend="compiled")
            dense = ResidualFamily(incumbent, seed, pair, mutable, matching_backend="dense")
            assert family.seed_movement_count() == dense.seed_movement_count()
            for mask in family.nonempty_masks():
                assert family.movement_count(mask) == migration_profile(incumbent, family.assignment(mask))["count"]
            for power in range(family.bits + 1):
                span = 1 << power
                for start in range(0, family.mask_count, span):
                    assert family.movement_lower_bound(start, start + span) == dense.movement_lower_bound(start, start + span)


def test_symmetric_family_keeps_seed_separate_from_representative():
    incumbent = (0, 1, 0, 1)
    seed = (0, 0, 1, 1)
    with matching_control(True):
        family = ResidualFamily(incumbent, seed, (0, 1), (0, 1, 2, 3), matching_backend="compiled")
    assert family.symmetric
    for mask in family.nonempty_masks():
        assert family.movement_count(mask) == migration_profile(incumbent, family.assignment(mask))["count"]
    assert family.seed_movement_count() == migration_profile(incumbent, seed)["count"]


def test_context_restores_production_class_after_error_and_nesting():
    original = compiled_matching.FamilyMatching
    with pytest.raises(RuntimeError, match="sentinel"):
        with matching_control(True):
            assert compiled_matching.FamilyMatching is FamilyMatching
            with matching_control(True):
                assert compiled_matching.FamilyMatching is FamilyMatching
            assert compiled_matching.FamilyMatching is FamilyMatching
            raise RuntimeError("sentinel")
    assert compiled_matching.FamilyMatching is original
    with matching_control():
        assert compiled_matching.FamilyMatching is original


@pytest.mark.parametrize("adaptive", [False, True])
def test_complete_rich_pools_and_adaptive_seeds_match_eager(adaptive):
    rng = np.random.default_rng(14023)
    hosts = 8
    incumbent = canonical_assignment(np.arange(hosts * 8) % hosts)
    seed = list(incumbent)
    for host, group in enumerate(host_memberships(incumbent)):
        seed[group[-1]] = (host + 1) % hosts
    seed = canonical_assignment(seed)
    panel = rng.uniform(.01, .10, size=(4, len(seed), 3))
    forecast = np.full((len(seed), 3), .04)
    config = {"top_p": 4, "migration_fraction": .2, "archive_mode": "balanced", "score_mode": "gain",
              "family_score_min_hosts": 0, "family_score_min_leaves": 1, "family_score_strategy": "leaf",
              "family_forecast_strategy": "leaf", "family_progressive_batch": 3, "query_batch_values": 1024}
    outputs, traces, cache = [], [], {}
    original = compiled_matching.FamilyMatching
    for plan in ("eager", "static", "compiled"):
        counters = {}
        def witness(value):
            execution.verify_witness(value, incumbent, forecast, panel, .51, counters, cache)
        with execution.profile_oracle(counters, cache):
            snapshots, stats = queries.run_plan(panel, forecast, incumbent, .51, config, plan,
                seed=seed, adaptive=adaptive, families=8, width=4, certificate_observer=witness)
        outputs.append(execution.encoded(snapshots))
        traces.append(execution.encoded([(row["seed_assignment"], row["pair"], row["mutable"])
                                        for row in stats["families"]]))
        if plan == "static":
            assert sum(row["counters"].get("matching_static_reductions", 0) for row in stats["families"]) > 0
        assert compiled_matching.FamilyMatching is original
        for reason in ("forecast", "migration", "score"):
            assert counters.get(reason + "_leaves", 0) == sum(
                row["counters"].get(reason + "_excluded_masks", 0) for row in stats["families"])
    assert len(set(outputs)) == len(set(traces)) == 1


def test_complete_execution_entrypoint_installs_and_restores_control(monkeypatch):
    original = compiled_matching.FamilyMatching
    def search(*args):
        assert compiled_matching.FamilyMatching is FamilyMatching
        return {"selected": None, "stats": {"family_execution": "certified", "cpu_seconds": 0., "selection_cpu_seconds": 0.}}
    monkeypatch.setattr(execution, "search_refined", search)
    case = {"config": {}, "incumbent": (0, 1), "forecast": None, "build": None,
            "selection": None, "capacity": 1., "seed": 0}
    _, admission, _ = execution.execute(case, "static")
    assert admission == {"admitted": False, "reason": "no_selected_action"}
    assert compiled_matching.FamilyMatching is original
