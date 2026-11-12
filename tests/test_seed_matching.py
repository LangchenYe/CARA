"""Independent exact checks for seed-universal matching compilation."""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import ResidualFamily
from cara.refinement import migration_profile
from cara.seed_matching import SeedMatchingCache, SeedMatchingIndex


def optimum(incumbent, candidate):
    hosts = len(set(incumbent))
    matrix = np.zeros((hosts, hosts), dtype=np.int64)
    np.add.at(matrix, (np.asarray(incumbent), np.asarray(candidate)), 1)
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return int(matrix[rows, columns].sum())


def test_seed_index_exhaustive_family_counts_and_dimension():
    rng = np.random.default_rng(130409)
    for hosts in range(2, 8):
        for _ in range(20):
            incumbent = canonical_assignment(np.repeat(np.arange(hosts), 3))
            seed = canonical_assignment(rng.permutation(incumbent))
            index = SeedMatchingIndex(incumbent, seed)
            groups = host_memberships(seed)
            q = min(Counter(incumbent).values())
            movements = len(seed) - optimum(incumbent, seed)
            for pair in [(0, hosts - 1), (hosts // 2 - 1, hosts // 2)]:
                mutable = tuple(sorted(groups[pair[0]][:2] + groups[pair[1]][:2]))
                family = ResidualFamily(incumbent, seed, pair, mutable)
                model = index.family(pair, mutable, family.symmetric, family.stats)
                assert len(model.kernel.rows) <= min(hosts, 2 + 3 * movements // (q + 1))
                assert model.seed_retained() == optimum(incumbent, seed)
                for mask in family.nonempty_masks():
                    signature = family._left_signature(mask)
                    right = tuple(total - left for total, left in zip(family._totals, signature))
                    assert model.retained(family.origins, signature, right) == optimum(
                        incumbent, family.assignment(mask))
                    labels = np.asarray(seed).copy()
                    for column, members in zip(pair, family.groups(mask)):
                        labels[list(members)] = column
                    matrix = np.zeros((hosts, hosts), dtype=np.int64)
                    np.add.at(matrix, (np.asarray(incumbent), labels), 1)
                    remaining_rows, remaining_columns = set(range(hosts)), set(range(hosts))
                    for row, column, weight, r_other, c_other, _ in model.kernel.fixed_edges:
                        assert int(matrix[row, column]) == weight
                        assert r_other >= max((int(matrix[row, other]) for other in remaining_columns
                                               if other != column), default=0)
                        assert c_other >= max((int(matrix[other, column]) for other in remaining_rows
                                               if other != row), default=0)
                        remaining_rows.remove(row)
                        remaining_columns.remove(column)
                    assert remaining_rows == set(model.kernel.rows)
                    assert remaining_columns == set(model.kernel.columns)


def test_seed_index_cycle_restores_only_edited_edges_and_rich_outputs():
    hosts, residents = 17, 8
    incumbent = canonical_assignment(np.repeat(np.arange(hosts), residents))
    labels = list(incumbent)
    for host in range(hosts):
        labels[host * residents] = (host + 1) % hosts
    seed = canonical_assignment(labels)
    index = SeedMatchingIndex(incumbent, seed)
    assert len(index.fixed_edges) == hosts
    groups = host_memberships(seed)
    for pair in [(0, 4), (3, 15), (8, 9)]:
        mutable = tuple(sorted(groups[pair[0]][:3] + groups[pair[1]][:3]))
        family = ResidualFamily(incumbent, seed, pair, mutable,
                                matching_backend="compiled", unique_physical_min_hosts=0)
        family._factor = index.family(pair, mutable, family.symmetric, family.stats)
        assert len(family._factor.kernel.rows) == 2
        assert family.stats["matching_seed_restored_edges"] == 2
        assert family.stats["matching_seed_fixed_edges"] == hosts - 2
        for mask in family.nonempty_masks():
            assignment = family.assignment(mask)
            assert family.movement_count(mask) == len(seed) - optimum(incumbent, assignment)
            physical = family.unique_physical_profile(mask, assignment)
            if physical is not None:
                assert physical == migration_profile(incumbent, assignment)


def test_seed_index_symmetric_families_and_tied_persistent_edges():
    # Include a zero-background family and a non-strict mass certificate.
    incumbent = canonical_assignment([0, 0, 0, 1, 1, 1, 2, 2, 2])
    for seed in [canonical_assignment([0, 0, 1, 0, 1, 1, 2, 2, 2]), incumbent]:
        groups = host_memberships(seed)
        for pair in [(0, 1), (1, 2), (2, 0)]:
            mutable = tuple(sorted(groups[pair[0]] + groups[pair[1]]))
            family = ResidualFamily(incumbent, seed, pair, mutable,
                                    matching_backend="compiled", unique_physical_min_hosts=0)
            family._factor = SeedMatchingIndex(incumbent, seed).family(
                pair, mutable, family.symmetric, family.stats)
            for mask in family.nonempty_masks():
                assignment = family.assignment(mask)
                assert family.movement_count(mask) == len(seed) - optimum(incumbent, assignment)
                physical = family.unique_physical_profile(mask, assignment)
                if physical is not None:
                    assert physical == migration_profile(incumbent, assignment)


def test_seed_cache_reuses_immutable_objects_and_evicts():
    incumbent = canonical_assignment([0, 0, 1, 1, 2, 2])
    first = canonical_assignment([0, 1, 1, 0, 2, 2])
    second = canonical_assignment([0, 0, 1, 2, 1, 2])
    cache = SeedMatchingCache(incumbent, max_seeds=1)
    index = cache.get(first)
    assert cache.get(first) is index
    cache.get(second)
    assert len(cache.indexes) == 1
    assert cache.get(first) is not index


def test_unchanged_origin_refines_a_weak_mass_edge_for_physical_outputs():
    # The (0,0) edge has w=3, row mass 5, column mass 4: its universal
    # mass test is an equality. Its true alternative maxima are only 1+1.
    incumbent = canonical_assignment([0] * 5 + [1] * 5 + [2] * 4)
    seed = canonical_assignment([0, 0, 0, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    pair, mutable = (1, 2), (6, 7, 10, 11)
    reference = ResidualFamily(incumbent, seed, pair, mutable,
                               matching_backend="compiled", unique_physical_min_hosts=0)
    indexed = ResidualFamily(incumbent, seed, pair, mutable,
                             matching_backend="compiled", unique_physical_min_hosts=0)
    index = SeedMatchingIndex(incumbent, seed)
    indexed._factor = index.family(pair, mutable, False, indexed.stats)
    assert index.fixed_edges[index.fixed_columns[0]][2:5] == (3, 2, 1)
    assert tuple(indexed._factor.kernel.fixed_edges)[0][2:5] == (3, 1, 1)
    for mask in indexed.nonempty_masks():
        assignment = indexed.assignment(mask)
        assert indexed.unique_physical_profile(mask, assignment) == reference.unique_physical_profile(mask, assignment)
