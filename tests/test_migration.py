"""Weighted physical-host matching against an independent exact oracle."""
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from cara.refinement import migration_profile


def test_weighted_count_ties_preserve_sub_ulp_cost_differences():
    incumbent = (0, 0, 1, 1, 2, 2)
    candidate = (0, 0, 1, 2, 0, 0)
    weights = np.array([1., 1e-20, 2e-20, 4e-20, 8e-20, 16e-20])
    result = migration_profile(incumbent, candidate, weights)
    exact_cost = sum((Fraction(float(w)) for a, b, w in zip(
        incumbent, result["aligned_assignment"], weights) if a != b), Fraction())
    oracle = []
    for mapping in permutations(range(3)):
        moved = [a != mapping[b] for a, b in zip(incumbent, candidate)]
        cost = sum((Fraction(float(w)) for w, change in zip(weights, moved)
                    if change), Fraction())
        oracle.append((sum(moved), cost))
    assert (result["count"], exact_cost) == min(oracle)


def test_weighted_matching_avoids_overflow_in_the_objective():
    incumbent = (0, 0, 1, 1, 2, 2)
    candidate = (0, 0, 1, 2, 0, 0)
    weights = np.array([1e308, 1e308, 1., 2., 4., 8.])
    result = migration_profile(incumbent, candidate, weights)
    # The fixed first host retains both enormous weights. Matching the two
    # singleton groups to hosts 2 and 1 moves VM 2 instead of the heavier VM 3.
    assert result["count"] == 3 and result["cost"] == 13.


def test_unrepresentable_matched_cost_raises_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="output range"):
        migration_profile((0, 0, 1, 1), (0, 1, 0, 1), np.full(4, 1e308))
