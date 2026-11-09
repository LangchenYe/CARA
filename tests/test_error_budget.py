"""Outward binary64 constants dominate the exact frozen recurrences."""
from fractions import Fraction
from random import Random

import pytest

from cara.action_query import _sum_error
from cara.bounded_query import projection_raw_allowance
from cara.error_budget import (migration_radii_upper, projection_raw_allowance_upper,
                               sum_error_upper, upper_float)


def test_outward_sum_bounds_dominate_the_geometric_rational_budget():
    counts = (1, 2, 3, 7, 63, 256, 4096, 2**20, 2**52, 2**53)
    operands = (Fraction(), Fraction(1, 3), Fraction(7, 10), Fraction(1),
                Fraction(1, 2**1075), Fraction(1, 2**1022), Fraction(2**400), Fraction(1, 2**400))
    for count in counts:
        for operand in operands:
            bound = sum_error_upper(count, operand)
            assert Fraction(bound) >= _sum_error(count, operand)
    rng = Random(3317)
    for _ in range(32):
        count = rng.randrange(1, 2**53 + 1)
        operand = Fraction(rng.randrange(1, 1000), 997) * Fraction(2) ** rng.randrange(-400, 401)
        assert Fraction(sum_error_upper(count, operand)) >= _sum_error(count, operand)
    assert sum_error_upper(0, 1.) == 0.
    with pytest.raises(ValueError):
        sum_error_upper(2**53 + 1, 1.)
    with pytest.raises(ValueError):
        sum_error_upper(3.5, 1.)
    with pytest.raises(OverflowError):
        sum_error_upper(2**53, Fraction(2**950))


def test_support_conversion_and_underflow_remain_outward():
    for value in (Fraction(1, 3), Fraction(1, 2**1075), Fraction(3, 2**1076)):
        assert Fraction(upper_float(value)) >= value
    # Positive products below the subnormal range still contribute an outward
    # allowance rather than being rounded silently to zero.
    assert sum_error_upper(2, Fraction(1, 2**1075)) > 0.
    with pytest.raises(ValueError):
        upper_float(-1.)


def test_outward_projection_budget_dominates_the_existing_rational_formula():
    for sizes, horizon in (((4, 7, 3), 8), ((1, 1, 1), 1), ((64, 64), 32)):
        for support in (Fraction(1, 3), Fraction(1), Fraction(1, 2**1075)):
            for radii in ((0.,) * len(sizes), tuple(2. ** (-45 - i) for i in range(len(sizes)))):
                for capacity in (.75, 1.5, 4., 2.**-400, 2.**400):
                    exact = projection_raw_allowance(sizes, radii, capacity, horizon, support)
                    bound = projection_raw_allowance_upper(sizes, radii, capacity, horizon, support)
                    assert Fraction(bound) >= Fraction(exact)


def test_delta_radii_dominate_the_exact_host_recurrences():
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1075)
    # Host sizes may change; some hosts have only incoming or outgoing moves.
    old, new, depart, arrive = (4, 7, 3, 1), (5, 5, 4, 1), (1, 3, 0, 0), (2, 1, 1, 0)
    for support in (Fraction(1, 3), Fraction(1), Fraction(1, 2**1075)):
        def error(count):
            return _sum_error(count, support) if count else Fraction()
        for kernel in ("canonical", "csr"):
            ref, candidate = migration_radii_upper(old, new, depart, arrive, support,
                                                  incumbent_reduction=kernel)
            for h, (r_old, r_new, d, a) in enumerate(zip(old, new, depart, arrive)):
                if not d and not a:
                    expected_ref = 2 * error(r_old) if kernel == "csr" else Fraction()
                    expected_candidate = expected_ref
                elif kernel == "csr":
                    expected_ref = 2 * error(r_old)
                    expected_candidate = ((1 + unit) * (error(r_old) + error(d + a))
                        + unit * r_new * support + tiny + error(r_new))
                else:
                    expected_ref = Fraction()
                    first = (1 + unit) * (error(r_old) + error(d)) + unit * (r_old - d) * support + tiny
                    expected_candidate = (1 + unit) * (first + error(a)) + unit * r_new * support + tiny + error(r_new)
                assert Fraction(ref[h]) >= expected_ref
                assert Fraction(candidate[h]) >= expected_candidate


def test_paired_budget_type_sharing_keeps_each_exact_projection_bound():
    from cara.error_budget import paired_raw_allowance_upper, sum_error_upper
    from cara.bounded_query import projection_raw_allowance
    sizes = ((2, 8, 2, 4), (4, 2, 8, 2))
    radii = tuple(tuple(2 * sum_error_upper(r, 1.) for r in side) for side in sizes)
    actual = paired_raw_allowance_upper(sizes, radii, 1.5, 8, 1.)
    assert actual[0] == actual[1]
    for side, radius, upper in zip(sizes, radii, actual):
        assert upper >= projection_raw_allowance(side, radius, 1.5, 8, 1.)
    import pytest
    with pytest.raises(ValueError):
        paired_raw_allowance_upper(sizes, radii + (radii[0],), 1.5, 8, 1.)


def test_late_projection_budget_dominates_the_exact_changed_program():
    from cara.bounded_query import projection_late_raw_allowance
    from cara.error_budget import paired_late_raw_allowance_upper
    for sizes, horizon in (((4, 7, 3), 8), ((1, 1, 1), 1), ((64, 64), 32)):
        for support in (Fraction(1, 3), Fraction(1), Fraction(1, 2**1075)):
            radii = tuple(2 * sum_error_upper(r, support) for r in sizes)
            for capacity in (.75, 1.5, 4., 2.**-400, 2.**400):
                exact = projection_late_raw_allowance(sizes, radii, capacity, horizon, support)
                bound = paired_late_raw_allowance_upper((sizes, sizes), (radii, radii),
                    capacity, horizon, support)
                assert bound[0] == bound[1]
                assert Fraction(bound[0]) >= Fraction(exact)


def test_late_normalization_rejects_an_overflowing_uniform_sum_only():
    from cara.bounded_query import projection_late_raw_allowance
    from cara.error_budget import paired_late_raw_allowance_upper
    sizes, radii = (4, 4), (1e293, 1e293)
    assert projection_raw_allowance_upper(sizes, radii, 1e307, 8, 1e307) < float('inf')
    with pytest.raises(OverflowError):
        projection_late_raw_allowance(sizes, radii, 1e307, 8, 1e307)
    with pytest.raises(OverflowError):
        paired_late_raw_allowance_upper((sizes, sizes), (radii, radii), 1e307, 8, 1e307)
