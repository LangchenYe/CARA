"""Frozen outward allowances for prefix-indexed canonical raw cells."""
from math import isfinite

from .error_budget import (UNIT, TINY, _add, _count, _div, _finite, _mul,
    _one_plus_unit, _sum_error, upper_float)


def window_raw_allowance_upper(sizes, capacity, horizon, length, demand_upper):
    """Bound indexed window raw loss against its original stored reduction.

    Both consumers use identical canonical load and per-hour raw-cell bits.
    Overflow requires falling back to the exact window consumer.  This budget
    does not change a population support or treat overlapping rows as iid.
    """
    sizes = tuple(_count(size) for size in sizes)
    hosts = _count(len(sizes))
    horizon, length = _count(horizon), _count(length)
    if horizon > length:
        raise ValueError("The horizon must fit in the indexed hour table")
    cap = float(capacity)
    if not isfinite(cap) or cap <= 0:
        raise ValueError("A positive finite stored capacity is required")
    terms = _count(hosts * horizon)
    bound = upper_float(demand_upper)
    load_max = max(_add(_mul(float(size), bound), _sum_error(size, bound))
                   for size in set(sizes))
    absolute = _div(load_max, cap)
    ratio_error = _add(_mul(UNIT, absolute), TINY)
    clip_error = _add(_add(_one_plus_unit(ratio_error),
        _mul(UNIT, _add(absolute, 1.))), TINY)
    cell_max = _finite(_add(absolute, clip_error))

    prefix_error = _sum_error(length, cell_max)
    _finite(_add(_mul(float(length), cell_max), prefix_error))
    window_max = _mul(float(horizon), cell_max)
    delta = _add(_add(_mul(2., _one_plus_unit(prefix_error)),
        _mul(UNIT, window_max)), TINY)
    indexed_operand = _add(window_max, delta)
    host_error = _sum_error(hosts, indexed_operand)
    indexed_sum_max = _finite(_add(_mul(float(hosts), indexed_operand), host_error))
    _finite(_div(indexed_sum_max, float(terms)))
    indexed_sum_error = _add(_mul(float(hosts), delta), host_error)
    normalization_error = _add(_mul(UNIT, cell_max), TINY)
    indexed_error = _add(_div(_one_plus_unit(indexed_sum_error), float(terms)),
                         normalization_error)

    canonical_sum_error = _sum_error(terms, cell_max)
    _finite(_add(_mul(float(terms), cell_max), canonical_sum_error))
    canonical_error = _add(_div(_one_plus_unit(canonical_sum_error), float(terms)),
                           normalization_error)
    return _finite(_add(indexed_error, canonical_error))
