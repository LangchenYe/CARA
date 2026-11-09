"""Positive outward binary64 evaluation of frozen arithmetic allowances.

These routines upper-bound the existing rational geometric-summation and
projection recurrences. They do not estimate errors from observed workload
values, change population gain supports, or introduce a statistical tolerance.
"""
from collections import Counter
from fractions import Fraction
from math import inf, isfinite, nextafter


UNIT = 2. ** -53
TINY = 2. ** -1074  # An outward representation of the half-subnormal allowance.


def _up(value):
    return nextafter(value, inf)


def _add(a, b):
    return _up(a + b) if a or b else 0.


def _mul(a, b):
    return _up(a * b) if a and b else 0.


def _div(a, b):
    return _up(a / b) if a else 0.


def _one_plus_unit(value):
    # The exact constant 1+2^-53 is not representable in binary64.
    return _add(value, _mul(UNIT, value))


def upper_float(value):
    """Convert a nonnegative support to an outward finite binary64 value."""
    exact = Fraction(value)
    if exact < 0:
        raise ValueError("A nonnegative arithmetic support is required")
    result = float(exact)
    if not isfinite(result):
        raise OverflowError("The arithmetic support exceeds finite binary64")
    if Fraction(result) < exact:
        result = nextafter(result, inf)
    if not isfinite(result):
        raise OverflowError("The outward arithmetic support exceeds finite binary64")
    return result


def _count(value, *, allow_zero=False):
    if int(value) != value or not (0 if allow_zero else 1) <= value <= 2**53:
        raise ValueError("Arithmetic counts must be integers not exceeding 2**53")
    return int(value)


def _finite(value):
    if not isfinite(value):
        raise OverflowError("The uniform arithmetic allowance exceeds finite binary64")
    return value


def _sum_error(count, operand):
    """Upper-bound S(count,operand), for an already outward operand bound."""
    if count < 2:
        return 0.
    steps = float(count - 1)
    eta = steps * UNIT
    # Counts <=2^53 make eta and 1-eta exact dyadic values, with 1-eta>0.
    gamma = _up(eta / (1. - eta))
    growth = _add(1., gamma)
    relative = _mul(_mul(gamma, float(count)), operand)
    underflow = _mul(_mul(TINY, steps), growth)
    return _finite(_add(relative, underflow))


def sum_error_upper(count, operand):
    """Outward counterpart of action_query._sum_error; zero terms give zero."""
    return _sum_error(_count(count, allow_zero=True), upper_float(operand))


def migration_radii_upper(old_sizes, new_sizes, departures, arrivals, demand_upper,
                          *, incumbent_reduction="csr"):
    """Return physical-host radii for the original stored old/new loads."""
    if incumbent_reduction not in ("canonical", "csr"):
        raise ValueError("The incumbent reduction is canonical or csr")
    values = tuple(tuple(_count(v, allow_zero=j >= 2) for v in x)
                   for j, x in enumerate((old_sizes, new_sizes, departures, arrivals)))
    if len({len(x) for x in values}) != 1 or not values[0]:
        raise ValueError("Complete aligned host-size and migration-count arrays required")
    bound, errors = upper_float(demand_upper), {0: 0.}
    def error(count):
        count = _count(count, allow_zero=True)
        if count not in errors:
            errors[count] = _sum_error(count, bound)
        return errors[count]
    rows, cache = list(zip(*values)), {}
    for r_old, r_new, depart, arrive in set(rows):
        _count(r_old); _count(r_new)
        if not 0 <= depart <= r_old or not 0 <= arrive <= r_new or r_old - depart + arrive != r_new:
            raise ValueError("Aligned resident counts and migration incidence disagree")
        reference = _mul(2., error(r_old)) if incumbent_reduction == "csr" else 0.
        if not depart and not arrive:
            candidate = reference
        elif incumbent_reduction == "csr":
            previous = _add(error(r_old), error(depart + arrive))
            approximate = _add(_add(_one_plus_unit(previous),
                _mul(UNIT, _mul(float(r_new), bound))), TINY)
            candidate = _add(approximate, error(r_new))
        else:
            previous = _add(error(r_old), error(depart))
            residual = _add(_add(_one_plus_unit(previous),
                _mul(UNIT, _mul(float(r_old - depart), bound))), TINY)
            previous = _add(residual, error(arrive))
            approximate = _add(_add(_one_plus_unit(previous),
                _mul(UNIT, _mul(float(r_new), bound))), TINY)
            candidate = _add(approximate, error(r_new))
        cache[(r_old, r_new, depart, arrive)] = (_finite(reference), _finite(candidate))
    return tuple(cache[row][0] for row in rows), tuple(cache[row][1] for row in rows)


def _projection_raw_classes(classes, capacity, horizon, demand_upper, *, late=False):
    """Compile one outward projection budget from repeated (size,radius) types."""
    hosts = sum(classes.values())
    cap, horizon = float(capacity), _count(horizon)
    if not isfinite(cap) or cap <= 0 or hosts < 1:
        raise ValueError("A positive finite capacity and complete host family are required")
    terms = _count(hosts * horizon)
    bound, largest, radius_counts = upper_float(demand_upper), {}, Counter()
    for (size, radius), count in classes.items():
        size = _count(size)
        if not isfinite(radius) or radius < 0:
            raise ValueError("One finite nonnegative radius per complete host required")
        largest[size] = max(largest.get(size, 0.), radius)
        radius_counts[radius] += count
    max_load = max(_add(_add(_mul(float(size), bound), _sum_error(size, bound)), radius)
                   for size, radius in largest.items())
    absolute = _div(max_load, cap)
    ratio = _add(_mul(UNIT, absolute), TINY)
    excess = _add(_add(_one_plus_unit(ratio), _mul(UNIT, _add(absolute, 1.))), TINY)
    total = _add(_mul(float(terms), excess), _sum_error(terms, _add(absolute, excess)))
    projection = _add(_add(_div(_one_plus_unit(total), float(terms)), _mul(UNIT, absolute)), TINY)
    radius_sum = 0.
    for radius, count in radius_counts.items():
        radius_sum = _add(radius_sum, _mul(float(count), radius))
    lipschitz = _div(_div(radius_sum, float(hosts)), cap)
    if not late:
        return _finite(_add(lipschitz, _mul(2., projection)))
    # The alternate consumer clips load-minus-capacity before normalization.
    # Its positive unnormalized sum and the first division must also be finite;
    # the early-normalization domain alone does not imply this extra contract.
    clipped_max = _up(max_load - cap) if max_load > cap else 0.
    clipped_error = _add(_mul(UNIT, clipped_max), TINY)
    operand = _add(clipped_max, clipped_error)
    summation = _sum_error(terms, operand)
    sum_max = _finite(_add(_mul(float(terms), operand), summation))
    _finite(_div(_div(sum_max, float(terms)), cap))
    total_late = _add(_mul(float(terms), clipped_error), summation)
    mean_error = _add(_add(_div(_one_plus_unit(total_late), float(terms)),
        _mul(UNIT, clipped_max)), TINY)
    projected_late = _add(_add(_div(_one_plus_unit(mean_error), cap),
        _mul(UNIT, _div(clipped_max, cap))), TINY)
    return _finite(_add(_add(lipschitz, projection), projected_late))


def _projection_classes(sizes, radii):
    sizes, radii = tuple(sizes), tuple(map(float, radii))
    if not sizes or len(sizes) != len(radii):
        raise ValueError("One finite nonnegative radius per complete host required")
    return Counter(zip(sizes, radii))


def projection_raw_allowance_upper(sizes, radii, capacity, horizon, demand_upper):
    """Outward counterpart of bounded_query.projection_raw_allowance."""
    return _projection_raw_classes(_projection_classes(sizes, radii), capacity, horizon, demand_upper)


def paired_raw_allowance_upper(sizes, radii, capacity, horizon, demand_upper):
    """Share identical frozen budget types between the two placement roots."""
    sizes, radii = tuple(sizes), tuple(radii)
    if len(sizes) != 2 or len(radii) != 2:
        raise ValueError("A complete paired view requires two size/radius families")
    cache, output = {}, []
    for side, radius in zip(sizes, radii):
        classes = _projection_classes(side, radius)
        key = frozenset(classes.items())
        if key not in cache:
            cache[key] = _projection_raw_classes(classes, capacity, horizon, demand_upper)
        output.append(cache[key])
    if len(output) != 2:
        raise ValueError("A complete paired view requires two size/radius families")
    return tuple(output)


def paired_late_raw_allowance_upper(sizes, radii, capacity, horizon, demand_upper):
    """Bound late-normalized centers against the original stored raw losses.

    An OverflowError requires using the original early consumer instead. This
    is a numerical physical-plan fallback, not a change to the action or rule.
    """
    sizes, radii = tuple(sizes), tuple(radii)
    if len(sizes) != 2 or len(radii) != 2:
        raise ValueError("A complete paired view requires two size/radius families")
    cache, output = {}, []
    for side, radius in zip(sizes, radii):
        classes = _projection_classes(side, radius)
        key = frozenset(classes.items())
        if key not in cache:
            cache[key] = _projection_raw_classes(classes, capacity, horizon, demand_upper, late=True)
        output.append(cache[key])
    return tuple(output)
