"""Exact evidence for one selected CARA action.

Burden arrays have columns J, V, R, S and contain deterministic binary64 replay
values in [0, 1]. Bets must be fixed before certification; fit_bets is intended
for a separate Build panel. Fresh scenes need the conditional-mean assumptions
in the paper. Finite certification concerns only its fixed table and draws an
independent uniform reveal order. Neither API establishes future trace safety.
"""

from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from math import comb

import numpy as np

M = 2**53
GRID = 256
SCORE_SCALE = 2**40
DENOMINATORS = (1, M + 1, M + 1, M + 1)
ENDPOINTS = ("J", "V", "R", "S")


def _alpha(value):
    value = value if isinstance(value, Fraction) else Fraction(str(value))
    if not 0 < value < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return value


def _burdens(values, *, name="burdens"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{name} must have shape (number of scenes, 4)")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    if np.any((values[:, 0] != 0) & (values[:, 0] != 1)):
        raise ValueError("J must contain exact zero/one occurrence bits")
    return values


def _bets(values):
    if values is None:
        return (128,) * 4  # A predeclared constant, never fitted on Certification.
    if len(values) != 4 or any(int(j) != j or not 0 <= j < GRID for j in values):
        raise ValueError("bets must contain four integer grid indices in [0, 255]")
    return tuple(int(j) for j in values)


def encoded_gains(reference, candidate):
    """Return integer gain rows and the four exact increment denominators."""
    reference = _burdens(reference, name="reference")
    candidate = _burdens(candidate, name="candidate")
    if candidate.shape != reference.shape:
        raise ValueError("paired burden arrays must have the same shape")
    rows = []
    for ref, cand in zip(reference, candidate):
        # Multiplication by 2**53 is exact for binary64 values in [0, 1].
        rows.append((int(ref[0]) - int(cand[0]),) + tuple(
            int(float(ref[e]) * M) - int(float(cand[e]) * M) - 1
            for e in range(1, 4)
        ))
    return rows, DENOMINATORS


def _product_tree(values):
    values = [int(value) for value in values]
    if not values:
        return 1
    while len(values) > 1:
        values = [values[k] * values[k + 1] for k in range(0, len(values) - 1, 2)] + (
            values[-1:] if len(values) % 2 else []
        )
    return values[0]


def _wealth_parts(z, denominator, index):
    """Unreduced exact product parts; decisions need no integer gcd."""
    denominator, index = int(denominator), int(index)
    if index == 0:
        return 1, 1
    scale = GRID * denominator
    groups = Counter(z)
    numerator = _product_tree(pow(scale + index * int(value), count) for value, count in groups.items())
    return numerator, pow(scale, len(z))


def _wealth(z, denominator, index):
    return Fraction(*_wealth_parts(z, denominator, index))


def fit_bets(reference, candidate):
    """Fit the smallest exact constant-bet maximizer on a Build panel."""
    rows, denominators = encoded_gains(reference, candidate)
    result = []
    for e, denominator in enumerate(denominators):
        groups = Counter(row[e] for row in rows if row[e] != 0)
        scale = GRID * denominator
        cache = {}

        def product(index):
            if index not in cache:
                cache[index] = _product_tree(pow(scale + index * z, count) for z, count in groups.items())
            return cache[index]

        # Floating arithmetic only proposes an index. Exact neighboring products
        # prove that it is the smallest maximizer; otherwise exact bisection runs.
        x = np.array([row[e] / denominator for row in rows], dtype=np.float64)
        low, high = 0, GRID - 1
        while low < high:
            middle = (low + high) // 2
            delta = np.log1p(((middle + 1) / GRID) * x).sum() - np.log1p((middle / GRID) * x).sum()
            if delta > 0:
                low = middle + 1
            else:
                high = middle
        proposal = low
        if ((proposal == 0 or product(proposal) > product(proposal - 1)) and
                (proposal == GRID - 1 or product(proposal + 1) <= product(proposal))):
            result.append(proposal)
            continue
        low, high = 0, GRID - 1
        while low < high:
            middle = (low + high) // 2
            if product(middle + 1) > product(middle):
                low = middle + 1
            else:
                high = middle
        result.append(low)
    return tuple(result)


def _binary_mixture_parts(beneficial, harmful):
    b, h = int(beneficial), int(harmful)
    if b != beneficial or h != harmful or b < 0 or h < 0:
        raise ValueError("discordance counts must be nonnegative integers")
    m, n = b + h, b + h + 1
    term = tail = 1
    for k in range(1, min(b, h) + 1):
        term = term * (n - k + 1) // k
        tail += term
    numerator = tail if b <= h else (1 << n) - tail
    return numerator, n * comb(m, b)


def binary_mixture(beneficial, harmful):
    """Uniform integral of (1+lambda)**b (1-lambda)**h, exactly."""
    return Fraction(*_binary_mixture_parts(beneficial, harmful))


_binary_mixture = binary_mixture


def log_cell(rational, *, max_precision=4096):
    """Resolve floor(2**40 log(q)) by outward decimal enclosures.

    q=1 is checked algebraically, including projected threshold equalities.
    An unresolved resource-limited computation raises rather than inventing a
    tie. Decimal.ln is correctly rounded; adjacent representable values enclose
    each result independently of its last rounding direction.
    """
    rational = Fraction(rational)
    if rational <= 0:
        raise ValueError("logarithm source must be positive")
    if rational == 1:
        return 0
    precision = 48
    while precision <= max_precision:
        with localcontext() as ctx:
            ctx.prec = precision
            ctx.rounding = ROUND_FLOOR
            low_q = Decimal(rational.numerator) / Decimal(rational.denominator)
            ctx.rounding = ROUND_CEILING
            high_q = Decimal(rational.numerator) / Decimal(rational.denominator)
            low_log = low_q.ln().next_minus()
            high_log = high_q.ln().next_plus()
            ctx.rounding = ROUND_FLOOR
            low_cell = int((low_log * SCORE_SCALE).to_integral_value(rounding=ROUND_FLOOR))
            ctx.rounding = ROUND_CEILING
            high_scaled = high_log * SCORE_SCALE
            high_cell = int(high_scaled.to_integral_value(rounding=ROUND_FLOOR))
            if low_cell == high_cell:
                return low_cell
        precision *= 2
    raise ArithmeticError("exact logarithm cell was not resolved within the precision limit")


def evidence_key(reference, candidate, bets=None, *, projection_ratio=1, alpha=0.05,
                 center=True, binary_mixture=True):
    """Four sorted exact score cells; append physical identity for total ties."""
    if int(projection_ratio) != projection_ratio or projection_ratio < 1:
        raise ValueError("projection_ratio must be a positive integer")
    ratio = int(projection_ratio)
    level = _alpha(alpha)
    rows, denominators = encoded_gains(reference, candidate)
    bets = fit_bets(reference, candidate) if bets is None else _bets(bets)
    sources = []
    for e, denominator in enumerate(denominators):
        z = [row[e] for row in rows]
        if e == 0 and binary_mixture:
            source = _binary_mixture(ratio * z.count(1), ratio * z.count(-1))
        else:
            source = _wealth(z, denominator, bets[e]) ** ratio
        sources.append(source * level if center else source)
    return tuple(sorted(log_cell(source) for source in sources))


def _candidate_source(candidate, n):
    if callable(candidate):
        return candidate
    array = np.asarray(candidate, dtype=np.float64)
    if array.shape != (n, 4):
        raise ValueError("candidate must match the reference shape")
    return lambda indices: array[indices]


def _crossed(wealth, level):
    return wealth.numerator * level.numerator >= wealth.denominator * level.denominator


def _result(admitted, rows_read, order, required, dominated, decision, totals=None):
    return {
        "admitted": bool(admitted),
        "rows_read": int(rows_read),
        "rows": int(rows_read),
        "order": [int(i) for i in order[:rows_read]],
        "required": [ENDPOINTS[e] for e in required],
        "dominated": [ENDPOINTS[e] for e in dominated],
        "decision": decision,
        "reason": decision,
        "gains_at_census": None if totals is None else [float(x) for x in totals],
        "exact_gains_at_census": None if totals is None else [str(x) for x in totals],
    }


def pareto_census(reference, candidate):
    """Exact original-burden Pareto comparison after one complete replay."""
    reference = _burdens(reference, name="reference")
    candidate = _burdens(candidate, name="candidate")
    if reference.shape != candidate.shape or not len(reference):
        raise ValueError("census needs nonempty paired burden arrays of the same shape")
    totals = []
    for e in range(4):
        counts = Counter(float(x) for x in reference[:, e])
        counts.subtract(float(x) for x in candidate[:, e])
        totals.append(sum((count * Fraction(value) for value, count in counts.items()), Fraction(0)))
    admitted = all(total >= 0 for total in totals) and any(total > 0 for total in totals)
    return _result(admitted, len(reference), range(len(reference)), list(range(4)), [], "census", totals)


def certify_finite(reference, candidate, bets=None, *, seed=0, batch=8, alpha=0.05,
                   anchor=2, use_dominance=True):
    """Random-order fixed-table admission with exact Pareto census priority.

    candidate may be a fixed burden array or callable(indices)->burden rows.
    The callable must replay the same fixed selected action at every request.
    """
    reference = _burdens(reference, name="reference")
    n = len(reference)
    if n < 1 or int(batch) != batch or batch < 1 or anchor not in range(4):
        raise ValueError("use a nonempty table, positive integer batch, and valid anchor")
    batch, anchor = int(batch), int(anchor)
    level, bets = _alpha(alpha), _bets(bets)
    fetch = _candidate_source(candidate, n)
    order = np.random.default_rng(seed).permutation(n)
    dominated = {e for e in range(4) if use_dominance and np.all(reference[:, e] == 1)}
    required = sorted((set(range(4)) - dominated) | {anchor})
    wealth = [Fraction(1) for _ in range(4)]
    sums, locked = [0] * 4, [False] * 4
    totals = [Fraction(0) for _ in range(4)]
    for start in range(0, n, batch):
        indices = order[start:min(start + batch, n)]
        cand = _burdens(fetch(indices), name="candidate batch")
        if cand.shape != (len(indices), 4):
            raise ValueError("candidate callback returned the wrong batch shape")
        ref = reference[indices]
        gains, _ = encoded_gains(ref, cand)
        for offset, row in enumerate(gains):
            time = start + offset
            remaining = n - time
            for e, denominator in enumerate(DENOMINATORS):
                totals[e] += Fraction(float(ref[offset, e])) - Fraction(float(cand[offset, e]))
                if e in required and not locked[e]:
                    a = denominator * remaining - sums[e]
                    if a < 0 or (a == 0 and row[e] > -denominator):
                        locked[e] = True  # Deterministic impossibility of the null.
                    elif a > 0:
                        numerator = GRID * a + bets[e] * (remaining * row[e] + sums[e])
                        wealth[e] *= Fraction(numerator, GRID * a)
                    # a=0 and x=-1 leaves wealth unchanged under the null.
                sums[e] += row[e]
        read = start + len(indices)
        if read == n:
            admitted = all(total >= 0 for total in totals) and any(total > 0 for total in totals)
            return _result(admitted, read, order, required, sorted(dominated), "census", totals)
        for e in required:
            locked[e] = locked[e] or _crossed(wealth[e], level)
        if all(locked[e] for e in required):
            return _result(True, read, order, required, sorted(dominated), "early")
    raise RuntimeError("unreachable finite-table completion")


def certify_fresh(reference, candidate, bets=None, *, batch=32, alpha=0.05,
                  binary_mixture=True):
    """Stopped strict four-endpoint test using unreduced integer products.

    Fixed and stopped calls share this exact kernel. Cross-multiplication avoids
    the unnecessary gcd cost of reducing very large evidence fractions.
    """
    reference = _burdens(reference, name="reference")
    n = len(reference)
    if n < 1 or int(batch) != batch or batch < 1:
        raise ValueError("use a nonempty scene panel and a positive integer batch")
    batch, level, bets = int(batch), _alpha(alpha), _bets(bets)
    fetch = _candidate_source(candidate, n)
    order = np.arange(n)
    numerators, denominators = [1] * 4, [1] * 4
    locked = [False] * 4
    beneficial = harmful = 0
    for start in range(0, n, batch):
        indices = order[start:min(start + batch, n)]
        cand = _burdens(fetch(indices), name="candidate batch")
        if cand.shape != (len(indices), 4):
            raise ValueError("candidate callback returned the wrong batch shape")
        gains, _ = encoded_gains(reference[indices], cand)
        for e, denominator in enumerate(DENOMINATORS):
            if locked[e]:
                continue
            z = [row[e] for row in gains]
            if e == 0 and binary_mixture:
                beneficial += z.count(1)
                harmful += z.count(-1)
                numerators[e], denominators[e] = _binary_mixture_parts(beneficial, harmful)
            else:
                numerator, divisor = _wealth_parts(z, denominator, bets[e])
                numerators[e] *= numerator
                denominators[e] *= divisor
            locked[e] = numerators[e] * level.numerator >= denominators[e] * level.denominator
        read = start + len(indices)
        if all(locked):
            return _result(True, read, order, list(range(4)), [], "strict")
    return _result(False, n, order, list(range(4)), [], "cap")
