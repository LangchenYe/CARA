"""Lower the original finite-table census to exact dyadic integer reductions."""
from fractions import Fraction

from .certification import _losses, _names


def _paired_total(reference, candidate):
    numerator, exponent = 0, 0
    for left, right in zip(reference, candidate):
        for value, sign in ((left, 1), (right, -1)):
            part, denominator = float(value).as_integer_ratio()
            shift = denominator.bit_length() - 1
            if shift > exponent:
                numerator <<= shift - exponent
                exponent = shift
            numerator += sign * (part << (exponent - shift))
    return numerator, exponent


def exact_dyadic_gate(reference, candidate, *, metric_names=None):
    """Return exactly the original exact_pareto_gate dictionary.

    Each input is accumulated separately; a rounded floating-point paired
    subtraction would change the target. Only final coordinate means need
    Fraction objects, including their original rounded and string outputs.
    """
    reference, candidate = _losses(reference, "reference"), _losses(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("paired loss matrices must have the same shape")
    names = _names(metric_names, reference.shape[1])
    totals = [_paired_total(reference[:, e], candidate[:, e]) for e in range(reference.shape[1])]
    rows = len(reference)
    means = [Fraction(numerator, (1 << exponent) * rows) for numerator, exponent in totals]
    admitted = all(value >= 0 for value, _ in totals) and any(value > 0 for value, _ in totals)
    return {"admitted": admitted, "rows": rows, "rows_read": rows,
        "decision": "exact_pareto_census", "reason": "exact_pareto_census",
        "metric_names": list(names), "gains": [float(value) for value in means],
        "exact_gains": [str(value) for value in means],
        "regressed_metrics": [name for name, (value, _) in zip(names, totals) if value < 0],
        "strict_metrics": [name for name, (value, _) in zip(names, totals) if value > 0],
        "claim_scope": "the supplied finite table only"}
