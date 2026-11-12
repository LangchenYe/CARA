"""Family-scoped load-error budgets shared by coarse and full certificates."""
from types import SimpleNamespace

import numpy as np

from .family_score_bounds import _ETA, _U, _add_hi, _mul_hi, _reduction, _sum_real_upper


def compile_uniform_radii(full):
    """Compile one outward radius pair for an immutable residual family."""
    cached = getattr(full, "_uniform_radius_payload", None)
    if cached is not None:
        return cached
    payload = SimpleNamespace(uniform_ready=False, uniform_radius_lo=None, uniform_radius_hi=None)
    full._uniform_radius_payload = payload
    full.uniform_radius_compilations = getattr(full, "uniform_radius_compilations", 0) + 1
    from .metric_cascade import IndicatorBounds
    if not IndicatorBounds(full).available:
        return payload
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        w = len(full.mutable)
        mutable_real_hi = _sum_real_upper(full.mutable_values, 0, w)
        _, growth_w, delta_w = _reduction(w)
        _, growth_r, delta_r = _reduction(full.resident_bound)
        subset_error = _add_hi(_mul_hi(mutable_real_hi, growth_w - 1.), delta_w)
        canonical_real_hi = _add_hi(full.background_hi, mutable_real_hi)
        canonical_error = _add_hi(_mul_hi(canonical_real_hi, growth_r - 1.), delta_r)
        point_max = _add_hi(full.background_lo, _add_hi(mutable_real_hi, subset_error))
        addition_error = _add_hi(_mul_hi(point_max, float(_U)), float(_ETA))
        background_width = _add_hi(full.background_hi, -full.background_lo)
        payload.uniform_radius_lo = _add_hi(_add_hi(subset_error, canonical_error), addition_error)
        payload.uniform_radius_hi = _add_hi(payload.uniform_radius_lo, background_width)
        end_max = _add_hi(point_max, payload.uniform_radius_hi)
    fields = (mutable_real_hi, subset_error, canonical_error, point_max,
              addition_error, background_width, payload.uniform_radius_lo,
              payload.uniform_radius_hi, end_max)
    payload.uniform_ready = (all(np.isfinite(value).all() for value in fields)
                             and bool(np.all(end_max <= np.finfo(float).max / 8.)))
    return payload


def leaf_load_enclosures(full, ranges):
    """Enclose canonical stored loads with two ordinary endpoint operations."""
    if not ranges or any(stop != start + 1 for start, stop in ranges):
        return None
    compiler = compile_uniform_radii(full)
    if not compiler.uniform_ready:
        return None
    q, w = len(ranges), len(full.mutable)
    if full.mutable_sum_index is None:
        selectors = np.zeros((q, 2, w))
        offset = int(full.symmetric)
        for row, (mask, _) in enumerate(ranges):
            if offset:
                selectors[row, 0, 0] = 1.
            for bit in range(full.bits):
                selectors[row, (mask >> bit) & 1, bit + offset] = 1.
        sums = (selectors.reshape(-1, w) @ full.mutable_flat).reshape(q, 2, full.scenes, full.horizon)
    else:
        sums = full.mutable_sum_index.sum_ranges(ranges, possible=False)
    center = full.background_lo + sums
    return np.maximum(center - compiler.uniform_radius_lo, 0.), center + compiler.uniform_radius_hi
