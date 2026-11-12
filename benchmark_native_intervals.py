"""CloudCons host-level interval metrics for the common probabilistic forecast."""
import numpy as np
from scipy.special import ndtri
from cara.data import host_loads, host_memberships

NAMES = ("PICP", "MPIW", "WinklerScore")


def predictive_metrics(demand, assignment, medians, scales, *, simulations=100,
                       confidence=.95, seed=42):
    """CloudCons' active-host average, strict interval formula and fixed RNG.

    Every host draws independent VM quantile levels. One VM's level is shared
    across its complete trajectory, as in the upstream quantile-forecast path.
    Canonical ascending VM order specifies resident summation reproducibly.
    """
    demand, medians, scales = map(np.asarray, (demand, medians, scales))
    if (demand.ndim != 2 or medians.shape != demand.shape
            or scales.shape != (len(demand),) or not 0 < confidence < 1
            or int(simulations) != simulations or simulations < 1
            or not all(np.isfinite(x).all() for x in (demand, medians, scales))
            or any(np.any(x < 0) for x in (demand, medians, scales))):
        raise ValueError("Finite nonnegative demand/median/scales and a valid simulation design required")
    groups = host_memberships(assignment)
    truth = host_loads(demand, assignment)
    values = []
    alpha = 1 - confidence
    active_index = 0
    for residents, actual in zip(groups, truth):
        if not np.any(actual != 0):
            continue
        rng = np.random.default_rng(seed + active_index)
        active_index += 1
        ids = np.asarray(residents)
        levels = rng.uniform(0., 1., (int(simulations), len(ids)))
        shifts = np.zeros_like(levels)
        # A zero-scale point distribution remains defined at the RNG's
        # possible zero endpoint; positive scales map that endpoint to zero
        # after the declared nonnegative clipping.
        np.multiply(ndtri(levels), scales[ids][None, :], out=shifts,
                    where=scales[ids][None, :] > 0)
        simulated = np.maximum(medians[ids][None, :, :] + shifts[:, :, None], 0.).sum(axis=1)
        lower, upper = np.percentile(simulated, [alpha * 50, (1 - alpha / 2) * 100], axis=0)
        width = upper - lower
        coverage = np.mean((actual >= lower) & (actual <= upper))
        score = width + (2 / alpha) * np.maximum(0., lower - actual)
        score += (2 / alpha) * np.maximum(0., actual - upper)
        values.append((coverage, np.mean(width), np.mean(score)))
    means = np.mean(values, axis=0) if values else np.zeros(3)
    return {name: float(value) for name, value in zip(NAMES, means)}
