"""A fully specified environment law with independent workload scenes.

Environment parameters are sampled once. No scene is centered or normalized
using other scenes; all Fit, Build, selection, certification, and evaluation
samples therefore follow the same conditional scene distribution.
"""
from dataclasses import dataclass
import numpy as np

LAW = {
    "vms": 48, "horizon": 8, "capacity": 1., "phase_groups": 4,
    "mean_range": [.15, .23], "amplitude_range": [.03, .09],
    "common_loading_range": [.015, .035], "group_loading_range": [.04, .10],
    "idiosyncratic_scale_range": [.015, .035], "factor_ar": .55,
    "groups": "independently permuted balanced phase and shock groups",
    "nonnegativity": "clip the sampled Gaussian demand at zero",
}


def _stationary_ar(rng, shape, rho):
    values = rng.standard_normal(shape)
    scale = np.sqrt(1. - rho * rho)
    for time in range(1, shape[-1]):
        values[..., time] = rho * values[..., time - 1] + scale * values[..., time]
    return values


@dataclass(frozen=True)
class Environment:
    seed: int
    means: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    groups: np.ndarray
    common_loadings: np.ndarray
    group_loadings: np.ndarray
    idiosyncratic_scales: np.ndarray

    def sample(self, count, role):
        if count <= 0:
            raise ValueError("The scene count must be positive")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(role)]))
        horizon = LAW["horizon"]
        common = _stationary_ar(rng, (count, horizon), LAW["factor_ar"])
        shocks = _stationary_ar(rng, (count, 4, horizon), LAW["factor_ar"])
        noise = rng.standard_normal((count, LAW["vms"], horizon))
        trajectory = self.means[:, None] + self.amplitudes[:, None] * np.sin(2. * np.pi * np.arange(horizon)[None, :] / horizon + self.phases[:, None])
        values = trajectory[None, :, :] + common[:, None, :] * self.common_loadings[None, :, None]
        values += shocks[:, self.groups, :] * self.group_loadings[None, :, None]
        values += noise * self.idiosyncratic_scales[None, :, None]
        return np.maximum(values, 0.)

    def parameters(self):
        return {name: getattr(self, name) for name in ("means", "amplitudes", "phases", "groups", "common_loadings", "group_loadings", "idiosyncratic_scales")}


def make_environment(seed):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    count = LAW["vms"]
    phases = rng.permutation(np.repeat(np.arange(4), count // 4)) * (np.pi / 2.)
    groups = rng.permutation(np.repeat(np.arange(4), count // 4))
    return Environment(int(seed), rng.uniform(*LAW["mean_range"], count), rng.uniform(*LAW["amplitude_range"], count), phases, groups, rng.uniform(*LAW["common_loading_range"], count), rng.uniform(*LAW["group_loading_range"], count), rng.uniform(*LAW["idiosyncratic_scale_range"], count))
