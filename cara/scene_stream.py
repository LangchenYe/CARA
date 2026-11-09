"""Independent, prefix-consistent workload scenes for paired what-if queries."""
import numpy as np

from .synthetic import LAW, _stationary_ar


SCENE_LAWS = {
    "complementary": {"amplitude_multiplier": 1., "common_multiplier": 1.,
        "group_multiplier": 1., "burst_probability": 0., "burst_amplitude": (0., 0.)},
    "bursty_common": {"amplitude_multiplier": 1., "common_multiplier": 3.,
        "group_multiplier": 1.25, "burst_probability": .12, "burst_amplitude": (.10, .25)},
    "weak_complementarity": {"amplitude_multiplier": .10, "common_multiplier": 2.,
        "group_multiplier": .25, "burst_probability": 0., "burst_amplitude": (0., 0.)},
}


class SceneStream:
    """Draw consecutive iid scenes conditional on fixed environment parameters.

    Separate random streams for common factors, group factors, idiosyncratic
    noise, burst occurrence and burst magnitude make every scene prefix
    invariant to how requests are batched. Each component keeps the original
    within-scene temporal law; scenes are neither centered nor rescaled jointly.
    """

    def __init__(self, environment, law, role):
        if law not in SCENE_LAWS or int(role) != role or role < 0:
            raise ValueError("a declared scene law and nonnegative integer role are required")
        self.environment, self.law, self.role = environment, law, int(role)
        self.rows = 0
        self.generators = [np.random.default_rng(np.random.SeedSequence(
            [int(environment.seed), self.role, 7319, component])) for component in range(5)]

    def next(self, count):
        if int(count) != count or count < 1:
            raise ValueError("a positive integer scene count is required")
        count = int(count)
        environment, settings = self.environment, SCENE_LAWS[self.law]
        horizon, vms = LAW["horizon"], len(environment.means)
        common = _stationary_ar(self.generators[0], (count, horizon), LAW["factor_ar"])
        groups = _stationary_ar(self.generators[1], (count, LAW["phase_groups"], horizon), LAW["factor_ar"])
        noise = self.generators[2].standard_normal((count, vms, horizon))
        trajectory = environment.means[:, None] + settings["amplitude_multiplier"] * environment.amplitudes[:, None] * np.sin(
            2 * np.pi * np.arange(horizon)[None, :] / horizon + environment.phases[:, None])
        values = trajectory[None, :, :] + settings["common_multiplier"] * common[:, None, :] * environment.common_loadings[None, :, None]
        values += settings["group_multiplier"] * groups[:, environment.groups, :] * environment.group_loadings[None, :, None]
        values += noise * environment.idiosyncratic_scales[None, :, None]
        if settings["burst_probability"]:
            event = self.generators[3].random((count, 1, 1)) < settings["burst_probability"]
            amplitude = self.generators[4].uniform(*settings["burst_amplitude"], (count, 1, 1))
            values += event * amplitude
        self.rows += count
        return np.clip(values, 0., 1.)


def sample_stream(environment, law, count, role, batch_size=32):
    """Materialize a finite role panel using the same consecutive scene stream."""
    if int(count) != count or count < 1 or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("positive scene count and batch size are required")
    stream = SceneStream(environment, law, role)
    return np.concatenate([stream.next(min(batch_size, count - start))
                           for start in range(0, count, batch_size)])
