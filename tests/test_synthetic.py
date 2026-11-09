import numpy as np
from cara.synthetic import LAW, make_environment


def test_scene_roles_reproduce_without_reusing_samples():
    environment = make_environment(73001)
    first = environment.sample(32, 1)
    assert first.shape == (32, 48, 8)
    assert np.array_equal(first, environment.sample(32, 1))
    assert not np.array_equal(first, environment.sample(32, 2))
    assert np.all(np.isfinite(first)) and np.all(first >= 0)
    assert np.array_equal(np.sort(environment.groups), np.repeat(np.arange(4), 12))
    assert np.array_equal(np.sort(environment.phases), np.repeat(np.arange(4), 12) * np.pi / 2.)
    assert np.all((environment.means >= LAW['mean_range'][0]) & (environment.means <= LAW['mean_range'][1]))
