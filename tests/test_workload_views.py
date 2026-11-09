import numpy as np
from cara.data import canonical_assignment, host_loads
from cara.refinement import panel_profile
from cara.workload_views import ProjectionReplay, host_projection, profile_from_projection


def test_projection_matches_full_replay_bitwise_across_boundaries():
    rng = np.random.default_rng(821)
    for rows, vms, times, hosts in ((1, 12, 1, 3), (25, 68, 8, 11), (7, 150, 24, 17)):
        panel = rng.uniform(0, .2, (rows, vms, times))
        panel[0, :, 0] = rng.choice([0., np.nextafter(.1, 0.), .1, np.nextafter(.1, 1.)], vms)
        incumbent = canonical_assignment(np.arange(vms) % hosts)
        cache = ProjectionReplay(panel, incumbent, .5)
        for _ in range(40):
            candidate = np.array(incumbent)
            left, right = rng.choice(vms, 2, replace=False)
            candidate[left], candidate[right] = candidate[right], candidate[left]
            candidate = canonical_assignment(candidate)
            actual, expected = cache.profile(candidate), panel_profile(panel, candidate, .5)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()


def test_projection_keeps_inactive_hosts_and_native_severity_semantics():
    loads = np.array([[[0., 0., 0.], [.5, 1., 1.5]], [[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [0., 0., 0.]]])
    from cara.refinement import profile_from_loads
    assert profile_from_projection(host_projection(loads, 1.)).tobytes() == profile_from_loads(loads, 1.).tobytes()
