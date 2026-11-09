import numpy as np
from cara.baseline_solver import scenario_mip
from cara.data import host_loads
from cara.repair import feasible


def test_scenario_solver_repairs_complementary_peaks_within_budget():
    incumbent = (0, 0, 1, 1)
    forecast = np.full((4, 1), .4)
    build = np.array([[[.7], [.7], [.1], [.1]], [[.1], [.1], [.7], [.7]]])
    result = scenario_mip(incumbent, forecast, build, build, 1., {'cpu_limit': 2., 'max_migrations': 2})
    selected = result['selected']
    assert selected is not None
    assert feasible(selected, forecast, 1., 2)
    assert np.maximum(host_loads(build, selected) - 1., 0.).sum() == 0
    assert result['stats']['solver_status'] == 0


def test_scenario_solver_zero_migration_preserves_incumbent():
    incumbent = (0, 0, 1, 1)
    forecast = np.full((4, 1), .4)
    build = np.array([[[.7], [.7], [.1], [.1]]])
    result = scenario_mip(incumbent, forecast, build, build, 1., {'cpu_limit': 2., 'max_migrations': 0})
    assert result['selected'] is None
