"""Complete-case and effective-policy checks for supplied and fresh comparisons."""
from copy import deepcopy
import json

import pytest

import analyze_family_study as analysis
from run_family_study import MAIN_METHODS, NATIVE_METRICS, case_grid, method_settings


def write_study(path, plan, records):
    path.mkdir(exist_ok=True)
    (path / 'study_plan.json').write_text(json.dumps(plan))
    (path / 'records.json').write_text(json.dumps(records))


@pytest.fixture
def comparison(tmp_path):
    protocol = {'primary': {'streams': [0], 'first_origin': 192, 'last_origin': 193,
                            'stride': 1, 'cases': 2},
                'decision_cohorts': {'toy': {'minimum_hours': 217, 'longitudinal_origins': [192, 193]}},
                'evaluation_union': {'unique_cases': 2}, 'longitudinal': {'cases': 2},
                'target_hours': 24, 'expected_rows': {'build': 1, 'selection': 1, 'audit': 1}}
    config = {'cpu_limit': 2., 'top_p': 2, 'family_matching_backend': 'compiled'}
    inputs = []
    for index, methods in enumerate(([m for m in MAIN_METHODS if m != 'Scenario-MIP'], ['Scenario-MIP'])):
        plan = {'study': 'complete_window_family_execution', 'phase': 'evaluation', 'protocol': protocol,
                'cases': case_grid(protocol, 'evaluation'), 'config': config, 'methods': methods,
                'method_settings': {method: method_settings(method, config) for method in methods},
                'proposal_allowance_seconds': 2., 'workers': 6 if index == 0 else 1,
                'inputs': ['toy'], 'rotation': 'cyclic method order', 'timing': 'process CPU'}
        records = []
        for position, case in enumerate(plan['cases']):
            policies = {method: {'metrics': dict.fromkeys(NATIVE_METRICS, 0.),
                'movement': {'count': 0, 'fraction': 0., 'cost': 0.},
                'timing': dict.fromkeys(analysis.TIMINGS, 0.),
                'search': dict.fromkeys(analysis.COUNTERS, 0), 'admission': {'admitted': False}}
                for method in ('FFD', *methods)}
            shift = position % len(methods)
            records.append({**case, 'config': config, 'n_vms': 2, 'capacity': 1.,
                'incumbent_assignment': [0, 1], 'panel_rows': protocol['expected_rows'],
                'policies': policies, 'method_order': methods[shift:] + methods[:shift],
                'trace_cache_hit': False, 'trace_loading_cpu_seconds': 0.,
                'trace_loading_wall_seconds': 0., 'common_forecast_incumbent_cpu_seconds': 0.,
                'evaluation_cpu_seconds': 0.})
        path = tmp_path / str(index)
        write_study(path, plan, records)
        inputs.append(path)
    plan, records = analysis.join_studies(inputs)
    path = tmp_path / 'complete'
    write_study(path, plan, records)
    return path, plan, records, inputs


def test_complete_comparison_roundtrip_preserves_every_policy_and_execution_group(comparison):
    path, plan, records, _ = comparison
    actual_plan, actual_records = analysis.join_studies([path])
    assert actual_plan == plan and actual_records == records
    assert len(records) == 2 and all(set(row['policies']) == {'FFD', *MAIN_METHODS} for row in records)
    group = plan['execution_groups'][str(plan['method_groups']['Scenario-MIP'])]
    assert group['methods'] == ['Scenario-MIP'] and group['workers'] == 1


def test_duplicate_methods_cannot_be_implicitly_replaced(comparison):
    _, _, _, inputs = comparison
    with pytest.raises(ValueError, match='one complete study'):
        analysis.join_studies([*inputs, inputs[0]])


@pytest.mark.parametrize('violation', ['duplicate_case', 'missing_policy', 'different_config',
                                    'incomplete_window', 'nonfinite_metric', 'different_effective_setting'])
def test_incomplete_or_inconsistent_comparison_is_rejected(comparison, violation):
    path, original_plan, original_records, _ = comparison
    plan, records = deepcopy(original_plan), deepcopy(original_records)
    if violation == 'duplicate_case':
        records[1] = records[0]
    elif violation == 'missing_policy':
        del records[0]['policies']['CARA']
    elif violation == 'different_config':
        records[0]['config']['top_p'] += 1
    elif violation == 'incomplete_window':
        records[0]['panel_rows']['build'] = 0
    elif violation == 'nonfinite_metric':
        records[0]['policies']['CARA']['metrics']['VR'] = float('nan')
    elif violation == 'different_effective_setting':
        plan['method_settings']['CARA'] = {**plan['method_settings']['CARA'], 'top_p': 3}
    write_study(path, plan, records)
    with pytest.raises(ValueError):
        analysis.load_complete(path)
