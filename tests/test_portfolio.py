"""Independent archive coverage and replay checks for the residual portfolio."""
import numpy as np
import pytest
from cara.refinement import _coverage_archive, _ordered_records, search_refined


def test_stable_archive_order_preserves_score_and_full_assignment_ties():
    rng = np.random.default_rng(421)
    rows = [{'key': tuple(rng.integers(-2, 3, 6)),
             'assignment': tuple(rng.integers(0, 30, 575))} for _ in range(60)]
    rows.extend({**rows[0], 'assignment': tuple(rng.integers(0, 30, 575))}
                for _ in range(20))
    expected = sorted(rows, key=lambda r: (r['key'], tuple(-v for v in r['assignment'])), reverse=True)
    assert _ordered_records(rows) == expected


def record(index, gains, moves):
    return {'assignment': (index,), 'relative_gains': gains,
            'key': tuple(sorted(gains)) + (-moves,), 'movement': {'count': moves}}


def test_archive_keeps_balanced_and_complementary_repairs_under_hard_cap():
    balanced = record(0, [.4, .4], 4)
    specialist = record(1, [.1, .9], 5)
    opposite = record(2, [.9, .1], 5)
    middle = [record(i, [.2, .3], 6) for i in range(3, 8)]
    archive = _coverage_archive([balanced, specialist, opposite, *middle], 4)
    assert len(archive) == 4
    assert {r['assignment'] for r in (balanced, specialist, opposite)} <= {r['assignment'] for r in archive}
    assert _coverage_archive([balanced, specialist], 1) == [balanced]


def test_archive_extremes_keep_original_ties_for_long_assignments():
    rng = np.random.default_rng(499)
    rows = [{"assignment": tuple(rng.integers(0, 30, 575)),
             "relative_gains": rng.integers(-2, 3, 6).tolist(),
             "key": tuple(rng.integers(-2, 3, 8)),
             "movement": {"count": int(rng.integers(0, 5))}}
            for _ in range(50)]
    ordered = _ordered_records(rows)
    extremes = [max(ordered, key=lambda r: (r["relative_gains"][e], r["key"],
                 tuple(-v for v in r["assignment"]))) for e in range(6)]
    extremes.append(min(ordered, key=lambda r: (r["movement"]["count"],
                    tuple(-v for v in r["key"]), r["assignment"])))
    for limit in range(1, 20):
        expected = ordered[:max(1, limit // 2)]
        for row in extremes + ordered:
            if len(expected) == limit:
                break
            if row not in expected:
                expected.append(row)
        assert _coverage_archive(rows, limit) == expected


@pytest.mark.parametrize('archive', ['balanced', 'coverage'])
def test_multiple_seed_families_preserve_global_constraints_and_exact_replay(archive):
    rng = np.random.default_rng(29)
    panel = rng.uniform(.1, .5, (16, 12, 8))
    forecast = np.full((12, 8), .25)
    incumbent = tuple(np.repeat(np.arange(4), 3))
    config = {'cpu_limit': .1, 'initial_broad': 0, 'mutable_limit': 4,
              'top_p': 8, 'migration_fraction': .5, 'diagnostic': False,
              'archive_mode': archive, 'seed_schedule': 'coverage',
              'replay_arms': ['exact'] * 8}
    results = [search_refined(incumbent, forecast, panel, panel.copy(), 1.,
               {**config, 'reuse_views': reuse, 'projected_views': projection},
               np.random.default_rng(7)) for reuse, projection in
               [(False, False), (True, False), (True, True)]]
    for result in results:
        families = [(tuple(h['seed_assignment']), tuple(h['pair']))
                    for h in result['stats']['history'] if h['arm'] == 'exact']
        assert len(set(families)) == len(families)
        assert len({seed for seed, _ in families}) > 1
        assert all(r['movement']['count'] <= 6 for r in result['candidates'])
    assert all(r['candidates'] == results[0]['candidates'] for r in results[1:])
    assert all(r['selected'] == results[0]['selected'] for r in results[1:])


def test_no_residual_control_has_same_joint_pool_and_only_frontier_work():
    rng = np.random.default_rng(13)
    panel = rng.uniform(.1, .5, (8, 8, 4))
    incumbent = (0, 0, 1, 1, 2, 2, 3, 3)
    result = search_refined(incumbent, np.full((8, 4), .3), panel, panel.copy(), 1.,
        {'cpu_limit': .1, 'frontier_seed': True, 'frontier_only': True,
         'migration_fraction': .5, 'replay_arms': ['frontier:30']},
        np.random.default_rng(8))
    assert result['stats']['blocks'] == 0
    assert all(h['arm'].startswith('frontier:') for h in result['stats']['history'])
    assert all('relative_gains' in r and len(r['relative_gains']) == 6
               for r in result['candidates'])
