from itertools import product
import numpy as np

from cara.data import burdens,canonical_assignment,host_loads
from cara.repair import ffd,feasible,search,ReplayCache,LazyReplay
from cara.admission import fit_bets,evidence_key


def test_cached_and_lazy_replay_preserve_full_shapes():
    rng=np.random.default_rng(71)
    for horizon in (1,3,8):
        panel=rng.lognormal(-2,2,(19,13,horizon))
        incumbent=tuple(i//3 for i in range(13))
        candidate=canonical_assignment(tuple((i+1)//3 for i in range(13)))
        full=burdens(panel,candidate,3)
        cache=ReplayCache(panel,incumbent,3)
        assert np.array_equal(cache(candidate),full)
        lazy=LazyReplay(panel,incumbent,candidate,3,reference_cache=cache)
        for indices in ([7],[4,0,11],[18,3,2,13,9]):
            assert np.array_equal(lazy(indices),full[indices])
        assert lazy.requested=={7,4,0,11,18,3,2,13,9}


def test_complete_residual_prefix_matches_independent_partition_oracle():
    rng=np.random.default_rng(92)
    forecast=np.full((6,4),.26)
    incumbent=(0,0,0,1,1,1)
    panel=rng.uniform(.05,.6,(12,6,4))
    config={"mode":"exact","local_rounds":1,"pair_limit":1,"mutable_limit":6,"top_p":16}
    found=search(incumbent,forecast,panel,panel,1.,config,np.random.default_rng(5))
    reference=burdens(panel,incumbent,1.)
    oracle={}
    for suffix in product((0,1),repeat=5):
        assignment=canonical_assignment((0,)+suffix)
        if assignment==incumbent or not feasible(assignment,forecast,1.,2):
            continue
        values=burdens(panel,assignment,1.)
        bets=fit_bets(reference,values)
        key=evidence_key(reference,values,bets=bets,center=False)
        oracle[assignment]=(key,bets)
    expected=sorted(oracle,key=lambda a:(oracle[a][0],tuple(-v for v in a)),reverse=True)[:16]
    assert [tuple(c['assignment']) for c in found['candidates']]==expected
    assert [tuple(c['bets']) for c in found['candidates']]==[oracle[a][1] for a in expected]


def test_dense_backgrounds_keep_all_mutable_orientations():
    forecast = np.full((8, 2), .15)
    incumbent = (0, 0, 0, 0, 1, 1, 1, 1)
    panel = np.full((6, 8, 2), .05)
    phase = np.linspace(0, 1, 6)
    for vm, scale in ((0, .2), (1, .1), (4, .18), (5, .12)):
        panel[:, vm, :] += scale * phase[:, None]
    config = {"mode": "exact", "local_rounds": 1, "pair_limit": 1,
              "mutable_limit": 4, "top_p": 64}
    found = search(incumbent, forecast, panel, panel, 1., config, np.random.default_rng(5))
    # The varying workloads select mutable VMs 0,1,4,5. Fixed residents 2,3 and
    # 6,7 distinguish both sides, including assignments with an empty mutable
    # side. A whole-pair symmetry restriction would discard valid partitions.
    expected = set()
    for sides in product((0, 1), repeat=4):
        labels = list(incumbent)
        for vm, side in zip((0, 1, 4, 5), sides):
            labels[vm] = side
        action = canonical_assignment(labels)
        if action != incumbent and feasible(action, forecast, 1., 2):
            expected.add(action)
    assert len(expected) == 15
    assert {tuple(c["assignment"]) for c in found["candidates"]} == expected


def test_lazy_replay_preserves_unsaturated_severity_and_threshold_bits():
    rng = np.random.default_rng(104)
    incumbent = tuple(i // 3 for i in range(14))
    candidate = canonical_assignment(tuple((i + 1) // 3 for i in range(14)))
    panel = rng.uniform(.330, .338, (23, 14, 7))
    full = burdens(panel, candidate, 1.)
    assert np.any((full[:, 3] > 0) & (full[:, 3] < 1))
    cache = ReplayCache(panel, incumbent, 1.)
    assert np.array_equal(cache(candidate), full)
    lazy = LazyReplay(panel, incumbent, candidate, 1., reference_cache=cache)
    for rows in ([9], [2, 8, 0], [22, 1, 14, 5, 7]):
        assert np.array_equal(lazy(rows), full[rows])


def test_reuse_preserves_entire_feedback_search():
    rng=np.random.default_rng(8)
    forecast=np.full((12,3),.2)
    incumbent=ffd(forecast,1.)
    panel=rng.uniform(.1,.4,(12,12,3))
    config={"broad_trials":24,"mutable_limit":6,"pair_limit":3,"local_rounds":2}
    cached=search(incumbent,forecast,panel,panel,1.,config,np.random.default_rng(31))
    full=search(incumbent,forecast,panel,panel,1.,dict(config,reuse=False),np.random.default_rng(31))
    assert cached['candidates']==full['candidates']
    assert cached['selected']==full['selected']
    assert cached['stats']['evaluations']==full['stats']['evaluations']
    assert cached['stats']['cache_hits']>0


def test_fixed_forecast_host_budget():
    f=np.array([[.6,.7],[.3,.2],[.4,.3],[.5,.4]])
    for best in (False,True):
        assignment=ffd(f,1,best_fit=best)
        assert len(set(assignment))==2
        assert np.all(host_loads(f,assignment)<=1+1e-12)
