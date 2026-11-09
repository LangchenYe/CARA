from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from cara.admission import certify_fresh, fit_bets
from cara.certification import (
    _DyadicMixture, certify_fresh_mixture, exact_pareto_gate,
    raw_migration_gain_bound, stored_raw_gain_bound,
)
from cara.data import burdens, host_loads


def test_mixture_parts_match_independent_fraction_sum_after_growth():
    values = [Fraction(1, 3), Fraction(-1, 7), Fraction(1, 4), Fraction(0),
              Fraction(-1), Fraction(1), Fraction(1, 9), Fraction(1, 5)]
    process = _DyadicMixture(minimum_components=1)
    for end, value in enumerate(values, 1):
        process.update([value])
        expected = Fraction(1, 2)
        for x in values[:end]:
            expected *= 1 + Fraction(255, 256) * x
        for j in range(1, len(process.small) + 1):
            wealth = Fraction(1, 2 ** (j + 1))
            for x in values[:end]:
                wealth *= 1 + Fraction(1, 2**j) * x
            expected += wealth
        assert Fraction(*process.parts()) == expected


def test_physical_zero_build_bet_failure_is_repaired():
    incumbent, candidate = (0, 0, 1, 1), (0, 1, 0, 1)
    build = np.full((8, 4, 1), .4)
    bets = fit_bets(burdens(build, incumbent, 1), burdens(build, candidate, 1))
    assert bets == (0, 0, 0, 0)
    scenes = np.tile(np.array([.7, .7, .1, .1])[None, :, None], (512, 1, 1))
    reference, action = burdens(scenes, incumbent, 1), burdens(scenes, candidate, 1)
    assert not certify_fresh(reference, action, bets)["admitted"]
    result = certify_fresh_mixture(reference, action, gain_bounds=(1,) * 4,
                                  noninferiority_margins=(0,) * 4, min_anchor_gain=0)
    assert result["admitted"] and result["rows"] < len(scenes)


def test_predeclared_margins_allow_occurrence_tie_with_strict_anchor():
    reference = np.tile([1, .5, .5, .02], (2048, 1))
    candidate = np.tile([1, .5, .3, .01], (2048, 1))
    result = certify_fresh_mixture(reference, candidate, gain_bounds=(1, 1, 1, .1),
                                  noninferiority_margins=(.01, .01, 0, .001),
                                  min_anchor_gain=.005, metric_names=("J", "V", "R", "raw"))
    assert result["admitted"] and result["rows"] < len(reference)
    assert result["anchor"] == "R"
    assert result["exact_thresholds"] == ["-1/100", "-1/100", "1/200", "-1/1000"]


def test_small_mixture_component_recovers_when_aggressive_bet_loses():
    # A 60/40 +/-1 alternative has positive mean, but lambda=255/256 has
    # negative expected log growth. A positive dyadic component must remain.
    signs = np.tile([1., 1., 1., 0., 0.], 256)
    reference, candidate = signs[:, None], (1 - signs)[:, None]
    result = certify_fresh_mixture(reference, candidate, gain_bounds=(1,),
                                  noninferiority_margins=(0,), anchor=0,
                                  min_anchor_gain=0, batch=5)
    assert result["admitted"] and result["rows"] < len(signs)
    assert .6 * np.log(1 + 255 / 256) + .4 * np.log(1 - 255 / 256) < 0


def test_noninferiority_violation_cannot_be_offset_by_anchor_gain():
    reference = np.tile([0., 1.], (1024, 1))
    candidate = np.tile([.02, 0.], (1024, 1))
    result = certify_fresh_mixture(reference, candidate, gain_bounds=(1., 1.),
                                  noninferiority_margins=(.01, 0.), anchor=1,
                                  min_anchor_gain=.005)
    assert not result["admitted"] and result["locked"] == [False, True]


def test_exact_gate_rejects_raw_regression_despite_capped_severity_gain():
    # The old clipped severity mean decreases 1 -> .6, while raw mean rises
    # .02 -> .1. The revised gate compares the uncapped losses exactly.
    raw_reference, raw_candidate = np.array([.02, .02]), np.array([.002, .198])
    assert np.minimum(raw_candidate / .01, 1).mean() < np.minimum(raw_reference / .01, 1).mean()
    reference = np.column_stack([np.ones(2)] * 3 + [raw_reference, np.ones(2), np.ones(2)])
    candidate = reference.copy()
    candidate[:, 2], candidate[:, 3] = .5, raw_candidate
    result = exact_pareto_gate(reference, candidate, metric_names=("J", "V", "VR", "raw", "VS", "PAR"))
    assert not result["admitted"]
    assert result["regressed_metrics"] == ["raw"]


def test_exact_gate_retains_ties_and_original_sub_ulp_aggregate_gains():
    reference = np.array([[1., .5], [1., .5]])
    candidate = reference.copy()
    candidate[0, 1] = np.nextafter(.5, 0)
    result = exact_pareto_gate(reference, candidate, metric_names=("J", "raw"))
    assert result["admitted"] and Fraction(result["exact_gains"][1]) > 0
    assert not exact_pareto_gate(reference, reference)["admitted"]


def test_mixture_null_level_over_all_fair_sign_sequences():
    false = 0
    for signs in product((0., 1.), repeat=8):
        reference = np.asarray(signs)[:, None]
        result = certify_fresh_mixture(reference, 1 - reference, gain_bounds=(1,),
                                      noninferiority_margins=(0,), anchor=0,
                                      min_anchor_gain=0, alpha=.2, batch=1,
                                      minimum_components=2)
        false += result["admitted"]
    assert Fraction(false, 2**8) <= Fraction(1, 5)


def test_gain_bound_violation_is_rejected_not_clipped():
    with pytest.raises(ValueError, match="predeclared bound"):
        certify_fresh_mixture([[.4]], [[0.]], gain_bounds=(.1,),
                             noninferiority_margins=(0,), anchor=0)


def test_lazy_fresh_requests_only_reported_rows():
    reference, candidate = np.ones((256, 1)), np.zeros((256, 1))
    requested = []

    def reader(indices):
        requested.extend(indices.tolist())
        return candidate[indices]

    result = certify_fresh_mixture(reference, reader, gain_bounds=(1,),
                                  noninferiority_margins=(0,), anchor=0, min_anchor_gain=0,
                                  batch=8)
    assert result["admitted"] and requested == list(range(result["rows"]))


def test_raw_migration_bound_and_stored_guard_on_all_small_assignments():
    assert raw_migration_gain_bound(2, 2, 1, 1) == 1
    rng = np.random.default_rng(911)
    scenes = rng.uniform(0, 1, (11, 4, 3))
    incumbent = (0, 0, 1, 1)

    def raw(assignment):
        loads = host_loads(scenes, assignment)
        return np.maximum(loads - 1, 0).sum(axis=(0, 2)) / 6

    reference = raw(incumbent)
    for assignment in product((0, 1), repeat=4):
        if len(set(assignment)) != 2:
            continue
        moved = min(sum(a != b for a, b in zip(incumbent, assignment)),
                    sum(a == b for a, b in zip(incumbent, assignment)))
        bound = stored_raw_gain_bound(moved, 2, 1, vm_count=4, horizon=3)
        assert bound >= float(raw_migration_gain_bound(moved, 2, 1))
        assert np.all(np.abs(reference - raw(assignment)) <= bound)
