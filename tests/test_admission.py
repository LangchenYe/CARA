from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np

from cara.admission import (
    DENOMINATORS, GRID, M, _wealth, binary_mixture, certify_finite,
    certify_fresh, encoded_gains, evidence_key, fit_bets, log_cell, pareto_census,
)


def test_conservative_encoding_uses_stored_values():
    ref = np.array([[1, 1, 0.5, 0], [0, 0, 0.25, 1]], dtype=float)
    cand = np.array([[0, 0, 0.5, 1], [1, 1, 0.75, 0]], dtype=float)
    gains, denominators = encoded_gains(ref, cand)
    assert denominators == DENOMINATORS
    assert gains[0] == (1, M - 1, -1, -M - 1)
    for i, row in enumerate(gains):
        for e in range(1, 4):
            original = Fraction(float(ref[i, e])) - Fraction(float(cand[i, e]))
            assert -1 <= Fraction(row[e], M + 1) <= Fraction(M, M + 1) * original


def test_binary_mixture_against_independent_polynomial_integral():
    for b in range(8):
        for h in range(8):
            integral = sum((Fraction(comb(b, j) * comb(h, k) * (-1)**k, j + k + 1)
                            for j in range(b + 1) for k in range(h + 1)), Fraction(0))
            assert binary_mixture(b, h) == integral


def test_bet_fitting_matches_exhaustive_grid_oracle():
    rng = np.random.default_rng(209)
    for _ in range(8):
        ref, cand = rng.random((2, 9, 4))
        ref[:, 0], cand[:, 0] = rng.integers(0, 2, (2, 9))
        rows, denominators = encoded_gains(ref, cand)
        selected = fit_bets(ref, cand)
        for e, denominator in enumerate(denominators):
            all_wealth = [_wealth([row[e] for row in rows], denominator, j) for j in range(GRID)]
            assert selected[e] == max(range(GRID), key=lambda j: all_wealth[j])
    ties = np.zeros((5, 4))
    assert fit_bets(ties, ties) == (0, 0, 0, 0)


def test_logarithm_zero_and_threshold_centering():
    assert log_cell(Fraction(1)) == 0
    assert log_cell(Fraction(3, 2) * Fraction(2, 3)) == 0
    assert log_cell(Fraction(2)) > 0
    assert log_cell(Fraction(1, 2)) < 0
    ref = np.array([[1, 0, 0, 0]], dtype=float)
    cand = np.zeros((1, 4))
    key = evidence_key(ref, cand, (0, 0, 0, 0), alpha=Fraction(2, 3))
    assert key[-1] == 0  # The projected binary margin equals zero exactly.


def test_finite_lazy_reader_requests_only_reported_rows():
    ref = np.ones((48, 4))
    cand = np.zeros((48, 4))
    fetched = []

    def reader(indices):
        fetched.extend(indices.tolist())
        return cand[indices]

    out = certify_finite(ref, reader, (255,) * 4, seed=17, batch=4)
    assert out["admitted"] and out["decision"] == "early"
    assert out["required"] == ["R"] and set(out["dominated"]) == {"J", "V", "R", "S"}
    assert out["rows_read"] < len(ref)
    assert fetched == out["order"] and len(set(fetched)) == len(fetched)


def test_census_overrides_an_earlier_false_coordinate_crossing():
    ref = np.array([[1, .4, .1, .1], [1, .4, .1, .1], [1, 0, .9, .9], [1, 0, .9, .9]])
    cand = np.array([[1, 0, .1, .1], [1, 0, .1, .1], [1, 1, 0, 0], [1, 1, 0, 0]])
    seed = next(s for s in range(100) if set(np.random.default_rng(s).permutation(4)[:2]) == {0, 1})
    out = certify_finite(ref, cand, (255,) * 4, seed=seed, batch=2, alpha=.8)
    # V crosses in the first batch while R and S remain tied; the full V total
    # is negative, so the final census must reject despite that earlier lock.
    assert not out["admitted"] and out["decision"] == "census"
    assert out["gains_at_census"][1] < 0


def test_exact_census_accepts_pareto_tie_and_rejects_no_change():
    ref = np.full((7, 4), .5)
    ref[:, 0] = 1
    cand = ref.copy()
    cand[:, 2] = .25
    out = certify_finite(ref, cand, (0,) * 4, batch=7)
    assert out["admitted"] and out["decision"] == "census"
    assert out["exact_gains_at_census"][2] == "7/4"
    assert not certify_finite(ref, ref, (255,) * 4, batch=7)["admitted"]
    direct = pareto_census(ref, cand)
    assert direct["admitted"] == out["admitted"]
    assert direct["exact_gains_at_census"] == out["exact_gains_at_census"]
    # A gain far below a usual floating comparison tolerance remains strict.
    tiny_ref = np.array([[1, .5, .5, .5]])
    tiny_cand = tiny_ref.copy()
    tiny_cand[0, 2] = np.nextafter(.5, 0.)
    assert pareto_census(tiny_ref, tiny_cand)["admitted"]


def test_finite_level_on_all_balanced_six_row_reveal_sequences():
    false = total = 0
    for positive in combinations(range(6), 3):
        ref = np.ones((6, 4))
        cand = np.ones((6, 4))
        ref[:, 2] = 0
        cand[:, 2] = 1
        ref[list(positive), 2] = 1
        cand[list(positive), 2] = 0
        # The other coordinates tie. All original totals are zero, so every
        # admission is false. Enumerating the balanced row labelings under one
        # fixed independent permutation enumerates their random-order law.
        false += certify_finite(ref, cand, (255,) * 4, seed=3, batch=1, alpha=.25)["admitted"]
        total += 1
    assert Fraction(false, total) <= Fraction(1, 4)


def test_fresh_level_on_all_eight_iid_fair_sign_sequences():
    false = total = 0
    for signs in product((0, 1), repeat=8):
        ref = np.repeat(np.asarray(signs, dtype=float)[:, None], 4, axis=1)
        cand = 1 - ref
        false += certify_fresh(ref, cand, (128,) * 4, batch=1, alpha=.2)["admitted"]
        total += 1
    assert Fraction(false, total) <= Fraction(1, 5)


def test_fresh_strict_admission_and_cap():
    ref, cand = np.ones((64, 4)), np.zeros((64, 4))
    out = certify_fresh(ref, cand, (128,) * 4, batch=4)
    assert out["admitted"] and out["rows_read"] < 64
    assert not certify_fresh(cand, cand, (128,) * 4, batch=4)["admitted"]
