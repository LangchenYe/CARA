"""The lowered census preserves complete dictionaries, not just signs."""
import numpy as np
import pytest

from cara.certification import exact_pareto_gate
from cara.dyadic_gate import exact_dyadic_gate


def test_exact_census_under_cancellation_and_widely_separated_exponents():
    tiny = np.nextafter(0., 1.)
    reference = np.array([[1., 2.**400, tiny, -1., 0., 1.],
        [2.**-53, tiny, tiny, 0., -0., 2.**-52],
        [2.**-53, -2.**400, tiny, 1., tiny, -1.]])
    candidate = np.array([[1., 2.**400, 0., 0., -0., 0.],
        [0., 0., tiny, 0., 0., 1.],
        [0., -2.**400, tiny, 0., 0., 0.]])
    names = ("J", "V", "VR", "raw", "VS", "PAR")
    assert exact_dyadic_gate(reference, candidate, metric_names=names) == exact_pareto_gate(
        reference, candidate, metric_names=names)


def test_full_dictionaries_match_on_random_dyadic_profiles():
    rng = np.random.default_rng(19937)
    for rows in (1, 3, 25, 128):
        for _ in range(8):
            reference = np.ldexp(rng.uniform(-1., 1., (rows, 6)), rng.integers(-1073, 401, (rows, 6)))
            candidate = np.ldexp(rng.uniform(-1., 1., (rows, 6)), rng.integers(-1073, 401, (rows, 6)))
            assert exact_dyadic_gate(reference, candidate) == exact_pareto_gate(reference, candidate)
            assert exact_dyadic_gate(reference, reference) == exact_pareto_gate(reference, reference)


@pytest.mark.parametrize("left,right,names", [([], [], None), ([[0.]], [[0., 1.]], None),
    ([[float('inf')]], [[0.]], None), ([[0., 1.]], [[0., 1.]], ("x", "x"))])
def test_same_invalid_inputs_are_rejected(left, right, names):
    for gate in (exact_pareto_gate, exact_dyadic_gate):
        with pytest.raises(ValueError):
            gate(left, right, metric_names=names)
