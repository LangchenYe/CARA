"""Check complete data, disjoint roles, native formulas, and retained observations."""
import json
import gzip
import os
from pathlib import Path
import numpy as np
import pytest
from cara.data import CAPACITIES, COHORT_SIZES, burdens, burdens_from_loads, canonical_assignment, host_loads, load_dataset, native_metrics, trace_intervals, trace_panels

DATA_ROOT = Path(os.environ.get("CARA_DATA_ROOT", Path(__file__).resolve().parents[1] / "data" / "raw" / "cloudcons"))
REFERENCE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retained_terminal.json.gz"


def test_canonical_placements_are_immutable_idempotent_tuple_values():
    import pickle
    labels = np.array([17, 17, -3, 9, -3])
    placement = canonical_assignment(labels)
    assert placement == (0, 0, 1, 2, 1)
    assert hash(placement) == hash((0, 0, 1, 2, 1))
    assert canonical_assignment(placement) is placement
    assert pickle.loads(pickle.dumps(placement)) == placement
    assert json.loads(json.dumps(placement)) == list(placement)
    labels[:] = 0
    assert placement == (0, 0, 1, 2, 1)


@pytest.mark.parametrize("name", CAPACITIES)
def test_complete_cohort_and_arrow_decoder(name):
    dataset = load_dataset(name, DATA_ROOT)
    assert dataset.n_vms == COHORT_SIZES[name]
    assert len(set(dataset.item_ids)) == dataset.n_vms
    for origin in [216, 228, 240, *range(288, 313)]:
        assert dataset.panel(origin).shape == (dataset.n_vms, 24)
    pa = pytest.importorskip("pyarrow")
    with pa.memory_map(str(DATA_ROOT / name / "data-00000-of-00001.arrow"), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    targets = table["target"].to_pylist()
    if isinstance(targets[0][0], list):
        targets = [row[0] for row in targets]
    assert all(np.array_equal(actual, expected) for actual, expected in zip(dataset.series, targets))
    assert tuple(table["item_id"].to_pylist()) == dataset.item_ids


def test_native_metrics_condition_on_violating_times_and_hosts():
    target = np.array([[2., 4., 0., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
    metrics = native_metrics(target, [0, 1, 2], 1.)
    assert metrics == {"hosts": 2, "Util": 1.25, "VR": .25, "VS": 2., "PAR": 1.6, "raw": .5}
    assert native_metrics(np.zeros((2, 3)), [0, 1], 1.)["hosts"] == 0


def test_burdens_and_canonical_host_reductions():
    panel = np.array([[[.8, .2], [.4, .5], [.2, .3]], [[.1, .8], [.3, .5], [.4, .1]]])
    assignment = [4, 4, 9]
    loads = host_loads(panel, assignment)
    assert loads.shape == (2, 2, 2)
    assert np.array_equal(loads, host_loads(panel, [17, 17, -3]))
    actual = burdens(panel, assignment, 1., kappa=.1)
    assert np.array_equal(actual, burdens_from_loads(loads, 1., kappa=.1))
    np.testing.assert_allclose(actual, [[1., .5, .25, .5], [1., .5, .25, .75]], atol=1e-15)


def test_chronological_roles_are_disjoint_and_complete():
    dataset = load_dataset("azure2019", DATA_ROOT)
    for origin in [216, 228, 240, 288, 312]:
        intervals = trace_intervals(origin)
        for role, (left, right) in intervals.items():
            for other, (left2, right2) in intervals.items():
                if role != other:
                    assert max(left, left2) >= min(right, right2)
        panels = trace_panels(dataset, origin)
        assert {role: panel.shape[0] for role, panel in panels.items()} == {"build": 24, "selection": 25, "audit": 57}
        assert np.array_equal(panels["audit"][0], dataset.panel(origin - 192, 8))
        assert np.array_equal(panels["audit"][-1], dataset.panel(origin - 136, 8))
        assert np.array_equal(panels["build"][0], dataset.panel(origin - 96, 8))
        assert np.array_equal(panels["build"][-1], dataset.panel(origin - 8, 8))
    with pytest.raises(ValueError):
        trace_panels(dataset, 191)
    with pytest.raises(ValueError):
        trace_panels(dataset, 336)


@pytest.mark.parametrize("name", CAPACITIES)
def test_all_800_retained_terminal_metric_sets(name):
    """Recompute data-dependent outcomes without claiming to rerun the old solver."""
    dataset = load_dataset(name, DATA_ROOT)
    with gzip.open(REFERENCE_FIXTURE, "rt") as handle:
        rows = json.load(handle)[name]
    assert [row["origin"] for row in rows] == list(range(288, 313))
    checked = 0
    for row in rows:
        target = dataset.panel(row["origin"], row["horizon"])
        assert len(row["policies"]) == 8
        for policy in row["policies"].values():
            assignment = policy["terminal_assignment"]
            assert len(assignment) == dataset.n_vms
            actual = native_metrics(target, assignment, dataset.capacity)
            for metric, value in actual.items():
                assert value == pytest.approx(policy["metrics"][metric], rel=0., abs=1e-12)
            checked += 1
    assert checked == 200
