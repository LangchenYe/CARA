from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from benchmark_hourly_queries import equal_outputs, family_queries, load_hourly, make_input, run_plan
from cara.data import host_memberships


def test_hourly_cpu_channel_alignment_and_deterministic_nan_interpolation(tmp_path):
    root = tmp_path / "huawei2025"
    root.mkdir()
    first = np.arange(30, dtype=float)
    first[1] = np.nan
    rows = [{"item_id": "item_2", "start": datetime(2025, 1, 1), "freq": "1H",
             "target": [first.tolist(), [999.] * 30]},
            {"item_id": "item_1", "start": datetime(2025, 1, 1), "freq": "1H",
             "target": [[50.] * 28, [888.] * 28]}]
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(root / "data-00000-of-00001.arrow"), "wb") as stream:
        with ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    matrix, stats = load_hourly(tmp_path, "huawei2025")
    assert matrix.shape == (2, 28) and np.all(matrix[0] == 50.)
    assert matrix[1, 1] == 1.
    assert stats["interpolated_values_in_common_interval"] == 1
    assert stats["cpu_channel"] == 0 and stats["raw_units_preserved"]


def test_hourly_queries_preserve_backgrounds_and_raw_capacity_units():
    matrix = np.arange(16 * 30, dtype=float).reshape(16, 30) + 1.
    panel, _, incumbent, capacity, starts = make_input(matrix, "full", 8)
    assert panel.shape == (23, 16, 8) and capacity > 1.
    assert np.array_equal(panel[0], matrix[:, :8])
    assert np.array_equal(panel[-1], matrix[:, -8:])
    assert np.array_equal(starts, np.arange(23))
    queries = family_queries(incumbent, (0, 1))
    assert len(queries) == 255
    original = host_memberships(incumbent)
    for query in queries:
        groups = host_memberships(query)
        assert len(groups) == 2 and all(4 <= len(group) <= 12 for group in groups)
        assert all(any(set(background[4:]) <= set(group) for group in groups) for background in original)


def test_all_hourly_query_plans_return_identical_canonical_bytes():
    matrix = np.random.default_rng(838).uniform(0, 40, (32, 30))
    reference = run_plan(matrix, "full", "full", families=2, query_batch=8)
    for plan in ("load", "project", "shared_scalar", "shared_batch", "shared_compiled"):
        result = run_plan(matrix, "full", plan, families=2, query_batch=8)
        equal_outputs(reference, result)
        assert result[2]["submitted_queries"] == reference[2]["submitted_queries"]


def test_projected_gather_budget_chunks_without_changing_logical_queries():
    matrix = np.random.default_rng(839).uniform(0, 40, (32, 30))
    reference = run_plan(matrix, "full", "full", families=1)
    bounded = run_plan(matrix, "full", "shared_batch", families=1,
                       profile_cells=100, trace_allocations=True)
    equal_outputs(reference, bounded)
    assert bounded[2]["effective_query_batch"] == 1
    assert bounded[2]["peak_traced_incremental_bytes"] > 0
    assert bounded[2]["peak_retained_views"] >= bounded[2]["retained_views"]
