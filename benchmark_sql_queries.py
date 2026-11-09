"""SQL join/aggregate reference for the same placement what-if query batches.

DuckDB input-table materialization is included. SQL SUM is not claimed to have
the canonical NumPy reduction sequence; the benchmark measures its observed
load, threshold and profile differences separately from exact execution plans.
"""
import argparse
import json
import os
from pathlib import Path
from time import process_time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pyarrow as pa
import duckdb

from benchmark_query_engine import candidates
from cara.data import CAPACITIES, canonical_assignment, host_loads, load_dataset
from cara.query_engine import SharedQueryReplay, _profiles
from cara.repair import _mutable, _pair_order, ffd
from cara.workload_views import host_projection


def profiles_from_batch_loads(loads, capacity):
    q, h, b, t = loads.shape
    columns = host_projection(loads.reshape(q * h, b, t), capacity)
    return _profiles(tuple(column.reshape(q, h, b) for column in columns))


def sql_queries(panel, assignments, capacity, intern_groups=False):
    """Materialize relations, join/aggregate, then apply the common profile."""
    start = process_time()
    b, n, t = panel.shape
    labels = np.asarray([canonical_assignment(assignment) for assignment in assignments], dtype=np.int64)
    q, h = len(labels), len(set(labels[0]))
    demand = pa.table({"i": np.tile(np.repeat(np.arange(n, dtype=np.int64), t), b),
                       "b": np.repeat(np.arange(b, dtype=np.int64), n * t),
                       "t": np.tile(np.arange(t, dtype=np.int64), b * n),
                       "x": panel.reshape(-1)})
    if intern_groups:
        group_ids, resident_ids, mapping, identities = [], [], [], {}
        for query, assignment in enumerate(labels):
            for host in range(h):
                residents = tuple(np.flatnonzero(assignment == host).tolist())
                if residents not in identities:
                    group = len(identities)
                    identities[residents] = group
                    group_ids.extend([group] * len(residents))
                    resident_ids.extend(residents)
                mapping.append((query, host, identities[residents]))
        memberships = pa.table({"g": np.asarray(group_ids, dtype=np.int64), "i": np.asarray(resident_ids, dtype=np.int64)})
        columns = np.asarray(mapping, dtype=np.int64)
        placement = pa.table({"q": columns[:, 0], "h": columns[:, 1], "g": columns[:, 2]})
    else:
        placement = pa.table({"q": np.repeat(np.arange(q, dtype=np.int64), n),
                             "i": np.tile(np.arange(n, dtype=np.int64), q), "h": labels.reshape(-1)})
    connection = duckdb.connect(config={"threads": "1", "memory_limit": "2GB"})
    try:
        connection.register("input_demand", demand)
        connection.register("input_placement", placement)
        connection.execute("CREATE TEMP TABLE demand AS SELECT * FROM input_demand")
        connection.execute("CREATE TEMP TABLE placement AS SELECT * FROM input_placement")
        if intern_groups:
            connection.register("input_membership", memberships)
            connection.execute("CREATE TEMP TABLE membership AS SELECT * FROM input_membership")
        materialization = process_time() - start
        query_start = process_time()
        if intern_groups:
            connection.execute("""
                CREATE TEMP TABLE group_load AS
                SELECT m.g, d.b, d.t, SUM(d.x) AS load
                FROM demand AS d JOIN membership AS m ON d.i = m.i
                GROUP BY m.g, d.b, d.t
            """)
            values = connection.execute("""
                SELECT p.q, p.h, l.b, l.t, l.load
                FROM placement AS p JOIN group_load AS l ON p.g = l.g
                ORDER BY p.q, p.h, l.b, l.t
            """).fetchnumpy()["load"]
        else:
            values = connection.execute("""
                SELECT p.q, p.h, d.b, d.t, SUM(d.x) AS load
                FROM demand AS d JOIN placement AS p ON d.i = p.i
                GROUP BY p.q, p.h, d.b, d.t
                ORDER BY p.q, p.h, d.b, d.t
            """).fetchnumpy()["load"]
        loads = np.asarray(values, dtype=np.float64).reshape(q, h, b, t)
        aggregation = process_time() - query_start
        profile_start = process_time()
        profiles = profiles_from_batch_loads(loads, capacity)
        profile_cpu = process_time() - profile_start
    finally:
        connection.close()
    return {"total_cpu_seconds": process_time() - start, "materialization_cpu_seconds": materialization,
            "aggregation_cpu_seconds": aggregation, "profile_cpu_seconds": profile_cpu,
            "demand_rows": b * n * t, "placement_rows": q * h if intern_groups else q * n,
            "membership_rows": len(resident_ids) if intern_groups else None,
            "distinct_resident_groups": len(identities) if intern_groups else None,
            "result_rows": q * h * b * t}, loads, profiles


def full_queries(panel, assignments, capacity):
    start = process_time()
    loads = np.stack([host_loads(panel, assignment) for assignment in assignments])
    profiles = profiles_from_batch_loads(loads, capacity)
    return {"total_cpu_seconds": process_time() - start}, loads, profiles


def shared_queries(panel, incumbent, assignments, capacity):
    start = process_time()
    session = SharedQueryReplay(panel, incumbent, capacity)
    profiles = session.profiles(assignments)
    cpu = process_time() - start
    costs = {"total_cpu_seconds": cpu, "retained_array_bytes": session.cached_view_bytes,
             "membership_hits": session.hits, "materialized_demand_values": session.materialized_values}
    # Loads are retrieved outside timing only to compare returned arithmetic.
    loads = session.loads_many(assignments)
    return costs, loads, profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", default=list(CAPACITIES), choices=list(CAPACITIES))
    parser.add_argument("--origins", nargs="+", type=int, default=[216, 228, 240])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    plans = ("sql", "sql_shared", "full", "shared")
    for name in args.cohorts:
        data = load_dataset(name)
        for origin in args.origins:
            panel = data.window_panel(origin, 192, 64, max_rows=24)
            forecast = data.forecast(origin)
            incumbent = ffd(forecast, data.capacity)
            pairs = _pair_order(panel, incumbent, data.capacity, np.random.default_rng(519 + origin))[:args.pairs]
            assignments = [assignment for pair in pairs
                for assignment, _ in candidates(forecast, incumbent, pair,
                    _mutable(panel, incumbent, pair, args.width, data.capacity), data.capacity, int(.2 * len(incumbent)))]
            record = {"cohort": name, "origin": origin, "N": data.n_vms, "H": len(set(incumbent)),
                      "B": len(panel), "T": panel.shape[2], "candidates": len(assignments)}
            if assignments:
                trials = {plan: [] for plan in plans}
                diagnostics = {name: [] for name in ("sql", "sql_shared")}
                for repetition in range(args.repetitions):
                    output = {}
                    for plan in plans[repetition % len(plans):] + plans[:repetition % len(plans)]:
                        costs, loads, profiles = (sql_queries(panel, assignments, data.capacity, plan == "sql_shared") if plan.startswith("sql") else
                            full_queries(panel, assignments, data.capacity) if plan == "full" else
                            shared_queries(panel, incumbent, assignments, data.capacity))
                        trials[plan].append(costs)
                        output[plan] = loads, profiles
                    reference, expected_profiles = output["full"]
                    if output["shared"][0].tobytes() != reference.tobytes() or output["shared"][1].tobytes() != expected_profiles.tobytes():
                        raise AssertionError("Shared batch changes canonical arithmetic")
                    for sql_name in diagnostics:
                        sql_loads, sql_profiles = output[sql_name]
                        diagnostics[sql_name].append({"maximum_absolute_load_difference": float(np.max(np.abs(sql_loads - reference))),
                            "strict_threshold_flips": int(np.sum((sql_loads > data.capacity) != (reference > data.capacity))),
                            "tolerant_threshold_flips": int(np.sum((sql_loads > data.capacity + 1e-12) != (reference > data.capacity + 1e-12))),
                            "maximum_absolute_profile_difference": float(np.max(np.abs(sql_profiles - expected_profiles))),
                            "canonical_load_bitwise_equal": sql_loads.tobytes() == reference.tobytes(),
                            "canonical_profile_bitwise_equal": sql_profiles.tobytes() == expected_profiles.tobytes()})
                median = {plan: float(np.median([value["total_cpu_seconds"] for value in values])) for plan, values in trials.items()}
                record.update({"trials": trials, "median_cpu_seconds": median, "sql_arithmetic_diagnostics": diagnostics["sql"],
                               "shared_sql_arithmetic_diagnostics": diagnostics["sql_shared"],
                               "shared_canonical_identity": True})
            else:
                record.update({"trials": {}, "median_cpu_seconds": {plan: 0. for plan in plans},
                               "sql_arithmetic_diagnostics": [], "shared_sql_arithmetic_diagnostics": [], "shared_canonical_identity": True})
            records.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"scope": "Identical fixed complete-family Build queries. "
                "SQL includes construction and materialization of D(i,b,t,x) and Q(q,i,h), join/group-by, "
                "and the same host-profile reduction as NumPy. Candidate filtering/matching are shared untimed inputs. "
                "SQL-shared interns complete resident-group memberships, aggregates each distinct group once, "
                "then maps group loads to every (query,host); construction and materialization of membership/mapping relations are timed. "
                "No claim is made that SQL SUM preserves canonical floating-point arithmetic. "
                "Zero-query cases issue no query and are retained with zero query cost.",
                "duckdb_version": duckdb.__version__, "records": records}, indent=2) + "\n")
            print(json.dumps({"cohort": name, "origin": origin, "candidates": len(assignments),
                              "median_cpu_seconds": record["median_cpu_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
