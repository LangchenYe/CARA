"""SQL and canonical reference operators for identical placement queries."""
from time import process_time
import numpy as np
import pyarrow as pa
import duckdb
from cara.data import canonical_assignment, host_loads
from cara.query_engine import SharedQueryReplay, _profiles
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
