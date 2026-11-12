"""Complete Build-query SQL context on twelve prescribed cohort settings.

Candidate enumeration and filtering are common untimed inputs. SQL keeps its
native SUM arithmetic; observed threshold and profile differences are reported.
This is a query-stage comparison, not a full-search or deployment baseline.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import json
from pathlib import Path
from time import process_time

import numpy as np
import duckdb
import pyarrow as pa

from benchmark_query_engine import candidates
from benchmark_sql_queries import full_queries, profiles_from_batch_loads, shared_queries, sql_queries
from cara.data import CAPACITIES, canonical_assignment, load_dataset
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.refinement import PROFILE_NAMES
from cara.repair import _mutable, _pair_order, ffd
from run_streaming_validation import save


ROOT = Path(__file__).resolve().parent
ORIGINS = (288, 300, 312)
PLANS = ("sql", "sql_shared", "sql_shared_resident", "full", "shared", "shared_resident")
WIDTH, FAMILY_LIMIT, REPETITIONS = 8, 4, 3
MAX_RESULT_BYTES = 64 << 20




def chunk_slices(queries, hosts, scenes, times):
    """Bound returned load arrays uniformly; never omit a candidate row."""
    if min(queries, hosts, scenes, times) < 0 or min(hosts, scenes, times) < 1:
        raise ValueError("Nonnegative query count and positive tensor dimensions required")
    per_query = hosts * scenes * times * np.dtype(np.float64).itemsize
    if per_query > MAX_RESULT_BYTES:
        raise ValueError("One result exceeds the declared load-array bound")
    width = max(1, MAX_RESULT_BYTES // per_query)
    return tuple((start, min(start + width, queries)) for start in range(0, queries, width))


def prepare_case(data, origin):
    began = process_time()
    panel = data.window_panel(origin, 192, 64)
    forecast = data.forecast(origin)
    incumbent = ffd(forecast, data.capacity)
    if panel.shape != (121, data.n_vms, 8):
        raise AssertionError("Every one of the 121 complete Build windows is required")
    pairs = _pair_order(panel, incumbent, data.capacity, np.random.default_rng(519 + origin))[:FAMILY_LIMIT]
    forecasts = ForecastViews(forecast, data.capacity, incumbent)
    assignments, family_rows = [], []
    for pair in pairs:
        mutable = _mutable(panel, incumbent, pair, WIDTH, data.capacity)
        rows = candidates(forecast, incumbent, pair, mutable, data.capacity,
                          int(.2 * len(incumbent)), forecast_views=forecasts)
        start = len(assignments)
        assignments.extend(assignment for assignment, _ in rows)
        family_rows.append({"pair": pair, "mutable": mutable,
                            "candidate_start": start, "candidate_stop": len(assignments)})
    return panel, incumbent, tuple(assignments), {"family_requests": family_rows,
        "common_input_cpu_seconds": process_time() - began}


def arithmetic_differences(loads, profiles, reference_loads, reference_profiles, capacity):
    if (loads.shape != reference_loads.shape or profiles.shape != reference_profiles.shape
            or loads.dtype != reference_loads.dtype or profiles.dtype != reference_profiles.dtype):
        raise AssertionError("Query returned a different complete result shape or dtype")
    load_bits = loads.view(np.uint64) != reference_loads.view(np.uint64)
    profile_bits = profiles.view(np.uint64) != reference_profiles.view(np.uint64)
    return {"load_values": loads.size, "profile_values": profiles.size,
        "different_load_bits": int(load_bits.sum()), "different_profile_bits": int(profile_bits.sum()),
        "different_profile_bits_by_metric": dict(zip(PROFILE_NAMES, map(int, profile_bits.sum(axis=(0, 1))))),
        "maximum_absolute_load_difference": float(np.max(np.abs(loads - reference_loads), initial=0.)),
        "maximum_absolute_profile_difference": float(np.max(np.abs(profiles - reference_profiles), initial=0.)),
        "strict_threshold_flips": int(np.count_nonzero((loads > capacity) != (reference_loads > capacity))),
        "tolerant_threshold_flips": int(np.count_nonzero((loads > capacity + 1e-12) != (reference_loads > capacity + 1e-12)))}


def aggregate_diagnostics(rows):
    maxima = ("maximum_absolute_load_difference", "maximum_absolute_profile_difference")
    counts = ("load_values", "profile_values", "different_load_bits", "different_profile_bits",
              "strict_threshold_flips", "tolerant_threshold_flips")
    result = {name: sum(row[name] for row in rows) for name in counts}
    result.update({name: max((row[name] for row in rows), default=0.) for name in maxima})
    result["different_profile_bits_by_metric"] = {name: sum(row["different_profile_bits_by_metric"][name]
                                                         for row in rows) for name in PROFILE_NAMES}
    result["canonical_load_bitwise_equal"] = result["different_load_bits"] == 0
    result["canonical_profile_bitwise_equal"] = result["different_profile_bits"] == 0
    return result


def resident_sql(panel, assignments, capacity, chunks, verify):
    """One immutable demand table and one global group aggregate per batch."""
    began = process_time()
    b, n, t = panel.shape
    demand = pa.table({"i": np.tile(np.repeat(np.arange(n, dtype=np.int64), t), b),
        "b": np.repeat(np.arange(b, dtype=np.int64), n * t),
        "t": np.tile(np.arange(t, dtype=np.int64), b * n), "x": panel.reshape(-1)})
    connection = duckdb.connect(config={"threads": "1", "memory_limit": "2GB"})
    diagnostics, chunk_costs, verification_cpu = [], [], 0.
    try:
        connection.register("input_demand", demand)
        connection.execute("CREATE TEMP TABLE demand AS SELECT * FROM input_demand")
        source_preparation = process_time() - began
        resident_began = process_time()
        labels = np.asarray([canonical_assignment(assignment) for assignment in assignments], dtype=np.int64)
        q, h = len(labels), len(set(labels[0]))
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
        columns = np.asarray(mapping, dtype=np.int64)
        membership = pa.table({"g": np.asarray(group_ids, dtype=np.int64), "i": np.asarray(resident_ids, dtype=np.int64)})
        placement = pa.table({"q": columns[:, 0], "h": columns[:, 1], "g": columns[:, 2]})
        connection.register("input_membership", membership)
        connection.register("input_placement", placement)
        connection.execute("CREATE TEMP TABLE membership AS SELECT * FROM input_membership")
        connection.execute("CREATE TEMP TABLE placement AS SELECT * FROM input_placement")
        membership_preparation = process_time() - resident_began
        tick = process_time()
        connection.execute("""
            CREATE TEMP TABLE group_load AS
            SELECT m.g, d.b, d.t, SUM(d.x) AS load
            FROM demand AS d JOIN membership AS m ON d.i = m.i
            GROUP BY m.g, d.b, d.t
        """)
        group_aggregation = process_time() - tick
        for start, stop in chunks:
            tick = process_time()
            query = """
                SELECT p.q, p.h, l.b, l.t, l.load
                FROM placement AS p JOIN group_load AS l ON p.g = l.g
            """
            if start == 0 and stop == q:
                values = connection.execute(query + " ORDER BY p.q, p.h, l.b, l.t").fetchnumpy()["load"]
            else:
                values = connection.execute(query + " WHERE p.q >= ? AND p.q < ? ORDER BY p.q, p.h, l.b, l.t",
                                            [start, stop]).fetchnumpy()["load"]
            loads = np.asarray(values, dtype=np.float64).reshape(stop - start, h, b, t)
            fetched = process_time()
            profiles = profiles_from_batch_loads(loads, capacity)
            profiled = process_time()
            chunk_costs.append({"fetch_cpu_seconds": fetched - tick,
                                "profile_cpu_seconds": profiled - fetched})
            checked = process_time()
            diagnostics.append(verify(start, stop, loads, profiles))
            verification_cpu += process_time() - checked
            del values, loads, profiles
        resident_cpu = process_time() - resident_began - verification_cpu
    finally:
        connection.close()
    total = process_time() - began - verification_cpu
    fetch_cpu = sum(row["fetch_cpu_seconds"] for row in chunk_costs)
    profile_cpu = sum(row["profile_cpu_seconds"] for row in chunk_costs)
    return {"cpu_seconds": total, "source_preparation_cpu_seconds": source_preparation,
        "membership_materialization_cpu_seconds": membership_preparation,
        "materialization_cpu_seconds": source_preparation + membership_preparation,
        "group_aggregation_cpu_seconds": group_aggregation,
        "aggregation_cpu_seconds": group_aggregation + fetch_cpu,
        "profile_cpu_seconds": profile_cpu, "resident_query_cpu_seconds": resident_cpu,
        "demand_materializations": 1, "distinct_resident_groups": len(identities),
        "demand_rows": b * n * t, "membership_rows": len(resident_ids),
        "placement_rows": q * h, "group_load_rows": len(identities) * b * t,
        "chunks": chunk_costs, "arithmetic": aggregate_diagnostics(diagnostics)}


def resident_shared(panel, incumbent, assignments, capacity, chunks, verify):
    """One bounded canonical registry and one complete profile call."""
    began = process_time()
    reader = SharedQueryReplay(panel, incumbent, capacity)
    ready = process_time()
    profiles = reader.profiles(assignments)
    finished = process_time()
    costs = {"cpu_seconds": finished - began, "source_preparation_cpu_seconds": ready - began,
        "resident_query_cpu_seconds": finished - ready,
        "retained_array_bytes": reader.cached_view_bytes,
        "membership_hits": reader.hits, "materialized_demand_values": reader.materialized_values}
    diagnostics = []
    for start, stop in chunks:
        # Only bounded inspection loads are materialized, after query timing.
        loads = reader.loads_many(assignments[start:stop])
        diagnostics.append(verify(start, stop, loads, profiles[start:stop]))
        del loads
    costs["arithmetic"] = aggregate_diagnostics(diagnostics)
    return costs


def run_case(data, origin, case_index):
    panel, incumbent, assignments, preparation = prepare_case(data, origin)
    b, n, t = panel.shape
    h, q = len(set(incumbent)), len(assignments)
    chunks = chunk_slices(q, h, b, t)
    aliases = {"sql_shared": "sql_shared_resident", "shared": "shared_resident"} if len(chunks) == 1 else {}
    active = tuple(plan for plan in PLANS if plan not in aliases) if q else ()
    records = []

    def verify(start, stop, loads, profiles):
        _, reference_loads, reference_profiles = full_queries(panel, assignments[start:stop], data.capacity)
        return arithmetic_differences(loads, profiles, reference_loads, reference_profiles, data.capacity)

    for repetition in range(REPETITIONS):
        shift = (case_index + repetition) % len(active) if active else 0
        order = active[shift:] + active[:shift]
        record = {"order": order, "plans": {}}
        for plan in order:
            gc.collect()
            if plan == "sql_shared_resident":
                values = resident_sql(panel, assignments, data.capacity, chunks, verify)
            elif plan == "shared_resident":
                values = resident_shared(panel, incumbent, assignments, data.capacity, chunks, verify)
            else:
                costs, diagnostics = [], []
                for start, stop in chunks:
                    batch = assignments[start:stop]
                    if plan.startswith("sql"):
                        cost, loads, profiles = sql_queries(panel, batch, data.capacity, plan == "sql_shared")
                    elif plan == "full":
                        cost, loads, profiles = full_queries(panel, batch, data.capacity)
                    else:
                        cost, loads, profiles = shared_queries(panel, incumbent, batch, data.capacity)
                    diagnostics.append(verify(start, stop, loads, profiles))
                    costs.append(cost)
                    del loads, profiles
                values = {"cpu_seconds": sum(row["total_cpu_seconds"] for row in costs),
                          "chunks": costs, "arithmetic": aggregate_diagnostics(diagnostics)}
                if plan.startswith("sql"):
                    for field in ("materialization_cpu_seconds", "aggregation_cpu_seconds", "profile_cpu_seconds"):
                        values[field] = sum(row[field] for row in costs)
            if plan in ("full", "shared", "shared_resident") and (
                    values["arithmetic"]["different_load_bits"] or values["arithmetic"]["different_profile_bits"]):
                raise AssertionError("A canonical NumPy control changed result bits")
            record["plans"][plan] = values
        for alias, original in aliases.items():
            record["plans"][alias] = {**record["plans"][original], "alias_of": original,
                "equivalence": "A single complete chunk executes the same cold full-batch physical operations; no duplicate timing arm."}
        if not q:
            record["plans"] = {plan: {"cpu_seconds": 0., "materialization_cpu_seconds": 0.,
                "aggregation_cpu_seconds": 0., "profile_cpu_seconds": 0.,
                "source_preparation_cpu_seconds": 0., "resident_query_cpu_seconds": 0.,
                "arithmetic": aggregate_diagnostics([])} for plan in PLANS}
        records.append(record)
    medians = {plan: {"cpu_seconds": float(np.median([row["plans"][plan]["cpu_seconds"] for row in records]))}
               for plan in PLANS}
    for plan in ("sql", "sql_shared", "sql_shared_resident"):
        for field in ("materialization_cpu_seconds", "aggregation_cpu_seconds", "profile_cpu_seconds"):
            medians[plan][field] = float(np.median([row["plans"][plan][field] for row in records]))
    for plan in ("sql_shared_resident", "shared_resident"):
        for field in ("source_preparation_cpu_seconds", "resident_query_cpu_seconds"):
            medians[plan][field] = float(np.median([row["plans"][plan][field] for row in records]))
    return {"cohort": data.name, "origin": origin, "N": n, "H": h, "B": b, "T": t,
        "candidates": q, "complete_family_count": len(preparation["family_requests"]),
        "status": "queried" if q else "zero_query", "chunks": chunks, "single_chunk_aliases": aliases,
        "maximum_returned_load_array_bytes": max(((stop - start) * h * b * t * 8 for start, stop in chunks), default=0),
        "input_panel_bytes": panel.nbytes, **preparation, "trials": records,
        "median_seconds": medians, "all_canonical_controls_exact": True}


def main(args):
    plan = {"cohorts": list(CAPACITIES), "origins": ORIGINS, "width": WIDTH,
        "requested_families": FAMILY_LIMIT, "pair_rule": "First min(4,H(H-1)/2) risk-ordered pairs; original jitter RNG seed 519+origin; original mutable-resident rule.",
        "build_role": {"start_lookback_hours": 192, "stop_lookback_hours": 64, "horizon": 8, "stride": 1, "rows": 121},
        "plans": PLANS, "repetitions": REPETITIONS, "max_returned_load_bytes": MAX_RESULT_BYTES,
        "common_input": "The complete original forecast-feasible, incumbent-relative movement-feasible ordered candidate multiset is constructed once outside all compared query timers. Zero-query settings remain with zero query cost.",
        "execution": "Existing cold SQL/helper controls are retained for multiple chunks. Resident SQL-Shared materializes D once, interns the entire candidate collection once, builds one native-SUM group_load relation, then fetches bounded result chunks. Resident canonical SharedQueryReplay initializes one bounded registry and computes the whole Q-by-B-by-6 profile batch once; only inspection loads are chunked outside timing. For one chunk, each cold helper and its resident counterpart are physically equivalent and share one measured arm.",
        "resident_boundary": "Resident SQL query CPU begins after immutable D and its connection are ready; it includes candidate membership/mapping preparation, group aggregation, fetch and profile work. Resident canonical query CPU begins after its reader constructor and includes membership planning and complete profile execution. Cold totals include both source preparation and query processing. Resident SQL verification intervals are explicitly excluded from its continuous CPU interval; all operator and loop work is retained.",
        "memory": "Each returned load tensor is at most 64 MiB. Cold helper plans charge repeated chunk preparation. Resident SQL retains its global group relation and resident canonical execution retains one bounded registry across the entire candidate batch. No candidate is dropped; at most one canonical reference chunk and one observed chunk are retained for diagnostics. Full profile arrays are Q-by-B-by-6, not Q-by-H-by-B-by-T.",
        "arithmetic": "SQL retains native SUM. Every returned load/profile bit difference and strict/tolerant threshold flip is counted per trial. SQL is not claimed to implement the canonical NumPy operation graph.",
        "scope": "Secondary query-stage database-engine context, not full CARA search, selection, certification, or deployment performance."}
    normalized = json.loads(json.dumps(plan))
    target = args.output / "study_plan.json"
    if target.exists() and json.loads(target.read_text()) != normalized:
        raise ValueError("The SQL stage benchmark has different inputs or settings")
    save(target, plan)
    if args.write_plan:
        return
    records = []
    for name in CAPACITIES:
        tick = process_time()
        data = load_dataset(name)
        save(args.output / f"{name}_loading.json", {"cpu_seconds": process_time() - tick, "N": data.n_vms})
        for origin in ORIGINS:
            checkpoint = args.output / "cases" / f"{name}_{origin}.json"
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text())
                if row["cohort"] != name or row["origin"] != origin:
                    raise ValueError("A SQL checkpoint belongs to another design")
            else:
                row = run_case(data, origin, len(records))
                save(checkpoint, row)
            records.append(row)
            save(args.output / "records.json", records)
            print(json.dumps({"cohort": name, "origin": origin, "queries": row["candidates"],
                              "median_seconds": row["median_seconds"]}), flush=True)
    save(args.output / "summary.json", {"complete": len(records) == 12,
        "settings": len(records), "zero_query_settings": sum(row["candidates"] == 0 for row in records),
        "total_queries": sum(row["candidates"] for row in records),
        "all_121_build_windows": all(row["B"] == 121 for row in records),
        "summed_median_seconds": {plan: sum(row["median_seconds"][plan]["cpu_seconds"] for row in records) for plan in PLANS},
        "all_canonical_controls_exact": all(row["all_canonical_controls_exact"] for row in records),
        "sql_semantics": "Inspect each recorded arithmetic diagnostic; observed equality is not a universal SQL floating-point guarantee."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-plan", action="store_true")
    main(parser.parse_args())
