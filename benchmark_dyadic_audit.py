"""Matched historical Audit helpers with independent population checks."""
import gc
from time import process_time
import numpy as np
from run_streaming_validation import save
from cara.certification import exact_pareto_gate
from cara.dyadic_gate import exact_dyadic_gate
from cara.data import load_dataset
from cara.refinement import PROFILE_NAMES, panel_profile
from cara.window_query import KeyedProfilePlan, materialize_windows, prepare_window_queries

PLANS = ("direct_original", "direct_dyadic", "keyed_original", "keyed_dyadic")
FIELDS = ("total_cpu_seconds", "candidate_cpu_seconds", "planning_cpu_seconds",
          "profile_cpu_seconds", "gate_cpu_seconds", "source_panel_reads", "aggregate_input_cells")


def same_gate(a, b):
    return (list(a) == list(b) and a == b and
        np.asarray(a["gains"], dtype=np.float64).tobytes() == np.asarray(b["gains"], dtype=np.float64).tobytes())


def source_for(record, datasets):
    data = datasets[record["dataset"]]
    matrix = data.panel(record["origin"] - 32, 32)
    matrix.setflags(write=False)
    return matrix, np.arange(25, dtype=np.int64), float(matrix.max())


def population_check(records, datasets):
    checked = 0
    for record in records:
        policy = record["policies"]["CARA"]
        if policy["selected_assignment"] is None:continue
        matrix, starts, _ = source_for(record, datasets)
        panel = materialize_windows(matrix, starts)
        reference = panel_profile(panel, record["incumbent_assignment"], record["capacity"])
        candidate = panel_profile(panel, policy["selected_assignment"], record["capacity"])
        old = exact_pareto_gate(reference, candidate, metric_names=PROFILE_NAMES)
        new = exact_dyadic_gate(reference, candidate, metric_names=PROFILE_NAMES)
        if not same_gate(old, policy["admission"]) or not same_gate(old, new):
            raise AssertionError((record["dataset"],record["origin"],record["seed_index"],"Population gate dictionary changed"))
        checked += 1
    return {"decisions":len(records),"selected_checked":checked,"no_action_retained":len(records)-checked,
            "all_fields_float_bits_strings_and_key_order_equal":True}


def run_plan(matrix, starts, candidate_assignment, capacity, upper, reference, plan):
    gc.collect()
    began = process_time()
    query = KeyedProfilePlan(candidate_assignment, capacity, 8,
        demand_upper=upper, reuse_hours=plan.startswith("keyed"))
    planned = process_time()
    candidate = query.historical_profile(matrix, starts)
    profiled = process_time()
    function = exact_dyadic_gate if plan.endswith("dyadic") else exact_pareto_gate
    gate = function(reference, candidate, metric_names=PROFILE_NAMES)
    elapsed = process_time() - began
    stats = dict(query.stats)
    return candidate,gate,{"total_cpu_seconds":elapsed,
        "candidate_cpu_seconds":profiled-began,"planning_cpu_seconds":planned-began,
        "profile_cpu_seconds":profiled-planned,"gate_cpu_seconds":elapsed-(profiled-began),
        "source_panel_reads":stats["source_panel_reads"],"aggregate_input_cells":stats["aggregate_input_cells"],
        "query_stats":stats}
