"""Complete matched Direct/Keyed by original/dyadic historical Audit gates."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import process_time

import numpy as np

from cara.certification import exact_pareto_gate
from cara.dyadic_gate import exact_dyadic_gate
from cara.data import load_dataset
from cara.refinement import PROFILE_NAMES, panel_profile
from cara.window_query import KeyedProfilePlan, materialize_windows, prepare_window_queries
from run_streaming_validation import save

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


def run(output, source):
    output=Path(output)
    records=json.loads(Path(source).read_text())
    if (len(records),sum(r["policies"]["CARA"]["selected_assignment"] is not None for r in records))!=(300,249):
        raise ValueError("Complete original300/249selected/51noaction population required")
    design={"phase":"complete_exact_dyadic_audit_factorial","plans":PLANS,"repetitions":5,
        "population":"Original300decisions/249selected/51noaction, no search rerun or filtering",
        "factorial":"Direct and Keyed candidate-only native6 plans each receive original and dyadic gates; neither factor receives an exclusive common optimization",
        "target":"Original exact_pareto_gate entire dictionary, floatbits, exactstrings, metricorder and keyorder unchanged",
        "timing":"Same untimed incumbent Audit reference for every arm. Each timed call includes cold candidateF0, sourcevalidation/profile and exactly one gate; gate-only interval is nested within that same timer, not a second query or additional total.",
        "common":"Dataset loading,32hour role extraction, upperbound and original incumbent reference construction recorded outside all method timers",
        "gate_support":"Finite stored binary64 values including native metric sentinels; exact integer alignment before each coordinate's final Fraction",
        "source_records_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    files=("benchmark_dyadic_audit.py","cara/dyadic_gate.py","cara/certification.py","cara/data.py",
        "cara/refinement.py","cara/workload_views.py","cara/window_query.py","cara/kernels/window.c",
        "cara/window_budget.py","cara/prefix_canonical.py","cara/kernels/prefix_canonical.c",
        "cara/fused_canonical.py","cara/kernels/fused_canonical.c")
    design["source_code"]={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    path=output/"study_plan.json"
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(design)):
        raise ValueError("Frozen source/input changed; use a new directory")
    save(path,design)
    datasets={}
    for name in dict.fromkeys(r["dataset"] for r in records):
        began=process_time();datasets[name]=load_dataset(name)
        save(output/(name+"_loading.json"),{"cpu_seconds":process_time()-began,"n_vms":datasets[name].n_vms})
    # This is a single untimed complete-population validation, before matched timing.
    save(output/"population_check.json",population_check(records,datasets))
    save(output/"common_preparation.json",prepare_window_queries())
    result=[]
    for index,record in enumerate(records):
        policy=record["policies"]["CARA"]
        identity={k:record[k] for k in ("dataset","origin","seed_index","seed")}
        candidate_assignment=policy["selected_assignment"]
        if candidate_assignment is None:
            result.append({"identity":identity,"status":"no_action","original_admission":policy["admission"],"trials":{},"medians":{}})
            save(output/"records.json",result);continue
        began=process_time();matrix,starts,upper=source_for(record,datasets)
        role_cpu=process_time()-began
        panel=materialize_windows(matrix,starts)
        began=process_time();reference=panel_profile(panel,record["incumbent_assignment"],record["capacity"])
        reference_cpu=process_time()-began
        target=panel_profile(panel,candidate_assignment,record["capacity"])
        expected=policy["admission"]
        trials={p:[] for p in PLANS}
        for repeat in range(5):
            shift=(index+repeat)%len(PLANS)
            for plan in PLANS[shift:]+PLANS[:shift]:
                candidate,gate,cost=run_plan(matrix,starts,candidate_assignment,record["capacity"],upper,reference,plan)
                if candidate.shape!=target.shape or candidate.tobytes()!=target.tobytes() or not same_gate(gate,expected):
                    raise AssertionError((identity,plan,"Candidate bytes or exact gate changed"))
                trials[plan].append(cost)
        medians={p:{k:float(np.median([c[k] for c in costs])) for k in FIELDS} for p,costs in trials.items()}
        result.append({"identity":identity,"status":"selected","N":len(candidate_assignment),
            "H":len(set(candidate_assignment)),"B":25,"L":32,"T":8,
            "role_source_prep_cpu_seconds":role_cpu,"reference_profile_cpu_seconds":reference_cpu,
            "original_admission":expected,"all_profile_bytes_and_original_gate_fields_equal":True,
            "trials":trials,"medians":medians})
        save(output/"records.json",result)
        print(json.dumps({"decision":index+1,"identity":identity,
            "actual_audit_ms":{p:1000*v["total_cpu_seconds"] for p,v in medians.items()},
            "nested_gate_ms":{p:1000*v["gate_cpu_seconds"] for p,v in medians.items()}}),flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=value for p,value in design["source_code"].items()):
        raise AssertionError("A frozen source file changed during execution")
    save(output/"summary.json",{"complete":True,"decisions":300,"selected":249,"no_action":51,
        "all_profile_bytes_and_original_gate_fields_equal":True,"sources_unchanged_after_execution":True})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=Path("results/icde2027_shared_main/records.json"))
    args=parser.parse_args();run(args.output,args.source)
