"""Regenerate frozen admission streams with interval-filtered exact evidence.

Actions, support bounds, thresholds and Evaluation arrays come from a completed
study. Search is not rerun. Every new certificate decision and generated prefix
must equal the original integer-backend result before its record is published.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import process_time

import numpy as np

from cara.migration_envelope import stored_action_gain_bounds
from cara.synthetic import make_environment
from run_streaming_validation import PairedWorkloadReader, diagnostic, save, summarize

ROOT = Path(__file__).resolve().parent
SEMANTIC = ("admitted", "rows", "rows_read", "decision", "reason", "metric_names", "anchor",
            "noninferiority_margins", "min_anchor_gain", "gain_bounds", "exact_gain_bounds",
            "exact_thresholds", "crossing_rows", "locked", "mixture_components", "claim_scope")


def binding(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _inspect(reader, bounds, cap, batch, plan):
    from cara.interval_certificate import certify_streaming_interval
    started = process_time()
    result = certify_streaming_interval(reader, cap, len(plan["fresh_metrics"]), gain_bounds=bounds,
        noninferiority_margins=plan["margins"], anchor=plan["fresh_metrics"].index(plan["anchor"]),
        min_anchor_gain=plan["minimum_anchor_gain"], alpha=plan["alpha"], batch=batch,
        metric_names=plan["fresh_metrics"])
    total = process_time() - started
    result.update({"total_cpu_seconds": total, "generation_cpu_seconds": reader.generation_cpu_seconds,
        "replay_cpu_seconds": reader.replay_cpu_seconds, "reader_cpu_seconds": reader.reader_cpu_seconds,
        "statistical_cpu_seconds": max(0., total - reader.reader_cpu_seconds),
        "peak_panel_values": reader.peak_panel_values, "generated_scene_rows": reader.rows,
        "generated_demand_values": reader.rows * plan["base_environment_law"]["vms"] * plan["base_environment_law"]["horizon"]})
    if result["rows"] != reader.rows or result["rows_read"] != reader.rows:
        raise AssertionError("Interval backend inspected rows other than those actually acquired")
    return result


def _evaluation(arrays, original, plan):
    means, checked = {}, 0
    for role, fresh_key, native_key in (("incumbent_outcomes", "evaluation_reference", "native_evaluation_reference"),
            ("candidate_outcomes", "evaluation_candidate", "native_evaluation_candidate")):
        if role == "candidate_outcomes" and original["status"] != "selected":
            continue
        fresh, native = arrays[fresh_key], arrays[native_key]
        for values, names in ((fresh, plan["fresh_metrics"]), (native, plan["native_metrics"])):
            if values.shape != (plan["counts"]["evaluation"], len(names)) or not np.isfinite(values).all():
                raise AssertionError("Invalid original independent Evaluation arrays")
        result = {name: float(fresh[:, i].mean()) for i, name in enumerate(plan["fresh_metrics"])}
        result.update({"native_" + name: float(native[:, i].mean()) for i, name in enumerate(plan["native_metrics"])})
        for name, value in result.items():
            if value != original[role][name]:
                raise AssertionError((original["method"], original["seed"], "Evaluation mean", name))
            checked += 1
        # Utilization has no per-scene source array in the original artifact.
        # Its unchanged recorded value is retained separately from checked means.
        result.update({name: original[role][name] for name in ("hosts", "fixed_H_utilization")})
        if result["hosts"] != len(set(original["incumbent_assignment"])):
            raise AssertionError("Original Evaluation changed the fixed host population")
        means[role] = result
    return means, checked


def one_environment(job):
    source, output, plan, new_binding, original_records = job
    source, output = Path(source), Path(output)
    first = next(iter(original_records.values()))["record"]
    environment = make_environment(first["seed"])
    completed = {}
    for method, item in original_records.items():
        original = item["record"]
        identity = f'{original["law"]}_{original["seed"]}'
        target = output / method / "records" / f"{identity}.json"
        array_file = source / method / "arrays" / f"{identity}.npz"
        raw = array_file.read_bytes()
        array_digest = sha256(raw).hexdigest()
        if target.exists():
            record = json.loads(target.read_text())
            if (record.get("study_plan_sha256") != new_binding
                    or record.get("original_record_sha256") != item["sha256"]
                    or record.get("original_arrays_sha256") != array_digest
                    or record.get("verification", {}).get("all_semantic_rules_equal") is not True):
                raise ValueError(f"Existing interval checkpoint has a different binding: {target}")
            completed[method] = record
            continue
        with np.load(BytesIO(raw), allow_pickle=False) as arrays:
            for name, expected in environment.parameters().items():
                if not np.array_equal(arrays[name], expected):
                    raise AssertionError("Frozen environment parameters differ from the original scene law")
            for key in ("incumbent_assignment", "candidate_assignment"):
                if key in arrays and not np.array_equal(arrays[key], original[key]):
                    raise AssertionError("Original saved arrays use a different frozen action")
            means, checked_means = _evaluation(arrays, original, plan)
            record = {key: original[key] for key in ("law", "seed", "method", "config", "status",
                "incumbent_assignment", "candidate_assignment", "movement", "counts", "action_source")}
            record.update({"study_plan_sha256": new_binding, "original_study_plan_sha256": plan["original_study_plan_sha256"],
                "original_record_sha256": item["sha256"], "original_arrays_sha256": array_digest,
                "original_arrays": os.path.relpath(array_file, output), "search_not_rerun": True,
                "evidence_backend": "directed interval filter with exact integer fallback", "rules": {}, **means})
            candidate = original["candidate_assignment"]
            prefix_rows = semantic_fields = 0
            if original["status"] == "selected":
                tick = process_time()
                footprint = stored_action_gain_bounds(original["incumbent_assignment"],
                    original["movement"]["aligned_assignment"], plan["base_environment_law"]["capacity"],
                    horizon=plan["base_environment_law"]["horizon"])
                bounds = footprint["gain_bounds"]
                if method == "No-footprint":
                    bounds = (1., 1., 1., bounds[3])
                bound_cpu = process_time() - tick
                if list(bounds) != original["gain_bounds"] or not np.array_equal(arrays["gain_bounds"], bounds):
                    raise AssertionError("Recomputed action support differs from the frozen predeclared bound")
                record.update({"gain_bounds": bounds, "bound_cpu_seconds": bound_cpu,
                    "touched_hosts": footprint["touched_hosts"], "touched_fraction": footprint["touched_fraction"]})
                if footprint["touched_hosts"] != original["touched_hosts"] or footprint["touched_fraction"] != original["touched_fraction"]:
                    raise AssertionError("Physical action footprint changed")
                for label, old_rule in original["rules"].items():
                    cap = plan["primary_cap"] if label.startswith("stopped") else int(label[5:])
                    reader = PairedWorkloadReader(environment, original["law"], plan["roles"]["certification"],
                        original["incumbent_assignment"], candidate, batch_size=plan["replay_batch"],
                        execution=plan["search_config"].get("certification_views", "shared"))
                    rule = _inspect(reader, bounds, cap,
                        plan["inspection_batch"] if label.startswith("stopped") else cap, plan)
                    for name in SEMANTIC:
                        if rule[name] != old_rule[name]:
                            raise AssertionError((method, identity, label, name, rule[name], old_rule[name]))
                        semantic_fields += 1
                    reference, action = reader.arrays()
                    for actual, key in ((reference, "certification_reference"), (action, "certification_candidate")):
                        expected = arrays[key][:reader.rows]
                        if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                            raise AssertionError((method, identity, label, "generated/replayed prefix differs"))
                    prefix_rows += reader.rows
                    record["rules"][label] = rule
            else:
                record["bound_cpu_seconds"] = 0.
                if candidate is not None or original["movement"] is not None:
                    raise AssertionError("Invalid original no-candidate fallback")
                for label, old_rule in original["rules"].items():
                    if old_rule["admitted"] or old_rule["rows"] or old_rule["decision"] != "no_candidate":
                        raise AssertionError("Original no-candidate fallback acquired evidence")
                    record["rules"][label] = {key: value for key, value in old_rule.items()
                        if key not in ("terminal_outcomes", "terminal_truth_approximation", "terminal_migrations")}
                    semantic_fields += sum(key in old_rule for key in SEMANTIC)
            reference = arrays["evaluation_reference"]
            action = arrays["evaluation_candidate"] if candidate is not None else reference
            if candidate is not None:
                record["candidate_truth_approximation"] = diagnostic(reference, action)
            for label, rule in record["rules"].items():
                rule["terminal_outcomes"] = means["candidate_outcomes"] if rule["admitted"] else means["incumbent_outcomes"]
                rule["terminal_migrations"] = original["movement"]["count"] if rule["admitted"] else 0
                rule["terminal_truth_approximation"] = diagnostic(reference, action if rule["admitted"] else reference)
                for key in ("terminal_outcomes", "terminal_migrations", "terminal_truth_approximation"):
                    if rule[key] != original["rules"][label][key]:
                        raise AssertionError((method, identity, label, key))
            record["verification"] = {"all_semantic_rules_equal": True, "all_generated_prefix_bytes_equal": True,
                "checked_rules": len(record["rules"]), "checked_semantic_fields": semantic_fields,
                "checked_paired_prefix_rows": prefix_rows, "checked_independent_evaluation_means": checked_means,
                "evaluation_scope": "Fresh/native means recomputed from original saved arrays; host count checked; "
                    "recorded fixed_H_utilization retained because its scene-level array was not saved."}
        save(target, record)
        completed[method] = record
    return completed


def run(source, output, workers=8, max_environments=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if workers < 1 or (max_environments is not None and max_environments < 1) or source == output:
        raise ValueError("Positive workers/case limit and a separate output directory are required")
    original_plan = json.loads((source / "study_plan.json").read_text())
    original_binding = binding(original_plan)
    expected = {(row["law"], row["seed"]) for row in original_plan["validation_environments"]}
    if len(expected) != len(original_plan["validation_environments"]) or not expected:
        raise ValueError("Original study has missing or duplicate designated environments")
    for relative, digest in original_plan["source_sha256"].items():
        if sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Original frozen scientific source changed: {relative}")
    sources, indexed = {}, {}
    for method in original_plan["methods"]:
        file = source / method / "records.json"
        raw = file.read_bytes()
        records = json.loads(raw)
        if len(records) != len(expected) or {(r["law"], r["seed"]) for r in records} != expected:
            raise ValueError(f"Original method is incomplete: {method}")
        sources[str(file.relative_to(source))] = sha256(raw).hexdigest()
        indexed[method] = {}
        for record in records:
            labels = {"stopped8192", *(f"fixed{cap}" for cap in original_plan["fixed_caps"])}
            if (record["study_plan_sha256"] != original_binding or record["method"] != method
                    or record["config"] != original_plan["search_config"] or record["counts"] != original_plan["counts"]
                    or set(record["rules"]) != labels or record["status"] not in ("selected", "no_candidate")
                    or (record["status"] == "selected") != (record["candidate_assignment"] is not None)):
                raise ValueError("Original record differs from its complete frozen design")
            digest = binding(record)
            compact = {key: value for key, value in record.items() if key not in ("search",)}
            indexed[method][record["law"], record["seed"]] = {"sha256": digest, "record": compact}
    if "No-footprint" in indexed and any(indexed["No-footprint"][key]["record"]["candidate_assignment"]
            != indexed["CARA"][key]["record"]["candidate_assignment"] for key in expected):
        raise AssertionError("Original No-footprint changes CARA's frozen actions")
    plan = {**original_plan, "original_study_plan": original_plan, "original_study_plan_sha256": original_binding,
        "original_record_files_sha256": sources, "original_study": os.path.relpath(source, output),
        "interval_backend_sha256": sha256((ROOT / "cara/interval_certificate.py").read_bytes()).hexdigest(),
        "interval_runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_operator": "Outward interval evidence filter with exact integer fallback; statistical rule and actions unchanged.",
        "execution": "Rerun actual on-demand generation and paired replay for every original stopped/fixed rule. "
            "Recompute each action bound once. Compare all semantic decision fields and every generated/replayed prefix byte "
            "with original integer results. Independent Evaluation arrays are referenced, checked and reused; search is not rerun."}
    new_binding = binding(plan)
    existing = output / "study_plan.json"
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError("Existing interval results have a different original study or backend")
    save(existing, plan)
    for method in plan["methods"]:
        save(output / method / "study_plan.json", {**plan, "search_method": method})
    identities = [(row["law"], row["seed"]) for row in original_plan["validation_environments"]]
    chosen = identities if max_environments is None else identities[:max_environments]
    jobs = [(str(source), str(output), plan, new_binding, {method: indexed[method][key] for method in plan["methods"]}) for key in chosen]
    completed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(one_environment, job) for job in jobs]):
            completed.append(future.result())
            progress = {"completed_environments": len(completed), "expected_environments": len(identities),
                       "complete": len(completed) == len(identities)}
            save(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    if len(completed) != len(identities):
        return {"complete": False, "completed_environments": len(completed), "expected_environments": len(identities)}
    summaries, verification = {}, {"complete": True, "all_semantic_rules_equal": True,
        "all_generated_prefix_bytes_equal": True, "original_study_plan_sha256": original_binding,
        "study_plan_sha256": new_binding, "methods": {}}
    for method in plan["methods"]:
        records = sorted((item[method] for item in completed), key=lambda row: (row["law"], row["seed"]))
        save(output / method / "records.json", records)
        summaries[method] = summarize(records)
        save(output / method / "summary.json", summaries[method])
        verification["methods"][method] = {"environments": len(records), **{key: sum(r["verification"][key] for r in records)
            for key in ("checked_rules", "checked_semantic_fields", "checked_paired_prefix_rows", "checked_independent_evaluation_means")}}
    save(output / "summary.json", summaries)
    save(output / "verification.json", verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Completed original integer-backend streaming study")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-environments", type=int, help="Smoke limit; incomplete studies never publish final aggregates")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.workers, args.max_environments), indent=2), flush=True)
