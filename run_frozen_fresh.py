"""Complete fixed-action support and stopping study on the original 96 laws.

Search is not rerun. Every timed arm regenerates its actually acquired full-VM
scene prefix and uses the same canonical paired reader and evidence backend.
Independent Evaluation means are diagnostics, not known population truths.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from fractions import Fraction
from functools import partial
import gc
import json
from math import inf, nextafter
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from cara.action_query import ActionSceneReader, canonical_fresh_losses, certify_action_queries
from cara.certification import stored_raw_gain_bound
from cara.data import canonical_assignment
from cara.fused_canonical import FusedCanonicalPlan, prepare_fused_canonical
from cara.interval_certificate import certify_streaming_interval
from cara.migration_envelope import stored_action_gain_bounds, stored_topology_gain_bound
from cara.native_aggregate import prepare_native
from cara.scene_stream import SCENE_LAWS, SceneStream
from cara.synthetic import LAW, make_environment
from run_streaming_validation import (FRESH_NAMES, MARGINS, PROFILE_NAMES, ROLES, VALIDATION,
                                      PairedWorkloadReader, diagnostic, outcome_means, save)


ROOT = Path(__file__).resolve().parent
SEMANTIC = ("admitted", "rows", "rows_read", "decision", "reason", "metric_names", "anchor",
            "noninferiority_margins", "min_anchor_gain", "gain_bounds", "exact_gain_bounds",
            "exact_thresholds", "crossing_rows", "locked", "mixture_components", "claim_scope")
ARMS = {
    "CARA": ("graph", 8192, 32),
    "NoComponent": ("no_component", 8192, 32),
    "NoGraph": ("count", 8192, 32),
    "NoBoolean": ("no_boolean", 8192, 32),
    "Global": ("global", 8192, 32),
    "Fixed2048": ("graph", 2048, 2048),
    "Fixed4096": ("graph", 4096, 4096),
    "Fixed8192": ("graph", 8192, 8192),
}




def action_contract(record):
    """The small complete input determining a fixed-action certification run."""
    return {name: record[name] for name in ("law", "seed", "status", "incumbent_assignment",
            "candidate_assignment", "movement", "gain_bounds") if name in record}


def load_contracts(source):
    source = Path(source)
    plan = json.loads((source / "study_plan.json").read_text())
    records = json.loads((source / "CARA/records.json").read_text())
    identities = {(row["law"], row["seed"]) for row in records}
    declared = {(row["law"], row["seed"]) for row in plan["validation_environments"]}
    counts = {"fit": 128, "build": 64, "selection": 128, "certification": 8192, "evaluation": 16384}
    if (len(records) != 96 or identities != set(VALIDATION) or declared != identities
            or len(plan["validation_environments"]) != 96 or plan["base_environment_law"] != LAW
            or plan["scene_laws"] != json.loads(json.dumps(SCENE_LAWS)) or plan["roles"] != ROLES
            or plan["counts"] != counts or plan["fresh_metrics"] != list(FRESH_NAMES)
            or plan["native_metrics"] != list(PROFILE_NAMES) or plan["margins"] != list(MARGINS)
            or plan["anchor"] != "R" or plan["minimum_anchor_gain"] != .005
            or plan["alpha"] != .05 or plan["primary_cap"] != 8192
            or plan["inspection_batch"] != 32 or plan["replay_batch"] != 32
            or plan["fixed_caps"] != [2048, 4096, 8192]):
        raise ValueError("The complete original 96-environment fresh contract is required")
    for row in records:
        old, new = row["incumbent_assignment"], row["candidate_assignment"]
        if (row["method"] != "CARA" or row["counts"] != counts
                or row["config"] != plan["search_config"]
                or len(old) != LAW["vms"] or tuple(old) != canonical_assignment(old)):
            raise ValueError("A frozen action disagrees with its original contract")
        if row["status"] == "no_candidate":
            if new is not None or row["movement"] is not None:
                raise ValueError("Invalid original incumbent fallback")
        elif row["status"] == "selected":
            aligned = row["movement"]["aligned_assignment"]
            if (new is None or len(new) != len(old) or set(aligned) != set(old)
                    or canonical_assignment(aligned) != tuple(new)
                    or sum(a != b for a, b in zip(old, aligned)) != row["movement"]["count"]):
                raise ValueError("The frozen physical alignment does not realize the selected action")
        else:
            raise ValueError("Unknown original action status")
    if sum(row["status"] == "selected" for row in records) != 90:
        raise ValueError("All 90 selected actions and six original fallbacks are required")
    return plan, sorted(records, key=lambda row: (row["law"], row["seed"]))


def support(record, mode):
    """Population bounds differing only in the specified support information.

    ``graph`` uses component cut/capacity bounds and changed-host Boolean
    bounds. ``no_component`` retains the same cuts with one global capacity
    cap. ``count`` uses moved count and the global cap, without migration-graph
    structure. ``no_boolean`` retains the component raw bound but uses unit
    Boolean bounds. ``global`` uses unit Boolean bounds and the global cap.
    """
    if mode not in {"graph", "no_component", "count", "no_boolean", "global"}:
        raise ValueError("Unknown fixed population support")
    old = record["incumbent_assignment"]
    aligned = record["movement"]["aligned_assignment"]
    hosts, capacity, horizon = len(set(old)), LAW["capacity"], LAW["horizon"]
    if mode in {"graph", "no_component"}:
        return tuple(stored_action_gain_bounds(old, aligned, capacity, horizon=horizon,
            component_conservation=mode == "graph")["gain_bounds"])
    if mode == "no_boolean":
        raw = stored_topology_gain_bound(old, aligned, capacity, horizon=horizon)
        return (1., 1., 1., raw)
    # Remove graph information without discarding the common fixed-H
    # conservation bound. Use the same stored replay allowance as CARA.
    roundoff = Fraction(stored_raw_gain_bound(0, hosts, capacity, vm_count=len(old), horizon=horizon))
    physical = Fraction(hosts - 1, hosts)
    if mode == "count":
        physical = min(physical, Fraction(record["movement"]["count"], hosts) / Fraction(capacity))
    raw = nextafter(float(physical + roundoff), inf)
    if mode == "global":
        return (1., 1., 1., raw)
    touched = {host for a, b in zip(old, aligned) if a != b for host in (a, b)}
    exact = min(Fraction(1), Fraction(len(touched), hosts) + Fraction(1, 2**52)) if touched else Fraction()
    ratio = float(exact)
    if Fraction(ratio) < exact:
        ratio = nextafter(ratio, inf)
    return (1. if touched else 0., ratio, ratio, raw)


def options(bounds):
    return dict(gain_bounds=bounds, noninferiority_margins=MARGINS, anchor=2,
                min_anchor_gain=.005, alpha=.05, metric_names=FRESH_NAMES)


def inspect_arm(record, environment, arm):
    mode, cap, inspection = ARMS[arm]
    started, wall = process_time(), perf_counter()
    bounds = support(record, mode)
    support_cpu = process_time() - started
    tick = process_time()
    reader = ActionSceneReader(environment, record["law"], ROLES["certification"],
        record["incumbent_assignment"], record["candidate_assignment"], LAW["capacity"],
        horizon=LAW["horizon"], execution="global", batch_size=32,
        plan_factory=partial(FusedCanonicalPlan, aligned_assignment=record["movement"]["aligned_assignment"]))
    setup_cpu = process_time() - tick
    rule = certify_action_queries(reader, cap, batch=inspection, evidence_engine="native", **options(bounds))
    cpu, wall = process_time() - started, perf_counter() - wall
    extras = {name: rule.pop(name) for name in ("query_stats", "reader_stats", "exact_filter_fallbacks")}
    if reader.rows != rule["rows"] or reader.rows != rule["rows_read"]:
        raise AssertionError("Evidence rows and acquired full-VM scenes disagree")
    return rule, {"cpu_seconds": cpu, "wall_seconds": wall, "support_cpu_seconds": support_cpu,
        "reader_setup_cpu_seconds": setup_cpu, "generation_cpu_seconds": reader.generation_cpu_seconds,
        "replay_cpu_seconds": reader.replay_cpu_seconds,
        "evidence_control_cpu_seconds": max(0., cpu - support_cpu - setup_cpu - reader.reader_cpu_seconds),
        "generated_rows": reader.rows,
        "unique_acquired_demand_values": reader.rows * LAW["vms"] * LAW["horizon"],
        "generated_demand_values": (reader.rows + reader.stats["fallback_rows_generated"]) * LAW["vms"] * LAW["horizon"],
        "fallback_generated_rows": reader.stats["fallback_rows_generated"], **extras}


def canonical_evidence(record, environment):
    """Every cap row, with an independent native-loss byte check at B=32."""
    old, new = record["incumbent_assignment"], record["candidate_assignment"]
    stream = SceneStream(environment, record["law"], ROLES["certification"])
    query = FusedCanonicalPlan(old, new, LAW["capacity"], LAW["horizon"],
                               aligned_assignment=record["movement"]["aligned_assignment"])
    reference, candidate = [], []
    for _ in range(8192 // 32):
        panel = stream.next(32)
        expected = np.stack([canonical_fresh_losses(panel, assignment, LAW["capacity"])
                             for assignment in (old, new)])
        # This fixed experiment has T=8 and B=32, the audited native contract.
        observed = query._run(panel, (True,) * 4, audit=True)[3]
        if observed.tobytes() != expected.tobytes():
            raise AssertionError("Native fresh losses differ from canonical reference bits")
        reference.append(expected[0])
        candidate.append(expected[1])
    return np.concatenate(reference), np.concatenate(candidate)


def evaluate(record, environment):
    candidate = record["candidate_assignment"] or record["incumbent_assignment"]
    reader = PairedWorkloadReader(environment, record["law"], ROLES["evaluation"],
        record["incumbent_assignment"], candidate, native=True)
    reader(np.arange(16384))
    reference, action = reader.arrays()
    native_old, native_new = np.concatenate(reader.native_reference), np.concatenate(reader.native_action)
    hosts = len(set(record["incumbent_assignment"]))
    means = {
        "incumbent": outcome_means(reference, native_old, np.concatenate(reader.reference_utilization), hosts),
        "candidate": outcome_means(action, native_new, np.concatenate(reader.action_utilization), hosts),
    }
    archived = {"incumbent": record["incumbent_outcomes"],
                "candidate": record.get("candidate_outcomes", record["incumbent_outcomes"])}
    comparison = {side: max(abs(means[side][name] - archived[side][name])
                           for name in means[side]) for side in means}
    return {"rows": reader.rows, "means": means, "paired_diagnostic": diagnostic(reference, action),
            "archived_mean_max_absolute_difference": comparison,
            "scope": "Independent Evaluation estimates; unknown population expectations and false-admission probabilities are not measured by these point estimates."}, {
                "evaluation_reference": reference, "evaluation_candidate": action,
                "native_evaluation_reference": native_old, "native_evaluation_candidate": native_new}


def summarize(records):
    result = {}
    for law in ("all", *SCENE_LAWS):
        rows = [row for row in records if law == "all" or row["law"] == law]
        selected = [row for row in rows if row["status"] == "selected"]
        group = {"environments": len(rows), "selected": len(selected), "fallbacks": len(rows) - len(selected), "arms": {}}
        for arm in ARMS:
            rules = [row["rules"][arm] for row in rows]
            admitted = [row for row in rows if row["rules"][arm]["admitted"]]
            group["arms"][arm] = {"admitted": len(admitted),
                "admitted_fraction_all": len(admitted) / len(rows),
                "admitted_fraction_selected": len(admitted) / len(selected) if selected else None,
                "mean_rows_all": float(np.mean([rule["rows"] for rule in rules])),
                "mean_rows_selected": float(np.mean([row["rules"][arm]["rows"] for row in selected])) if selected else None,
                "mean_cpu_seconds_all": float(np.mean([row["medians"][arm]["cpu_seconds"] for row in rows])),
                "locked_counts": [sum(row["rules"][arm].get("locked", [False] * 4)[e] for row in selected) for e in range(4)],
                "mean_crossing_or_cap_selected": [float(np.mean([
                    row["rules"][arm]["crossing_rows"][e] or ARMS[arm][1] for row in selected])) if selected else None for e in range(4)],
                "mean_terminal_outcomes": {name: float(np.mean([
                    row["evaluation"]["means"]["candidate" if row["rules"][arm]["admitted"] else "incumbent"][name]
                    for row in rows])) for name in rows[0]["evaluation"]["means"]["incumbent"]},
                "admitted_evaluation_mean_target_disagreements": sum(
                    not row["evaluation"]["paired_diagnostic"]["estimated_all_targets_hold"] for row in admitted)}
        result[law] = group
    return {"scope": "Controlled fresh-source fixed-action support and stopping comparison. Evaluation target disagreements are diagnostics, not known false admissions.",
            "all_96_contracts_complete": len(records) == 96, "groups": result}


def main(args):
    if args.repetitions < 1:
        raise ValueError("Positive serial timing repetitions are required")
    original_plan, original_records = load_contracts(args.source)
    plan = {"source": str(args.source), "source_study": original_plan,
         "contracts": len(original_records),
        "selected": 90, "no_action_fallbacks": 6, "search_rerun": False, "arms": ARMS,
        "support_variants": {
            "CARA": "Component-wise directed-cut/conservation raw bound; changed-host Boolean bounds.",
            "NoComponent": "Same directed cuts and changed-host Boolean bounds; one global conservation cap.",
            "NoGraph": "Moved-count raw bound with global conservation cap; changed-host Boolean bounds.",
            "NoBoolean": "Same component-wise raw bound as CARA; unit Boolean bounds.",
            "Global": "Global conservation raw cap and unit Boolean bounds.",
            "Fixed2048/4096/8192": "Same bounds as CARA; inspection only at the stated fixed cap."},
        "repetitions": args.repetitions, "keep_evaluation_arrays": args.keep_evaluation_arrays,
        "population_support": {"demand_interval": [0., 1.],
            "source": "SceneStream.next applies np.clip(values, 0., 1.) to every generated scene; this fixed population bound is not estimated from observed samples."},
        "execution": "All arms regenerate their actual 48-VM iid scene prefixes and use identical fused canonical paired queries plus native directed-evidence evaluation with exact fallback. Complete support, reader construction, generation, replay and evidence CPU are timed; common kernel compilation and immutable environment construction are separate.",
        "verification": "All 8192 certification loss rows per selected action are independently reconstructed and byte-compared with native losses. Every arm's full semantic result is compared with the Decimal-filter/exact-integer-fallback array oracle outside timing. All 16384 independent Evaluation rows are regenerated for each of 96 environments, including six fallbacks.",
        "statistics": "One action, support and thresholds frozen before fresh evidence. Conditional iid laws are synthetic controls; no iid or future-law claim is attached to overlapping historical trace windows. Repetitions reuse prefixes for CPU measurements and are not independent statistical trials."}
    path = args.output / "study_plan.json"
    normalized = json.loads(json.dumps(plan))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError("Existing fixed-action study has different inputs or settings")
    save(path, plan)
    if args.write_plan:
        return
    save(args.output / "common_preparation.json", {"aggregate": prepare_native(), "canonical": prepare_fused_canonical()})
    records = []
    for index, original in enumerate(original_records):
        identity = f'{original["law"]}_{original["seed"]}'
        checkpoint = args.output / "cases" / f"{identity}.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
            if record["action_contract"] != action_contract(original):
                raise ValueError("Checkpoint belongs to another fixed-action study")
            records.append(record)
            continue
        environment = make_environment(original["seed"])
        record = {"law": original["law"], "seed": original["seed"], "status": original["status"],
                  "action_contract": action_contract(original), "rules": {}, "trials": [], "medians": {}}
        saved = {}
        if original["status"] == "selected":
            supports = {mode: support(original, mode) for mode in {value[0] for value in ARMS.values()}}
            if tuple(supports["no_component"]) != tuple(original["gain_bounds"]):
                raise AssertionError("The global-conservation control must retain its predeclared support")
            if any(new > old for new, old in zip(supports["graph"], supports["no_component"])):
                raise AssertionError("Component conservation cannot enlarge the declared support")
            reference, candidate = canonical_evidence(original, environment)
            saved.update(certification_reference=reference, certification_candidate=candidate)
            record["supports"] = supports
            record["archived_graph_semantic_equal"] = {}
            for arm, (mode, cap, inspection) in ARMS.items():
                expected = certify_streaming_interval(lambda indices: (reference[indices], candidate[indices]),
                    cap, 4, batch=inspection, **options(supports[mode]))
                actual, _ = inspect_arm(original, environment, arm)
                if any(actual[name] != expected[name] for name in SEMANTIC):
                    raise AssertionError((identity, arm, "Current canonical evidence oracle mismatch"))
                record["rules"][arm] = actual
                if mode in {"graph", "no_component"}:
                    old_key = "stopped8192" if arm in {"CARA", "NoComponent"} else f"fixed{cap}"
                    record["archived_graph_semantic_equal"][arm] = all(
                        actual[name] == original["rules"][old_key][name] for name in SEMANTIC)
            record["canonical_loss_rows_checked"] = len(reference)
            record["all_current_semantic_oracles_equal"] = True
            arms = tuple(ARMS)
            for repetition in range(args.repetitions):
                shift = (index + repetition) % len(arms)
                order = arms[shift:] + arms[:shift]
                trial = {"order": order, "arms": {}}
                for arm in order:
                    gc.collect()
                    rule, cost = inspect_arm(original, environment, arm)
                    if rule != record["rules"][arm]:
                        raise AssertionError("A timed fresh admission changed its frozen result")
                    trial["arms"][arm] = cost
                record["trials"].append(trial)
            for arm in ARMS:
                record["medians"][arm] = {name: float(np.median([
                    trial["arms"][arm][name] for trial in record["trials"]]))
                    for name in ("cpu_seconds", "wall_seconds", "support_cpu_seconds", "reader_setup_cpu_seconds",
                                 "generation_cpu_seconds", "replay_cpu_seconds", "evidence_control_cpu_seconds")}
        else:
            record["rules"] = {arm: {"admitted": False, "rows": 0, "rows_read": 0,
                "decision": "no_candidate", "reason": "no_candidate"} for arm in ARMS}
            record["medians"] = {arm: {"cpu_seconds": 0., "wall_seconds": 0., "support_cpu_seconds": 0.,
                "reader_setup_cpu_seconds": 0., "generation_cpu_seconds": 0., "replay_cpu_seconds": 0.,
                "evidence_control_cpu_seconds": 0.} for arm in ARMS}
        # All rule decisions precede the independent Evaluation stream.
        evaluation, evaluation_arrays = evaluate(original, environment)
        record["evaluation"] = evaluation
        if args.keep_evaluation_arrays:
            saved.update(evaluation_arrays)
        if saved:
            (args.output / "arrays").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output / "arrays" / f"{identity}.npz", **saved)
        save(checkpoint, record)
        records.append(record)
        save(args.output / "records.json", records)
        print(json.dumps({"completed": len(records), "identity": identity, "status": record["status"],
                          "rows": {arm: record["rules"][arm]["rows"] for arm in ARMS}}), flush=True)
    save(args.output / "records.json", records)
    save(args.output / "summary.json", summarize(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/fresh_actions"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-evaluation-arrays", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    main(parser.parse_args())
