"""Independent whole-family cost calibration for unique physical mappings.

Only integer resident families are used. Every measured invocation constructs a
fresh family and 16 complete rich records. This script never writes a release
configuration or reads benchmark quality outcomes.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import json
from pathlib import Path
from time import process_time_ns

import numpy as np

from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import ResidualFamily
from cara.refinement import migration_profile


KINDS = ("incumbent", "cyclic_strict", "cyclic_weak")


def source(hosts, kind):
    incumbent = canonical_assignment(np.arange(hosts * 8) % hosts)
    seed = list(incumbent)
    moved = {"incumbent": 0, "cyclic_strict": 1, "cyclic_weak": 4}[kind]
    if moved:
        for host, residents in enumerate(host_memberships(incumbent)):
            for vm in residents[-moved:]:
                seed[vm] = (host + 1) % hosts
    seed = canonical_assignment(seed)
    pair = (0, hosts - 1)
    groups = host_memberships(seed)
    mutable = tuple(sorted(groups[pair[0]][:4] + groups[pair[1]][:4]))
    # Eight free residents and nonempty backgrounds yield exactly 256 masks.
    masks = tuple(range(0, 256, 17))
    return incumbent, seed, pair, mutable, masks


def construct(arguments, enabled):
    incumbent, seed, pair, mutable, masks = arguments
    family = ResidualFamily(incumbent, seed, pair, mutable, matching_backend="compiled",
                            use_unique_physical=enabled, unique_physical_min_hosts=0)
    records, full_calls = [], 0
    for mask in masks:
        candidate = family.assignment(mask)
        count = family.movement_count(mask)
        physical = family.unique_physical_profile(mask, candidate) if family.use_unique_physical else None
        if physical is None:
            physical = migration_profile(incumbent, candidate)
            full_calls += 1
        records.append({"assignment": candidate, "movement_count": count, "movement": physical})
    return records, {**family.stats, "physical_matching_calls": full_calls}


def signature(records):
    return tuple((record["assignment"], record["movement_count"],
                  tuple((name, value.hex() if isinstance(value, float) else value)
                        for name, value in sorted(record["movement"].items()))) for record in records)


def run(args):
    if args.repetitions < 3 or args.families_per_trial < 1 or any(size < 3 for size in args.hosts):
        raise ValueError("Use at least three rotations, one family per trial, and at least three hosts")
    protocol = {"study": "independent_integer_family_physical_cost", "hosts": args.hosts,
                "kinds": KINDS, "residents_per_host": 8, "retained_records_per_family": 16,
                "mutable_residents": 8, "repetitions": args.repetitions,
                "families_per_trial": args.families_per_trial,
                "timed_scope": "fresh compiled family descriptor, 16 canonical assignments, count-signature evaluation, and all five rich physical fields",
                "excluded_from_timing": "integer source construction, warmup, garbage collection, and exact rich-field comparison",
                "order": "alternate original and unique-first order by host index, kind index, and repetition",
                "selection_rule": "smallest host count with median original/unique >= 1.10 and every paired ratio > 1.0 for both eligible kinds at that count and every larger tested count",
                "selection_kinds": ["incumbent", "cyclic_strict"],
                "fallback_control": "cyclic_weak is reported but cannot establish a speedup because it intentionally uses the original matcher",
                "reads_formal_benchmark_outcomes": False, "writes_release_configuration": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for host_index, hosts in enumerate(args.hosts):
        for kind_index, kind in enumerate(KINDS):
            arguments = source(hosts, kind)
            expected, baseline_stats = construct(arguments, False)
            warm, unique_stats = construct(arguments, True)
            oracle = signature(expected)
            if signature(warm) != oracle:
                raise AssertionError("The unique path changed a complete rich physical record")
            if any(record["movement_count"] != record["movement"]["count"] for record in expected):
                raise AssertionError("The compiled count disagrees with its original rich matcher")
            trials = []
            for repetition in range(args.repetitions):
                order = (False, True) if (host_index + kind_index + repetition) % 2 == 0 else (True, False)
                elapsed = {}
                for enabled in order:
                    gc.collect()
                    started = process_time_ns()
                    outputs = [construct(arguments, enabled) for _ in range(args.families_per_trial)]
                    elapsed["unique" if enabled else "original"] = (process_time_ns() - started) / (1e9 * args.families_per_trial)
                    if any(signature(records) != oracle for records, _ in outputs):
                        raise AssertionError("A measured invocation changed a complete rich record")
                    del outputs
                trials.append({"order": ["unique" if enabled else "original" for enabled in order],
                               "seconds_per_family": elapsed, "speedup": elapsed["original"] / elapsed["unique"]})
            ratios = [trial["speedup"] for trial in trials]
            row = {"H": hosts, "N": hosts * 8, "kind": kind, "all_rich_fields_exact": True,
                   "original_counters": baseline_stats, "unique_counters": unique_stats,
                   "median_seconds": {name: float(np.median([trial["seconds_per_family"][name] for trial in trials]))
                                      for name in ("original", "unique")},
                   "median_paired_speedup": float(np.median(ratios)), "min_paired_speedup": min(ratios),
                   "stable_gain": bool(np.median(ratios) >= 1.10 and min(ratios) > 1.0), "trials": trials}
            rows.append(row)
            print(json.dumps({name: row[name] for name in ("H", "kind", "median_paired_speedup", "min_paired_speedup", "stable_gain")}), flush=True)
    recommended = next((hosts for hosts in sorted(args.hosts)
                        if all(row["stable_gain"] for row in rows if row["H"] >= hosts
                               and row["kind"] in protocol["selection_kinds"])), None)
    result = {"complete": True, "protocol": protocol, "all_rich_fields_exact": True,
              "recommended_min_hosts": recommended, "rows": rows}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"recommended_min_hosts": recommended, "settings": len(rows), "all_rich_fields_exact": True}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hosts", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--families-per-trial", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
