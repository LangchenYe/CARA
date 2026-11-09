"""Select an assignment-kernel crossover on synthetic integer families only."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import json
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import linear_sum_assignment

from cara.assignment_replay import NativeTwoColumnReplay, prepare_assignment_replay


def family(size, kind):
    if kind == "cycle":
        weights = np.zeros((size, size), dtype=np.int64)
        rows = np.arange(size)
        weights[rows, rows] = 7
        weights[rows, (rows + 1) % size] = 1
    else:
        rng = np.random.default_rng(791104 + size)
        weights = rng.integers(0, 101, size=(size, size), dtype=np.int64)
        rows = np.arange(size)
        weights[rows, rows] = np.maximum(weights[rows, rows], 4)
    columns = np.asarray([size // 3, 2 * size // 3], dtype=np.int64)
    fixed = weights[:, columns].T.copy()
    fixed[0, columns[0]] -= 4
    fixed[1, columns[1]] -= 4
    members = []
    for first in range(5):
        for second in range(5):
            member = fixed.copy()
            member[0, columns[0]] += first
            member[1, columns[0]] += 4 - first
            member[0, columns[1]] += second
            member[1, columns[1]] += 4 - second
            members.append(member)
    return weights, columns, members


def measure(weights, columns, members, plan):
    started = process_time()
    outputs = []
    if plan == "replay":
        compiled = NativeTwoColumnReplay(weights)
        for member in members:
            outputs.append(compiled.replay(columns, member)[0])
    else:
        for member in members:
            matrix = weights.copy()
            matrix[:, columns] = member.T
            rows, selected = linear_sum_assignment(matrix, maximize=True)
            outputs.append(int(matrix[rows, selected].sum()))
    return outputs, process_time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reproduced/assignment_calibration.json"))
    args = parser.parse_args()
    prepare_assignment_replay()
    sizes = [8, 16, 32, 64, 128, 256]
    records = []
    for kind_index, kind in enumerate(("cycle", "dense")):
        for size_index, size in enumerate(sizes):
            weights, columns, members = family(size, kind)
            expected, _ = measure(weights, columns, members, "direct")
            trials = []
            for repetition in range(3):
                shift = (kind_index + size_index + repetition) % 2
                order = ["direct", "replay"]
                order = order[shift:] + order[:shift]
                values = {}
                for plan in order:
                    outputs, cpu = measure(weights, columns, members, plan)
                    assert outputs == expected
                    values[plan] = cpu
                trials.append({"order": order, "cpu_seconds": values})
            medians = {plan: float(np.median([trial["cpu_seconds"][plan] for trial in trials]))
                       for plan in ("direct", "replay")}
            records.append({"kind": kind, "size": size, "signatures": 25,
                            "medians": medians, "trials": trials,
                            "replay_speedup": medians["direct"] / medians["replay"],
                            "all_exact": True})
    # A monotone conservative crossover must win both kinds at every tested
    # size above it. The <=2 algebraic branch remains separate.
    threshold = None
    for cutoff in sizes:
        later = [record for record in records if record["size"] > cutoff]
        if later and all(record["replay_speedup"] >= 1 for record in later):
            threshold = cutoff
            break
    report = {"scope": "Independent synthetic kernel calibration; no benchmark rows or quality outcomes used.",
              "protocol": "25 complete origin-count signatures per family; 3 serial rotated repetitions; process CPU; native base compilation, state setup, validation and all replay calls included; C compilation prewarmed; direct includes candidate matrix construction.",
              "sizes": sizes, "kinds": ["cycle", "dense"], "seed": 791104,
              "direct_up_to_size": threshold, "records": records}
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"direct_up_to_size": threshold,
          "medians": [{"kind": row["kind"], "n": row["size"], **row["medians"],
                       "speedup": row["replay_speedup"]} for row in records]}, indent=2))


if __name__ == "__main__":
    main()
