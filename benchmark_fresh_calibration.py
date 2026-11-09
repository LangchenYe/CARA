"""Null calibration of CARA's exact four-coordinate fresh certificate.

These are bounded-loss simulations with analytically zero expected gains,
not cloud workload forecasts. Every run calls the production rational
certificate; no approximate evidence or empirical-null classification is used.
"""
from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import beta

from cara.certification import certify_fresh_mixture


LAWS = ("symmetric", "rare_positive", "single_null")
NAMES = ("loss_0", "loss_1", "loss_2", "loss_3")


def paired_losses(law, rows, rng):
    """Return dyadic losses with an exact zero-mean null coordinate."""
    if law == "rare_positive":
        # (1/16)*(15/16) + (15/16)*(-1/16) = 0 exactly.
        gain = np.where(rng.integers(0, 16, size=rows) == 0, 15 / 16, -1 / 16)
    elif law in ("symmetric", "single_null"):
        gain = np.where(rng.integers(0, 2, size=rows) == 0, 1 / 2, -1 / 2)
    else:
        raise ValueError(f"Unknown calibration law: {law}")
    if law == "single_null":
        gains = np.full((rows, 4), 1 / 4)
        # A failing non-anchor alone must control the joint release.
        gains[:, 1] = gain
    else:
        # Complete endpoint dependence is intentional.
        gains = np.repeat(gain[:, None], 4, axis=1)
    reference, candidate = (1 + gains) / 2, (1 - gains) / 2
    # All generated values and their differences are exactly dyadic.
    return reference, candidate


def one_replicate(law, replica, *, seed, cap, batch, alpha):
    law_index = LAWS.index(law)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(law_index, replica)))
    reference, candidate = paired_losses(law, cap, rng)
    result = certify_fresh_mixture(
        reference, candidate, gain_bounds=(1,) * 4,
        noninferiority_margins=(0,) * 4, anchor=2, min_anchor_gain=0,
        alpha=alpha, batch=batch, metric_names=NAMES,
    )
    return {"law": law, "replica": replica, "false_admission": bool(result["admitted"]),
            "rows": int(result["rows"]), "crossing_rows": result["crossing_rows"]}


def summarize(rows, confidence):
    count = len(rows)
    if not count:
        raise ValueError("A calibration summary needs at least one replicate")
    false = sum(row["false_admission"] for row in rows)
    upper = 1.0 if false == count else float(beta.ppf(confidence, false + 1, count - false))
    return {"replicates": count, "false_admissions": false,
            "false_admission_fraction": false / count,
            "one_sided_clopper_pearson_confidence": confidence,
            "one_sided_clopper_pearson_upper": upper,
            "mean_rows": float(np.mean([row["rows"] for row in rows])),
            "median_rows": float(np.median([row["rows"] for row in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fresh_calibration"))
    parser.add_argument("--laws", nargs="+", choices=LAWS, default=list(LAWS))
    parser.add_argument("--replicates", type=int, default=1024)
    parser.add_argument("--start-replica", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20271113)
    parser.add_argument("--cap", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--alpha", type=Fraction, default=Fraction(1, 20))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--progress", type=int, default=128)
    args = parser.parse_args()
    if (args.replicates < 1 or args.start_replica < 0 or args.seed < 0
            or args.cap < 1 or args.batch < 1 or args.progress < 0
            or not 0 < args.alpha < 1 or not 0 < args.confidence < 1
            or len(set(args.laws)) != len(args.laws)):
        parser.error("Use positive counts, distinct laws, and probabilities strictly between zero and one")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment": "bounded-loss null calibration",
        "laws": args.laws, "replicates_per_law": args.replicates,
        "start_replica": args.start_replica, "seed": args.seed,
        "row_cap": args.cap, "inspection_batch": args.batch,
        "alpha": str(args.alpha), "targets": [0] * 4, "gain_bounds": [1] * 4,
        "anchor": 2, "confidence": args.confidence,
        "certificate": "cara.certification.certify_fresh_mixture",
        "law_definitions": {
            "symmetric": "All four gains equal: +1/2 or -1/2, each with probability 1/2.",
            "rare_positive": "All four gains equal: +15/16 with probability 1/16, otherwise -1/16.",
            "single_null": "Coordinate 1 is symmetric +/-1/2; other coordinates equal +1/4."},
        "inference": "Each independent stream is one replicate; all tested joint strict-gain claims are false."}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    groups = {}
    with (args.output / "replicates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("law", "replica", "false_admission", "rows", "crossing_rows"))
        writer.writeheader()
        for law in args.laws:
            rows, started = [], process_time()
            for offset in range(args.replicates):
                record = one_replicate(law, args.start_replica + offset, seed=args.seed,
                                       cap=args.cap, batch=args.batch, alpha=args.alpha)
                rows.append(record)
                writer.writerow({**record, "crossing_rows": json.dumps(record["crossing_rows"])})
                if args.progress and (offset + 1) % args.progress == 0:
                    handle.flush()
                    print(f"{law}: {offset + 1}/{args.replicates} replicates completed", flush=True)
            groups[law] = summarize(rows, args.confidence)
            groups[law]["run_cpu_seconds"] = process_time() - started
            (args.output / "summary.json").write_text(json.dumps({"protocol": protocol, "groups": groups}, indent=2) + "\n")
            print(json.dumps({"law": law, **groups[law]}), flush=True)


if __name__ == "__main__":
    main()
