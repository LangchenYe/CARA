"""Reproduce CARA's complete experiments with the released configuration.

Examples:
    python reproduce.py all --dry-run
    python reproduce.py assignment physical development
    python reproduce.py quality analysis
    python reproduce.py generality execution queries

Development and both matching calibrations report their results without changing
the released configuration. Development explicitly disables unique physical-map
recovery to reproduce the executor used for the published controller grid. All
study stages execute sequentially.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


STAGES = ("data", "test", "assignment", "physical", "development", "quality", "analysis", "generality",
          "execution", "queries", "fresh", "calibration", "audit", "sql", "figures")
QUALITY_METHODS = ("CARA", "Component-CARA", "Compact-CARA", "Eager-CARA", "Joint-SA", "Anneal", "Risk-SA",
                   "Broad", "Swap", "No-residual", "No-frontier", "Best-seed", "Raw-order")
QUERY_PLANS = ("eager", "compact", "sparse", "no_factorization", "no_late_matching",
               "no_progressive", "no_score", "no_matching_reuse", "compiled_no_unique_maps", "compiled")
EXECUTION_PLANS = ("eager", "compact", "components", "compiled_no_unique_maps",
                   "no_late_matching", "no_forecast", "no_matching_reuse", "compiled")
GENERALITY_PLANS = ("components", "sparse", "static", "replay", "locked", "compiled_no_unique_maps", "compiled")
HOURLY_SOURCES = ("azure2019", "borg2019_d_trace", "borg2019_e_trace", "huawei2025")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", choices=("all", *STAGES), default=["all"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    parser.add_argument("--workers", type=int, help="Override development/quality workers (released defaults: 6/5)")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--fixed-paths", type=Path, help="Override the supplied complete-family execution paths")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.workers is not None and args.workers < 1) or args.repetitions < 1:
        parser.error("Workers and repetitions must be positive")
    if "all" in args.stages and len(args.stages) != 1:
        parser.error("Use all alone or list individual stages")
    selected = list(STAGES) if "all" in args.stages or not args.stages else args.stages
    if len(set(selected)) != len(selected):
        parser.error("Specify each stage once")
    root = args.root.resolve()
    output = args.output.resolve() if args.output.is_absolute() else root / args.output
    config = root / "configs/release/family_frozen.json"
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    python = sys.executable
    logs = output / "logs"

    def run(stage, script, *arguments):
        command = [python, str(root / script), *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / (stage + ".log")).open("a") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

    for stage in selected:
        if stage == "data":
            run(stage, "download_data.py")
        elif stage == "test":
            command = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp=" + str(output / "test_tmp")]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, env=environment, check=True)
        elif stage == "assignment":
            run(stage, "calibrate_assignment.py", "--output", output / "assignment_calibration.json")
        elif stage == "physical":
            run(stage, "calibrate_physical_maps.py", "--output", output / "physical_gate_calibration")
        elif stage == "development":
            destination = output / "parameters"
            workers = args.workers or 6
            development_base = destination / "development_base.json"
            raw = json.loads(config.read_text())
            development_settings = {**raw.get("config", raw), "family_unique_physical": False}
            print("# Development-only base: " + str(development_base)
                  + " (family_unique_physical=False)", flush=True)
            if not args.dry_run:
                if development_base.exists() and json.loads(development_base.read_text()) != development_settings:
                    parser.error("Output contains a different development base; choose a new --output directory")
                destination.mkdir(parents=True, exist_ok=True)
                development_base.write_text(json.dumps(development_settings, indent=2) + "\n")
            run(stage, "family_parameter_study.py", "prepare", "--base-config", development_base,
                "--workers", workers, "--output", destination)
            if args.dry_run:
                specification = json.loads((root / "configs/development/family_parameter_grid.json").read_text())
                identities = [f"w{width:02d}_{engine}_f{int(100 * fraction):02d}" for width in specification["mutable_limit"]
                              for engine in specification["frontier_engine"] for fraction in specification["frontier_fraction"]]
            else:
                grid = json.loads((destination / "grid_plan.json").read_text())
                specification = grid["specification"]
                identities = [arm["id"] for arm in grid["configurations"]]
            for identity in identities:
                run(stage, "run_family_study.py", "--phase", "development", "--config", destination / "configs" / (identity + ".json"),
                    "--methods", specification["method"], "--seeds", *specification["streams"], "--workers", workers,
                    "--output", destination / "runs" / identity)
            run(stage, "family_parameter_study.py", "summarize", "--output", destination)
        elif stage == "quality":
            run(stage, "run_family_study.py", "--phase", "evaluation", "--config", config,
                "--methods", *QUALITY_METHODS, "--workers", args.workers or 5, "--output", output / "quality")
            run(stage, "run_family_study.py", "--phase", "evaluation", "--config", config,
                "--methods", "Scenario-MIP", "--workers", 1, "--output", output / "solver")
        elif stage == "analysis":
            run(stage, "analyze_family_study.py", "--inputs", output / "quality", output / "solver",
                "--output", output / "analysis", "--case", "huawei2025", 297, 0)
        elif stage == "generality":
            run(stage, "benchmark_family_generality.py", "--config", config, "--sources", *HOURLY_SOURCES,
                "--kinds", "cycles", "dispersed", "concentrated", "adaptive", "--spans", 8, 32, 128, "full",
                "--residents", 1, "--plans", *GENERALITY_PLANS, "--families", 8, "--adaptive-families", 16,
                "--width", 8, "--repetitions", args.repetitions, "--memory", "--output", output / "generality")
        elif stage == "execution":
            replay = args.fixed_paths or root / "results/strengthened_execution/frozen_paths.json"
            replay = replay.resolve() if replay.is_absolute() else root / replay
            target = output / "execution/frozen_paths.json"
            if args.fixed_paths and not args.dry_run and not replay.is_file():
                parser.error("The requested fixed-path file does not exist")
            if replay.is_file() or args.dry_run:
                print("Reuse supplied execution paths if present: " + str(replay), flush=True)
                if not args.dry_run:
                    if target.exists() and target.read_bytes() != replay.read_bytes():
                        parser.error("Output contains different frozen paths; choose a new --output directory")
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(replay, target)
            run(stage, "benchmark_family_execution.py", "--phase", "evaluation", "--config", config,
                "--plans", *EXECUTION_PLANS, "--repetitions", args.repetitions, "--output", output / "execution")
        elif stage == "queries":
            run(stage, "benchmark_family_queries.py", "--config", config, "--plans", *QUERY_PLANS,
                "--sources", *HOURLY_SOURCES, "--repetitions", args.repetitions, "--memory", "--output", output / "queries")
        elif stage == "fresh":
            run(stage, "run_frozen_fresh.py", "--source", root / "results/icde2027_shared_streaming",
                "--repetitions", args.repetitions, "--output", output / "fresh")
        elif stage == "calibration":
            run(stage, "benchmark_fresh_calibration.py", "--laws", "symmetric", "rare_positive", "single_null",
                "--replicates", 1024, "--start-replica", 0, "--seed", 20271113, "--cap", 1024,
                "--batch", 32, "--alpha", "1/20", "--confidence", .95, "--output", output / "calibration")
        elif stage == "audit":
            run(stage, "benchmark_family_audit.py", "--source", output / "quality", "--output", output / "audit")
        elif stage == "sql":
            run(stage, "benchmark_family_sql.py", "--output", output / "sql")
        elif stage == "figures":
            run(stage, "draw_strengthened_results.py", "--method", "--parameters", output / "parameters",
                "--quality", output / "analysis", "--case", output / "analysis", "--queries", output / "queries",
                "--generality", output / "generality", "--output", output / "figures")


if __name__ == "__main__":
    main()
