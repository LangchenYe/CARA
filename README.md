# CARA

Code and experiments for **CARA: Exact Repair Queries with One-Action Certification for Cloud Consolidation**.

CARA evaluates related placement repairs by sharing workload aggregates and migration constraints across a repair family. It compiles a family-wide matching core, reuses assignment updates, and constructs full profiles and physical mappings when they are needed by the retained candidate pool. The selected action is checked separately by historical or fresh-scene certification.

## Setup

The experiments use Python 3.12 on Linux and a C99 compiler, available as `cc` or through `CC`. No GPU is needed. Native kernels compile on first use and are cached in the system temporary directory.

```sh
git clone https://github.com/LangchenYe/CARA.git
cd CARA
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python download_data.py
python -m pytest -q
```

Use `requirements.txt` if you do not need pytest. The downloader retrieves eight public CloudCons Arrow files, about 128 MB. The raw data are downloaded separately. Sources and terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); data options are described in [data/README.md](data/README.md).

## Run the experiments

Run commands from the repository root. The controller settings are in [family_frozen.json](configs/release/family_frozen.json), and the evaluation protocol is in [family_protocol.json](configs/release/family_protocol.json).

```sh
# Inspect the commands, then run all stages.
python reproduce.py all --dry-run
python reproduce.py all
```

The driver runs stages in sequence, limits numerical libraries to one thread per worker, and runs Scenario-MIP separately. It writes outputs and logs to `reproduced/`. Run timing experiments without competing numerical jobs.

Stages can also be selected individually:

```sh
python reproduce.py data test
python reproduce.py assignment physical development
python reproduce.py quality analysis
python reproduce.py generality execution queries
python reproduce.py fresh calibration audit sql
python reproduce.py figures
```

`--output` chooses another output directory. `--workers` sets the development and quality worker counts; their defaults are six and five. Timing studies use three serial rotated repetitions, controlled by `--repetitions`.

The development stage evaluates twelve controller configurations on 24 contexts each. It disables unique physical-map recovery to match the executor used in the parameter study. Quality evaluation uses the final configuration: residual width 12, a half-budget swap frontier, direct matching through core size 64, and physical-map recovery from 16 hosts. Calibration stages report their results without changing these settings.

Quality evaluation covers 354 cohort/origin/search-stream contexts, with 14 methods and the FFD reference. The primary and daily views contain 300 and 78 decisions, with 24 shared contexts. The generality study covers 19 settings and 184 repair families. Fixed-operation and hourly-query studies compare eight and ten execution plans, respectively. Fresh certification replays 96 supplied action contracts, including six no-action cases.

## Included results

The `results/` directory contains the recorded observations and study protocols:

| Path under `results/` | Contents |
| --- | --- |
| `strengthened_parameters/` | Development grid, records, and rankings |
| `strengthened_quality/` | Thirteen heuristic and ablation policies on all 354 contexts |
| `strengthened_solver/` | Scenario-MIP on the same contexts |
| `strengthened_quality_maps/` | Reruns of CARA, No-frontier, Best-seed, and Raw-order with unique physical-map recovery |
| `strengthened_analysis/` | Combined metrics, paired differences, and the Huawei case study |
| `strengthened_generality/` | Seven execution plans across all 19 settings |
| `strengthened_execution/` | Fixed-operation comparison and replay paths |
| `strengthened_queries/` | Hourly-source query comparison |
| `strengthened_audit/` | Historical-admission comparison |
| `assignment_calibration.json` | Assignment-kernel calibration |
| `physical_gate_calibration/` | Physical-map recovery calibration |
| `revision_fresh/` | Fresh-action certification |
| `revision_null_calibration/` | Null-calibration protocol and 3,072 replicate outcomes |
| `revision_sql/` | SQL and canonical-query controls |
| `icde2027_shared_streaming/` | Action inputs for fresh certification |

To reconstruct the supplied quality comparison, combine the original quality and solver studies with the four-policy rerun:

```sh
python analyze_family_study.py \
  --inputs results/strengthened_quality results/strengthened_solver \
  --replacement results/strengthened_quality_maps \
  --output reproduced/analysis --case huawei2025 297 0
```

A new `reproduce.py quality analysis` run evaluates all heuristic policies with the final configuration and combines them with a separate solver run; it does not need a replacement input. The audit stage normally reads these new quality results. To audit the supplied final CARA actions instead:

```sh
python benchmark_family_audit.py \
  --source results/strengthened_quality_maps \
  --output reproduced/supplied_audit
```

The execution stage reuses `results/strengthened_execution/frozen_paths.json`. Use `--fixed-paths` to select another replay file.

## Figures and derivations

Plot the included results without rerunning the experiments:

```sh
python draw_strengthened_results.py --method \
  --parameters results/strengthened_parameters \
  --quality results/strengthened_analysis \
  --case results/strengthened_analysis \
  --queries results/strengthened_queries \
  --generality results/strengthened_generality \
  --output reproduced/figures
```

The plotting script also exports the plotted observations as JSON and CSV. Repetition ranges describe the recorded timings; they are not confidence intervals.

Mathematical derivations are provided as [PDF](theory/derivations.pdf) and [LaTeX source](theory/derivations.tex).

## Code layout

| Files | Purpose |
| --- | --- |
| `cara/matching_core.py` | Family-wide fixed-edge elimination and support decomposition |
| `cara/compiled_matching.py`, `cara/assignment_replay.py`, `cara/kernels/assignment.c` | Residual-core matching and two-column assignment replay |
| `cara/family_certificates.py`, `cara/family_score_bounds.py` | Family certificates, score bounds, and physical-map recovery |
| `cara/query_engine.py`, `cara/kernels/` | Shared query evaluation and native kernels |
| `cara/refinement.py` | Repair controller and candidate retention |
| `cara/certification.py`, `cara/migration_envelope.py`, `cara/scene_stream.py` | One-action certification and action-dependent supports |
| `benchmark_*.py`, `run_*.py` | Experiment entry points |
| `tests/` | Matching, query, controller, and certificate tests |

Source code is licensed under the [MIT License](LICENSE). Third-party data and dependencies retain their own licenses.
