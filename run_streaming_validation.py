"""Independent-scene evaluation with on-demand paired workload queries."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import json
from pathlib import Path
from time import process_time

import numpy as np

from cara.baseline_anneal import search_anneal
from cara.certification import certify_fresh_mixture, certify_streaming_mixture
from cara.data import burdens_from_loads, host_loads
from cara.migration_envelope import stored_action_gain_bounds
from cara.query_engine import SharedQueryReplay
from cara.refinement import PROFILE_NAMES, migration_profile, profile_from_loads, search_refined, search_swap
from cara.repair import ffd
from cara.scene_stream import SCENE_LAWS, SceneStream, sample_stream
from cara.synthetic import LAW, make_environment


FRESH_NAMES = ("J", "V", "R", "raw_fixed_H")
ROLES = {"fit": 11, "build": 12, "selection": 13, "proposal": 14,
         "certification": 15, "evaluation": 16}
MARGINS = (.01, .01, 0., .001)
METHODS = ("CARA", "Swap", "Anneal", "Joint-SA", "Broad", "Best-seed", "No-batch", "No-footprint")
VALIDATION = [(law, seed) for law, first in zip(SCENE_LAWS, (246001, 247001, 248001))
              for seed in range(first, first + 32)]
DEVELOPMENT = [("complementary", seed) for seed in range(205001, 205004)] + [
    ("bursty_common", seed) for seed in range(205004, 205007)] + [
    ("weak_complementarity", seed) for seed in range(205007, 205009)]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n")
    temporary.replace(path)


def fresh_from_loads(loads, capacity):
    first = burdens_from_loads(loads, capacity)[:, :3]
    raw = np.maximum(loads / capacity - 1., 0.).sum(axis=(0, 2)) / (len(loads) * loads.shape[2])
    return np.column_stack((first, raw))


class PairedWorkloadReader:
    """Generate and replay only requested consecutive rows at fixed batch shape."""

    def __init__(self, environment, law, role, incumbent, candidate, *, batch_size=32,
                 execution="shared", native=False):
        if int(batch_size) != batch_size or batch_size < 1 or execution not in ("shared", "full"):
            raise ValueError("positive replay batch size and shared/full execution required")
        self.stream = SceneStream(environment, law, role)
        self.incumbent, self.candidate = tuple(incumbent), tuple(candidate)
        self.capacity, self.batch_size = LAW["capacity"], int(batch_size)
        self.execution, self.native = execution, bool(native)
        self.reference, self.action, self.native_reference, self.native_action = [], [], [], []
        self.reference_utilization, self.action_utilization = [], []
        self.generation_cpu_seconds = self.replay_cpu_seconds = self.reader_cpu_seconds = 0.
        self.peak_panel_values = 0

    @property
    def rows(self):
        return self.stream.rows

    def __call__(self, indices):
        started = process_time()
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or not np.array_equal(indices, np.arange(self.rows, self.rows + len(indices)))):
            raise ValueError("paired reader requires consecutive, previously unread scene indices")
        reference, action = [], []
        for start in range(0, len(indices), self.batch_size):
            count = min(self.batch_size, len(indices) - start)
            tick = process_time()
            panel = self.stream.next(count)
            self.generation_cpu_seconds += process_time() - tick
            self.peak_panel_values = max(self.peak_panel_values, panel.size)
            # Padding fixes the numerical reduction shape even for a short tail.
            replay_panel = (panel if count == self.batch_size else
                            np.pad(panel, ((0, self.batch_size - count), (0, 0), (0, 0))))
            tick = process_time()
            if self.execution == "shared":
                views = SharedQueryReplay(replay_panel, self.incumbent, self.capacity)
                loads = views.loads_many((self.incumbent, self.candidate))
            else:
                loads = np.stack([host_loads(replay_panel, assignment)
                                  for assignment in (self.incumbent, self.candidate)])
            ref, act = (fresh_from_loads(value, self.capacity)[:count] for value in loads)
            if self.native:
                self.native_reference.append(profile_from_loads(loads[0], self.capacity)[:count])
                self.native_action.append(profile_from_loads(loads[1], self.capacity)[:count])
                self.reference_utilization.append(loads[0].mean(axis=(0, 2))[:count] / self.capacity)
                self.action_utilization.append(loads[1].mean(axis=(0, 2))[:count] / self.capacity)
            self.replay_cpu_seconds += process_time() - tick
            reference.append(ref)
            action.append(act)
            self.reference.append(ref)
            self.action.append(act)
        returned = np.concatenate(reference), np.concatenate(action)
        self.reader_cpu_seconds += process_time() - started
        return returned

    def arrays(self):
        return np.concatenate(self.reference), np.concatenate(self.action)


def inspect(reader, bounds, cap, inspection_batch):
    started = process_time()
    result = certify_streaming_mixture(reader, cap, 4, gain_bounds=bounds,
        noninferiority_margins=MARGINS, anchor=2, min_anchor_gain=.005,
        alpha=.05, batch=inspection_batch, metric_names=FRESH_NAMES)
    total = process_time() - started
    result.update({"total_cpu_seconds": total,
        "generation_cpu_seconds": reader.generation_cpu_seconds,
        "replay_cpu_seconds": reader.replay_cpu_seconds,
        "statistical_cpu_seconds": max(0., total - reader.reader_cpu_seconds),
        "reader_cpu_seconds": reader.reader_cpu_seconds,
        "peak_panel_values": reader.peak_panel_values,
        "generated_scene_rows": reader.rows,
        "generated_demand_values": reader.rows * LAW["vms"] * LAW["horizon"]})
    assert reader.rows == result["rows"]
    return result


def outcome_means(fresh, native, utilization, hosts):
    result = {name: float(fresh[:, i].mean()) for i, name in enumerate(FRESH_NAMES)}
    result.update({"native_" + name: float(native[:, i].mean()) for i, name in enumerate(PROFILE_NAMES)})
    result.update({"hosts": hosts, "fixed_H_utilization": float(utilization.mean())})
    return result


def diagnostic(reference, candidate):
    gains = reference - candidate
    means = gains.mean(axis=0)
    errors = gains.std(axis=0, ddof=1) / np.sqrt(len(gains))
    targets = np.array([-.01, -.01, .005, -.001])
    return {"mean_gains": means.tolist(), "standard_errors": errors.tolist(),
            "pointwise_95_intervals": np.column_stack((means - 1.96 * errors, means + 1.96 * errors)).tolist(),
            "estimated_all_targets_hold": bool(np.all(means > targets))}


def search_method(label, incumbent, forecast, panels, config, seed):
    settings = dict(config)
    if label == "Best-seed":
        settings["seed_schedule"] = "best"
    if label == "No-batch":
        settings["batch_queries"] = False
    if label == "Broad":
        settings.update({"mode": "broad", "frontier_seed": False})
    if label == "Joint-SA":
        settings["anneal_objective"] = "joint"
    engine = search_swap if label == "Swap" else search_anneal if label in ("Anneal", "Joint-SA") else search_refined
    rng = np.random.default_rng(np.random.SeedSequence([seed, ROLES["proposal"], 7319]))
    return engine(incumbent, forecast, panels["build"], panels["selection"],
                  LAW["capacity"], settings, rng)


def one_environment(job):
    law, seed, config, methods, counts, caps, output, _context = job
    output = Path(output)
    study_settings = json.loads(json.dumps({"config": config, "methods": methods, "counts": counts, "fixed_caps": caps}))
    paths = {method: output / method / "records" / f"{law}_{seed}.json" for method in methods}
    if all(path.exists() for path in paths.values()):
        records = {method: json.loads(path.read_text()) for method, path in paths.items()}
        if any(record.get("study_settings") != study_settings for record in records.values()):
            raise ValueError("existing records differ from the frozen study design")
        return records
    environment = make_environment(seed)
    panels = {role: sample_stream(environment, law, counts[role], ROLES[role])
              for role in ("fit", "build", "selection")}
    forecast = panels["fit"].mean(axis=0)
    incumbent = ffd(forecast, LAW["capacity"])
    proposals = {method: search_method(method, incumbent, forecast, panels, config, seed)
                 for method in methods if method != "No-footprint"}
    if "No-footprint" in methods:
        proposals["No-footprint"] = proposals["CARA"]
    records, saved_arrays = {}, {}
    # Every proposal is fixed before certification or independent Evaluation.
    for method in methods:
        proposal, candidate = proposals[method], proposals[method]["selected"]
        movement = None if candidate is None else migration_profile(incumbent, candidate)
        record = {"law": law, "seed": seed, "method": method, "config": config,
            "study_settings": study_settings, "status": "no_candidate" if candidate is None else "selected",
            "incumbent_assignment": incumbent, "candidate_assignment": candidate,
            "movement": movement, "search": proposal["stats"], "rules": {}, "counts": counts,
            "action_source": "CARA" if method == "No-footprint" else method}
        saved = {**environment.parameters(), "incumbent_assignment": incumbent}
        if candidate is not None:
            tick = process_time()
            footprint = stored_action_gain_bounds(incumbent, movement["aligned_assignment"],
                LAW["capacity"], horizon=LAW["horizon"])
            bounds = footprint["gain_bounds"]
            if method == "No-footprint":
                bounds = (1., 1., 1., bounds[3])
            record.update({"gain_bounds": bounds, "bound_cpu_seconds": process_time() - tick,
                "touched_hosts": footprint["touched_hosts"], "touched_fraction": footprint["touched_fraction"]})
            reader = PairedWorkloadReader(environment, law, ROLES["certification"], incumbent, candidate,
                execution=config.get("certification_views", "shared"))
            record["rules"]["stopped8192"] = inspect(reader, bounds, counts["certification"], 32)
            stopped_reference, stopped_candidate = reader.arrays()
            for cap in caps:
                reader = PairedWorkloadReader(environment, law, ROLES["certification"], incumbent, candidate,
                    execution=config.get("certification_views", "shared"))
                record["rules"][f"fixed{cap}"] = inspect(reader, bounds, cap, cap)
                if cap == counts["certification"]:
                    reference, action = reader.arrays()
                    assert np.array_equal(reference[:len(stopped_reference)], stopped_reference)
                    assert np.array_equal(action[:len(stopped_candidate)], stopped_candidate)
                    saved.update({"candidate_assignment": candidate,
                        "certification_reference": reference, "certification_candidate": action,
                        "gain_bounds": bounds})
        else:
            for rule in ("stopped8192",) + tuple(f"fixed{cap}" for cap in caps):
                record["rules"][rule] = {"admitted": False, "rows": 0, "rows_read": 0,
                    "generated_scene_rows": 0, "generated_demand_values": 0,
                    "reason": "no_candidate", "decision": "no_candidate",
                    "total_cpu_seconds": 0., "generation_cpu_seconds": 0.,
                    "replay_cpu_seconds": 0., "statistical_cpu_seconds": 0.,
                    "reader_cpu_seconds": 0., "peak_panel_values": 0}
        records[method], saved_arrays[method] = record, saved
    # Evaluation is a distinct role and cannot affect any proposal or certificate.
    for method in methods:
        record, saved = records[method], saved_arrays[method]
        candidate = record["candidate_assignment"]
        reader = PairedWorkloadReader(environment, law, ROLES["evaluation"], incumbent,
            incumbent if candidate is None else candidate,
            execution=config.get("certification_views", "shared"), native=True)
        reader(np.arange(counts["evaluation"]))
        reference, action = reader.arrays()
        native_reference, native_action = np.concatenate(reader.native_reference), np.concatenate(reader.native_action)
        incumbent_means = outcome_means(reference, native_reference,
            np.concatenate(reader.reference_utilization), len(set(incumbent)))
        candidate_means = outcome_means(action, native_action,
            np.concatenate(reader.action_utilization), len(set(incumbent)))
        record["incumbent_outcomes"] = incumbent_means
        record["evaluation_generation_cpu_seconds"] = reader.generation_cpu_seconds
        record["evaluation_replay_cpu_seconds"] = reader.replay_cpu_seconds
        if candidate is not None:
            record["candidate_outcomes"] = candidate_means
            record["candidate_truth_approximation"] = diagnostic(reference, action)
        saved.update({"evaluation_reference": reference, "native_evaluation_reference": native_reference})
        if candidate is not None:
            saved.update({"evaluation_candidate": action, "native_evaluation_candidate": native_action})
        for rule in record["rules"].values():
            rule["terminal_outcomes"] = candidate_means if rule["admitted"] else incumbent_means
            rule["terminal_migrations"] = record["movement"]["count"] if rule["admitted"] else 0
            rule["terminal_truth_approximation"] = diagnostic(reference, action if rule["admitted"] else reference)
        path = output / method / "arrays" / f"{law}_{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **saved)
        save(paths[method], record)
    return records


def summarize(records):
    report = {}
    for law in ("all",) + tuple(SCENE_LAWS):
        subset = [record for record in records if law == "all" or record["law"] == law]
        if not subset:
            continue
        result = {"environments": len(subset),
                  "selected": sum(row["status"] == "selected" for row in subset), "rules": {}}
        for label in subset[0]["rules"]:
            rules = [record["rules"][label] for record in subset]
            result["rules"][label] = {"admitted": sum(rule["admitted"] for rule in rules),
                "mean_rows_all_environments": float(np.mean([rule["rows"] for rule in rules])),
                "mean_terminal_migrations": float(np.mean([rule["terminal_migrations"] for rule in rules])),
                "mean_terminal_outcomes": {name: float(np.mean([rule["terminal_outcomes"][name] for rule in rules]))
                                           for name in rules[0]["terminal_outcomes"]},
                "accepted_estimated_target_failures": sum(rule["admitted"] and not rule["terminal_truth_approximation"]["estimated_all_targets_hold"] for rule in rules),
                **{"mean_" + name: float(np.mean([rule[name] for rule in rules]))
                   for name in ("total_cpu_seconds", "generation_cpu_seconds", "replay_cpu_seconds",
                                "statistical_cpu_seconds", "generated_demand_values", "peak_panel_values")}}
        report[law] = result
    return report


def study_plan(config, methods, phase, caps):
    environments = DEVELOPMENT if phase == "development" else VALIDATION
    counts = {"fit": 128, "build": 64, "selection": 128, "certification": 8192, "evaluation": 16384}
    if phase == "smoke":
        environments, counts = DEVELOPMENT[:1], {"fit": 64, "build": 32, "selection": 64,
                                                  "certification": 512, "evaluation": 512}
        caps = (512,)
    if max(caps) != counts["certification"] or any(cap < 1 or cap % 32 for cap in caps):
        raise ValueError("fixed caps must be positive multiples of 32 and include the primary cap")
    return {"phase": "validation" if phase == "plan" else phase,
        "validation_environments": [{"law": law, "seed": seed} for law, seed in environments],
        "methods": methods, "search_config": config, "base_environment_law": LAW,
        "scene_laws": SCENE_LAWS, "roles": ROLES, "counts": counts,
        "fresh_metrics": FRESH_NAMES, "native_metrics": PROFILE_NAMES,
        "margins": MARGINS, "anchor": "R", "minimum_anchor_gain": .005, "alpha": .05,
        "primary_cap": counts["certification"], "inspection_batch": 32, "replay_batch": 32,
        "fixed_caps": caps,
        "sampling": "Separate seeded streams for common, group, noise, burst occurrence and magnitude. Consecutive scene prefixes are invariant to batch partition. Roles are distinct; scenes are conditionally iid.",
        "timing": "Every inspected paired batch is generated and replayed on demand. Fixed caps use the same 32-scene replay shape but inspect evidence only at the cap. Total CPU includes acquisition, replay and exact statistical arithmetic; action bounds are separately timed.",
        "controls": "No-footprint reuses CARA's frozen candidate and changes only V/R supports to 1; other named controls run their own predeclared search before certification."}
