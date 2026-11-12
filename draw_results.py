"""Draw the paper figures only from complete recorded studies.

Generality accepts the raw study directory, its records.json, or the complete
normalized JSON/CSV exported by this script. The CSV has a matching metadata
file so a missing setting, method, or repetition cannot silently disappear.
An explicit layout preview contains no experimental values and is PNG only.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
SCHEMA = "cara-generality"
SOURCES = ("azure2019", "borg2019_d", "borg2019_e", "huawei2025")
QUERY_SOURCES = ("azure2019", "borg2019_d_trace", "borg2019_e_trace", "huawei2025")
SOURCE_NAMES = dict(zip(SOURCES, ("Azure", "Borg-d", "Borg-e", "Huawei")))
SOURCE_NAMES.update(zip(QUERY_SOURCES, ("Azure", "Borg-d", "Borg-e", "Huawei")))
SHORT_NAMES = dict(zip(QUERY_SOURCES, ("AZ", "BD", "BE", "HW")))
PLANS = ("components", "sparse", "static", "replay", "locked", "compiled_no_unique_maps", "no_bootstrap", "compiled")
APPEARANCE = {
    "components": ("Component-CARA", "#697681", "s"),
    "sparse": ("Sparse-JV", "#c07437", "^"),
    "static": ("Static-Reduction", "#8b6a98", "D"),
    "replay": ("Dynamic replay", "#3166a0", "o"),
    "locked": ("Core/direct", "#849547", "v"),
    "compiled_no_unique_maps": ("No unique maps", "#b35665", "X"),
    "no_bootstrap": ("Delayed acquisition", "#9a7a42", "P"),
    "compiled": ("CARA", "#127b79", "o"),
}
BLUE, TEAL, GREY, ORANGE = "#3166a0", "#127b79", "#697681", "#c07437"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3,
                     "axes.labelsize": 7.4, "axes.titlesize": 8.1,
                     "xtick.labelsize": 6.7, "ytick.labelsize": 6.8,
                     "legend.fontsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.linewidth": .65})


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (name + ".pdf"))
    fig.savefig(output / (name + ".png"), dpi=240)
    plt.close(fig)


def identity(setting):
    return tuple(setting.get(key) for key in ("source", "kind", "span", "residents"))


def three(values):
    return {name: float(function(values)) for name, function in
            (("min", np.min), ("median", np.median), ("max", np.max))}


def normalize(raw, plan, summary):
    settings = []
    for row in raw:
        if not row.get("complete") or not all(row.get(name) for name in
            ("all_profiles_exact", "all_family_pools_exact", "all_exclusion_leaves_checked", "all_adaptive_seeds_exact")):
            raise ValueError("Every plotted setting must have completed its exact-output checks")
        families = row["structure"]["families"]
        if len(families) != row["complete_families"]:
            raise ValueError("Incomplete structural family records")
        fields = {"component_rows": [family["component_rows_before_peeling"] for family in families],
                  "core_rows": [family["square_core_rows_after_peeling"] for family in families],
                  "active_core_rows": [family["active_rows_after_peeling"] for family in families],
                  "bound_rows": [family["migration_core_bound"]["bound_kernel_rows"] for family in families],
                  "locked_edges": [family["locked_edges"] for family in families]}
        samples = {name: [float(trial["plans"][name]["cpu_seconds"]) for trial in row["trials"]]
                   for name in PLANS}
        for name, values in samples.items():
            if not values or not np.isclose(np.median(values), row["median_seconds"][name]["cpu_seconds"], rtol=1e-12, atol=1e-12):
                raise ValueError("Recorded CPU medians do not match the complete trials")
        settings.append({"setting": row["setting"], **{key: row[key] for key in ("N", "H", "B", "T", "complete_families")},
                         "structure": {name: three(values) for name, values in fields.items()}, "samples": samples})
    keys = ("study", "sources", "kinds", "spans", "span_all_sources", "residents_per_host", "plans",
            "families", "adaptive_families", "width", "repetitions")
    return {"schema": SCHEMA, "study_plan": {key: plan[key] for key in keys},
            "summary": {key: summary[key] for key in ("complete", "settings", "all_exact", "all_core_bounds_hold")},
            "structure_definitions": {
                "component_rows": "Rows in all active support components before fixed-edge elimination.",
                "core_rows": "Rows in the global square core after fixed-edge elimination, including unchanged residual components.",
                "active_core_rows": "Rows only in residual components that contain an edited column; these components are updated per signature."},
            "settings": settings}


def validate(data, expected_settings):
    if data.get("schema") != SCHEMA:
        raise ValueError("Unknown normalized generality schema")
    plan, summary, rows = data["study_plan"], data["summary"], data["settings"]
    if not all(summary.get(key) for key in ("complete", "all_exact", "all_core_bounds_hold")):
        raise ValueError("Only complete, verified generality studies can produce a paper figure")
    if len(rows) != summary["settings"] or len(rows) != expected_settings:
        raise ValueError(f"Expected all {expected_settings} settings; received {len(rows)}")
    if set(plan["sources"]) != set(QUERY_SOURCES) or set(plan["plans"]) != set(PLANS):
        raise ValueError("The paper comparison requires every source and all eight prescribed plans")
    if set(plan["kinds"]) != {"cycles", "dispersed", "concentrated", "adaptive"} or plan["repetitions"] < 1:
        raise ValueError("The full timed seed-kind grid is required")
    hosts = {}
    for row in rows:
        source = row["setting"]["source"]
        if hosts.setdefault(source, row["H"]) != row["H"]:
            raise ValueError("A source changed its complete host count")
        if set(row["samples"]) != set(PLANS):
            raise ValueError("A method is missing from a setting")
        for values in row["samples"].values():
            if len(values) != plan["repetitions"] or not all(np.isfinite(value) and value > 0 for value in values):
                raise ValueError("Every method needs all positive finite trial times")
        for bounds in row["structure"].values():
            if not 0 <= bounds["min"] <= bounds["median"] <= bounds["max"]:
                raise ValueError("Invalid structural range")
        if min(row["structure"][name]["min"] for name in ("component_rows", "core_rows", "active_core_rows")) <= 0:
            raise ValueError("Matching-row logarithms require positive recorded dimensions")
    expected = set()
    for source in plan["sources"]:
        spans = plan["spans"] if source == "azure2019" or plan["span_all_sources"] else ["full"]
        for span in spans:
            span = hosts[source] if span == "full" else min(hosts[source], int(span))
            expected.add((source, "cycles", span, plan["residents_per_host"]))
        expected.update((source, kind, None, None) for kind in ("dispersed", "concentrated", "adaptive"))
    actual = [identity(row["setting"]) for row in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Settings do not equal the complete declared source/seed grid")
    return data


def from_csv(source):
    data = read(source.with_suffix(".metadata.json"))
    grouped = {}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            setting = {"source": row["source"], "kind": row["kind"]}
            if row["span"]:
                setting.update(span=int(row["span"]), residents=int(row["residents"]))
            structural = {name: {measure: float(row[name + "_" + measure]) for measure in ("min", "median", "max")}
                          for name in ("component_rows", "core_rows", "active_core_rows", "bound_rows", "locked_edges")}
            header = {"setting": setting, **{key: int(row[key]) for key in ("N", "H", "B", "T", "complete_families")},
                      "structure": structural}
            group = grouped.setdefault(identity(setting), {"header": header, "samples": {name: {} for name in PLANS}})
            if group["header"] != header or row["plan"] not in PLANS:
                raise ValueError("Inconsistent setting metadata or method in the CSV")
            samples = group["samples"][row["plan"]]
            trial = int(row["trial_index"])
            if trial in samples:
                raise ValueError("Repeated method/trial row in the CSV")
            samples[trial] = float(row["cpu_seconds"])
    data["settings"] = []
    for group in grouped.values():
        samples = {}
        for name, values in group["samples"].items():
            if sorted(values) != list(range(data["study_plan"]["repetitions"])):
                raise ValueError("The CSV omits a declared method trial")
            samples[name] = [values[index] for index in sorted(values)]
        data["settings"].append({**group["header"], "samples": samples})
    return data


def load_generality(source, expected_settings=19):
    source = Path(source)
    if source.suffix == ".csv":
        data = from_csv(source)
    elif source.is_file():
        data = read(source)
        if isinstance(data, list):
            data = normalize(data, read(source.parent / "study_plan.json"), read(source.parent / "summary.json"))
    else:
        data = normalize(read(source / "records.json"), read(source / "study_plan.json"), read(source / "summary.json"))
    return validate(data, expected_settings)


def export_generality(data, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "generality_plot_data.json", data)
    write_json(output / "generality_plot_data.metadata.json", {key: value for key, value in data.items() if key != "settings"})
    records = []
    for setting in data["settings"]:
        header = {key: setting["setting"].get(key, "") for key in ("source", "kind", "span", "residents")}
        header.update({key: setting[key] for key in ("N", "H", "B", "T", "complete_families")})
        header.update({name + "_" + measure: value for name, triple in setting["structure"].items() for measure, value in triple.items()})
        for name in PLANS:
            records.extend({**header, "plan": name, "trial_index": index, "cpu_seconds": value}
                           for index, value in enumerate(setting["samples"][name]))
    with (output / "generality_plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def partitions(rows):
    cycles = sorted((row for row in rows if row["setting"]["kind"] == "cycles"),
                    key=lambda row: (QUERY_SOURCES.index(row["setting"]["source"]), row["setting"]["span"]))
    mixed = sorted((row for row in rows if row["setting"]["kind"] in ("dispersed", "concentrated")),
                   key=lambda row: (("dispersed", "concentrated").index(row["setting"]["kind"]), QUERY_SOURCES.index(row["setting"]["source"])))
    adaptive = sorted((row for row in rows if row["setting"]["kind"] == "adaptive"),
                      key=lambda row: QUERY_SOURCES.index(row["setting"]["source"]))
    return cycles, mixed, adaptive


def tick_labels(rows):
    result = []
    for row in rows:
        setting = row["setting"]
        if setting["kind"] == "cycles":
            detail = "full" if setting["span"] == row["H"] else str(setting["span"])
        else:
            detail = {"concentrated": "conc.", "dispersed": "disp.", "adaptive": ""}[setting["kind"]]
        result.append(SHORT_NAMES[setting["source"]] + ("\n" + detail if detail else ""))
    return result


def axes_layout(groups):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.0), sharex="col",
                             gridspec_kw={"height_ratios": (1.13, 1.), "width_ratios": tuple(map(len, groups))})
    fig.subplots_adjust(left=.068, right=.995, bottom=.155, top=.827, hspace=.46, wspace=.27)
    for column, rows in enumerate(groups):
        for ax in axes[:, column]:
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#dfe4e7", linewidth=.45)
            ax.tick_params(axis="both", length=2.5, pad=2)
            ax.set_xlim(-.53, len(rows) - .47)
        axes[1, column].set_xticks(range(len(rows)), tick_labels(rows))
        axes[0, column].tick_params(axis="x", bottom=False, labelbottom=False)
    for ax, title in zip(axes[0], ("(a) Cyclic seeds", "(b) Dispersed and concentrated", "(c) Adaptive")):
        ax.set_title(title, loc="left", pad=4)
    for ax, title in zip(axes[1], ("(d) Matching core", "(e) Matching core", "(f) Matching core")):
        ax.set_title(title, loc="left", pad=4)
    axes[0, 0].set_ylabel("CPU / Component-CARA")
    axes[1, 0].set_ylabel("Matching rows")
    handles = [Line2D([], [], color=APPEARANCE[name][1], marker=APPEARANCE[name][2], linestyle="none",
                      markerfacecolor="white" if name == "replay" else APPEARANCE[name][1],
                      markersize=4.2, markeredgewidth=.8, label=APPEARANCE[name][0]) for name in PLANS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .995), ncol=4,
               frameon=False, handletextpad=.35, columnspacing=1.0, borderaxespad=0.)
    core_handles = [Line2D([], [], color=GREY, marker="o", markerfacecolor="white", linestyle="-", linewidth=.65,
                           markersize=3.7, label="Active support"),
                    Line2D([], [], color=TEAL, marker="o", linestyle="-", linewidth=.65,
                           markersize=3.7, label="Active core"),
                    Line2D([], [], color=ORANGE, marker="^", markerfacecolor="white", linestyle="none",
                           markersize=3.7, label="Migration bound"),
                    Line2D([], [], linestyle="none", label="AZ: Azure  BD: Borg-d  BE: Borg-e  HW: Huawei")]
    fig.legend(handles=core_handles, loc="lower center", bbox_to_anchor=(.53, -.004), ncol=4,
               frameon=False, columnspacing=.8, handlelength=1.25, borderaxespad=0., fontsize=6.4)
    return fig, axes


def log_ticks(ax, low, high, choices):
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator([value for value in choices if low <= value <= high]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=0)


def generality(data, output):
    groups = partitions(data["settings"])
    fig, axes = axes_layout(groups)
    ratios = [value / np.median(row["samples"]["components"]) for row in data["settings"]
              for values in row["samples"].values() for value in values]
    cpu_low, cpu_high = min(ratios) / 1.28, max(ratios) * 1.28
    row_high = max(row["structure"][name]["max"] for row in data["settings"] for name in ("component_rows", "active_core_rows")) * 1.45
    offsets = np.linspace(-.38, .38, len(PLANS))
    for column, rows in enumerate(groups):
        cpu, core = axes[:, column]
        log_ticks(cpu, cpu_low, cpu_high, (.1, .2, .3, .5, .75, 1., 1.5, 2., 3., 5., 10., 30., 100.))
        row_low = min(row["structure"][name]["min"] for row in data["settings"] for name in ("component_rows", "active_core_rows"))
        log_ticks(core, row_low / 1.35, row_high, (min(2, row_low), 10, 100, 1000, 10000))
        cpu.axhline(1., color=GREY, linewidth=.7, linestyle=(0, (3, 3)), zorder=1)
        for index, row in enumerate(rows):
            reference = np.median(row["samples"]["components"])
            for offset, name in zip(offsets, PLANS):
                values = np.asarray(row["samples"][name]) / reference
                middle = np.median(values)
                label, color, marker = APPEARANCE[name]
                cpu.errorbar(index + offset, middle, yerr=[[middle - values.min()], [values.max() - middle]],
                             marker=marker, color=color, markerfacecolor="white" if name == "replay" else color,
                             markeredgewidth=.75, markersize=3.1 if name == "compiled" else 2.7,
                             linestyle="none", elinewidth=.65, capsize=1.3, capthick=.65, zorder=4 if name == "compiled" else 3)
            before, after = row["structure"]["component_rows"], row["structure"]["active_core_rows"]
            core.plot([index, index], [before["median"], after["median"]],
                      color="#b4bdc4", linewidth=1., zorder=1)
            for values, color, face, offset in ((before, GREY, "white", -.09), (after, TEAL, TEAL, .09)):
                middle = values["median"]
                core.errorbar(index + offset, middle,
                              yerr=[[middle - values["min"]], [values["max"] - middle]],
                              color=color, marker="o", markerfacecolor=face, markeredgewidth=.75,
                              markersize=3.7, elinewidth=.8, capsize=1.4, capthick=.8, linestyle="none", zorder=3)
            bound = row["structure"]["bound_rows"]
            middle = bound["median"]
            core.errorbar(index + .24, middle,
                          yerr=[[middle - bound["min"]], [bound["max"] - middle]],
                          color=ORANGE, marker="^", markerfacecolor="white", markeredgewidth=.75,
                          markersize=3.7, elinewidth=.75, capsize=1.3, capthick=.75, linestyle="none", zorder=4)
        if column == 1:
            split = sum(row["setting"]["kind"] == "dispersed" for row in rows) - .5
            for ax in (cpu, core):
                ax.axvline(split, color="#c7cfd4", linewidth=.65, linestyle=":")
    save(fig, output, "generality")
    export_generality(data, output)


def layout_preview(output):
    rows = [{"setting": {"source": source, "kind": "cycles", "span": span}, "H": 1350 if source == "azure2019" else 1}
            for source in QUERY_SOURCES for span in ((8, 32, 128, 1350) if source == "azure2019" else (1,))]
    rows += [{"setting": {"source": source, "kind": kind}, "H": 1}
             for kind in ("dispersed", "concentrated", "adaptive") for source in QUERY_SOURCES]
    fig, axes = axes_layout(partitions(rows))
    for ax in axes.flat:
        ax.set_yticks([])
        ax.text(.5, .45, "Awaiting complete\nrecorded results", transform=ax.transAxes,
                ha="center", va="center", color="#89969f", fontsize=7)
    fig.text(.5, .505, "Layout preview — no experimental values", ha="center", fontsize=7.2, color="#8b4c40")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "generality_layout_preview.png", dpi=220)
    plt.close(fig)


def parameters(source, output):
    path = source / "analysis/summary.json" if (source / "analysis/summary.json").exists() else source / "summary.json"
    data = read(path)
    rows = data["ranking"]
    widths = sorted({row["settings"]["mutable_limit"] for row in rows})
    fractions = sorted({row["settings"]["frontier_fraction"] for row in rows})
    engines = sorted({row["settings"]["frontier_engine"] for row in rows}, reverse=True)
    if not data["complete"] or len(rows) != data["configurations"] or len(rows) != len(widths) * len(fractions) * len(engines):
        raise ValueError("The complete development parameter grid is required")
    values = [100 * row["score"] for row in rows]
    fig, axes = plt.subplots(1, len(engines), figsize=(3.48, 1.96), layout="constrained", squeeze=False)
    selected = rows[0]["settings"]
    for ax, engine in zip(axes[0], engines):
        matrix = np.array([[next(100 * row["score"] for row in rows if
                         row["settings"]["mutable_limit"] == width and row["settings"]["frontier_fraction"] == fraction
                         and row["settings"]["frontier_engine"] == engine) for fraction in fractions] for width in widths])
        ax.imshow(matrix, cmap="Blues", vmin=min(values) - 2, vmax=max(values) + 3, aspect="auto")
        for i, width in enumerate(widths):
            for j, fraction in enumerate(fractions):
                ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if (matrix[i, j] - min(values) + 2) / (max(values) - min(values) + 5) > .65 else "#193249")
                if (width, fraction, engine) == (selected["mutable_limit"], selected["frontier_fraction"], selected["frontier_engine"]):
                    ax.add_patch(Rectangle((j - .48, i - .48), .96, .96, fill=False, edgecolor=ORANGE, linewidth=1.7))
        ax.set_xticks(range(len(fractions)), [f"{100 * value:g}%" for value in fractions])
        ax.set_yticks(range(len(widths)), widths)
        ax.set_xlabel("Frontier share")
        ax.set_title("Swap frontier" if engine == "swap" else "Joint-SA frontier", fontsize=8.1)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel("Residual width")
    fig.suptitle("Development: mean reduction in native losses", fontsize=8.1)
    save(fig, output, "parameters")


def quality(source, output):
    data = read(source / "summary.json")
    if not data["complete"] or not {"CARA", "Component-CARA"}.issubset(data["methods"]):
        raise ValueError("The completed CARA / Component-CARA comparison is required")
    rows = {(row["cohort"], row["method"], row["stream"]): row for row in data["group_summaries"] if row["view"] == "primary"}
    metrics = ("VR", "VS", "PAR", "raw")
    methods = (("Component-CARA", GREY), ("CARA", BLUE))
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.12), sharey=True)
    fig.subplots_adjust(left=.071, right=.995, bottom=.18, top=.74, wspace=.16)
    extrema, export = [0.], []
    for ax, source_name in zip(axes, SOURCES):
        reference = rows[source_name, "FFD", "all"]
        streams = sorted({stream for cohort, method, stream in rows if cohort == source_name and method == "CARA" and stream != "all"})
        for offset, (method, color) in zip((-.18, .18), methods):
            current = rows[source_name, method, "all"]
            gain = np.array([100 * (reference[metric] - current[metric]) / max(abs(reference[metric]), 1e-8) for metric in metrics])
            trials = np.array([[100 * (rows[source_name, "FFD", stream][metric] - rows[source_name, method, stream][metric])
                                / max(abs(rows[source_name, "FFD", stream][metric]), 1e-8) for metric in metrics] for stream in streams])
            error = np.array([np.maximum(0., gain - trials.min(axis=0)), np.maximum(0., trials.max(axis=0) - gain)])
            ax.bar(np.arange(len(metrics)) + offset, gain, width=.32, color=color, label=method,
                   yerr=error, error_kw={"elinewidth": .65, "capsize": 1.3, "capthick": .65})
            extrema.extend(trials.ravel())
            export.extend({"source": source_name, "method": method, "metric": metric, "gain_percent": float(value),
                           "stream_min": float(low), "stream_max": float(high)}
                          for metric, value, low, high in zip(metrics, gain, trials.min(axis=0), trials.max(axis=0)))
        ax.set_title(SOURCE_NAMES[source_name], pad=5)
        ax.set_xticks(range(4), ("VR", "VS", "PAR", "Excess"))
        ax.axhline(0., color=GREY, linewidth=.6)
        ax.grid(axis="y", color="#dfe4e7", linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=2.5)
    lo, hi = min(extrema), max(extrema)
    pad = max(3., .08 * (hi - lo))
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Reduction from FFD (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.54, .99))
    save(fig, output, "quality")
    write_json(output / "quality_plot_data.json", export)


def main_table(source, output):
    data = read(source / "summary.json")
    if not data["complete"]:
        raise ValueError("The complete benchmark analysis is required")
    rows = {row["method"]: row for row in data["cohort_macro"] if row["view"] == "primary"}
    if set(rows) != {"FFD", *data["methods"]}:
        raise ValueError("A declared comparator is absent from the main table")
    preferred = ("FFD", "Swap", "Anneal", "Risk-SA", "Joint-SA", "Broad", "Scenario-MIP", "Eager-CARA", "Compact-CARA",
                 "Component-CARA", "No-residual", "No-frontier", "Best-seed", "Raw-order")
    methods = [name for name in preferred if name in rows] + sorted(set(rows) - {*preferred, "CARA"}) + ["CARA"]
    metrics = ("hosts", "Util", "VR", "VS", "PAR", "raw", "PICP", "MPIW", "WinklerScore", "moved_fraction")
    precision = {"hosts": 2, "Util": 2, "VR": 4, "VS": 5, "PAR": 3, "raw": 5,
                 "PICP": 4, "MPIW": 4, "WinklerScore": 4, "moved_fraction": 2}
    minima = {metric: min(row[metric] for row in rows.values()) for metric in ("VR", "VS", "PAR", "raw", "WinklerScore")}
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3.8pt}",
             r"\caption{Primary comparison over all " + str(data["view_cases"]["primary"]) + r" decisions, including incumbent fallbacks. Cohorts receive equal weight. Lower VR, VS, PAR, excess, and Winkler scores are better; column minima are bold. Coverage and interval width are reported jointly. Movement is the percentage of workloads actually moved after admission.}",
             r"\label{tab:main}", r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"Method & Hosts & Util. & VR & VS & PAR & Excess & PICP & MPIW & Winkler & Move (\%)\\", r"\midrule"]
    for method in methods:
        values = []
        for metric in metrics:
            value = rows[method][metric] * (100 if metric == "moved_fraction" else 1)
            text = f"{value:.{precision[metric]}f}"
            if metric in minima and abs(rows[method][metric] - minima[metric]) <= 1e-12:
                text = r"\textbf{" + text + "}"
            values.append(text)
        label = r"\textbf{CARA}" if method == "CARA" else method
        lines.append(label + " & " + " & ".join(values) + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "main_table.tex").write_text("\n".join(lines) + "\n")


def method(output):
    source = HERE / "draw_method.py"
    if not source.exists():
        raise FileNotFoundError("Keep draw_method.py beside this plotting script")
    specification = importlib.util.spec_from_file_location("cara_method_figure", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with plt.rc_context():
        module.draw(output / "method.pdf", output / "method.png")


def case_study(source, output):
    from matplotlib.colors import TwoSlopeNorm
    case = read(source / "case_study.json")
    if (source / "summary.json").exists() and not read(source / "summary.json")["complete"]:
        raise ValueError("The case must come from a completed study")
    matrices = {name: np.full((case["hosts"], case["target_hours"]), np.nan) for name in ("incumbent", "CARA")}
    with (source / "loads.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for name in matrices:
                matrices[name][int(row["host"]) - 1, int(row["hour"])] = float(row[name]) / case["capacity"]
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("The case heatmaps require every host-hour observation")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.72), layout="constrained",
                             gridspec_kw={"width_ratios": (1., 1., 1.03)})
    upper = max(1.05, max(float(matrix.max()) for matrix in matrices.values()))
    norm = TwoSlopeNorm(vmin=0., vcenter=1., vmax=upper)
    ticks = sorted(set((0, case["target_hours"] // 4, case["target_hours"] // 2,
                        3 * case["target_hours"] // 4, case["target_hours"] - 1)))
    for ax, name, title in zip(axes[:2], matrices, ("(a) Incumbent", "(b) CARA")):
        plot = ax.pcolormesh(np.arange(case["target_hours"] + 1) - .5,
                             np.arange(case["hosts"] + 1) + .5, matrices[name],
                             cmap="RdBu_r", norm=norm, shading="flat", rasterized=False)
        ax.set_title(title)
        ax.set_xticks(ticks)
        ax.set_xlabel("Target hour")
    axes[0].set_ylabel("Physical host")
    axes[1].tick_params(axis="y", labelleft=False)
    colorbar = fig.colorbar(plot, ax=axes[:2], label="Load / capacity", shrink=.88, pad=.02, fraction=.035)
    colorbar.set_ticks((0., .5, 1., upper), labels=("0", "0.5", "1.0", f"{upper:.2f}"))
    ax = axes[2]
    for name, color, style in (("FFD", GREY, "--"), ("CARA", BLUE, "-")):
        ax.step(np.arange(case["target_hours"]), case["overloaded_hosts_by_hour"][name], where="mid",
                color=color, linestyle=style, linewidth=1.3, label="Incumbent" if name == "FFD" else "CARA")
    ax.set_title("(c) Overloaded hosts")
    ax.set_xlabel("Target hour")
    ax.set_xticks(ticks)
    ax.set_ylim(bottom=-.2)
    ax.grid(axis="y", linewidth=.45, color="#dfe4e7")
    ax.legend(frameon=False, loc="upper left")
    save(fig, output, "case")


def query_phases(source, output):
    """Sum the disjoint phases of each source/plan's median-total CPU trial."""
    summary, rows = read(source / "summary.json"), read(source / "records.json")
    if not summary["complete"] or not summary["all_exact"] or summary["sources"] != 4 or {row["source"] for row in rows} != set(QUERY_SOURCES):
        raise ValueError("All four verified hourly-source studies are required")
    available = set.intersection(*(set(row["median_seconds"]) for row in rows))
    final = "indexed" if "indexed" in available else "compiled" if "compiled" in available else "certified"
    left_plans = (("eager", "Eager"), ("compact", "Compact"), ("sparse", "Sparse-JV"), (final, "CARA"))
    right_plans = (("no_factorization", "No factorization"), ("no_late_matching", "Early physical maps"),
                   ("no_progressive", "Fixed cutoff"), ("no_score", "No score filter"),
                   ("no_matching_reuse", "No count reuse"), ("compiled_no_unique_maps", "No unique maps"),
                   ("sparse", "Sparse-JV"), (final, "CARA"))
    if not set(dict(left_plans + right_plans)).issubset(available):
        raise ValueError("The full hourly-plan ablation grid is required")
    values, provenance = {name: np.zeros(4) for name in dict(left_plans + right_plans)}, []
    for row in sorted(rows, key=lambda item: QUERY_SOURCES.index(item["source"])):
        for name in values:
            trials = [(index, trial["plans"][name]) for index, trial in enumerate(row["trials"])]
            index, trial = sorted(trials, key=lambda pair: pair[1]["cpu_seconds"])[len(trials) // 2]
            if not np.isclose(trial["cpu_seconds"], row["median_seconds"][name]["cpu_seconds"], rtol=1e-12, atol=1e-12):
                raise ValueError("Use an odd number of complete repetitions for phase decomposition")
            phases = [sum(family[field] for family in trial["families"])
                      for field in ("predicate_cpu_seconds", "profile_cpu_seconds", "retention_cpu_seconds")]
            other = trial["cpu_seconds"] - sum(phases)
            if other < -1e-10:
                raise ValueError("The recorded phase intervals overlap")
            phases = np.array([*phases, max(0., other)])
            values[name] += phases
            provenance.append({"source": row["source"], "plan": name, "trial_index": index,
                               "total_cpu_seconds": trial["cpu_seconds"],
                               "predicate_profile_retention_other_seconds": phases.tolist()})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.72), layout="constrained",
                              gridspec_kw={"width_ratios": (.82, 1.27)})
    totals = np.array([values[name].sum() for name, _ in left_plans])
    ax = axes[0]
    ax.barh(np.arange(len(left_plans)), totals, height=.57, color=["#a0abb4", GREY, TEAL, BLUE])
    for position, total in enumerate(totals):
        ax.text(total + .025 * totals.max(), position, f"{total:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(left_plans)), [label for _, label in left_plans])
    ax.set_xlim(0, totals.max() * 1.25)
    ax.set_title("(a) Complete queries, all four sources", pad=6)
    ax = axes[1]
    matrix = np.array([values[name] for name, _ in right_plans])
    offsets = np.zeros(len(right_plans))
    for column, label, color in ((0, "Predicates", ORANGE), (1, "Profiles", BLUE),
                                 (2, "Acquisition/retention", TEAL), (3, "Other", "#b8c1c8")):
        ax.barh(np.arange(len(right_plans)), matrix[:, column], left=offsets, height=.61, color=color, label=label)
        offsets += matrix[:, column]
    for position, total in enumerate(offsets):
        ax.text(total + .025 * offsets.max(), position, f"{total:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(right_plans)), [label for _, label in right_plans])
    ax.set_xlim(0, offsets.max() * 1.18)
    ax.set_title("(b) Query phases, all four sources", pad=6)
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("CPU time (s)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe4e7", linewidth=.45)
        ax.tick_params(axis="y", length=0)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    save(fig, output, "query_phases")
    write_json(output / "query_phase_plot_data.json", {"sources": list(QUERY_SOURCES),
        "aggregation": "Sum the disjoint phase costs from each source/plan's median-total CPU trial; every displayed total equals the sum of the four source medians.",
        "trials": provenance, "sum_predicate_profile_retention_other_seconds": {name: value.tolist() for name, value in values.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generality", type=Path)
    parser.add_argument("--expected-settings", type=int, default=19)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--method", action="store_true")
    parser.add_argument("--layout-preview", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    if not any((args.generality, args.parameters, args.quality, args.case, args.queries, args.method, args.layout_preview)):
        parser.error("Choose a complete recorded study or the explicitly empty layout preview")
    if args.method:
        method(args.output)
    if args.layout_preview:
        layout_preview(args.output)
    if args.parameters:
        parameters(args.parameters, args.output)
    if args.quality:
        quality(args.quality, args.output)
        main_table(args.quality, args.output)
    if args.case:
        case_study(args.case, args.output)
    if args.queries:
        query_phases(args.queries, args.output)
    if args.generality:
        generality(load_generality(args.generality, args.expected_settings), args.output)


if __name__ == "__main__":
    main()
