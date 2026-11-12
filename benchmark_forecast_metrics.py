"""Forecast diagnostics shared by every consolidation policy.

Evaluate the unchanged trailing-24-hour forecast on every distinct target
context in the consolidation protocol. MASE and the nine-quantile normalized
CRPS approximation follow CloudCons' executable forecasting metrics. This is
the common forecast used by the repair comparison, not its forecasting-model
leaderboard across multiple frequencies and horizons.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from cara.data import load_dataset


ROOT = Path(__file__).resolve().parent
QUANTILES = np.arange(1, 10, dtype=np.float64) / 10


def forecast_totals(history, truth, medians, scales, *, seasonality=24):
    """Sufficient totals for GluonTS MASE and mean weighted quantile loss.

    The MASE denominator is each VM's mean absolute seasonal difference over
    its complete pre-origin history. Undefined zero-scale entries are masked,
    as in the upstream masked-array evaluator, and their count is returned.
    Quantiles use the same zero-clipped Gaussian family as the host intervals.
    """
    history, truth, medians, scales = map(np.asarray, (history, truth, medians, scales))
    if (history.ndim != 2 or truth.ndim != 2 or history.shape[0] != truth.shape[0]
            or medians.shape != truth.shape or scales.shape != (len(truth),)
            or not isinstance(seasonality, int) or seasonality < 1
            or history.shape[1] <= seasonality or not truth.size
            or any(not np.isfinite(x).all() or np.any(x < 0)
                   for x in (history, truth, medians, scales))):
        raise ValueError("Finite aligned histories, targets, and forecast parameters are required")
    seasonal_scale = np.abs(history[:, seasonality:] - history[:, :-seasonality]).mean(axis=1)
    valid = seasonal_scale > 0
    absolute_error = np.abs(truth - medians)
    scaled_error = absolute_error[valid] / seasonal_scale[valid, None]
    forecasts = np.maximum(medians[None, :, :]
        + ndtri(QUANTILES)[:, None, None] * scales[None, :, None], 0.)
    residuals = truth[None, :, :] - forecasts
    quantile_loss = 2 * np.maximum(QUANTILES[:, None, None] * residuals,
                                 (QUANTILES[:, None, None] - 1) * residuals)
    return {
        "target_cells": int(truth.size),
        "mase_cells": int(valid.sum()) * truth.shape[1],
        "zero_scale_vm_contexts": int((~valid).sum()),
        "absolute_error_sum": float(absolute_error.sum()),
        "scaled_absolute_error_sum": float(scaled_error.sum()),
        "absolute_label_sum": float(np.abs(truth).sum()),
        "quantile_loss_sums": quantile_loss.sum(axis=(1, 2)).tolist(),
    }


def metrics(totals):
    cells, scaled_cells = totals["target_cells"], totals["mase_cells"]
    quantile_sum = float(np.mean(totals["quantile_loss_sums"]))
    return {
        "MASE": totals["scaled_absolute_error_sum"] / scaled_cells if scaled_cells else None,
        "CRPS": quantile_sum / totals["absolute_label_sum"] if totals["absolute_label_sum"] else None,
        "MAE": totals["absolute_error_sum"] / cells,
        "mean_quantile_loss": quantile_sum / cells,
    }


def aggregate(records):
    totals = {key: sum(row["totals"][key] for row in records)
              for key in ("target_cells", "mase_cells", "zero_scale_vm_contexts",
                          "absolute_error_sum", "scaled_absolute_error_sum", "absolute_label_sum")}
    totals["quantile_loss_sums"] = np.sum([row["totals"]["quantile_loss_sums"]
                                         for row in records], axis=0).tolist()
    return {"contexts": len(records), **totals, **metrics(totals)}


def run(protocol_path, data_root, output):
    protocol = json.loads(Path(protocol_path).read_text())
    primary = set(range(protocol["primary"]["first_origin"],
                        protocol["primary"]["last_origin"] + 1,
                        protocol["primary"]["stride"]))
    horizon = protocol["target_hours"]
    records = []
    for name, metadata in protocol["decision_cohorts"].items():
        data = load_dataset(name, data_root)
        daily = set(metadata["longitudinal_origins"])
        if data.n_vms != metadata["vms"] or data.min_length != metadata["minimum_hours"]:
            raise ValueError("The decision data disagree with the declared evaluation cohort")
        for origin in sorted(primary | daily):
            history = data.panel(0, origin)
            truth = data.panel(origin, horizon)
            medians = data.forecast(origin, horizon=horizon)
            scales = data.panel(origin - 24, 24).std(axis=1, ddof=1)
            totals = forecast_totals(history, truth, medians, scales)
            records.append({"dataset": name, "origin": origin,
                "views": (["primary"] if origin in primary else [])
                         + (["longitudinal"] if origin in daily else []),
                "totals": totals, "metrics": metrics(totals)})
    expected = protocol["evaluation_union"]["unique_cases"] // len(protocol["primary"]["streams"])
    if len(records) != expected:
        raise AssertionError("Every unique cohort/origin forecast must be evaluated exactly once")
    summaries = {}
    for view in ("primary", "longitudinal", "union"):
        selected = [row for row in records if view == "union" or view in row["views"]]
        cohorts = {name: aggregate([row for row in selected if row["dataset"] == name])
                   for name in protocol["decision_cohorts"]}
        summaries[view] = {"cohorts": cohorts, "pooled": aggregate(selected),
            "equal_cohort": {metric: float(np.mean([value[metric] for value in cohorts.values()]))
                             for metric in ("MASE", "CRPS", "MAE", "mean_quantile_loss")}}
    design = {
        "contexts": len(records), "forecast_identical_for_all_policies": True,
        "forecast": "Trailing-24-hour mean repeated for 24 target hours; the zero-clipped Gaussian quantiles use the preceding-24-hour sample standard deviation (ddof=1), exactly as the shared host-interval diagnostics.",
        "MASE": "Mean over VM-target cells of absolute median forecast error divided by the VM's mean absolute lag-24 difference over its complete pre-origin history. Zero-scale VM contexts are masked and counted, following the upstream masked-array metric.",
        "CRPS": "CloudCons executable approximation: average over q=0.1,...,0.9 of the sum of twice the pinball loss, divided by the sum of absolute target values. This is GluonTS MeanWeightedSumQuantileLoss, not an unnormalized exact CRPS integral.",
        "aggregation": "Each distinct cohort/origin appears once, regardless of search stream or placement policy. Cohort metrics aggregate the underlying sufficient sums over every context; equal_cohort averages the four cohort metrics.",
        "scope": "Forecast diagnostics on the consolidation decision targets; the separate CloudCons forecasting leaderboard uses additional frequencies/horizons and seasonal-naive normalization.",
        "sources": [
            "https://github.com/dmwyd/CloudCons/blob/main/forecasting_bench/chronos2_eval.py",
            "https://github.com/dmwyd/CloudCons/blob/main/forecasting_bench/get_results.py",
            "https://github.com/awslabs/gluonts/blob/v0.15.1/src/gluonts/ev/stats.py",
            "https://github.com/awslabs/gluonts/blob/v0.15.1/src/gluonts/ev/ts_stats.py"],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("protocol", design), ("records", records), ("summary", summaries)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (output / "cohort_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("view", "dataset", "contexts", "target_cells",
            "zero_scale_vm_contexts", "MASE", "CRPS", "MAE", "mean_quantile_loss"))
        writer.writeheader()
        for view, summary in summaries.items():
            for name, value in summary["cohorts"].items():
                writer.writerow({"view": view, "dataset": name,
                    **{key: value[key] for key in writer.fieldnames if key not in {"view", "dataset"}}})
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/release/family_protocol.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons")
    parser.add_argument("--output", type=Path, default=ROOT / "results/forecast_metrics")
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.data_root, args.output)["union"], indent=2))
