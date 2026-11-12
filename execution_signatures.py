"""Timing-independent signatures for complete search results."""
import json
import numpy as np


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def canonical_json(value):
    return json.dumps(serial(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_history(history):
    return [{k: serial(v) for k, v in row.items()
             if "cpu" not in k.lower() and "seconds" not in k.lower()}
            for row in history]


def result_signature(result):
    stats = result["stats"]
    stable_counts = {k: stats.get(k) for k in ("evaluations", "blocks", "broad_trials",
                    "migration_rejections", "forecast_rejections", "selection_candidates",
                    "selection_guard_passed", "migration_limit")}
    return canonical_json({"selected": result.get("selected"), "bets": result.get("bets"),
           "candidates": result.get("candidates"), "movement": result.get("movement"),
           "history": stable_history(stats["history"]), "counts": stable_counts,
           "selection_reason": result.get("selection_reason")})
