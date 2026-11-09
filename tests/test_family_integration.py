"""Whole-controller oracles for compact and certified residual execution."""
from copy import deepcopy

import numpy as np
import pytest

from cara.refinement import migration_profile, search_refined
from cara.repair import feasible


def run_modes(*, score="gain", archive="balanced", weighted=False, certify_score=False):
    rng = np.random.default_rng(29)
    panel = rng.uniform(.1, .5, (16, 12, 8))
    forecast = np.full((12, 8), .25)
    incumbent = tuple(np.repeat(np.arange(4), 3))
    config = {"cpu_limit": .1, "initial_broad": 0, "mutable_limit": 4,
              "top_p": 4, "migration_fraction": .5, "diagnostic": False,
              "archive_mode": archive, "seed_schedule": "coverage",
              "score_mode": score, "shared_queries": True, "batch_queries": True,
              "family_certificate_score": certify_score, "family_score_min_leaves": 2,
              "replay_arms": ["exact", "broad", "exact", "exact", "broad", "exact", "exact", "exact"],
              "broad_batch": 8}
    if weighted:
        config.update(migration_weights=np.linspace(.25, 2., 12), migration_cost_limit=5.)
    observed, results, witnesses = {}, {}, []

    def observe(witness):
        family = witness["family"]
        for mask in witness["active_masks"]:
            assignment = family.assignment(mask)
            if witness["reason"] == "forecast":
                assert not feasible(assignment, forecast, 1., 4)
            elif witness["reason"] == "migration":
                movement = migration_profile(incumbent, assignment)
                assert movement["count"] > witness["migration_limit"]
        witnesses.append(witness["reason"])

    for mode in ("eager", "compact", "certified"):
        snapshots = []
        settings = {**config, "family_execution": mode,
                    "family_observer": lambda records, details: snapshots.append(
                        (deepcopy(records), (details["seed_assignment"], details["pair"], details["mutable"]))),
                    "certificate_observer": observe}
        results[mode] = search_refined(incumbent, forecast, panel, panel.copy(), 1.,
                                      settings, np.random.default_rng(7))
        observed[mode] = snapshots
    for mode in ("compact", "certified"):
        assert observed[mode] == observed["eager"]
        for key in ("candidates", "selected", "movement", "selection_reason"):
            assert results[mode][key] == results["eager"][key]
        assert results[mode]["stats"]["family_execution"] == mode
        assert results[mode]["stats"]["blocks"] == results["eager"]["stats"]["blocks"]
    return results, witnesses


@pytest.mark.parametrize("score,archive,weighted", [
    ("gain", "balanced", False), ("gain", "coverage", False),
    ("gain", "balanced", True), ("raw", "balanced", False),
    ("evidence", "balanced", False)])
def test_compact_and_predicate_certificates_preserve_controller(score, archive, weighted):
    results, _ = run_modes(score=score, archive=archive, weighted=weighted)
    work = results["compact"]["stats"]["family_work"]
    assert work["label_materializations"] <= work["forecast_passed_masks" if weighted else "profiled_masks"]


def test_all_certificates_preserve_complete_family_pools():
    run_modes(certify_score=True)


def test_tied_zero_profiles_preserve_canonical_assignment_order():
    inc = (0, 0, 0, 1, 1, 1, 2, 2, 2)
    panel, forecast = np.zeros((8, 9, 8)), np.zeros((9, 8))
    results = [search_refined(inc, forecast, panel, panel, 1.,
        {"cpu_limit": .01, "initial_broad": 0, "mutable_limit": 4, "top_p": 4,
         "shared_queries": True, "batch_queries": True, "migration_fraction": 1.,
         "replay_arms": ["exact"] * 3, "family_execution": mode,
         "family_certificate_score": False}, np.random.default_rng(37))
        for mode in ("eager", "compact", "certified")]
    assert results[0]["candidates"]
    assert all(result["candidates"] == results[0]["candidates"] for result in results)
    assert all(result["selected"] is None for result in results)
