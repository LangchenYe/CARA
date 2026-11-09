import numpy as np
import pytest

from cara.certification import certify_fresh_mixture
from cara.synthetic import make_environment
import run_streaming_validation as runner
from run_streaming_validation import PairedWorkloadReader, inspect, one_environment, study_plan


def test_real_paired_reader_preserves_scene_losses_between_inspection_schedules():
    environment = make_environment(145006)
    old = tuple(i // 4 for i in range(48))
    new = list(old)
    new[0], new[4] = new[4], new[0]
    full = PairedWorkloadReader(environment, "bursty_common", 15, old, new, execution="full")
    full_reference, full_candidate = full(np.arange(65))
    shared = PairedWorkloadReader(environment, "bursty_common", 15, old, new, execution="shared")
    parts = [shared(np.arange(start, end)) for start, end in ((0, 17), (17, 49), (49, 65))]
    assert np.array_equal(full_reference, np.concatenate([part[0] for part in parts]))
    assert np.array_equal(full_candidate, np.concatenate([part[1] for part in parts]))
    assert full.peak_panel_values <= 32 * 48 * 8
    assert shared.rows == 65
    with pytest.raises(ValueError, match="previously unread"):
        shared(np.arange(2))


def test_paired_reader_stopping_avoids_later_scene_generation():
    class ConstantReader:
        def __init__(self):
            self.rows = self.peak_panel_values = 0
            self.generation_cpu_seconds = self.replay_cpu_seconds = self.reader_cpu_seconds = 0.

        def __call__(self, indices):
            self.rows += len(indices)
            return np.ones((len(indices), 4)), np.zeros((len(indices), 4))

    reader = ConstantReader()
    result = inspect(reader, (1,) * 4, 512, 32)
    assert result["admitted"] and reader.rows == result["rows"] < 512
    reference = np.ones((512, 4))
    expected = certify_fresh_mixture(reference, np.zeros_like(reference), gain_bounds=(1,) * 4,
        noninferiority_margins=(.01, .01, 0., .001), anchor=2, min_anchor_gain=.005,
        alpha=.05, batch=32, metric_names=("J", "V", "R", "raw_fixed_H"))
    assert result["rows"] == expected["rows"]


def test_final_validation_design_is_disjoint_and_complete():
    plan = study_plan({}, ["CARA"], "validation", (8192,))
    assert len(plan["validation_environments"]) == 96
    assert len({(row["law"], row["seed"]) for row in plan["validation_environments"]}) == 96
    assert min(row["seed"] for row in plan["validation_environments"]) >= 246001
    assert plan["counts"]["evaluation"] == 16384


def test_no_candidate_retains_incumbent_without_generating_certification(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "search_method", lambda *args: {"selected": None, "stats": {"cpu_seconds": 0.}})
    counts = {"fit": 32, "build": 32, "selection": 32, "certification": 32, "evaluation": 32}
    records = one_environment(("complementary", 145007, {}, ["CARA", "No-footprint"],
                               counts, [32], tmp_path, "test"))
    for record in records.values():
        assert record["status"] == "no_candidate"
        for rule in record["rules"].values():
            assert rule["rows"] == rule["generated_scene_rows"] == 0
            assert not rule["admitted"] and rule["terminal_migrations"] == 0
            assert rule["terminal_outcomes"] == record["incumbent_outcomes"]
