"""Prediction-interval diagnostics follow the benchmark's conditional means."""
import numpy as np
import pytest

from benchmark_native_intervals import predictive_metrics


def test_zero_scale_forecast_recovers_exact_coverage_and_winkler_penalties():
    demand = np.array([[.5, 1.5], [.25, .75]])
    forecast = np.array([[.5, 1.], [.25, .5]])
    result = predictive_metrics(demand, (0, 1), forecast, np.zeros(2))
    assert result["PICP"] == .5
    assert result["MPIW"] == 0.
    assert result["WinklerScore"] == pytest.approx(7.5)
    empty = predictive_metrics(np.zeros_like(demand), (0, 1), forecast, np.zeros(2))
    assert empty == {"PICP": 0., "MPIW": 0., "WinklerScore": 0.}


def test_probability_forecast_is_deterministic_and_omits_inactive_hosts():
    demand = np.array([[.5, .7, .8], [.4, .3, .2], [0., 0., 0.]])
    median = np.array([[.6, .6, .6], [.3, .3, .3], [2., 2., 2.]])
    scales = np.array([.2, .1, .5])
    first = predictive_metrics(demand, (0, 1, 2), median, scales)
    assert first == predictive_metrics(demand, (0, 1, 2), median, scales)
    assert first == predictive_metrics(demand[:2], (0, 1), median[:2], scales[:2])
    assert 0 <= first["PICP"] <= 1 and first["MPIW"] > 0
    assert first["WinklerScore"] >= first["MPIW"]
