import json

import pytest

from cara.query_feedback import QueryFeedback


def test_unseen_seeds_follow_balanced_order_and_unavailable_seeds_are_ignored():
    first, second, third = (0, 0, 1), (0, 1, 0), (0, 1, 1)
    feedback = QueryFeedback()
    assert feedback.choose([first, second, third]) == first
    feedback.update(first, False, .01)
    assert feedback.choose([first, second, third]) == second
    feedback.update(second, True, .01)
    assert feedback.choose([first, second, third]) == third
    assert feedback.choose([first, second]) == second


def test_completed_successes_are_preferred_at_equal_cost():
    first, second = (0, 0, 1), (0, 1, 0)
    feedback = QueryFeedback()
    for _ in range(8):
        feedback.update(first, False, .01)
        feedback.update(second, True, .01)
    assert feedback.choose([first, second]) == second


def test_cost_is_used_without_unbounded_reward_for_nearly_free_failures():
    cheap, expensive = (0, 0, 1), (0, 1, 0)
    feedback = QueryFeedback()
    for _ in range(8):
        feedback.update(cheap, False, 1e-12)
        feedback.update(expensive, True, .01)
    assert feedback.choose([cheap, expensive]) == expensive
    assert feedback.state()["cost_floor_seconds"] >= .005
    equal_success = QueryFeedback()
    for i in range(8):
        equal_success.update(cheap, bool(i % 2), .01)
        equal_success.update(expensive, bool(i % 2), .1)
    assert equal_success.choose([expensive, cheap]) == cheap


def test_state_is_json_serializable_and_invalid_observations_are_rejected():
    feedback = QueryFeedback()
    feedback.update((0, 1), True, 0.)
    feedback.choose([(0, 1)])
    json.dumps(feedback.state(include_arms=True), allow_nan=False)
    with pytest.raises(ValueError):
        feedback.choose([])
    with pytest.raises(ValueError):
        feedback.update((0, 1), True, float("nan"))
    with pytest.raises(ValueError):
        feedback.update((0, 1), 1, .01)
