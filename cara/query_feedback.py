"""Build-only feedback for allocating complete residual workload queries."""
from dataclasses import dataclass
from math import isfinite, log, sqrt
from statistics import median


@dataclass
class _Arm:
    families: int = 0
    improvements: int = 0
    cpu_seconds: float = 0.

    @property
    def mean_cpu(self):
        return self.cpu_seconds / self.families if self.families else 0.


class QueryFeedback:
    """Allocate families by an optimistic improvement probability per CPU cost.

    Unseen seeds are tried in the supplied balanced order. Completed-family
    outcomes update a smoothed Bernoulli success rate; an exploration bonus
    keeps unsuccessful, lightly sampled seeds eligible. A median cost floor
    prevents nearly free failures from receiving arbitrarily large scores.
    This is a bounded search heuristic, not a regret or optimality guarantee.
    """

    def __init__(self, exploration=.35, minimum_cost=1e-4, cost_floor_fraction=1.):
        if (not isfinite(exploration) or exploration < 0
                or not isfinite(minimum_cost) or minimum_cost <= 0
                or not isfinite(cost_floor_fraction) or cost_floor_fraction <= 0):
            raise ValueError("finite exploration and positive finite cost floors required")
        self.exploration, self.minimum_cost = float(exploration), float(minimum_cost)
        self.cost_floor_fraction = float(cost_floor_fraction)
        self.arms = {}
        self.families = self.improvements = 0
        self.last_reason, self.last_rank, self.last_score = None, None, None

    def _cost_floor(self):
        means = [arm.mean_cpu for arm in self.arms.values() if arm.families]
        return max(self.minimum_cost, self.cost_floor_fraction * median(means)) if means else self.minimum_cost

    def _optimistic_probability(self, arm):
        probability = (arm.improvements + 1) / (arm.families + 2)
        bonus = self.exploration * sqrt(log(2 + self.families) / (arm.families + 1))
        return min(1., probability + bonus)

    def choose(self, ordered_seed_assignments):
        seeds = list(dict.fromkeys(tuple(seed) for seed in ordered_seed_assignments))
        if not seeds:
            raise ValueError("at least one eligible seed is required")
        for rank, seed in enumerate(seeds):
            arm = self.arms.get(seed)
            if arm is None or not arm.families:
                self.last_reason, self.last_rank, self.last_score = "unseen", rank, None
                return seed
        floor = self._cost_floor()
        scores = [self._optimistic_probability(self.arms[seed]) /
                  max(self.arms[seed].mean_cpu, floor) for seed in seeds]
        rank = max(range(len(seeds)), key=lambda index: (scores[index], -index))
        self.last_reason, self.last_rank, self.last_score = "optimistic_rate", rank, scores[rank]
        return seeds[rank]

    def update(self, seed, improved_bool, cpu):
        if not isinstance(improved_bool, bool) or not isfinite(cpu) or cpu < 0:
            raise ValueError("a Boolean completed-family outcome and finite nonnegative CPU are required")
        arm = self.arms.setdefault(tuple(seed), _Arm())
        arm.families += 1
        arm.improvements += int(improved_bool)
        arm.cpu_seconds += float(cpu)
        self.families += 1
        self.improvements += int(improved_bool)

    def state(self, include_arms=False):
        result = {"seeds": len(self.arms), "families": self.families,
                  "improvements": self.improvements, "cost_floor_seconds": self._cost_floor(),
                  "last_choice_reason": self.last_reason, "last_choice_rank": self.last_rank,
                  "last_optimistic_rate": self.last_score}
        if include_arms:
            result["arms"] = [{"assignment": [int(value) for value in seed], "families": arm.families,
                               "improvements": arm.improvements, "cpu_seconds": arm.cpu_seconds,
                               "mean_cpu_seconds": arm.mean_cpu,
                               "optimistic_probability": self._optimistic_probability(arm)}
                              for seed, arm in self.arms.items()]
        return result

    diagnostics = state
