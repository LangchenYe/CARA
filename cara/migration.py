"""Exact lexicographic host matching for optional per-VM movement weights."""
import numpy as np


def weighted_host_matching(counts, incumbent, candidate, weights):
    """Maximize retained VM count, then exact stored binary64 retained weight.

    Every finite binary64 weight is dyadic. A common power-of-two denominator
    turns the secondary objective into integers; a coefficient larger than the
    sum of all weights makes the primary count objective strictly dominant.
    The integer Hungarian algorithm avoids a floating-point tie perturbation.
    """
    ratios = [float(value).as_integer_ratio() for value in weights]
    denominator = max(value[1] for value in ratios)
    integer_weights = [numerator * (denominator // divisor)
                       for numerator, divisor in ratios]
    hosts = len(counts)
    retained = [[0] * hosts for _ in range(hosts)]
    for old, new, weight in zip(incumbent, candidate, integer_weights):
        retained[int(old)][int(new)] += weight
    priority = sum(integer_weights) + 1
    costs = [[-int(counts[i, j]) * priority - retained[i][j]
              for j in range(hosts)] for i in range(hosts)]
    # Potentials and reduced costs are arbitrary-precision Python integers.
    row_potential, column_potential = [0] * (hosts + 1), [0] * (hosts + 1)
    matched_row, previous = [0] * (hosts + 1), [0] * (hosts + 1)
    for row in range(1, hosts + 1):
        matched_row[0], column = row, 0
        minimum, used = [None] * (hosts + 1), [False] * (hosts + 1)
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta, next_column = None, 0
            for other in range(1, hosts + 1):
                if used[other]:
                    continue
                reduced = (costs[active_row - 1][other - 1]
                           - row_potential[active_row] - column_potential[other])
                if minimum[other] is None or reduced < minimum[other]:
                    minimum[other], previous[other] = reduced, column
                if delta is None or minimum[other] < delta:
                    delta, next_column = minimum[other], other
            for other in range(hosts + 1):
                if used[other]:
                    row_potential[matched_row[other]] += delta
                    column_potential[other] -= delta
                elif minimum[other] is not None:
                    minimum[other] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while column:
            prior = previous[column]
            matched_row[column] = matched_row[prior]
            column = prior
    columns = np.empty(hosts, dtype=int)
    for column in range(1, hosts + 1):
        columns[matched_row[column] - 1] = column - 1
    return np.arange(hosts), columns
