"""Action-specific excess-gain bounds from a physical migration graph.

The graph is fixed after host matching and before certification scenes open.
Its maximum directed cut bounds the positive part of every possible load
change. Small support components are enumerated exactly; a reciprocal-edge
upper bound handles larger components without weakening validity.
"""
from fractions import Fraction
from math import isfinite, lcm, nextafter

import numpy as np

from .certification import _rational, stored_raw_gain_bound


def migration_cut_bound(incumbent, aligned_candidate, demand_upper=1,
                        *, exact_component_limit=18):
    """Return an exact rational upper bound on the graph's directed max-cut.

    Assignments must already refer to the same physical hosts. A supplied
    bound for each VM must hold on every possible fresh scene, not merely on
    observed samples. Enumeration is exponential only in the size of a small
    connected component; larger components use sum(max(W_uv, W_vu)).
    """
    old, new = tuple(incumbent), tuple(aligned_candidate)
    if (not old or len(old) != len(new)
            or any(not isinstance(v, (int, np.integer)) or v < 0 for v in old + new)
            or set(old) != set(new)):
        raise ValueError("complete assignments on the same physical hosts are required")
    if int(exact_component_limit) != exact_component_limit or exact_component_limit < 1:
        raise ValueError("exact_component_limit must be a positive integer")
    if np.ndim(demand_upper) == 0:
        bounds = (_rational(demand_upper),) * len(old)
    else:
        bounds = tuple(map(_rational, demand_upper))
    if len(bounds) != len(old) or any(value < 0 for value in bounds):
        raise ValueError("nonnegative population demand bounds must identify every VM")
    edges = {}
    for source, target, bound in zip(old, new, bounds):
        if source != target and bound:
            edges[source, target] = edges.get((source, target), Fraction()) + bound
    graph = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
    remaining, components = set(graph), []
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            vertex = pending.pop()
            if vertex in component:
                continue
            component.add(vertex)
            pending.extend(graph[vertex] - component)
        remaining -= component
        components.append(tuple(sorted(component)))
    total, details = Fraction(), []
    for component in components:
        position = {vertex: j for j, vertex in enumerate(component)}
        weights = {(position[u], position[v]): weight for (u, v), weight in edges.items()
                   if u in position and v in position}
        if len(component) <= exact_component_limit:
            denominator = lcm(*(v.denominator for v in weights.values()))
            integer_edges = [(1 << u, 1 << v, int(w * denominator))
                             for (u, v), w in weights.items()]
            # Both orientations are required: a directed cut need not equal its
            # complement. Reversing the entire graph, however, preserves max-cut.
            maximum = max(sum(w for u, v, w in integer_edges
                              if not mask & u and mask & v)
                          for mask in range(1 << len(component)))
            value, exact = Fraction(maximum, denominator), True
        else:
            pairs = {tuple(sorted(pair)) for pair in weights}
            pair_bound = sum((max(weights.get((u, v), Fraction()),
                                  weights.get((v, u), Fraction())) for u, v in pairs), Fraction())
            total_weight = sum(weights.values(), Fraction())
            balance = [Fraction()] * len(component)
            for (source, target), weight in weights.items():
                balance[source] -= weight
                balance[target] += weight
            # For any cut, 2*entering = crossing + net entering. Crossing is at
            # most total_weight, and net entering at most half the L1 imbalance.
            balance_bound = total_weight / 2 + sum(map(abs, balance), Fraction()) / 4
            denominator = lcm(*(v.denominator for v in weights.values()))
            scaled = balance_bound * denominator
            balance_bound = Fraction(scaled.numerator // scaled.denominator, denominator)
            value = min(pair_bound, balance_bound)
            exact = False
        total += value
        details.append({"hosts": list(component), "exact": exact, "bound": str(value)})
    return {"bound": total, "exact": all(c["exact"] for c in details),
            "moved_count": sum(u != v for u, v in zip(old, new)),
            "total_moved_demand_bound": sum((b for u, v, b in zip(old, new, bounds) if u != v), Fraction()),
            "components": details}


def stored_topology_gain_bound(incumbent, aligned_candidate, capacity, *, horizon,
                               demand_upper=1, exact_component_limit=18,
                               component_conservation=True):
    """Bound stored fixed-H raw gain using each migration component.

    A component with v hosts conserves its own demand, so its total excess
    can change by at most v-1 capacity units. Combining that cap with its
    directed-cut bound before adding components is stronger than a global
    conservation cap. Set ``component_conservation=False`` for that control.
    """
    result = migration_cut_bound(incumbent, aligned_candidate, demand_upper,
                                exact_component_limit=exact_component_limit)
    return _stored_graph_gain_bound(result, len(incumbent), len(set(incumbent)),
                                    capacity, horizon, demand_upper,
                                    component_conservation=component_conservation)


def _stored_graph_gain_bound(result, vm_count, hosts, capacity, horizon, demand_upper,
                             *, component_conservation=True):
    capacity = _rational(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    upper = (_rational(demand_upper) if np.ndim(demand_upper) == 0
             else max(map(_rational, demand_upper)))
    # The zero-movement call gives an outward-rounded bound on the two replay
    # errors alone. Keeping that excess rounding is conservative.
    roundoff = stored_raw_gain_bound(0, hosts, capacity, vm_count=vm_count,
                                     horizon=horizon, demand_upper=upper)
    if component_conservation:
        physical = sum((min(Fraction(component["bound"]) / capacity,
                            Fraction(len(component["hosts"]) - 1))
                        for component in result["components"]), Fraction()) / hosts
    else:
        physical = min(result["bound"] / (hosts * capacity), Fraction(hosts - 1, hosts))
    bound = physical + Fraction(roundoff)
    stored = nextafter(float(bound), float("inf"))
    if not isfinite(stored):
        raise ValueError("the topology arithmetic bound must be finite")
    return stored


def stored_action_gain_bounds(incumbent, aligned_candidate, capacity, *, horizon,
                              demand_upper=1, exact_component_limit=18,
                              component_conservation=True):
    """Predeclared (J, V, R, raw) bounds for a frozen physical action.

    Only hosts incident to a moved VM can change Boolean overload projections.
    Each stored Boolean ratio contributes at most 2**-53 absolute rounding, so
    two paired ratios add at most 2**-52. The raw coordinate combines the
    component migration cuts and conserved-capacity envelopes with its replay
    allowance. ``component_conservation=False`` retains the global-cap control.
    """
    graph = migration_cut_bound(incumbent, aligned_candidate, demand_upper,
                                exact_component_limit=exact_component_limit)
    capacity = _rational(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("a positive integer horizon is required")
    hosts = len(set(incumbent))
    if hosts * int(horizon) > 2**53:
        raise ValueError("Boolean normalization integers must be exactly representable")
    touched = sorted({host for old, new in zip(incumbent, aligned_candidate)
                      if old != new for host in (old, new)})
    if not touched:
        return {"gain_bounds": (0., 0., 0., 0.), "touched_hosts": [],
                "touched_fraction": 0., "graph": graph}
    ratio = min(Fraction(1), Fraction(len(touched), hosts) + Fraction(1, 2**52))
    stored_ratio = float(ratio)
    if Fraction(stored_ratio) < ratio:
        stored_ratio = nextafter(stored_ratio, float("inf"))
    raw = _stored_graph_gain_bound(graph, len(incumbent), hosts,
                                   capacity, horizon, demand_upper,
                                   component_conservation=component_conservation)
    return {"gain_bounds": (1., stored_ratio, stored_ratio, raw),
            "touched_hosts": touched, "touched_fraction": len(touched) / hosts,
            "graph": graph}
