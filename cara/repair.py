"""Complete residual repair and adaptive broad proposals on disjoint Build data."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import process_time
import numpy as np

from .data import canonical_assignment, host_memberships, host_loads, burdens_from_loads, burdens
from .admission import evidence_key, fit_bets


def ffd(forecast, capacity, best_fit=False):
    forecast = np.asarray(forecast, dtype=float)
    if forecast.ndim != 2 or np.any(forecast < 0) or np.any(~np.isfinite(forecast)):
        raise ValueError("Expected finite nonnegative VM forecasts")
    assignment = np.full(len(forecast), -1, dtype=int)
    loads = []
    for vm in np.argsort(-forecast.max(axis=1), kind="stable"):
        if np.any(forecast[vm] > capacity + 1e-12):
            raise ValueError("One VM exceeds a host's forecast capacity")
        feasible = [h for h, load in enumerate(loads) if np.all(load + forecast[vm] <= capacity + 1e-12)]
        if feasible:
            host = max(feasible, key=lambda h: float(np.mean(loads[h]))) if best_fit else feasible[0]
            loads[host] += forecast[vm]
        else:
            host = len(loads)
            loads.append(forecast[vm].copy())
        assignment[vm] = host
    return canonical_assignment(assignment)


def feasible(assignment, forecast, capacity, host_count):
    assignment = canonical_assignment(assignment)
    return len(set(assignment)) == host_count and bool(np.all(host_loads(forecast, assignment) <= capacity + 1e-12))


@dataclass(frozen=True)
class Candidate:
    assignment: tuple
    bets: tuple
    key: tuple

    def record(self):
        return {"assignment": self.assignment, "bets": self.bets, "key": self.key}


class ReplayCache:
    """One immutable panel and seed; reuse complete host membership sets."""
    def __init__(self, panel, seed, capacity, h0=None):
        self.panel = np.asarray(panel, dtype=np.float64)
        self.seed = canonical_assignment(seed)
        self.capacity, self.h0 = capacity, h0 or len(set(seed))
        self.groups = dict(zip(host_memberships(self.seed), host_loads(self.panel, seed)))
        self.hits = self.changed_values = self.full_values = 0

    def loads(self, assignment):
        assignment = canonical_assignment(assignment)
        arrays = []
        b, n, t = self.panel.shape
        self.full_values += b * n * t
        for members in host_memberships(assignment):
            if members in self.groups:
                self.hits += 1
                arrays.append(self.groups[members])
            else:
                arrays.append(self.panel[:, list(members), :].sum(axis=1))
                self.changed_values += b * len(members) * t
        return np.stack(arrays)

    def __call__(self, assignment):
        return burdens_from_loads(self.loads(assignment), self.capacity, self.h0)


class LazyReplay:
    """Candidate-row access with full-panel reduction shapes and canonical sums."""
    def __init__(self, panel, incumbent, candidate, capacity, h0=None, reference_cache=None):
        self.panel = np.asarray(panel, dtype=np.float64)
        self.reference = reference_cache or ReplayCache(self.panel, incumbent, capacity, h0)
        if self.reference.panel is not self.panel or self.reference.seed != canonical_assignment(incumbent):
            raise ValueError("A shared replay cache must use this exact panel and incumbent")
        self.candidate = canonical_assignment(candidate)
        self.members = host_memberships(self.candidate)
        self.capacity, self.h0 = capacity, h0 or len(self.members)
        self.requested = set()
        self.read_values = 0

    def __call__(self, indices):
        indices = np.asarray(indices, dtype=int)
        b, n, t = self.panel.shape
        loads = np.zeros((len(self.members), b, t), dtype=np.float64)
        for h, members in enumerate(self.members):
            if members in self.reference.groups:
                loads[h, indices, :] = self.reference.groups[members][indices]
            else:
                # Advanced membership indexing in complete replay is VM-major.
                # Preserve that shape and layout rather than compressing rows.
                work = np.zeros((len(members), b, t), dtype=np.float64)
                work[:, indices, :] = self.panel[indices[:, None], np.asarray(members)[None, :], :].transpose(1, 0, 2)
                loads[h] = work.transpose(1, 0, 2).sum(axis=1)
                self.read_values += len(indices) * len(members) * t
        self.requested.update(map(int, indices))
        return burdens_from_loads(loads, self.capacity, self.h0)[indices]


def _retain(pool, candidate, count):
    pool[candidate.assignment] = candidate
    if len(pool) > count:
        keys = sorted(pool, key=lambda a: (pool[a].key, tuple(-v for v in a)), reverse=True)[:count]
        kept = {a: pool[a] for a in keys}
        pool.clear()
        pool.update(kept)


def _ordered(pool):
    return sorted(pool.values(), key=lambda c: (c.key, tuple(-v for v in c.assignment)), reverse=True)


def select_candidate(candidates, reference, selection, capacity):
    if not candidates:
        return None
    records = [c.record() if isinstance(c, Candidate) else c for c in candidates]
    ref = burdens(selection, reference, capacity)
    gains = np.asarray([(ref - burdens(selection, r["assignment"], capacity)).mean(axis=0) for r in records])
    ranks = np.empty_like(gains)
    for e in range(4):
        x = gains[:, e]
        ranks[:, e] = ((x[:, None] > x[None, :]).sum(axis=1) + .5 * (x[:, None] == x[None, :]).sum(axis=1)) / len(x)
    # Midranks have no burden-specific weights and are evaluated on held-out rows.
    best = max(range(len(records)), key=lambda j: (tuple(sorted(ranks[j])), -j))
    return records[best]


def _pair_order(panel, assignment, capacity, rng):
    loads = host_loads(panel, assignment)
    risk = np.maximum(loads / capacity - 1, 0).mean(axis=(1, 2))
    pairs = list(combinations(range(len(loads)), 2))
    # Random tie-breaking is fixed by the proposal RNG and uses no later data.
    jitter = rng.random(len(pairs)) * 1e-10
    return [pairs[j] for j in sorted(range(len(pairs)), key=lambda j: (risk[pairs[j][0]] + risk[pairs[j][1]] + jitter[j]), reverse=True)]


def _mutable(panel, assignment, pair, count, capacity):
    labels = np.asarray(assignment)
    loads = host_loads(panel, assignment)
    chosen = []
    for position, h in enumerate(pair):
        members = np.flatnonzero(labels == h)
        excess = np.maximum(loads[h] / capacity - 1, 0)
        demand = panel[:, members, :]
        weights = (demand * excess[:, None, :]).mean(axis=(0, 2)) + .1 * demand.std(axis=(0, 2))
        budget = count // 2 + (position == 0 and count % 2)
        chosen.extend(members[np.argsort(-weights, kind="stable")[:budget]].tolist())
    if len(chosen) < count:
        other = [i for i in np.flatnonzero(np.isin(labels, pair)) if i not in chosen]
        chosen.extend(other[:count - len(chosen)])
    return tuple(sorted(chosen))


def _destroy_repair(seed, panel, forecast, capacity, rng, operator):
    labels = np.asarray(seed, dtype=int).copy()
    hosts = len(set(seed))
    pair = rng.choice(hosts, size=min(hosts, 2 + (operator == 2)), replace=False)
    members = np.flatnonzero(np.isin(labels, pair))
    if len(members) < 2:
        return None
    widths = (4, 8, 16)
    size = min(len(members), widths[operator])
    mutable = rng.choice(members, size=size, replace=False)
    labels[mutable] = -1
    remaining = [np.flatnonzero(labels == h).tolist() for h in pair]
    prediction = [forecast[m].sum(axis=0) if m else np.zeros(forecast.shape[1]) for m in remaining]
    realized = [panel[:, m, :].sum(axis=1) if m else np.zeros((len(panel), panel.shape[2])) for m in remaining]
    # A noisy regret preference supplies broad diverse seeds. Only the exact
    # certificate-aligned key decides which complete proposals are retained.
    order = sorted(mutable, key=lambda i: -float(forecast[i].max()) * rng.uniform(.8, 1.2))
    for vm in order:
        options = []
        for j, h in enumerate(pair):
            if np.all(prediction[j] + forecast[vm] <= capacity + 1e-12):
                before = np.maximum(realized[j] / capacity - 1., 0.)
                after = np.maximum((realized[j] + panel[:, vm, :]) / capacity - 1., 0.)
                loss = float((after * after - before * before).mean())
                options.append((loss * rng.uniform(.85, 1.15), j, h))
        if not options:
            return None
        _, j, h = min(options)
        labels[vm] = h
        prediction[j] += forecast[vm]
        realized[j] += panel[:, vm, :]
    return canonical_assignment(labels)


def search(inc, forecast, build, selection, capacity, config, rng):
    """Keep a complete local prefix; all scores compare with the fixed incumbent.

    Complete enumeration is used for the small trace residuals. It deliberately
    avoids unsound floating-point partial-load pruning. ``mode='broad'`` is an
    adaptive destroy-and-repair comparator using the identical final score.
    """
    started = process_time()
    inc = canonical_assignment(inc)
    host_count = len(set(inc))
    top = int(config.get("top_p", 16))
    mode = config.get("mode", "cara")
    reference = burdens(build, inc, capacity)
    pool, seen = {}, {inc}
    stats = {"evaluations": 0, "seed_evaluations": 0, "local_evaluations": 0, "blocks": 0, "cache_hits": 0, "changed_values": 0, "full_values": 0}

    def evaluate(assignment, cache, known_feasible=False):
        assignment = canonical_assignment(assignment)
        if assignment in seen or (not known_feasible and not feasible(assignment, forecast, capacity, host_count)):
            return False
        seen.add(assignment)
        values = cache(assignment) if cache is not None else burdens(build, assignment, capacity)
        bets = tuple(map(int, fit_bets(reference, values)))
        key = tuple(map(int, evidence_key(reference, values, bets=bets, center=False, binary_mixture=True)))
        candidate = Candidate(assignment, bets, key)
        old = _ordered(pool)[0].key if pool else None
        _retain(pool, candidate, top)
        stats["evaluations"] += 1
        return old is None or key > old

    weights = np.ones(3)
    tries = 0
    if mode != "exact" and host_count >= 2:
        budget = config.get("cpu_limit")
        eval_limit = config.get("eval_limit")
        limit = int(config.get("broad_trials", 128))
        if budget is not None or eval_limit is not None:
            limit = max(limit, 100000)
        while tries < limit:
            if budget is not None and tries > 0 and process_time() - started >= budget:
                break
            if eval_limit is not None and stats["evaluations"] >= eval_limit:
                break
            ordered = _ordered(pool)
            seed = inc if not ordered or rng.random() < .2 else ordered[int(rng.integers(min(len(ordered), 4)))].assignment
            operator = int(rng.choice(3, p=weights / weights.sum()))
            proposal = _destroy_repair(seed, build, forecast, capacity, rng, operator)
            if proposal is not None:
                improved = evaluate(proposal, None)
                weights[operator] = .95 * weights[operator] + .05 * (5. if improved else 1.)
            tries += 1
            if eval_limit is not None and tries >= max(2000, 100 * eval_limit):
                break
    stats["seed_evaluations"] = stats["evaluations"]
    stats["seed_cpu_seconds"] = process_time() - started
    seed_candidates = _ordered(pool)
    seed_selected = select_candidate(seed_candidates, inc, selection, capacity)
    visited_blocks = set()
    if mode != "broad" and host_count >= 2:
        for round_index in range(int(config.get("local_rounds", 2))):
            ranked = _ordered(pool)
            seed = ranked[0].assignment if ranked else inc
            cache = ReplayCache(build, seed, capacity) if config.get("reuse", True) else None
            pairs = _pair_order(build, seed, capacity, rng)
            completed = 0
            for pair in pairs:
                mutable = _mutable(build, seed, pair, int(config.get("mutable_limit", 8)), capacity)
                block_id = (seed, pair, mutable)
                if block_id in visited_blocks:
                    continue
                visited_blocks.add(block_id)
                labels = np.asarray(seed, dtype=int)
                fixed = [i for i in np.flatnonzero(np.isin(labels, pair)) if i not in mutable]
                symmetric = not fixed
                # Both dense sides have fixed identities unless both backgrounds
                # are empty; no other symmetry restriction is valid.
                decisions = len(mutable) - int(symmetric)
                for mask in range(1 << decisions):
                    proposal = labels.copy()
                    for j, vm in enumerate(mutable):
                        side = 0 if symmetric and j == 0 else (mask >> (j - int(symmetric))) & 1
                        proposal[vm] = pair[side]
                    changed_groups = [np.flatnonzero(proposal == h) for h in pair]
                    if any(len(group) == 0 for group in changed_groups):
                        continue
                    if any(np.any(forecast[group].sum(axis=0) > capacity + 1e-12) for group in changed_groups):
                        continue
                    evaluate(proposal, cache, known_feasible=True)
                stats["blocks"] += 1
                completed += 1
                if completed >= int(config.get("pair_limit", 6)):
                    break
            if cache:
                stats["cache_hits"] += cache.hits
                stats["changed_values"] += cache.changed_values
                stats["full_values"] += cache.full_values
    candidates = _ordered(pool)
    chosen = select_candidate(candidates, inc, selection, capacity)
    stats["local_evaluations"] = stats["evaluations"] - stats["seed_evaluations"]
    stats["cpu_seconds"] = process_time() - started
    stats["local_cpu_seconds"] = stats["cpu_seconds"] - stats["seed_cpu_seconds"]
    return {"selected": chosen["assignment"] if chosen else None, "bets": chosen["bets"] if chosen else None, "candidates": [c.record() for c in candidates], "seed_candidates": [c.record() for c in seed_candidates], "seed_selected": seed_selected["assignment"] if seed_selected else None, "seed_bets": seed_selected["bets"] if seed_selected else None, "stats": stats}
