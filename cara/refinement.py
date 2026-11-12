"""Migration-constrained, diagnostic residual refinement for CARA.

Diagnostics and all search feedback use Build only. Each completed family is
resolved by canonical evaluation or a certified exclusion preserving retention.
Selection fixes one action; this module does not inspect admission evidence.
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import combinations
from time import process_time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .admission import fit_bets
from .data import burdens, burdens_from_loads, canonical_assignment, host_loads, host_memberships
from .repair import ReplayCache, _destroy_repair, _mutable, _pair_order, feasible

PROFILE_NAMES = ("J", "V", "VR", "raw", "VS", "PAR")


class _FullReplay:
    """Full canonical replay for the controlled, fixed-path execution ablation."""
    def __init__(self, panel):
        self.panel = np.asarray(panel)
        self.hits = self.changed_values = self.full_values = 0

    def loads(self, assignment):
        self.full_values += self.panel.size
        self.changed_values += self.panel.size
        return host_loads(self.panel, assignment)


def _replay(panel, seed, capacity, reuse=True, project=False):
    # Full replay does not pay for an unused seed materialization. Cached replay
    # charges that materialization when its first seed view is constructed.
    if not reuse:
        return _FullReplay(panel)
    if project == "shared":
        from .query_engine import SharedQueryReplay
        return SharedQueryReplay(panel, seed, capacity)
    if project == "auto":
        # Fixed shape rule, not a learned cost model: short windows retain the
        # cheaper load-only path; larger row-time products use host projections.
        shape = np.shape(panel)
        project = shape[0] * shape[2] >= 256
    if project:
        from .workload_views import ProjectionReplay
        return ProjectionReplay(panel, seed, capacity)
    return ReplayCache(panel, seed, capacity)


def _profile_replay(cache, assignment, capacity):
    if hasattr(cache, "profile"):
        return cache.profile(assignment), None
    loads = cache.loads(assignment)
    return profile_from_loads(loads, capacity), loads


def migration_profile(incumbent, candidate, weights=None):
    """Minimum moved-VM count, breaking matching ties by migration cost.

    A *single* physical host matching determines both reported quantities.
    Weights describe per-VM movement cost; their units are supplied by the caller.
    The matching is count-first, not an unconstrained minimum-cost matching.
    """
    incumbent = np.asarray(canonical_assignment(incumbent), dtype=int)
    candidate = np.asarray(canonical_assignment(candidate), dtype=int)
    if incumbent.shape != candidate.shape or incumbent.ndim != 1 or not len(incumbent):
        raise ValueError("Complete nonempty assignments must have the same shape")
    h = len(set(incumbent))
    if len(set(candidate)) != h:
        raise ValueError("Migration matching requires the same host count")
    explicit_weights = weights is not None
    weights = np.ones(len(incumbent)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != incumbent.shape or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Movement weights must be finite, nonnegative and VM-aligned")
    counts = np.zeros((h, h), dtype=np.int64)
    np.add.at(counts, (incumbent, candidate), 1)
    if explicit_weights:
        from .migration import weighted_host_matching
        old_hosts, new_hosts = weighted_host_matching(counts, incumbent, candidate, weights)
    else:
        # Preserve the original fast unit-weight path and its deterministic
        # host mapping. Count ties have identical unit movement costs.
        score = counts.astype(float) + counts / (2 * len(incumbent))
        old_hosts, new_hosts = linear_sum_assignment(-score)
    mapping = np.empty(h, dtype=int)
    mapping[new_hosts] = old_hosts
    aligned = mapping[candidate]
    moved = aligned != incumbent
    with np.errstate(over="ignore"):
        moved_cost = float(weights[moved].sum())
    if not np.isfinite(moved_cost):
        raise ValueError("The matched movement cost exceeds the finite binary64 output range")
    return {"count": int(moved.sum()), "fraction": float(moved.mean()),
            "cost": moved_cost, "aligned_assignment": tuple(map(int, aligned)),
            "moved_vms": tuple(map(int, np.flatnonzero(moved)))}


def profile_from_loads(loads, capacity, tolerance=1e-12):
    """Per-scene J, V, native VR, uncapped raw excess, native VS and PAR.

    J/V retain CARA's tolerance.  Native VR/VS use strict capacity and exclude
    zero-load hosts exactly as data.native_metrics does.  Native VR is *not*
    silently identified with CARA's tolerance-based, fixed-H duration R.
    """
    loads = np.asarray(loads, dtype=float)
    if loads.ndim != 3 or not len(loads) or capacity <= 0 or not np.isfinite(loads).all() or np.any(loads < 0):
        raise ValueError("Expected finite nonnegative (hosts, scenes, time) loads")
    active = np.any(loads != 0, axis=2)
    active_count = active.sum(axis=0)
    native_overload = loads > capacity
    overload = loads > capacity + tolerance
    excess = np.maximum(loads / capacity - 1., 0.)
    violating_times = native_overload.sum(axis=2)
    affected = violating_times > 0
    severity = np.divide(excess.sum(axis=2), violating_times,
                         out=np.zeros_like(violating_times, dtype=float), where=affected)
    vs = np.divide(severity.sum(axis=0), affected.sum(axis=0),
                   out=np.zeros(loads.shape[1]), where=affected.any(axis=0))
    vr = np.divide(native_overload.mean(axis=2).sum(axis=0), active_count,
                   out=np.full(loads.shape[1], -1.), where=active_count > 0)
    raw = np.divide(excess.mean(axis=2).sum(axis=0), active_count,
                    out=np.zeros(loads.shape[1]), where=active_count > 0)
    peaks = loads.max(axis=2)
    mean_peak = np.divide(peaks.sum(axis=0), active_count,
                          out=np.zeros(loads.shape[1]), where=active_count > 0)
    par = np.divide(peaks.max(axis=0), mean_peak,
                    out=np.full(loads.shape[1], -1.), where=mean_peak > 0)
    return np.column_stack((overload.any(axis=(0, 2)), overload.any(axis=2).mean(axis=0),
                            vr, raw, vs, par))


def panel_profile(panel, assignment, capacity):
    return profile_from_loads(host_loads(panel, assignment), capacity)


def _coordinates(names):
    if not names or any(name not in PROFILE_NAMES for name in names):
        raise ValueError("Objective names must be a nonempty subset of PROFILE_NAMES")
    return tuple(PROFILE_NAMES.index(name) for name in names)


def _coverage_archive(records, limit):
    """Keep balanced repairs and complementary coordinate specialists.

    The balanced half uses CARA's existing order. Each remaining slot protects
    a coordinate extreme, with that same order breaking ties. No Selection or
    Admission value enters the archive. This is a bounded proposal heuristic,
    not a Pareto-completeness or optimization guarantee.
    """
    ordered = _ordered_records(records)
    if len(ordered) <= limit:
        return ordered
    retained = ordered[:max(1, limit // 2)]
    included = {r["assignment"] for r in retained}
    if len(retained) == limit:
        return retained
    # max/min retain the first equal item. The existing balanced order already
    # supplies both tie breaks, without repeatedly constructing N-label keys.
    extremes = [max(ordered, key=lambda r: r["relative_gains"][e])
                for e in range(len(ordered[0]["relative_gains"]))]
    extremes.append(min(ordered, key=lambda r: r["movement"]["count"]))
    for record in extremes + ordered:
        if record["assignment"] not in included:
            retained.append(record)
            included.add(record["assignment"])
        if len(retained) == limit:
            break
    return retained


def _ordered_records(records):
    """Descending score, then ascending placement, without negating every VM.

    Python's stable second sort preserves the placement order for score ties.
    The result is identical to sorting by (score, negative-placement) in reverse.
    """
    return sorted(sorted(records, key=lambda r: r["assignment"]),
                  key=lambda r: r["key"], reverse=True)


def _relative_gains(reference, candidate, names):
    indices = _coordinates(names)
    means = reference.mean(axis=0)
    scale = np.maximum(np.abs(means[list(indices)]), 1e-8)
    return (reference[:, indices] - candidate[:, indices]).mean(axis=0) / scale


def diagnostic_weights(reference, candidate, names=("VR", "raw", "VS")):
    """Find the weakest Build coordinate and emphasize its harmful rows.

    If no row is harmful, emphasize the least-improved rows.  A 25% uniform
    component avoids forgetting the rest of Build.  These are heuristic search
    weights, never sampling probabilities for a statistical certificate.
    """
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != len(PROFILE_NAMES):
        raise ValueError("Diagnostics require paired six-coordinate profiles")
    relative = _relative_gains(reference, candidate, names)
    name = names[int(np.argmin(relative))]
    gain = reference[:, PROFILE_NAMES.index(name)] - candidate[:, PROFILE_NAMES.index(name)]
    hard = np.maximum(-gain, 0.)
    if not np.any(hard):
        scale = max(float(np.std(gain)), 1e-12)
        hard = np.exp(-(gain - gain.min()) / scale)
    weights = .25 / len(gain) + .75 * hard / hard.sum()
    return name, weights


def diagnostic_blocks(panel, seed, reference, capacity, mutable_limit, names=("VR", "raw", "VS"),
                      excluded_pairs=(), pair_limit=None):
    """Return every host pair in a deterministic diagnostic priority order."""
    labels = np.asarray(seed)
    loads = host_loads(panel, seed)
    bottleneck, weights = diagnostic_weights(reference, profile_from_loads(loads, capacity), names)
    excess = np.maximum(loads / capacity - 1., 0.)
    responsibility = (loads > capacity).astype(float) if bottleneck == "VR" else excess
    risk = (responsibility.mean(axis=2) * weights[None, :]).sum(axis=1)
    excluded_pairs = set(excluded_pairs)
    pairs = sorted((p for p in combinations(range(len(loads)), 2) if p not in excluded_pairs),
                   key=lambda p: (-(risk[p[0]] + risk[p[1]]), p))
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    blocks = []
    for pair in pairs:
        selected = []
        for side, host in enumerate(pair):
            members = np.flatnonzero(labels == host)
            demand = panel[:, members, :]
            # VM attribution uses demands on precisely the rows and host-time
            # cells implicated by the diagnostic, plus a variation tie-break.
            relevance = responsibility[host] + .05
            score = (demand * relevance[:, None, :] * weights[:, None, None]).mean(axis=2).sum(axis=0)
            score += .01 * demand.std(axis=(0, 2))
            budget = mutable_limit // 2 + (side == 0 and mutable_limit % 2)
            selected.extend(members[np.argsort(-score, kind="stable")[:budget]].tolist())
        if len(selected) < mutable_limit:
            selected.extend(i for i in np.flatnonzero(np.isin(labels, pair))
                            if i not in selected)
            selected = selected[:mutable_limit]
        blocks.append((pair, tuple(sorted(map(int, selected)))))
    return blocks, {"bottleneck": bottleneck, "maximum_row_weight": float(weights.max()),
                    "effective_rows": float(1. / np.square(weights).sum())}


def _preview_blocks(blocks, panel, seed, forecast, capacity, names, pair_limit=8, rows=8, reuse=True, project=False):
    """Order a short pair list with cheap swap rollouts on a fixed Build sketch.

    All blocks are returned.  The preview is not a feasibility filter, score
    bound or statistical certificate.  Its overhead is part of proposal CPU.
    """
    indices = np.linspace(0, len(panel) - 1, min(rows, len(panel)), dtype=int)
    sketch = panel[indices]
    cache = _replay(sketch, seed, capacity, reuse, project)
    base, _ = _profile_replay(cache, seed, capacity)
    prediction = host_loads(forecast, seed)
    labels = np.asarray(seed)
    scored = []
    for position, (pair, mutable) in enumerate(blocks[:pair_limit]):
        best = tuple([0.] * len(names))
        left = [vm for vm in mutable if labels[vm] == pair[0]]
        right = [vm for vm in mutable if labels[vm] == pair[1]]
        for vm in left:
            for other in right:
                if (np.any(prediction[pair[0]] - forecast[vm] + forecast[other] > capacity + 1e-12)
                        or np.any(prediction[pair[1]] - forecast[other] + forecast[vm] > capacity + 1e-12)):
                    continue
                proposal = labels.copy()
                proposal[vm], proposal[other] = pair[1], pair[0]
                prof, _ = _profile_replay(cache, proposal, capacity)
                score = tuple(sorted(map(float, _relative_gains(base, prof, names))))
                best = max(best, score)
        scored.append((best, position, (pair, mutable)))
    ordered = [row[2] for row in sorted(scored, key=lambda row: (row[0], -row[1]), reverse=True)]
    ordered.extend(blocks[pair_limit:])
    return ordered, {"preview_pairs": len(scored), "preview_rows": len(indices),
                     "best_preview_gain": list(max((row[0] for row in scored), default=()))}


def _constraints(incumbent, config):
    if "max_migrations" in config:
        limit = int(config["max_migrations"])
        if limit != config["max_migrations"]:
            raise ValueError("max_migrations must be an integer")
    else:
        fraction = float(config.get("migration_fraction", .2))
        if not 0 <= fraction <= 1:
            raise ValueError("migration_fraction must be in [0, 1]")
        limit = int(np.floor(len(incumbent) * fraction))
    if not 0 <= limit <= len(incumbent):
        raise ValueError("Migration count is outside the workload size")
    weights = config.get("migration_weights")
    # Validate weights even when the search does not produce a candidate.
    migration_profile(incumbent, incumbent, weights)
    cost_limit = float(config.get("migration_cost_limit", np.inf))
    if np.isnan(cost_limit) or cost_limit < 0:
        raise ValueError("Migration cost limit must be nonnegative")
    return limit, weights, cost_limit


def _select(records, incumbent, selection, capacity, config):
    if not records:
        return None, {"selection_candidates": 0, "selection_guard_passed": 0}
    reference = panel_profile(selection, incumbent, capacity)
    guard_names = tuple(config.get("selection_guard_names", PROFILE_NAMES))
    guard = _coordinates(guard_names)
    objectives = tuple(config.get("objective_names", PROFILE_NAMES))
    tolerance = float(config.get("selection_tolerance", 1e-12))
    if not 0 <= tolerance <= 1e-10:
        raise ValueError("Selection tolerance is only a numerical tolerance")
    penalty = float(config.get("selection_dispersion", 0.))
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Selection dispersion penalty must be finite and nonnegative")
    eligible, gains = [], []
    if config.get("batch_queries", False):
        from .query_engine import SharedQueryReplay
        reader = SharedQueryReplay.from_config(selection, incumbent, capacity, config)
        profiles = reader.profiles([record["assignment"] for record in records])
    else:
        profiles = [panel_profile(selection, record["assignment"], capacity) for record in records]
    for record, candidate in zip(records, profiles):
        mean_gain = (reference - candidate).mean(axis=0)
        if np.all(mean_gain[list(guard)] >= -tolerance) and np.any(mean_gain[list(guard)] > tolerance):
            eligible.append(record)
            relative = _relative_gains(reference, candidate, objectives)
            if penalty:
                indices = _coordinates(objectives)
                scale = np.maximum(np.abs(reference.mean(axis=0)[list(indices)]), 1e-8)
                relative = relative - penalty * ((reference[:, indices] - candidate[:, indices]) / scale).std(axis=0)
            gains.append(relative)
    if not eligible:
        return None, {"selection_candidates": len(records), "selection_guard_passed": 0}
    gains = np.asarray(gains)
    ranks = np.empty_like(gains)
    for column in range(gains.shape[1]):
        values = gains[:, column]
        ranks[:, column] = ((values[:, None] > values[None, :]).sum(axis=1)
                            + .5 * (values[:, None] == values[None, :]).sum(axis=1)) / len(values)
    selection_order = config.get("selection_order", "rank")
    if selection_order == "gain":
        ranks = gains
    elif selection_order != "rank":
        raise ValueError("Unknown held-out selection order")
    chosen = max(range(len(eligible)), key=lambda j: (tuple(sorted(ranks[j])),
                 -eligible[j]["movement"]["count"], -eligible[j]["movement"]["cost"], -j))
    return eligible[chosen], {"selection_candidates": len(records), "selection_guard_passed": len(eligible)}


def search_refined(inc, forecast, build, selection, capacity, config, rng):
    """CARA with global migration constraints and diagnostic exact refinement.

    Supported ablations: diagnostic=False, sketch_preview=False, mode='broad',
    frontier_seed=False, score_mode='gain'/'raw'/'evidence'. Gain scores use deterministic stored arithmetic and
    a 2**40 lattice; they are not confidence bounds.  Each declared residual
    block completes before checking CPU.  replay_arms fixes the execution path.
    projected_views='auto' uses projections exactly when Build rows*time >=256;
    this fixed execution choice depends only on the current panel shape.
    """
    started = process_time()
    inc = canonical_assignment(inc)
    forecast, build, selection = map(np.asarray, (forecast, build, selection))
    hosts = len(set(inc))
    if not feasible(inc, forecast, capacity, hosts):
        raise ValueError("The incumbent must satisfy the shared forecast constraint")
    budget = float(config.get("cpu_limit", 2.))
    top = int(config.get("top_p", 16))
    width = int(config.get("mutable_limit", 8))
    fraction = float(config.get("broad_fraction", .25))
    mode = config.get("mode", "cara")
    score_mode = config.get("score_mode", "gain")
    archive_mode = config.get("archive_mode", "balanced")
    seed_schedule = config.get("seed_schedule", "best")
    family_execution = config.get("family_execution", "eager")
    if budget <= 0 or top < 1 or width < 1 or not 0 <= fraction <= 1 or mode not in ("cara", "broad", "exact"):
        raise ValueError("Invalid CPU, pool, block or allocation configuration")
    if score_mode not in ("gain", "raw", "evidence"):
        raise ValueError("score_mode must be gain, raw or evidence")
    if archive_mode not in ("balanced", "coverage") or seed_schedule not in ("best", "coverage"):
        raise ValueError("Unknown archive or residual seed schedule")
    if family_execution not in ("eager", "compact", "certified"):
        raise ValueError("family_execution must be eager, compact or certified")
    if family_execution != "eager" and not (config.get("shared_queries", False) and config.get("batch_queries", False)):
        raise ValueError("Compact families require shared_queries and batch_queries")
    objectives = tuple(config.get("objective_names", PROFILE_NAMES))
    _coordinates(objectives)
    migration_limit, movement_weights, cost_limit = _constraints(inc, config)
    reference = panel_profile(build, inc, capacity)
    from .gain_circuit import GainsContext
    gain_context = GainsContext(reference, objectives)
    reference_burdens = burdens(build, inc, capacity)
    membership_index = None
    if family_execution != "eager":
        from .family_certificates import MembershipIndex
        membership_index = MembershipIndex()

    def seen_key(assignment):
        return assignment if membership_index is None else membership_index.assignment_key(assignment)

    pool, seen, caches, completed, exhausted = {}, {seen_key(inc)}, OrderedDict(), set(), set()
    shared_cache = None
    forecast_views = None
    family_score_context = None
    seed_compiler = None
    if config.get("family_seed_compilation", False) and config.get("family_matching_backend") == "compiled":
        from .seed_matching import SeedMatchingCache
        seed_compiler = SeedMatchingCache(inc)
    if config.get("shared_queries", False):
        from .query_engine import SharedQueryReplay, ForecastViews
        shared_cache = SharedQueryReplay.from_config(build, inc, capacity, config)
        caches[inc] = shared_cache
        forecast_views = ForecastViews.from_config(forecast, capacity, inc, config)
    weights = np.ones(3)
    seed_visits = {}
    feedback = None
    if config.get("feedback_queries", False):
        from .query_feedback import QueryFeedback
        feedback = QueryFeedback()
    replay = config.get("replay_arms")
    stats = {"evaluations": 0, "blocks": 0, "broad_trials": 0, "migration_rejections": 0,
             "forecast_rejections": 0, "history": [], "cache_hits": 0, "projection_hits": 0, "planning_cpu_seconds": 0.,
             "changed_values": 0, "full_values": 0, "migration_limit": migration_limit,
             "objective_names": objectives, "profile_names": PROFILE_NAMES,
             "family_execution": family_execution, "family_work": {}}

    def ordered():
        return _ordered_records(pool.values())

    def residual_seed():
        ranked = ordered()
        if seed_schedule == "best" or not ranked:
            return ranked[0]["assignment"] if ranked else inc
        available = [r for r in ranked if r["assignment"] not in exhausted]
        if not available:
            return ranked[0]["assignment"]
        # A small portfolio avoids spending every residual family around one
        # balanced composition. Completed-family feedback rotates exploration;
        # newly retained seeds enter naturally as their Build outcomes improve.
        available = _coverage_archive(available, min(4, len(available)))
        rank = {r["assignment"]: j for j, r in enumerate(ranked)}
        if feedback is not None:
            available.sort(key=lambda r: rank[r["assignment"]])
            return feedback.choose([r["assignment"] for r in available])
        return min(available, key=lambda r: (seed_visits.get(r["assignment"], 0),
                   rank[r["assignment"]]))["assignment"]

    def cache_for(seed):
        if seed not in caches:
            caches[seed] = (shared_cache.fork(seed) if shared_cache is not None else
                _replay(build, seed, capacity, config.get("reuse_views", True), config.get("projected_views", False)))
            if len(caches) > 4:
                _, cache = caches.popitem(last=False)
                for stat, attr in (("cache_hits", "hits"), ("changed_values", "changed_values"), ("full_values", "full_values")):
                    stats[stat] += getattr(cache, attr)
                stats["projection_hits"] += getattr(cache, "projection_hits", 0)
        caches.move_to_end(seed)
        return caches[seed]

    def score_candidate(assignment, cache, profile, loads, movement, gains=None):
        if gains is None:
            gains = gain_context.scalar(profile)
        gain_key = tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains)) if score_mode != "raw" else ()
        if score_mode == "evidence":
            from .admission import evidence_key
            if loads is None:
                loads = cache.loads(assignment)
            values = burdens_from_loads(loads, capacity)
            bets = fit_bets(reference_burdens, values)
            key = tuple(map(int, evidence_key(reference_burdens, values, bets, center=False))) + gain_key
        elif score_mode == "raw":
            # Optimize continuous workload excess. The operational vector is
            # protected by the unchanged held-out selector and admission gate.
            means = profile.mean(axis=0)
            key = tuple(-int(np.floor(means[e] * (1 << 40))) for e in (3, 2, 4, 5))
        else:
            key = gain_key
        key += (-movement["count"], -movement["cost"])
        return key, gains

    def retain_scored(assignment, profile, movement, key, gains):
        old = max(record["key"] for record in pool.values()) if pool else None
        record = {"assignment": assignment, "key": key, "movement": movement,
                  "build_means": profile.mean(axis=0).tolist(), "relative_gains": gains.tolist()}
        pool[assignment] = record
        if len(pool) > top:
            retained = _coverage_archive(list(pool.values()), top) if archive_mode == "coverage" else ordered()[:top]
            pool.clear()
            pool.update((r["assignment"], r) for r in retained)
        stats["evaluations"] += 1
        return old is None or key > old

    def evaluate(assignment, cache, known_feasible=False, prepared=None, gains=None):
        assignment = canonical_assignment(assignment)
        identity = seen_key(assignment)
        if identity in seen:
            return False
        seen.add(identity)
        movement = migration_profile(inc, assignment, movement_weights) if prepared is None else prepared[2]
        if movement["count"] > migration_limit or movement["cost"] > cost_limit:
            stats["migration_rejections"] += 1
            return False
        if not known_feasible and not (forecast_views.feasible(assignment, hosts) if forecast_views is not None
                                      else feasible(assignment, forecast, capacity, hosts)):
            stats["forecast_rejections"] += 1
            return False
        if prepared is None:
            if score_mode == "evidence":
                loads = cache.loads(assignment)
                profile = profile_from_loads(loads, capacity)
            else:
                profile, loads = _profile_replay(cache, assignment, capacity)
        else:
            profile, loads, _ = prepared
        key, gains = score_candidate(assignment, cache, profile, loads, movement, gains)
        return retain_scored(assignment, profile, movement, key, gains)

    def broad_batch():
        for _ in range(int(config.get("broad_batch", 32))):
            ranked = ordered()
            seed = inc if not ranked or rng.random() < .2 else ranked[int(rng.integers(min(len(ranked), 4)))]["assignment"]
            operator = int(rng.choice(3, p=weights / weights.sum()))
            proposal = _destroy_repair(seed, build, forecast, capacity, rng, operator)
            if proposal is not None and len(set(proposal)) == hosts:
                improved = evaluate(proposal, cache_for(seed))
                weights[operator] = .95 * weights[operator] + .05 * (5. if improved else 1.)
            stats["broad_trials"] += 1

    def compact_family(seed, pair, mutable, cache):
        nonlocal family_score_context
        from .family_certificates import FamilyCertificates, ResidualFamily
        family = ResidualFamily(inc, seed, pair, mutable,
                                reuse_migration_signatures=config.get("family_migration_reuse", True),
                                factor_matching=config.get("family_factor_matching", True),
                                matching_backend=config.get("family_matching_backend"),
                                use_unique_physical=config.get("family_unique_physical", True),
                                unique_physical_min_hosts=config.get("family_unique_physical_min_hosts", 16),
                                seed_matching_index=None if seed_compiler is None else seed_compiler.get(seed))
        certificates = FamilyCertificates(family, forecast_views)
        certified = family_execution == "certified"
        observer = config.get("certificate_observer")

        def observe(witness):
            if observer is not None:
                observer({**witness, "family": family, "reference_profile": reference,
                          "objective_names": objectives})

        passed = certificates.forecast_masks(
            certify=certified and config.get("family_certificate_forecast", True),
            observer=observe if observer is not None else None,
            strategy=config.get("family_forecast_strategy", "tree"))
        active = tuple(mask for mask in passed if family.identity(mask, membership_index) not in seen)
        cutoff = min(record["key"] for record in pool.values()) if len(pool) == top else None
        bootstrap_bounds = (config.get("family_bootstrap_bounds", False)
                            and config.get("family_progressive", True)
                            and config.get("family_score_strategy", "tree") == "leaf")
        from .family_score_bounds import should_bound_profiles
        can_score = (certified and config.get("family_certificate_score", True)
                     and (cutoff is not None or bootstrap_bounds)
                     and archive_mode == "balanced"
                     and score_mode == "gain" and movement_weights is None
                     and should_bound_profiles(config, hosts, build.shape[2],
                         sum(len(family.seed_groups[h]) for h in pair)))
        minimum = int(config.get("family_score_min_leaves", 8))
        score_bounds = None
        if can_score and len(active) >= minimum:
            from .family_score_bounds import FamilyScoreContext
            if family_score_context is None:
                family_score_context = FamilyScoreContext.from_reader(cache, reference, objective_names=objectives,
                    compiled_views=config.get("family_bound_views", False),
                    compiled_subsets=config.get("family_subset_views", False),
                    compiled_bounds=config.get("family_compiled_bounds", False))
            score_bounds = family_score_context.for_family(seed, pair, mutable,
                seed_loads=None if config.get("family_bound_views", False) else cache.loads(seed))
        count_limit = (min(migration_limit, cost_limit) if movement_weights is None else migration_limit)
        surviving, excluded = certificates.prune(active, migration_limit=count_limit,
            cutoff=cutoff, score_bounds=score_bounds,
            certify_migration=certified and config.get("family_certificate_migration", True),
            certify_score=can_score, score_min_leaves=minimum,
            score_strategy=config.get("family_score_strategy", "tree"),
            metric_cascade=config.get("family_metric_cascade", False),
            staged_acquisition=config.get("family_staged_acquisition", False), top_p=top,
            observer=observe if observer is not None else None)
        for witness in excluded:
            for mask in witness["active_masks"]:
                seen.add(family.identity(mask, membership_index))
            if witness["reason"] == "migration":
                stats["migration_rejections"] += len(witness["active_masks"])
        actions, movements, assignments, pending_mask = [], [], {}, {}
        rich_matchings = 0
        for mask in surviving:
            seen.add(family.identity(mask, membership_index))
            if movement_weights is None:
                count = family.movement_count(mask)
                movement = {"count": count, "cost": float(count)}
            else:
                assignments[mask] = family.assignment(mask)
                movement = migration_profile(inc, assignments[mask], movement_weights)
                rich_matchings += 1
            if movement["count"] > migration_limit or movement["cost"] > cost_limit:
                stats["migration_rejections"] += 1
                continue
            actions.append(mask)
            movements.append(movement)
        progressive = (can_score and config.get("family_score_strategy", "tree") == "leaf"
                       and config.get("family_progressive", True) and certificates.has_score_schedule)
        batches = (certificates.score_batches(actions,
                   lambda: min(record["key"] for record in pool.values()) if len(pool) == top else None,
                   batch_size=config.get("family_progressive_batch", 32),
                   observer=observe if observer is not None else None) if progressive else (actions,))
        movement_by_mask = dict(zip(actions, movements))
        profiled = 0
        for batch in batches:
            profiles = cache.profiles_groups([family.memberships(mask) for mask in batch])
            batch_gains = gain_context.batch(profiles)
            profiled += len(batch)
            for mask, profile, gains in zip(batch, profiles, batch_gains):
                movement = movement_by_mask[mask]
                assignment = assignments.get(mask)
                if score_mode == "evidence" and assignment is None:
                    assignment = family.assignment(mask)
                key, gains = score_candidate(assignment, cache, profile, None, movement, gains)
                # A strict score loss cannot enter a full balanced pool. Exact ties
                # still materialize their canonical labels for the original tie break.
                if archive_mode == "balanced" and len(pool) == top and key < min(r["key"] for r in pool.values()):
                    stats["evaluations"] += 1
                    continue
                if assignment is None:
                    assignment = family.assignment(mask)
                if movement_weights is None and not config.get("family_defer_physical", True):
                    physical = family.unique_physical_profile(mask, assignment) if family.use_unique_physical else None
                    if physical is None:
                        physical = migration_profile(inc, assignment)
                        rich_matchings += 1
                    if physical["count"] != movement["count"] or physical["cost"] != movement["cost"]:
                        raise ArithmeticError("Compact migration count disagrees with the physical matching")
                    movement = physical
                retain_scored(assignment, profile, movement, key, gains)
                if (family.use_unique_physical and assignment in pool
                        and "aligned_assignment" not in movement):
                    pending_mask[assignment] = mask
                    if len(pending_mask) > top:
                        pending_mask = {a: m for a, m in pending_mask.items() if a in pool}
        # Only complete-family outputs are visible to the next operation.
        # A provisional entrant evicted within this family needs no physical
        # permutation. A strict family certificate resolves unique mappings;
        # every uncertain or tied mapping retains the original physical solver.
        if movement_weights is None:
            for assignment, record in pool.items():
                if "aligned_assignment" in record["movement"]:
                    continue
                physical = (family.unique_physical_profile(pending_mask[assignment], assignment)
                            if family.use_unique_physical else None)
                if physical is None:
                    physical = migration_profile(inc, assignment)
                    rich_matchings += 1
                if physical["count"] != record["movement"]["count"]:
                    raise ArithmeticError("Factored movement disagrees with physical matching")
                record["movement"] = physical
        work = {**family.stats, **certificates.stats, "declared_masks": family.mask_count,
                "forecast_passed_masks": len(passed), "duplicate_masks": len(passed) - len(active),
                "profiled_masks": profiled, "physical_matching_calls": rich_matchings}
        if score_bounds is not None:
            work.update(score_bounds.diagnostics())
        for name, value in work.items():
            stats["family_work"][name] = stats["family_work"].get(name, 0) + value
        return work

    def exact_block():
        seed = residual_seed()
        plan_started = process_time()
        excluded = {pair for old_seed, pair, _ in completed if old_seed == seed}
        pair_limit = int(config.get("preview_pairs", 8)) if config.get("sketch_preview", False) else 1
        if config.get("diagnostic", True):
            blocks, diagnostics = diagnostic_blocks(build, seed, reference, capacity, width,
                  tuple(config.get("diagnostic_names", ("VR", "raw", "VS"))), excluded, pair_limit)
        else:
            pairs = [pair for pair in _pair_order(build, seed, capacity, rng) if pair not in excluded][:pair_limit]
            blocks = [(pair, _mutable(build, seed, pair, width, capacity)) for pair in pairs]
            diagnostics = {"bottleneck": "mean_excess"}
        if config.get("sketch_preview", False) and blocks:
            blocks, preview = _preview_blocks(blocks, build, seed, forecast, capacity, objectives,
                 int(config.get("preview_pairs", 8)), int(config.get("preview_rows", 8)),
                 config.get("reuse_views", True), config.get("projected_views", False))
            diagnostics.update(preview)
        planning = process_time() - plan_started
        stats["planning_cpu_seconds"] += planning
        diagnostics["planning_cpu"] = planning
        if not blocks:
            exhausted.add(seed)
            return False, diagnostics
        pair, mutable = blocks[0]
        seed_visits[seed] = seed_visits.get(seed, 0) + 1
        completed.add((seed, pair, mutable))
        labels = np.asarray(seed, dtype=int)
        fixed = [i for i in np.flatnonzero(np.isin(labels, pair)) if i not in mutable]
        symmetric = not fixed
        cache = cache_for(seed)
        before = stats["evaluations"]
        if family_execution != "eager":
            diagnostics["family_work"] = compact_family(seed, pair, mutable, cache)
            masks = ()
        else:
            masks = range(1 << (len(mutable) - int(symmetric)))
        pending = []
        pending_seen = set()
        if family_execution == "eager" and config.get("batch_queries", False) and forecast_views is not None:
            backgrounds = [tuple(i for i in np.flatnonzero(labels == host) if i not in mutable)
                           for host in pair]
            group_sets = []
            for mask in masks:
                sides = [[], []]
                for j, vm in enumerate(mutable):
                    side = 0 if symmetric and j == 0 else (mask >> (j - int(symmetric))) & 1
                    sides[side].append(vm)
                group_sets.append(tuple(tuple(sorted(backgrounds[side] + tuple(sides[side])))
                                        for side in range(2)))
            fits = forecast_views.groups_feasible_many(group_sets)
            masks = [mask for mask, fits_one in zip(masks, fits) if fits_one]
        for mask in masks:
            proposal = labels.copy()
            for j, vm in enumerate(mutable):
                side = 0 if symmetric and j == 0 else (mask >> (j - int(symmetric))) & 1
                proposal[vm] = pair[side]
            groups = [np.flatnonzero(proposal == h) for h in pair]
            if any(not len(g) for g in groups):
                continue
            if config.get("batch_queries", False) and hasattr(cache, "profiles"):
                assignment = canonical_assignment(proposal)
                if assignment not in seen and assignment not in pending_seen:
                    pending.append(assignment)
                    pending_seen.add(assignment)
            elif not any(np.any(forecast[g].sum(axis=0) > capacity + 1e-12) for g in groups):
                evaluate(proposal, cache, True)
        if pending:
            admissible = ([True] * len(pending) if forecast_views is not None else
                          [feasible(a, forecast, capacity, hosts) for a in pending])
            actions, movements = [], []
            for assignment, fits in zip(pending, admissible):
                if not fits:
                    continue
                movement = migration_profile(inc, assignment, movement_weights)
                if movement["count"] > migration_limit or movement["cost"] > cost_limit:
                    seen.add(assignment)
                    stats["migration_rejections"] += 1
                    continue
                actions.append(assignment)
                movements.append(movement)
            profiles = cache.profiles(actions)
            batch_gains = gain_context.batch(profiles)
            for assignment, movement, profile, gains in zip(actions, movements, profiles, batch_gains):
                evaluate(assignment, cache, True, (profile, None, movement), gains)
        stats["blocks"] += 1
        diagnostics.update({"pair": tuple(map(int, pair)), "mutable": tuple(map(int, mutable)),
                            "eligible_scored": stats["evaluations"] - before})
        if seed_schedule == "coverage" and config.get("record_seed_assignments", True):
            diagnostics["seed_assignment"] = seed
        if config.get("family_observer") is not None:
            config["family_observer"](ordered(), {"seed_assignment": seed, **diagnostics})
        return True, diagnostics

    # Charge warm-start attempts to the same proposal budget.  They remain
    # atomic batches, with explicit overshoot rather than hidden free work.
    frontier_enabled = bool(config.get("frontier_seed", False))
    frontier_fraction = float(config.get("frontier_fraction", .5))
    if frontier_enabled and not 0 < frontier_fraction < 1:
        raise ValueError("frontier_fraction must lie strictly between zero and one")
    frontier_remaining = budget - (process_time() - started)
    if frontier_enabled and hosts > 1 and (replay is not None or frontier_remaining > 0):
        frontier_allowance = budget if config.get("frontier_only", False) else budget * frontier_fraction
        if replay is None:
            frontier_allowance = min(frontier_allowance, frontier_remaining)
        frontier_config = {**config, "cpu_limit": frontier_allowance}
        if replay is not None:
            if not replay or not replay[0].startswith("frontier:"):
                raise ValueError("A recorded frontier must identify its attempt count")
            frontier_config["max_attempts"] = int(replay[0].split(":", 1)[1])
        else:
            frontier_config.pop("max_attempts", None)
        tick = process_time()
        frontier_engine = search_swap
        if config.get("frontier_engine", "swap") == "anneal":
            from .baseline_anneal import search_anneal
            frontier_engine = search_anneal
            frontier_config["anneal_objective"] = config.get("frontier_objective", "joint")
            frontier_config["record_cooling"] = True
            if config.get("replay_frontier_progress") is not None:
                frontier_config["replay_cooling"] = config["replay_frontier_progress"]
        frontier = frontier_engine(inc, forecast, build, build, capacity, frontier_config, rng,
            on_candidate=lambda assignment, cache, profile, loads, movement:
                evaluate(assignment, cache, True, (profile, loads, movement)),
            select_action=False, query_context=(shared_cache, forecast_views))
        history = {"arm": f"frontier:{frontier['stats']['attempts']}",
                   "cpu": process_time() - tick,
                   "accepted_steps": frontier["stats"]["accepted_steps"],
                   "eligible_scored": frontier["stats"]["evaluations"],
                   "frontier_assignment": frontier["frontier_assignment"]}
        if "cooling_progress" in frontier["stats"]:
            history["cooling_progress"] = frontier["stats"]["cooling_progress"]
        stats["history"].append(history)
        stats["frontier_cpu_seconds"] = history["cpu"]
        stats["frontier_evaluations"] = frontier["stats"]["evaluations"]
        for key in ("cache_hits", "projection_hits", "changed_values", "full_values"):
            stats[key] += frontier["stats"][key]
    initial = int(config.get("initial_broad", 64)) if mode != "exact" and not frontier_enabled else 0
    batch_size = int(config.get("broad_batch", 32))
    if batch_size <= 0 or initial < 0:
        raise ValueError("Broad attempt counts must be valid")
    initial_batches = 0
    for _ in range((initial + batch_size - 1) // batch_size if hosts > 1 else 0):
        if replay is None and process_time() - started >= budget:
            break
        broad_batch()
        initial_batches += 1
    stats["initial_broad_batches"] = initial_batches
    stats["initial_cpu_seconds"] = process_time() - started
    while hosts > 1 and not config.get("frontier_only", False) and (len(stats["history"]) < len(replay) if replay is not None else process_time() - started < budget):
        seed = residual_seed()
        if replay is None and mode == "exact" and seed in exhausted:
            break
        if replay is not None:
            arm = replay[len(stats["history"])]
        elif mode == "broad" or seed in exhausted or (mode != "exact" and not frontier_enabled and process_time() - started < budget * fraction):
            arm = "broad"
        else:
            arm = "exact"
        before = max(record["key"] for record in pool.values()) if pool else None
        tick = process_time()
        if arm == "broad":
            broad_batch()
            details = {}
        elif arm in ("exact", "exact_unavailable"):
            available, details = exact_block()
            actual_arm = "exact" if available else "exact_unavailable"
            if replay is not None and arm != actual_arm:
                raise ValueError("Recorded residual-family availability changed")
            arm = actual_arm
        else:
            raise ValueError("Unknown replay arm")
        after = max(record["key"] for record in pool.values()) if pool else None
        elapsed = process_time() - tick
        improved = after is not None and (before is None or after > before)
        if feedback is not None and arm == "exact":
            recorded = config.get("replay_feedback_costs")
            measured = elapsed if recorded is None else recorded[len(stats["history"])]
            feedback.update(seed, improved, measured)
            details["feedback_cpu"] = measured
        stats["history"].append({"arm": arm, "cpu": elapsed,
                                  "top_improved": improved, **details})
    records = ordered()
    proposal_cpu = process_time() - started
    selected, selection_stats = _select(records, inc, selection, capacity, config)
    bets = None if selected is None else fit_bets(reference_burdens, burdens(build, selected["assignment"], capacity))
    for cache in caches.values():
        stats["cache_hits"] += cache.hits
        stats["projection_hits"] += getattr(cache, "projection_hits", 0)
        stats["changed_values"] += cache.changed_values
        stats["full_values"] += cache.full_values
    stats.update(selection_stats)
    if shared_cache is not None:
        stats.update({"view_peak_bytes": shared_cache._registry.peak_view_bytes,
                      "view_evictions": shared_cache._registry.evictions})
    if forecast_views is not None:
        stats["forecast_view_hits"] = forecast_views.hits
    if feedback is not None:
        stats["query_feedback"] = feedback.state()
    stats.update({"cpu_seconds": proposal_cpu, "selection_cpu_seconds": process_time() - started - proposal_cpu,
                  "allowance_seconds": budget, "proposal_overshoot_seconds": max(0., proposal_cpu - budget),
                  "local_cpu_seconds": sum(r["cpu"] for r in stats["history"] if r["arm"].startswith("exact"))})
    return {"selected": None if selected is None else selected["assignment"], "bets": bets,
            "candidates": records, "movement": None if selected is None else selected["movement"],
            "stats": stats, "selection_reason": "no_eligible_action" if selected is None else "six_coordinate_noninferiority"}


def search_swap(inc, forecast, build, selection, capacity, config, rng, *, on_candidate=None,
                select_action=True, query_context=None):
    """Independent first-improvement swap/move descent on Build raw excess.

    No CARA diagnostic, evidence order or residual enumeration is used.  The
    same forecast/migration constraints and final held-out selector apply.
    Multiple passes use shuffled VM-pair order; CPU includes every proposal.
    This is a transparent optimization baseline, not a named published method.
    """
    started = process_time()
    inc = canonical_assignment(inc)
    hosts = len(set(inc))
    budget = float(config.get("cpu_limit", 2.))
    limit, weights, cost_limit = _constraints(inc, config)
    top = int(config.get("top_p", 16))
    reference = panel_profile(build, inc, capacity)
    current, current_profile = inc, reference
    pool, seen = {}, {inc}
    stats = {"evaluations": 0, "attempts": 0, "accepted_steps": 0, "migration_rejections": 0,
             "migration_limit": limit, "blocks": 0, "history": [], "cache_hits": 0,
             "projection_hits": 0, "changed_values": 0, "full_values": 0}
    max_attempts = config.get("max_attempts")
    attempts_limit = np.inf if max_attempts is None else int(max_attempts)
    shared_cache = None
    forecast_views = None
    if config.get("shared_queries", False):
        from .query_engine import SharedQueryReplay, ForecastViews
        shared_cache, forecast_views = (query_context if query_context is not None and query_context[0] is not None
                                       else (SharedQueryReplay.from_config(build, inc, capacity, config),
                                             ForecastViews.from_config(forecast, capacity, inc, config)))
    while hosts > 1 and stats["attempts"] < attempts_limit and (max_attempts is not None or process_time() - started < budget):
        labels = np.asarray(current)
        cache = (shared_cache.fork(current) if shared_cache is not None else
                 _replay(build, current, capacity, config.get("reuse_views", True), config.get("projected_views", False)))
        vm_order = rng.permutation(len(inc))
        improved = False
        for left in range(len(vm_order)):
            if improved:
                break
            vm = int(vm_order[left])
            # Single moves and pair swaps cover distinct feasible local steps.
            alternatives = [(None, h) for h in range(hosts) if h != labels[vm]]
            alternatives.extend((int(other), int(labels[other])) for other in vm_order[left + 1:] if labels[other] != labels[vm])
            for other, target in alternatives:
                if stats["attempts"] >= attempts_limit or (max_attempts is None and process_time() - started >= budget):
                    break
                stats["attempts"] += 1
                proposal = labels.copy()
                proposal[vm] = target
                if other is not None:
                    proposal[other] = labels[vm]
                proposal = canonical_assignment(proposal)
                if proposal in seen or len(set(proposal)) != hosts:
                    continue
                seen.add(proposal)
                movement = migration_profile(inc, proposal, weights)
                if movement["count"] > limit or movement["cost"] > cost_limit:
                    stats["migration_rejections"] += 1
                    continue
                if not (forecast_views.feasible(proposal, hosts) if forecast_views is not None else
                        feasible(proposal, forecast, capacity, hosts)):
                    continue
                profile, loads = _profile_replay(cache, proposal, capacity)
                stats["evaluations"] += 1
                if on_candidate is not None:
                    on_candidate(proposal, cache, profile, loads, movement)
                loss = float(profile[:, 3].mean())
                record = {"assignment": proposal, "movement": movement,
                          "key": (-loss, -movement["count"], -movement["cost"]),
                          "build_means": profile.mean(axis=0).tolist()}
                # A CARA frontier streams every candidate into the shared pool;
                # it does not need a second, discarded raw-only retained pool.
                if select_action or on_candidate is None:
                    pool[proposal] = record
                    retained = _ordered_records(pool.values())[:top]
                    pool = {r["assignment"]: r for r in retained}
                if loss < float(current_profile[:, 3].mean()) - 1e-12:
                    current, current_profile = proposal, profile
                    stats["accepted_steps"] += 1
                    improved = True
                    break
            if stats["attempts"] >= attempts_limit or (max_attempts is None and process_time() - started >= budget):
                break
        stats["cache_hits"] += cache.hits
        stats["projection_hits"] += getattr(cache, "projection_hits", 0)
        stats["changed_values"] += cache.changed_values
        stats["full_values"] += cache.full_values
        if not improved:
            break
    proposal_cpu = process_time() - started
    records = _ordered_records(pool.values())
    selected, selection_stats = _select(records, inc, selection, capacity, config) if select_action else (None, {"selection_candidates": 0, "selection_guard_passed": 0})
    bets = None if selected is None else fit_bets(burdens(build, inc, capacity), burdens(build, selected["assignment"], capacity))
    stats.update(selection_stats)
    stats.update({"cpu_seconds": proposal_cpu, "selection_cpu_seconds": process_time() - started - proposal_cpu,
                  "allowance_seconds": budget})
    return {"selected": None if selected is None else selected["assignment"], "bets": bets,
            "candidates": records, "movement": None if selected is None else selected["movement"],
            "frontier_assignment": current, "stats": stats}


def search_refined_legacy(inc, forecast, build, selection, capacity, config, rng):
    """Named ablation retaining the pre-frontier CARA proposal schedule."""
    return search_refined(inc, forecast, build, selection, capacity,
                          {**config, "frontier_seed": False}, rng)
