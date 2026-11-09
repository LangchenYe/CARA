"""Compact two-host residual families and exact exclusion certificates.

Certificates exclude only candidates unable to pass a required predicate or
enter the balanced pool. Optimistic scheduling consults its exact cutoff between
profile batches. Uncertain cases use the canonical forecast and profile readers.
"""
from bisect import bisect_left
from math import inf, nextafter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import canonical_assignment, host_memberships
from .error_budget import sum_error_upper


class MembershipIndex:
    """Collision-safe partition identities from interned complete resident sets."""

    def __init__(self):
        self.groups = []
        self._ids = {}

    def group_id(self, group):
        group = tuple(group)
        identity = self._ids.get(group)
        if identity is None:
            identity = len(self.groups)
            self._ids[group] = identity
            self.groups.append(group)
        return identity

    def groups_key(self, memberships):
        """The supplied groups are in canonical minimum-VM order."""
        return tuple(self.group_id(group) for group in memberships)

    def assignment_key(self, assignment):
        return self.groups_key(host_memberships(assignment))


class ResidualFamily:
    """One immutable seed and the original ordered two-host mask family."""

    def __init__(self, incumbent, seed, pair, mutable, *, reuse_migration_signatures=True,
                 factor_matching=True, matching_backend=None, use_unique_physical=True,
                 unique_physical_min_hosts=16):
        self.incumbent, self.seed = canonical_assignment(incumbent), canonical_assignment(seed)
        self.n_vms, self.hosts = len(self.seed), len(set(self.seed))
        self.pair, self.mutable = tuple(map(int, pair)), tuple(map(int, mutable))
        if (not self.n_vms or self.n_vms > 2**50 or len(self.incumbent) != self.n_vms
                or len(set(self.incumbent)) != self.hosts or len(self.pair) != 2
                or len(set(self.pair)) != 2 or any(h < 0 or h >= self.hosts for h in self.pair)
                or not self.mutable or self.mutable != tuple(sorted(set(self.mutable)))
                or any(vm < 0 or vm >= self.n_vms or self.seed[vm] not in self.pair for vm in self.mutable)):
            raise ValueError("A residual family requires equal-host complete assignments and ordered mutable residents")
        self.seed_groups = host_memberships(self.seed)
        mutable_set = set(self.mutable)
        self.backgrounds = tuple(tuple(vm for vm in self.seed_groups[h] if vm not in mutable_set)
                                 for h in self.pair)
        self.symmetric = not any(self.backgrounds)
        self.free_vms = self.mutable[int(self.symmetric):]
        self.bits, self.mask_count = len(self.free_vms), 1 << len(self.free_vms)
        self.initial_groups = (tuple(sorted(self.backgrounds[0] + self.mutable[:int(self.symmetric)])),
                               self.backgrounds[1])
        self.common_groups = tuple(group for h, group in enumerate(self.seed_groups) if h not in self.pair)
        self._common_minima = tuple(group[0] for group in self.common_groups)
        self._groups, self._identities = {}, {}
        self._common_ids = {}
        self.origins = tuple(sorted({self.incumbent[vm] for vm in self.free_vms}))
        positions = {old_host: j for j, old_host in enumerate(self.origins)}
        self._bit_origins = tuple(positions[self.incumbent[vm]] for vm in self.free_vms)
        totals = [0] * len(self.origins)
        for position in self._bit_origins:
            totals[position] += 1
        self._totals = tuple(totals)
        self.reuse_migration_signatures = bool(reuse_migration_signatures)
        self._migration, self._migration_masks, self._migration_bounds = {}, {}, {}
        self._seed_movement = None
        self.stats = {"group_pairs": 0, "label_materializations": 0,
                      "migration_matching_calls": 0, "migration_signature_hits": 0,
                      "migration_mask_cache_hits": 0,
                      "migration_bound_calls": 0, "migration_bound_hits": 0,
                      "matching_component_solves": 0, "matching_matrix_cells": 0,
                      "matching_active_rows": 0, "matching_active_columns": 0,
                      "matching_unchanged_components": 0}
        self._factor = None
        backend = matching_backend or ("factorized" if factor_matching else "dense")
        if backend not in ("factorized", "dense", "sparse", "compiled", "replay", "locked"):
            raise ValueError("Unknown matching backend")
        if (int(unique_physical_min_hosts) != unique_physical_min_hosts
                or unique_physical_min_hosts < 0):
            raise ValueError("The unique physical mapping gate must be a nonnegative integer")
        self.use_unique_physical = bool(use_unique_physical and backend == "compiled"
                                       and self.hosts >= unique_physical_min_hosts)
        self._unique_physical = None
        if backend != "dense":
            if backend in ("compiled", "replay", "locked"):
                from .compiled_matching import FamilyMatching
                self._factor = FamilyMatching(self.incumbent, self.seed, self.pair,
                    self.mutable, self.symmetric, self.stats,
                    lock_edges=backend != "replay", replay=backend != "locked")
            elif backend == "sparse":
                from .sparse_matching import FamilyMatching
            else:
                from .factorized_matching import FamilyMatching
            if self._factor is None:
                self._factor = FamilyMatching(self.incumbent, self.seed, self.pair,
                                              self.mutable, self.symmetric, self.stats)
        else:
            old, labels = np.asarray(self.incumbent), np.asarray(self.seed)
            self._seed_counts = np.zeros((self.hosts, self.hosts), dtype=np.int64)
            np.add.at(self._seed_counts, (old, labels), 1)
            self._fixed_counts = self._seed_counts.copy()
            for vm in self.mutable:
                self._fixed_counts[self.incumbent[vm], self.seed[vm]] -= 1
            if self.symmetric:
                self._fixed_counts[self.incumbent[self.mutable[0]], self.pair[0]] += 1

    def _mask(self, mask):
        if int(mask) != mask or not 0 <= mask < self.mask_count:
            raise ValueError("Mask is outside this residual family")
        return int(mask)

    def _range(self, start, stop):
        if int(start) != start or int(stop) != stop:
            raise ValueError("A subtree has integer mask endpoints")
        start, stop = int(start), int(stop)
        span = stop - start
        if span <= 0 or span & (span - 1) or start < 0 or stop > self.mask_count or start % span:
            raise ValueError("A subtree must be an aligned dyadic mask interval")
        return start, stop, span.bit_length() - 1

    def nonempty_masks(self):
        left, right = map(len, self.initial_groups)
        return tuple(mask for mask in range(self.mask_count)
                     if left + self.bits - mask.bit_count() and right + mask.bit_count())

    def groups(self, mask):
        mask = self._mask(mask)
        groups = self._groups.get(mask)
        if groups is None:
            sides = [list(group) for group in self.initial_groups]
            for bit, vm in enumerate(self.free_vms):
                sides[(mask >> bit) & 1].append(vm)
            groups = tuple(tuple(sorted(group)) for group in sides)
            self._groups[mask] = groups
            self.stats["group_pairs"] += 1
        return groups

    def _merge(self, mask, common, changed):
        groups = self.groups(mask)
        if not all(groups):
            raise ValueError("An empty host is outside the complete-host family")
        order = (0, 1) if groups[0][0] < groups[1][0] else (1, 0)
        first = bisect_left(self._common_minima, groups[order[0]][0])
        second = bisect_left(self._common_minima, groups[order[1]][0])
        return (common[:first] + (changed[order[0]],) + common[first:second]
                + (changed[order[1]],) + common[second:])

    def memberships(self, mask):
        return self._merge(mask, self.common_groups, self.groups(mask))

    def identity(self, mask, index):
        key = (index, int(mask))
        identity = self._identities.get(key)
        if identity is None:
            if index not in self._common_ids:
                self._common_ids[index] = index.groups_key(self.common_groups)
            changed = tuple(index.group_id(group) for group in self.groups(mask))
            identity = self._merge(mask, self._common_ids[index], changed)
            self._identities[key] = identity
        return identity

    def assignment(self, mask):
        labels = list(self.seed)
        mask = self._mask(mask)
        if self.symmetric:
            labels[self.mutable[0]] = self.pair[0]
        for bit, vm in enumerate(self.free_vms):
            labels[vm] = self.pair[(mask >> bit) & 1]
        self.stats["label_materializations"] += 1
        return canonical_assignment(labels)

    def unique_physical_profile(self, mask, candidate, weights=None):
        """Return an exact rich action when its physical permutation is unique.

        The caller supplies the canonical assignment already materialized for
        this mask. None requests the original physical matcher. Explicit
        movement weights always retain that original matching policy.
        """
        if not self.use_unique_physical or weights is not None:
            return None
        if self._unique_physical is None:
            self._unique_physical = _UniquePhysicalMap(self)
            self.stats["physical_unique_descriptors"] = 1
            self.stats["physical_unique_eligible_families"] = int(self._unique_physical.eligible)
        descriptor = self._unique_physical
        if not descriptor.eligible:
            return None
        result = descriptor.permutation(self._left_signature(mask), self.stats)
        if result is None:
            return None
        seed_mapping, retained = result
        columns = self._merge(mask, descriptor.common_columns, self.pair)
        mapping = seed_mapping[np.asarray(columns, dtype=int)]
        incumbent = np.asarray(self.incumbent, dtype=int)
        candidate = np.asarray(candidate, dtype=int)
        aligned = mapping[candidate]
        moved = aligned != incumbent
        count = int(moved.sum())
        if count != self.n_vms - retained:
            raise ArithmeticError("The unique physical permutation disagrees with its count certificate")
        weights = np.ones(self.n_vms)
        with np.errstate(over="ignore"):
            cost = float(weights[moved].sum())
        self.stats["physical_unique_profiles"] = self.stats.get("physical_unique_profiles", 0) + 1
        return {"count": count, "fraction": float(moved.mean()), "cost": cost,
                "aligned_assignment": tuple(map(int, aligned)),
                "moved_vms": tuple(map(int, np.flatnonzero(moved)))}

    def forced_groups(self, start, stop):
        start, stop, free = self._range(start, stop)
        sides = [list(group) for group in self.initial_groups]
        for bit in range(free, self.bits):
            sides[(start >> bit) & 1].append(self.free_vms[bit])
        return tuple(sorted(sides[0])), tuple(sorted(sides[1])), self.free_vms[:free]

    def _left_signature(self, mask):
        mask = self._mask(mask)
        counts = list(self._totals)
        while mask:
            low = mask & -mask
            counts[self._bit_origins[low.bit_length() - 1]] -= 1
            mask ^= low
        return tuple(counts)

    def _retained(self, matrix):
        # Same stored score as the original unit-weight movement matcher. The
        # bounded-count scope keeps the integer count objective dominant.
        score = matrix.astype(float) + matrix / (2 * self.n_vms)
        rows, columns = linear_sum_assignment(-score)
        self.stats["migration_matching_calls"] += 1
        self.stats["matching_matrix_cells"] += matrix.size
        return int(matrix[rows, columns].sum())

    def _signature_retained(self, left, right):
        if self._factor is not None:
            self.stats["migration_matching_calls"] += 1
            return self._factor.retained(self.origins, left, right)
        counts = self._fixed_counts.copy()
        for row, first, second in zip(self.origins, left, right):
            counts[row, self.pair[0]] += first
            counts[row, self.pair[1]] += second
        return self._retained(counts)

    def movement_count(self, mask):
        """Exact optimum count, not a physical tie mapping or weighted cost."""
        mask = self._mask(mask)
        if mask in self._migration_masks:
            self.stats["migration_mask_cache_hits"] += 1
            return self._migration_masks[mask]
        signature = self._left_signature(mask)
        if self.reuse_migration_signatures and signature in self._migration:
            self.stats["migration_signature_hits"] += 1
            result = self._migration[signature]
        else:
            right = tuple(total - left for left, total in zip(signature, self._totals))
            result = self.n_vms - self._signature_retained(signature, right)
            if self.reuse_migration_signatures:
                self._migration[signature] = result
        self._migration_masks[mask] = result
        return result

    def seed_movement_count(self):
        if self._seed_movement is None:
            if self._factor is not None:
                self.stats["migration_matching_calls"] += 1
                retained = self._factor.seed_retained()
            else:
                retained = self._retained(self._seed_counts)
            self._seed_movement = self.n_vms - retained
        return self._seed_movement

    def movement_lower_bound(self, start, stop):
        """Best possible count in the subtree before forecast/nonempty guards."""
        start, stop, free = self._range(start, stop)
        if not free:
            return self.movement_count(start)
        left = list(self._left_signature(start))
        unresolved = [0] * len(self.origins)
        for position in self._bit_origins[:free]:
            unresolved[position] += 1
            left[position] -= 1
        signature = (tuple(left), tuple(unresolved))
        self.stats["migration_bound_calls"] += 1
        if signature in self._migration_bounds:
            self.stats["migration_bound_hits"] += 1
            return self._migration_bounds[signature]
        first = tuple(forced + pending for forced, pending in zip(left, unresolved))
        second = tuple(total - forced for forced, total in zip(left, self._totals))
        result = self.n_vms - self._signature_retained(first, second)
        self._migration_bounds[signature] = result
        return result


class _ForecastBounds:
    """Mandatory-load lower bounds for every completion of a mask interval."""

    def __init__(self, family, views):
        self.family, self.views = family, views
        source = views.forecast
        self.available = family.bits <= 16
        if not self.available:
            return
        self.base = np.stack([source[list(group)].sum(axis=0) if group else np.zeros(source.shape[1])
                              for group in family.initial_groups])
        self.sums = np.zeros((family.mask_count, source.shape[1]))
        with np.errstate(over="ignore", invalid="ignore"):
            for bit, vm in enumerate(family.free_vms):
                width = 1 << bit
                self.sums[width:2 * width] = self.sums[:width] + source[vm]
        residents = family.seed_groups[family.pair[0]] + family.seed_groups[family.pair[1]]
        count = max(map(len, family.backgrounds)) + len(family.mutable)
        try:
            error = sum_error_upper(count, float(source[list(residents)].max(initial=0.)))
            self.radius = nextafter(2. * error, inf) if error else 0.
        except (OverflowError, ValueError):
            self.available = False
            return
        self.available = bool(np.isfinite(self.base).all() and np.isfinite(self.sums).all()
                              and np.isfinite(self.radius))

    def impossible(self, ranges):
        if not self.available:
            return np.zeros(len(ranges), dtype=bool)
        left, right = [], []
        all_bits = self.family.mask_count - 1
        for start, stop in ranges:
            start, stop, free = self.family._range(start, stop)
            left.append(all_bits ^ ((1 << free) - 1) ^ start)
            right.append(start)
        return self._impossible_indices(left, right)

    def impossible_masks(self, masks):
        if not self.available:
            return np.zeros(len(masks), dtype=bool)
        right = np.asarray(masks, dtype=np.intp)
        left = (self.family.mask_count - 1) ^ right
        return self._impossible_indices(left, right)

    def _impossible_indices(self, left, right):
        with np.errstate(over="ignore", invalid="ignore"):
            a = self.base[0] + self.sums[left]
            b = self.base[1] + self.sums[right]
            lower_a = np.nextafter(a - self.radius, -np.inf)
            lower_b = np.nextafter(b - self.radius, -np.inf)
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        threshold = self.views.capacity + self.views.tolerance
        return finite & (np.any(lower_a > threshold, axis=1) | np.any(lower_b > threshold, axis=1))


class _UniquePhysicalMap:
    """Strict family edges and a unique global two-row completion.

    Signature choices contain only an orientation and an integer value. At
    most two immutable seed-column permutations are retained by this family.
    """

    def __init__(self, family):
        self.eligible = False
        self.choices, self.permutations = {}, {}
        model = family._factor
        incumbent_model = getattr(model, "_incumbent_plan", None)
        self.identity_fixed = incumbent_model is not None
        if self.identity_fixed:
            self.core_rows = tuple(sorted(incumbent_model.rows, key=incumbent_model.rows.get))
            self.core_columns = tuple(sorted(family.pair))
            self.base = incumbent_model.base.copy()
            self.fixed_retained = incumbent_model.fixed_retained
            self.fixed_edges = ()
        else:
            kernel = getattr(model, "kernel", None)
            if kernel is None or len(kernel.rows) != 2 or len(kernel.columns) != 2:
                return
            if any(weight <= row_other + column_other
                   for _, _, weight, row_other, column_other, _ in kernel.fixed_edges):
                return
            self.core_rows, self.core_columns = tuple(kernel.rows), tuple(kernel.columns)
            if set(self.core_columns) != set(family.pair):
                raise ArithmeticError("A physical family certificate may not eliminate edited columns")
            self.fixed_edges = kernel.fixed_edges
            self.fixed_retained = kernel.fixed_value
            self.base = np.zeros((2, 2), dtype=np.int64)
            rows = {row: index for index, row in enumerate(self.core_rows)}
            columns = {column: index for index, column in enumerate(self.core_columns)}
            for variable in model.variables:
                for row, local in variable.rows.items():
                    for side, column in enumerate(variable.changed):
                        self.base[rows[row], columns[column]] = variable.base_columns[side, local]
        self.hosts, self.pair = family.hosts, family.pair
        self.origins, self.totals = family.origins, family._totals
        self.common_columns = tuple(column for column in range(family.hosts) if column not in family.pair)
        self.rows = {row: index for index, row in enumerate(self.core_rows)}
        self.columns = {column: index for index, column in enumerate(self.core_columns)}
        self.eligible = True

    def permutation(self, signature, stats):
        if signature in self.choices:
            stats["physical_unique_signature_hits"] = stats.get("physical_unique_signature_hits", 0) + 1
            choice = self.choices[signature]
        else:
            matrix = self.base.copy()
            for old, left, total in zip(self.origins, signature, self.totals):
                row = self.rows.get(old)
                if row is not None:
                    matrix[row, self.columns[self.pair[0]]] += left
                    matrix[row, self.columns[self.pair[1]]] += total - left
            straight = int(matrix[0, 0]) + int(matrix[1, 1])
            crossed = int(matrix[0, 1]) + int(matrix[1, 0])
            choice = None if straight == crossed else (straight > crossed, self.fixed_retained + max(straight, crossed))
            self.choices[signature] = choice
        if choice is None:
            stats["physical_unique_ties"] = stats.get("physical_unique_ties", 0) + 1
            return None
        orientation, retained = choice
        if orientation not in self.permutations:
            mapping = np.arange(self.hosts, dtype=int) if self.identity_fixed else np.full(self.hosts, -1, dtype=int)
            for row, column, *_ in self.fixed_edges:
                mapping[column] = row
            selected = self.core_rows if orientation else self.core_rows[::-1]
            for column, row in zip(self.core_columns, selected):
                mapping[column] = row
            mapping.flags.writeable = False
            self.permutations[orientation] = mapping
            stats["physical_unique_maps_cached"] = len(self.permutations)
        return self.permutations[orientation], retained


class FamilyCertificates:
    """Exact exclusions with optional optimistic scheduling of retained leaves."""

    def __init__(self, family, forecast_views):
        self.family, self.forecast_views = family, forecast_views
        if len(family.seed) != forecast_views.forecast.shape[0]:
            raise ValueError("Family and forecast must identify the same VMs")
        self.exclusions = []
        self.leaf_upper_keys = {}
        self.stats = {"forecast_nodes": 0, "forecast_bound_batches": 0,
                      "forecast_excluded_masks": 0,
                      "forecast_exact_masks": 0, "migration_nodes": 0,
                      "score_nodes": 0, "migration_excluded_masks": 0,
                      "score_excluded_masks": 0, "progressive_batches": 0,
                      "progressive_excluded_masks": 0}

    def _record(self, reason, start, stop, masks, observer, **details):
        witness = {"reason": reason, "start": int(start), "stop": int(stop),
                   "active_masks": tuple(map(int, masks)), **details}
        self.exclusions.append(witness)
        self.stats[reason + "_excluded_masks"] += len(masks)
        if observer is not None:
            observer(witness)
        return witness

    def score_batches(self, masks, cutoff, *, batch_size=32, observer=None):
        """Yield optimistic-key batches, consulting the exact pool between batches.

        A missing upper key is positive infinity. A finite upper key equal to
        the cutoff still requires profiling because the placement tie may win.
        The consumer must mark every supplied mask seen before this iterator.
        """
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("A positive integer profile batch size is required")
        batch_size = int(batch_size)
        pending = sorted(masks, key=lambda mask: (
            self.leaf_upper_keys.get(mask) is None,
            self.leaf_upper_keys.get(mask) or (), -mask), reverse=True)
        for start in range(0, len(pending), batch_size):
            optimistic = self.leaf_upper_keys.get(pending[start])
            current = cutoff()
            if optimistic is not None and current is not None and optimistic < tuple(current):
                remaining = tuple(sorted(pending[start:]))
                self._record("score", remaining[0], remaining[-1] + 1, remaining, observer,
                    movement_lower_bound=0, optimistic_key=optimistic, cutoff=tuple(current),
                    bound_scope="maximum remaining optimistic key after exact batch retention")
                self.stats["progressive_excluded_masks"] += len(remaining)
                return
            self.stats["progressive_batches"] += 1
            yield tuple(pending[start:start + batch_size])

    def _forecast_leaf(self, masks, bounds, observer):
        rejected = bounds.impossible_masks(masks)
        self.stats["forecast_nodes"] += len(masks)
        self.stats["forecast_bound_batches"] += 1
        counts = np.concatenate(([0], np.cumsum(rejected)))
        pending = [(0, self.family.mask_count, 0, len(masks))]
        while pending:
            start, stop, low, high = pending.pop()
            count = int(counts[high] - counts[low])
            if not count:
                continue
            if count == high - low:
                self._record("forecast", start, stop, masks[low:high], observer,
                    bound_scope="listed active leaves; batched mandatory-load certificates")
                continue
            middle = (start + stop) // 2
            split = bisect_left(masks, middle, low, high)
            if split < high:
                pending.append((middle, stop, split, high))
            if split > low:
                pending.append((start, middle, low, split))
        return tuple(mask for mask, reject in zip(masks, rejected) if not reject)

    def forecast_masks(self, certify=True, observer=None, *, strategy="tree"):
        if strategy not in ("tree", "leaf"):
            raise ValueError("Unknown forecast certificate execution strategy")
        masks = self.family.nonempty_masks()
        if certify and masks:
            bounds = _ForecastBounds(self.family, self.forecast_views)
            if bounds.available and strategy == "leaf":
                masks = self._forecast_leaf(masks, bounds, observer)
            pending = [(0, self.family.mask_count, 0, len(masks))] if bounds.available and strategy == "tree" else []
            exact = []
            while pending:
                rejected = bounds.impossible([(start, stop) for start, stop, _, _ in pending])
                self.stats["forecast_nodes"] += len(pending)
                self.stats["forecast_bound_batches"] += 1
                following = []
                for (start, stop, low, high), reject in zip(pending, rejected):
                    if reject:
                        self._record("forecast", start, stop, masks[low:high], observer)
                    elif stop - start == 1:
                        exact.extend(masks[low:high])
                    else:
                        middle = (start + stop) // 2
                        split = bisect_left(masks, middle, low, high)
                        if split > low:
                            following.append((start, middle, low, split))
                        if split < high:
                            following.append((middle, stop, split, high))
                pending = following
            if bounds.available and strategy == "tree":
                masks = tuple(sorted(exact))
        self.stats["forecast_exact_masks"] += len(masks)
        passing = self.forecast_views.groups_feasible_many([self.family.groups(mask) for mask in masks])
        return tuple(mask for mask, fits in zip(masks, passing) if fits)

    def _migration_classes(self, masks, counts, limit, observer):
        """Compile the shared exact class predicate before the score consumer."""
        pending = [(0, self.family.mask_count, 0, len(masks))]
        passing, excluded = [], []
        while pending:
            start, stop, low, high = pending.pop()
            values = [counts[mask] for mask in masks[low:high]]
            lower, upper = min(values), max(values)
            self.stats["migration_nodes"] += 1
            if lower > limit:
                excluded.append(self._record("migration", start, stop, masks[low:high], observer,
                    movement_lower_bound=int(lower), migration_limit=float(limit),
                    bound_scope="active masks; exact incumbent-histogram equivalence classes"))
            elif upper <= limit:
                passing.extend(masks[low:high])
            else:
                middle = (start + stop) // 2
                split = bisect_left(masks, middle, low, high)
                if split < high:
                    pending.append((middle, stop, split, high))
                if split > low:
                    pending.append((start, middle, low, split))
        return tuple(passing), excluded

    def prune(self, active_masks, *, migration_limit, cutoff=None, score_bounds=None,
              certify_migration=True, certify_score=True, score_min_leaves=8,
              score_strategy="tree", migration_strategy="classes", observer=None):
        self.leaf_upper_keys.clear()
        masks = tuple(map(int, active_masks))
        if masks != tuple(sorted(set(masks))) or any(mask < 0 or mask >= self.family.mask_count for mask in masks):
            raise ValueError("Active masks must be unique and in original numeric order")
        if not masks:
            return (), ()
        if score_min_leaves < 1 or int(score_min_leaves) != score_min_leaves:
            raise ValueError("The score proof requires a positive active-leaf threshold")
        if np.isnan(migration_limit) or migration_limit < 0:
            raise ValueError("A nonnegative unit-movement limit is required")
        if score_strategy not in ("tree", "leaf") or migration_strategy not in ("classes", "tree"):
            raise ValueError("Unknown certificate execution strategy")
        scoring = bool(certify_score and cutoff is not None and score_bounds is not None)
        migration = bool(certify_migration)
        counts, excluded = None, []
        if (migration and migration_strategy == "classes") or scoring:
            counts = {mask: self.family.movement_count(mask) for mask in masks}
        if migration and migration_strategy == "classes":
            masks, excluded = self._migration_classes(masks, counts, migration_limit, observer)
            migration = False
        elif migration and self.family.seed_movement_count() + len(self.family.mutable) <= migration_limit:
            migration = False
        if not masks:
            return (), tuple(excluded)
        if not migration and not scoring:
            return masks, tuple(excluded)
        if scoring and score_strategy == "leaf":
            # A single fine batch avoids repeatedly probing loose ancestors.
            # Matching counts are the same shared exact class predicate used
            # by compact execution, not a new migration optimization claim.
            if migration:
                masks, rejected = self._migration_classes(masks, counts, migration_limit, observer)
                excluded.extend(rejected)
            if len(masks) < score_min_leaves:
                return masks, tuple(excluded)
            movement = [counts[mask] for mask in masks]
            keys = score_bounds.upper_keys([(mask, mask + 1) for mask in masks],
                movement_lower_bounds=movement, cost_lower_bounds=movement)
            if len(keys) != len(masks):
                raise ValueError("One optimistic score key is required per active leaf")
            self.stats["score_nodes"] += len(masks)
            passing = []
            for mask, optimistic in zip(masks, keys):
                if optimistic is not None and len(optimistic) != len(cutoff):
                    raise ValueError("Optimistic and retained keys must use the same coordinates")
                self.leaf_upper_keys[mask] = None if optimistic is None else tuple(optimistic)
                if optimistic is not None and tuple(optimistic) < tuple(cutoff):
                    excluded.append(self._record("score", mask, mask + 1, (mask,), observer,
                        movement_lower_bound=int(counts[mask]), optimistic_key=tuple(optimistic), cutoff=tuple(cutoff)))
                else:
                    passing.append(mask)
            return tuple(passing), tuple(excluded)
        pending = [(0, self.family.mask_count, 0, len(masks))]
        surviving = []
        while pending:
            remaining, lower = [], []
            for start, stop, low, high in pending:
                bound = (min(counts[mask] for mask in masks[low:high]) if counts is not None else
                         self.family.movement_lower_bound(start, stop) if migration else 0)
                self.stats["migration_nodes"] += int(migration)
                if migration and bound > migration_limit:
                    excluded.append(self._record("migration", start, stop, masks[low:high], observer,
                                                 movement_lower_bound=int(bound), migration_limit=float(migration_limit)))
                else:
                    remaining.append((start, stop, low, high))
                    lower.append(bound)
            selected = [j for j, (_, _, low, high) in enumerate(remaining)
                        if scoring and high - low >= score_min_leaves]
            keys = {}
            if selected:
                ranges = [(remaining[j][0], remaining[j][1]) for j in selected]
                movement = [lower[j] for j in selected]
                computed = score_bounds.upper_keys(ranges, movement_lower_bounds=movement,
                                                   cost_lower_bounds=movement)
                if len(computed) != len(selected):
                    raise ValueError("One optimistic score key is required per subtree")
                keys = dict(zip(selected, computed))
                self.stats["score_nodes"] += len(selected)
            following = []
            for j, (start, stop, low, high) in enumerate(remaining):
                optimistic = keys.get(j)
                if optimistic is not None and len(optimistic) != len(cutoff):
                    raise ValueError("Optimistic and retained keys must use the same coordinates")
                if optimistic is not None and tuple(optimistic) < tuple(cutoff):
                    excluded.append(self._record("score", start, stop, masks[low:high], observer,
                        movement_lower_bound=int(lower[j]), optimistic_key=tuple(optimistic), cutoff=tuple(cutoff)))
                elif stop - start == 1 or (not migration and (not scoring or high - low < score_min_leaves)):
                    surviving.extend(masks[low:high])
                else:
                    middle = (start + stop) // 2
                    split = bisect_left(masks, middle, low, high)
                    if split > low:
                        following.append((start, middle, low, split))
                    if split < high:
                        following.append((middle, stop, split, high))
            pending = following
        return tuple(sorted(surviving)), tuple(excluded)
