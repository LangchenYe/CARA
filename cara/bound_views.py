"""Incrementally maintained summaries for repair-family score certificates.

The tree encloses real sums of *stored* host projections.  Its stable resident
slots need not follow canonical host order: only certificates read these sums;
exact retained profiles continue to use their original canonical reductions.
"""
import numpy as np

from .data import canonical_assignment, host_memberships


class BoundViewIndex:
    """One seed's certificate views, updated by changed resident memberships."""

    FANOUT = 16

    def __init__(self, reader, seed):
        self.reader = reader._registry
        self.scenes, self.workloads, self.horizon = self.reader.panel.shape
        if max(self.horizon, self.workloads) > 2**50:
            raise ArithmeticError("The certificate count range is unavailable")
        self.stats = {"compilations": 1, "seed_updates": 0, "changed_groups": 0,
                      "aggregate_nodes": 0, "family_queries": 0, "projected_groups": 0}
        self.seed = None
        self.update(seed)

    def _leaf_values(self, groups):
        values = self.reader._obtain(groups)
        counts, numeric = [], []
        for group in groups:
            active, tolerant, affected, frequency, excess, severity, peaks = values[group][1]
            # RN(RN(n/T)*T) lies within 1/2 of the integer n for T<=2**50.
            native_count = np.rint(frequency * self.horizon).astype(np.int64)
            counts.append(np.stack((active, tolerant, affected, native_count)).astype(np.int64))
            numeric.append(np.stack((frequency, excess, severity, peaks)))
        counts, numeric = np.stack(counts), np.stack(numeric)
        if not np.isfinite(numeric).all():
            raise ArithmeticError("Finite host projections are required for a certificate index")
        self.stats["projected_groups"] += len(groups)
        return counts, numeric

    def _refresh_node(self, level, node):
        from .family_score_bounds import _sum_real_lower, _sum_real_upper

        previous, current = self.levels[level - 1], self.levels[level]
        lo, hi = node * self.FANOUT, min((node + 1) * self.FANOUT, len(previous["count"]))
        count = hi - lo
        current["count"][node] = previous["count"][lo:hi].sum(axis=0)
        current["lower"][node] = _sum_real_lower(previous["lower"][lo:hi], 0, count)
        current["upper"][node] = _sum_real_upper(previous["upper"][lo:hi], 0, count)
        peaks = previous["top"][lo:hi].transpose(0, 2, 1).reshape(-1, self.scenes)
        slots = previous["slot"][lo:hi].transpose(0, 2, 1).reshape(-1, self.scenes)
        take = np.argpartition(peaks, max(0, len(peaks) - 3), axis=0)[-3:]
        chosen = np.take_along_axis(peaks, take, axis=0)
        identities = np.take_along_axis(slots, take, axis=0)
        current["top"][node] = chosen.T
        current["slot"][node] = identities.T
        self.stats["aggregate_nodes"] += 1

    def update(self, seed):
        seed = canonical_assignment(seed)
        if len(seed) != self.workloads:
            raise ValueError("The certificate index needs a complete seed")
        if self.seed == seed:
            return self
        groups = host_memberships(seed)
        if not groups or len(groups) * self.horizon > 2**50:
            raise ArithmeticError("The certificate count range is unavailable")
        if self.seed is None or len(groups) != len(self.groups):
            self.slots = {group: j for j, group in enumerate(groups)}
            counts, numeric = self._leaf_values(groups)
            hosts, scenes = len(groups), self.scenes
            tops = np.zeros((hosts, scenes, 3))
            ids = np.full((hosts, scenes, 3), -1, dtype=np.int64)
            tops[:, :, 0] = numeric[:, 3]
            ids[:, :, 0] = np.arange(hosts)[:, None]
            self.levels = [{"count": counts, "lower": numeric,
                            "upper": numeric, "top": tops, "slot": ids}]
            size = hosts
            while size > 1:
                size = (size + self.FANOUT - 1) // self.FANOUT
                self.levels.append({"count": np.zeros((size, 4, scenes), dtype=np.int64),
                    "lower": np.zeros((size, 4, scenes)), "upper": np.zeros((size, 4, scenes)),
                    "top": np.zeros((size, scenes, 3)),
                    "slot": np.full((size, scenes, 3), -1, dtype=np.int64)})
            changed = range(hosts)
        else:
            incoming = set(groups)
            removed = [group for group in self.slots if group not in incoming]
            added = [group for group in groups if group not in self.slots]
            free = [self.slots.pop(group) for group in removed]
            if len(free) != len(added):
                raise ArithmeticError("Resident slots must preserve the host count")
            changed = free
            if added:
                counts, numeric = self._leaf_values(added)
                leaves = self.levels[0]
                for row, (group, slot) in enumerate(zip(added, free)):
                    self.slots[group] = slot
                    leaves["count"][slot] = counts[row]
                    leaves["lower"][slot] = numeric[row]
                    leaves["top"][slot, :, 0] = numeric[row, 3]
            self.stats["seed_updates"] += 1
            self.stats["changed_groups"] += len(added)
        for level in range(1, len(self.levels)):
            changed = sorted({slot // self.FANOUT for slot in changed})
            for node in changed:
                self._refresh_node(level, node)
        root = self.levels[-1]
        if (not np.isfinite(root["lower"]).all() or not np.isfinite(root["upper"]).all()
                or not np.isfinite(root["top"]).all()):
            raise ArithmeticError("Finite certificate summaries are unavailable")
        self.seed, self.groups = seed, groups
        self.canonical_slots = tuple(self.slots[group] for group in groups)
        return self

    def common(self, pair):
        """Snapshot bounds for all resident groups except the two edited ones."""
        from .family_score_bounds import _down

        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("Two distinct edited groups are required")
        first, second = (self.canonical_slots[h] for h in pair)
        leaves, root = self.levels[0], self.levels[-1]
        counts = root["count"][0] - leaves["count"][first] - leaves["count"][second]
        lower = _down(_down(root["lower"][0] - leaves["lower"][first]) - leaves["lower"][second])
        upper = np.maximum(np.nextafter(np.nextafter(root["upper"][0] - leaves["upper"][first], np.inf)
                                        - leaves["upper"][second], np.inf), 0.)
        tops, slots = root["top"][0], root["slot"][0]
        peak = np.where((slots != first) & (slots != second), tops, 0.).max(axis=1)
        groups = tuple(self.groups[h] for h in pair)
        obtained = self.reader._obtain(groups)
        loads = np.stack([obtained[group][0] for group in groups])
        self.stats["family_queries"] += 1
        return {"active": counts[0], "tolerant": counts[1], "affected": counts[2],
                "native_count": counts[3], "frequency_lo": lower[0], "excess_lo": lower[1],
                "severity_lo": lower[2], "peak_hi": upper[3], "peak_max": peak,
                "all_scenes_active": bool(np.all(root["count"][0, 0] > 0)),
                "loads": loads}

    @property
    def stored_bytes(self):
        arrays = {id(value): value for level in self.levels for value in level.values()}
        return sum(value.nbytes for value in arrays.values())
