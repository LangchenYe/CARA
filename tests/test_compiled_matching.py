from itertools import permutations

import numpy as np
import pytest

from cara.data import canonical_assignment
from cara.family_certificates import ResidualFamily


def _count(incumbent, assignment):
    hosts = len(set(incumbent))
    counts = np.zeros((hosts, hosts), dtype=np.int64)
    np.add.at(counts, (incumbent, assignment), 1)
    retained = max(sum(int(counts[i, j]) for i, j in enumerate(order))
                   for order in permutations(range(hosts)))
    return len(incumbent) - retained


@pytest.mark.parametrize('backend', ['compiled', 'locked', 'replay'])
def test_complete_mixed_families_and_relaxed_subtrees(backend):
    rng = np.random.default_rng(81726)
    for trial in range(25):
        hosts = 2 + trial % 4
        n = hosts * 4
        incumbent = tuple(np.tile(np.arange(hosts), 4))
        seed = np.array(incumbent)
        seed[hosts:] = rng.integers(hosts, size=n-hosts)
        seed = canonical_assignment(seed)
        pair = tuple(map(int, rng.choice(hosts, size=2, replace=False)))
        residents = np.flatnonzero(np.isin(seed, pair))
        mutable = tuple(sorted(map(int, rng.choice(residents, size=min(6, len(residents)), replace=False))))
        family = ResidualFamily(incumbent, seed, pair, mutable, matching_backend=backend)
        dense = ResidualFamily(incumbent, seed, pair, mutable, matching_backend='dense')
        assert family.seed_movement_count() == _count(family.incumbent, family.seed)
        for mask in family.nonempty_masks():
            assert family.movement_count(mask) == _count(family.incumbent, family.assignment(mask))
        for exponent in range(family.bits + 1):
            width = 1 << exponent
            for start in range(0, family.mask_count, width):
                assert family.movement_lower_bound(start, start + width) == dense.movement_lower_bound(start, start + width)


@pytest.mark.parametrize('backend', ['compiled', 'locked', 'replay'])
def test_connected_support_and_physical_label_permutation(backend):
    hosts, width = 96, 8
    incumbent = tuple(i // width for i in range(hosts * width))
    seed = list(incumbent)
    for old in range(hosts):
        seed[old * width + width - 1] = (old + 1) % hosts
    seed = canonical_assignment(seed)
    mutable = tuple(vm for vm in range(len(seed)) if seed[vm] in (0, 1))[:8]
    family = ResidualFamily(incumbent, seed, (0, 1), mutable, matching_backend=backend)
    control = ResidualFamily(incumbent, seed, (0, 1), mutable, matching_backend='sparse')
    assert family.seed_movement_count() == control.seed_movement_count()
    for mask in family.nonempty_masks():
        assert family.movement_count(mask) == control.movement_count(mask)
    if backend != 'replay':
        assert family.stats['matching_locked_edges'] >= hosts - 4
        assert family.stats['matching_core_rows'] <= 4
    else:
        assert family.stats['matching_dual_replays'] > 0


def test_unsafe_native_range_uses_exact_count_fallback(monkeypatch):
    from cara.assignment_replay import NativeTwoColumnReplay
    from cara.compiled_matching import _Variable

    def decline(self, weights):
        raise ValueError('Outside the certified native range')

    monkeypatch.setattr(NativeTwoColumnReplay, '__init__', decline)
    entries = {(i, i): 1 + i % 5 for i in range(65)}
    stats = {}
    model = _Variable(tuple(range(65)), tuple(range(65)), entries, entries, (0, 1), stats, True)
    assert model.retained((), (), ()) == sum(entries.values())
    assert model.retained_seed() == sum(entries.values())
    assert stats['matching_exact_fallbacks'] == 1
