"""Select partial or full certificate acquisition using an exact pool cutoff."""
from .aggregate_bounds import AggregateBounds
from .family_score_bounds import FamilyScoreBounds


MINIMUM_CANDIDATES = 1024
SAMPLE_SIZE = 64
MINIMUM_REJECTION = .125


def _keys(certificates, partial, masks, movement):
    moved = [movement[mask] for mask in masks]
    values = partial.upper_keys([(mask, mask + 1) for mask in masks], moved, moved)
    certificates.stats["aggregate_bound_nodes"] = certificates.stats.get("aggregate_bound_nodes", 0) + len(masks)
    return values


def screen(certificates, partial, masks, cutoff, movement, observer):
    masks = list(masks)
    if cutoff is None or not masks:
        return masks, ()
    count = min(SAMPLE_SIZE, len(masks))
    sample = [masks[(j * len(masks)) // count] for j in range(count)]
    values = _keys(certificates, partial, sample, movement)
    keys = dict(zip(sample, values))
    rejected_sample = sum(key is not None and tuple(key) < tuple(cutoff) for key in values)
    if rejected_sample / len(sample) < MINIMUM_REJECTION:
        certificates.stats["staged_sample_fallbacks"] = certificates.stats.get("staged_sample_fallbacks", 0) + 1
    else:
        rest = [mask for mask in masks if mask not in keys]
        keys.update(zip(rest, _keys(certificates, partial, rest, movement)))
    rejected = [mask for mask in masks if keys.get(mask) is not None and tuple(keys[mask]) < tuple(cutoff)]
    if not rejected:
        return masks, ()
    witness = certificates._record("score", min(rejected), max(rejected) + 1, rejected, observer,
        movement_lower_bound=0, optimistic_key=max(tuple(keys[mask]) for mask in rejected),
        cutoff=tuple(cutoff), bound_scope="partial native-profile bounds over listed active leaves")
    certificates.stats["aggregate_excluded_masks"] = certificates.stats.get("aggregate_excluded_masks", 0) + len(rejected)
    rejected = set(rejected)
    return [mask for mask in masks if mask not in rejected], (witness,)


def prepare(certificates, masks, movement, cutoff, bounds, top_p, fine_options, observer):
    """Return a staged plan, or None to retain the ordinary leaf plan."""
    if (len(masks) <= MINIMUM_CANDIDATES or not isinstance(bounds, FamilyScoreBounds)
            or not bounds.available):
        return None
    if (cutoff is None and len(masks) * bounds.horizon
            < top_p * (bounds.hosts + bounds.horizon * bounds.resident_bound)):
        certificates.stats["staged_shape_fallbacks"] = certificates.stats.get("staged_shape_fallbacks", 0) + 1
        return None
    partial = AggregateBounds(bounds)
    if not partial.available:
        return None
    if cutoff is not None:
        passing, excluded = screen(certificates, partial, masks, cutoff, movement, observer)
        return passing, excluded, False
    certificates._staged_state = (partial, movement, int(top_p), fine_options)
    return masks, (), True


def batches(certificates, masks, cutoff, *, batch_size, observer):
    partial, movement, top_p, fine_options = certificates._staged_state
    certificates._staged_state = None
    order = sorted(masks, key=lambda mask: (movement[mask], mask))
    pilot = order[:min(len(order), top_p)]
    for start in range(0, len(pilot), batch_size):
        batch = tuple(pilot[start:start + batch_size])
        certificates.stats["progressive_batches"] += 1
        certificates.stats["staged_pilot_profiles"] = certificates.stats.get("staged_pilot_profiles", 0) + len(batch)
        yield batch
    exact = set(pilot)
    remaining = [mask for mask in order if mask not in exact]
    remaining, _ = screen(certificates, partial, remaining, cutoff(), movement, observer)
    if remaining:
        surviving, _ = certificates.prune(sorted(remaining), cutoff=cutoff(), observer=observer, **fine_options)
        yield from certificates.score_batches(surviving, cutoff, batch_size=batch_size, observer=observer)
