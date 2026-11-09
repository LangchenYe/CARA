"""Factor a complete membership-query plan before bounded output reduction.

The shared registry retains its configured limits. A plan additionally owns a
candidate-to-membership index and one projection table for its distinct groups.
Only the final host gather is chunked, so output workspace limits do not cause
repeated membership planning or projection-table reconstruction.
"""
import numpy as np

from .data import host_memberships
from .query_engine import _profiles


def compiled_profiles(replay, assignments=None, profile_cells=1 << 20, *, memberships=None, diagnostics=None):
    """Return canonical (candidate, scene, metric) profiles in supplied order.

    ``profile_cells`` limits the candidate/host/scene gather, with one candidate
    as its minimum unit. The complete plan's index and group table are separate
    from this workspace and from the bounded persistent registry. Diagnostics
    account for named numerical arrays, not Python objects or process RSS.
    """
    if int(profile_cells) != profile_cells or profile_cells < 1:
        raise ValueError("profile_cells must be a positive integer")
    if (assignments is None) == (memberships is None):
        raise ValueError("Supply assignments or canonical memberships, but not both")
    q = len(assignments if memberships is None else memberships)
    b, n, _ = replay.panel.shape
    if not q:
        if diagnostics is not None:
            diagnostics.update({"distinct_memberships": 0, "candidate_membership_index_bytes": 0,
                "projection_table_bytes": 0, "maximum_projection_gather_bytes": 0,
                "unretained_view_array_bytes": 0, "result_array_bytes": 0,
                "maximum_accounted_array_bytes": 0, "output_chunks": 0})
        return np.empty((0, b, 6), dtype=np.float64)
    first = host_memberships(assignments[0]) if memberships is None else memberships[0]
    hosts = len(first)
    lookup = np.empty((q, hosts), dtype=np.intp)
    identities, unique = {}, []
    for query in range(q):
        groups = first if query == 0 else (host_memberships(assignments[query]) if memberships is None else memberships[query])
        if len(groups) != hosts or sum(map(len, groups)) != n:
            raise ValueError("A batch requires complete assignments with equal host count")
        for host, residents in enumerate(groups):
            group = identities.get(residents)
            if group is None:
                group = len(unique)
                identities[residents] = group
                unique.append(residents)
            lookup[query, host] = group
    replay.full_values += replay.panel.size * q
    before = replay.hits
    values = replay._obtain(unique)
    # Preserve the original full-batch counter definition although the plan
    # submits each distinct membership only once to the physical constructor.
    duplicates = q * hosts - len(unique)
    replay.group_queries += duplicates
    replay.hits += duplicates
    replay.projection_hits += replay.hits - before
    table = tuple(np.stack([values[group][1][e] for group in unique]) for e in range(7))
    table_bytes = sum(column.nbytes for column in table)
    registry_arrays = {id(array) for loads, projected in replay.views.values()
                       for array in (loads, *projected)}
    unretained_bytes = sum(array.nbytes for loads, projected in values.values()
                          for array in (loads, *projected) if id(array) not in registry_arrays)
    accounted = lookup.nbytes + table_bytes + unretained_bytes
    del values
    output = np.empty((q, b, 6), dtype=np.float64)
    batch_size = max(1, int(profile_cells) // max(1, hosts * b))
    gather_bytes = chunks = 0
    for start in range(0, q, batch_size):
        ids = lookup[start:start + batch_size]
        columns = tuple(column[ids] for column in table)
        block = _profiles(columns)
        output[start:start + len(ids)] = block
        gather_bytes = max(gather_bytes, sum(column.nbytes for column in columns))
        accounted = max(accounted, lookup.nbytes + table_bytes + output.nbytes +
                        sum(column.nbytes for column in columns) + block.nbytes)
        chunks += 1
        del columns, block
    if diagnostics is not None:
        diagnostics.update({"distinct_memberships": len(unique),
            "candidate_membership_index_bytes": int(lookup.nbytes),
            "projection_table_bytes": int(table_bytes),
            "maximum_projection_gather_bytes": int(gather_bytes),
            "unretained_view_array_bytes": int(unretained_bytes),
            "result_array_bytes": int(output.nbytes),
            "maximum_accounted_array_bytes": int(accounted),
            "output_chunks": chunks, "effective_query_batch": batch_size,
            "array_accounting_scope": "Named plan index, group projection table, unretained load/projection views, "
                "final output, gathered projection columns and returned profile block. Does not include persistent "
                "registry arrays, input panel, Python objects or internal reduction temporaries; not process RSS."})
    return output
