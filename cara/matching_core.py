"""Exact family-wide fixed-edge elimination before matching.

Only columns unchanged by the repair family are candidates for elimination.
The sparse upper envelope contains every edge value of every family member.
For fixed edge (i,j), w_ij >= max_{k!=j} U_ik + max_{l!=i} U_lj permits
the edge to be fixed in an optimum by a two-edge exchange.  A disjoint batch
can be eliminated sequentially because removal only decreases competitors.
The procedure preserves an optimum value, not its physical tie permutation.
"""
from dataclasses import dataclass


@dataclass
class Kernel:
    rows: tuple
    columns: tuple
    upper: dict
    fixed_value: int
    fixed_edges: tuple
    rounds: int
    components: tuple
    original_support_edges: int


def _top_two(values):
    first = second = 0
    arg = None
    for key, value in values.items():
        if value > first:
            second, first, arg = first, value, key
        elif value == first:
            second = first
            if arg is None or key < arg:
                arg = key
        elif value > second:
            second = value
    return first, second, arg


def support_components(rows, columns, edges):
    """Include isolated vertices and rectangular components explicitly."""
    row_set, column_set = set(rows), set(columns)
    parent = {(0, row): (0, row) for row in rows}
    parent.update({(1, column): (1, column) for column in columns})

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (row, column), value in edges.items():
        if value > 0 and row in row_set and column in column_set:
            first, second = root((0, row)), root((1, column))
            if first != second:
                parent[second] = first
    groups = {}
    for side, values in ((0, rows), (1, columns)):
        for value in values:
            groups.setdefault(root((side, value)), [[], []])[side].append(value)
    return tuple((tuple(left), tuple(right)) for left, right in groups.values())


def peel_fixed_edges(hosts, upper, fixed, changed_columns):
    """Return an exact residual kernel without allocating an H-by-H array.

    ``upper`` maps nonzero possible edges to integral family upper bounds.
    ``fixed`` maps edges of unchanged columns to their exact fixed weights.
    Missing edges have weight zero.  The initial assignment is square; after
    paired elimination the residual is square, although its support components
    may be rectangular or have an empty side.
    """
    changed = set(changed_columns)
    if hosts < 1 or any(not 0 <= c < hosts for c in changed):
        raise ValueError("Invalid host count or changed column")
    row_edges = {row: {} for row in range(hosts)}
    column_edges = {column: {} for column in range(hosts)}
    for (row, column), value in upper.items():
        if not (0 <= row < hosts and 0 <= column < hosts and value >= 0 and int(value) == value):
            raise ValueError("Expected finite nonnegative integral edge upper bounds")
        if value:
            row_edges[row][column] = int(value)
            column_edges[column][row] = int(value)
    for (row, column), value in fixed.items():
        if column in changed or value < 0 or int(value) != value or upper.get((row, column), 0) != value:
            raise ValueError("Only exact positive edges of unchanged columns may be fixed")
    original_edges = sum(map(len, row_edges.values()))
    fixed_value, fixed_edges, rounds = 0, [], 0
    while True:
        row_max = {row: _top_two(edges) for row, edges in row_edges.items()}
        column_max = {column: _top_two(edges) for column, edges in column_edges.items()}
        candidates = []
        for column, edges in column_edges.items():
            if column in changed:
                continue
            c_first, c_second, c_arg = column_max[column]
            for row, value in edges.items():
                weight = fixed.get((row, column), 0)
                if not weight:
                    continue
                r_first, r_second, r_arg = row_max[row]
                r_other = r_second if r_arg == column else r_first
                c_other = c_second if c_arg == row else c_first
                if weight >= r_other + c_other:
                    candidates.append((-weight, row, column, r_other, c_other))
        if not candidates:
            break
        chosen, used_rows, used_columns = [], set(), set()
        for negative, row, column, r_other, c_other in sorted(candidates):
            if row not in used_rows and column not in used_columns:
                chosen.append((row, column, -negative, r_other, c_other))
                used_rows.add(row)
                used_columns.add(column)
        rounds += 1
        for row, column, weight, r_other, c_other in chosen:
            fixed_edges.append((row, column, weight, r_other, c_other, rounds))
            fixed_value += weight
            for other_column in tuple(row_edges[row]):
                del column_edges[other_column][row]
            del row_edges[row]
            for other_row in tuple(column_edges[column]):
                del row_edges[other_row][column]
            del column_edges[column]
    rows, columns = tuple(row_edges), tuple(column_edges)
    remaining = {(row, column): value for row, edges in row_edges.items() for column, value in edges.items()}
    return Kernel(rows, columns, remaining, fixed_value, tuple(fixed_edges), rounds,
                  support_components(rows, columns, remaining), original_edges)

