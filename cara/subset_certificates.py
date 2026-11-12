"""Meet-in-the-middle sum views for outward profile certificates.

The stored entries and answers are rounded sums, not canonical host loads.
The existing gamma-w enclosures are therefore still applied by the consumer.
"""
import ctypes
import numpy as np


_FUNCTION = None


def _kernel():
    global _FUNCTION
    if _FUNCTION is None:
        from .native_runtime import load_kernel
        library, _ = load_kernel("subset_sums")
        function = library.cara_subset_sum_views
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED"))
        fixed = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        function.argtypes = [f64, f64, fixed, i64, i64, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, f64]
        function.restype = None
        _FUNCTION = function
    return _FUNCTION


class MutableSumIndex:
    """Share two half-subset relations across bound fields and both host sides.

    ``sum_ranges`` returns mandatory loads, and optionally possible loads, as
    q-by-2-by-B-by-T arrays.  For a batch consisting only of leaves, the two
    results alias.  Small requests retain a selector product; construction is
    lazy and capped by ``max_bytes``.
    """
    def __init__(self, values, *, symmetric=False, max_bytes=32 << 20, minimum_ranges=None):
        self.values = np.ascontiguousarray(values, dtype=np.float64)
        if self.values.ndim != 3 or not 1 <= len(self.values) <= 52:
            raise ValueError("Expected one to 52 mutable B-by-T demand rows")
        self.width, self.scenes, self.horizon = self.values.shape
        self.symmetric = bool(symmetric)
        self.offset = int(self.symmetric)
        self.bits = self.width - self.offset
        self.mask_count = 1 << self.bits
        self.split = self.bits // 2
        self.low_mask = (1 << self.split) - 1
        self.max_bytes = int(max_bytes)
        self.minimum_ranges = (max(128, 1 << ((self.bits + 1) // 2))
                               if minimum_ranges is None else int(minimum_ranges))
        self._requests_seen = 0
        if self.max_bytes < 0 or self.minimum_ranges < 1:
            raise ValueError("Nonnegative memory and positive request thresholds required")
        self.flat = self.values.reshape(self.width, -1)
        self.tables = None
        self.stats = {"mutable_sum_builds": 0, "mutable_sum_queries": 0,
                      "mutable_sum_half_rows": 0, "mutable_sum_bytes": 0,
                      "mutable_sum_selector_fallbacks": 0}

    def _range(self, start, stop):
        if int(start) != start or int(stop) != stop:
            raise ValueError("A subtree has integer mask endpoints")
        start, stop = int(start), int(stop)
        span = stop - start
        if span <= 0 or span & (span - 1) or start < 0 or stop > self.mask_count or start % span:
            raise ValueError("An aligned dyadic range is required")
        return start, stop, span.bit_length() - 1

    def _ensure(self, ranges):
        if self.tables is not None:
            return True
        rows = (1 << self.split) + (1 << (self.bits - self.split))
        required = rows * self.flat.shape[1] * self.flat.itemsize
        self._requests_seen += ranges
        if self._requests_seen < self.minimum_ranges or required > self.max_bytes:
            return False
        free = self.flat[self.offset:]
        tables = []
        for part in (free[:self.split], free[self.split:]):
            table = np.zeros((1 << len(part), self.flat.shape[1]), dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                for bit, value in enumerate(part):
                    width = 1 << bit
                    table[width:2 * width] = table[:width] + value
            tables.append(table)
        self.tables = tuple(tables)
        self.stats["mutable_sum_builds"] += 1
        self.stats["mutable_sum_half_rows"] = rows
        self.stats["mutable_sum_bytes"] = required
        return True

    def _sides(self, left, right):
        left, right = np.ascontiguousarray(left, dtype=np.int64), np.ascontiguousarray(right, dtype=np.int64)
        result = np.empty((len(left), 2, self.scenes, self.horizon), dtype=np.float64)
        _kernel()(self.tables[0], self.tables[1], self.flat[0], left, right, len(left),
                  self.flat.shape[1], self.split, int(self.symmetric),
                  result.reshape(2 * len(left), -1))
        return result

    def sum_ranges(self, ranges, possible=True):
        ranges = tuple(self._range(start, stop) for start, stop in ranges)
        q = len(ranges)
        if not q:
            empty = np.empty((0, 2, self.scenes, self.horizon))
            return (empty, empty) if possible else empty
        leaves = all(free == 0 for _, _, free in ranges)
        if self._ensure(q):
            self.stats["mutable_sum_queries"] += 1
            starts = np.asarray([start for start, _, _ in ranges], dtype=np.intp)
            unresolved = np.asarray([(1 << free) - 1 for _, _, free in ranges], dtype=np.intp)
            all_bits = self.mask_count - 1
            mandatory = self._sides(all_bits ^ starts ^ unresolved, starts)
            if not possible:
                return mandatory
            return mandatory, mandatory if leaves else self._sides(all_bits ^ starts, starts | unresolved)
        self.stats["mutable_sum_selector_fallbacks"] += 1
        mandatory = np.zeros((q, 2, self.width))
        allowed = np.zeros_like(mandatory) if possible and not leaves else mandatory
        for row, (start, _, free) in enumerate(ranges):
            if self.symmetric:
                mandatory[row, 0, 0] = 1.
                allowed[row, 0, 0] = 1.
            for bit in range(self.bits):
                j = bit + self.offset
                if bit >= free:
                    mandatory[row, (start >> bit) & 1, j] = 1.
                    allowed[row, (start >> bit) & 1, j] = 1.
                elif possible:
                    allowed[row, :, j] = 1.
        if possible and not leaves:
            selectors = np.concatenate((mandatory.reshape(-1, self.width), allowed.reshape(-1, self.width)))
            sums = (selectors @ self.flat).reshape(2, q, 2, self.scenes, self.horizon)
            return sums[0], sums[1]
        output = (mandatory.reshape(-1, self.width) @ self.flat).reshape(q, 2, self.scenes, self.horizon)
        return (output, output) if possible else output

