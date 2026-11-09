"""Strict native control retaining canonical host/time projection layout.

The native summation order is audited against the installed reference on every
formal input batch. The scalar reduction case B=T=1 uses the original compiled
plan because NumPy may use a pairwise resident reduction there. This control
does not assert numerical equivalence for an arbitrary future NumPy backend.
"""
import ctypes

import numpy as np

from .action_query import ActionQueryPlan, canonical_fresh_losses
from .compiled_admission import CompiledGlobalPlan
from .data import host_loads


_FUNCTION = None
_METADATA = None


def prepare_native_canonical():
    global _FUNCTION, _METADATA
    if _FUNCTION is None:
        from .native_runtime import load_kernel
        library, metadata = load_kernel("canonical")
        function = library.cara_canonical
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=("C_CONTIGUOUS", "ALIGNED"))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        function.argtypes = [f64, i64, i64, i64, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t, f64, f64]
        function.restype = None
        _FUNCTION, _METADATA = function, metadata
    return dict(_METADATA)


def _same_bits(left, right):
    return np.array_equal(np.asarray(left).view(np.uint64), np.asarray(right).view(np.uint64))


class NativeCanonicalPlan(ActionQueryPlan):
    """Same C source scan as full aggregation, canonical H,B,T output sinks."""
    def __init__(self, *args, aligned_assignment=None, **kwargs):
        kwargs["raw_error_bound"] = False
        super().__init__(*args, **kwargs)
        self.physical_layout(aligned_assignment)
        self.old_labels = np.asarray(self.incumbent, dtype=np.int64)
        self.new_labels = np.asarray(self.candidate, dtype=np.int64)
        old_position = {group: h for h, group in enumerate(self.old)}
        self.common_source = np.asarray([old_position.get(group, -1) for group in self.new], dtype=np.int64)
        self._fallback = None
        self.stats.update(source_workspace_values=0, aggregate_input_cells=0,
                          source_panel_reads=0, scalar_canonical_fallback_batches=0)
        if _FUNCTION is None:
            prepare_native_canonical()

    def canonical_loads(self, panel):
        panel = np.ascontiguousarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        b, n, t = panel.shape
        if b == t == 1:
            return host_loads(panel, self.incumbent), host_loads(panel, self.candidate)
        old = np.empty((self.hosts, b, t))
        new = np.empty_like(old)
        _FUNCTION(panel, self.old_labels, self.new_labels, self.common_source,
                  b, n, self.hosts, t, old, new)
        return old, new

    def _project(self, loads, active):
        b, t = loads.shape[1:]
        result = np.zeros((b, 4))
        if any(active[:3]):
            bits = loads > self.capacity + self.tolerance
            if active[0]:
                result[:, 0] = bits.any(axis=(0, 2))
            if active[1]:
                result[:, 1] = bits.any(axis=2).sum(axis=0) / self.hosts
            if active[2]:
                result[:, 2] = bits.sum(axis=(0, 2)) / (self.hosts * t)
        if active[3]:
            result[:, 3] = np.maximum(loads / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * t)
        return result

    def audit(self, panel):
        """Untimed byte check of every load and stored loss on this input."""
        loads = self.canonical_loads(panel)
        for actual, assignment in zip(loads, (self.incumbent, self.candidate)):
            if not _same_bits(actual, host_loads(panel, assignment)):
                raise AssertionError("Native canonical control differs from canonical load bits")
            losses = self._project(actual, (True,) * 4)
            if not _same_bits(losses, canonical_fresh_losses(panel, assignment, self.capacity, self.tolerance)):
                raise AssertionError("Native canonical control differs from canonical loss bits")
        return {"rows": len(panel), "load_values": sum(a.size for a in loads),
                "loss_values": 8 * len(panel), "all_load_and_loss_bits_equal": True,
                "scalar_fallback": len(panel) == self.horizon == 1}

    def evaluate_global(self, panel, active=(True, True, True, True)):
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        b, n, t = panel.shape
        self.stats["query_batches"] += 1
        self.stats["global_batches"] += 1
        self.stats["aggregate_input_cells"] += (n + self.support_residents) * b * t
        self.stats["global_operand_cells"] += (n + self.support_residents) * b * t
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        if b == t == 1:
            if self._fallback is None:
                self._fallback = CompiledGlobalPlan(self.incumbent, self.candidate,
                    self.capacity, self.horizon, demand_upper=self._raw_parameters[1],
                    tolerance=self.tolerance, raw_error_bound=False)
            self.stats["scalar_canonical_fallback_batches"] += 1
            self.stats["source_panel_reads"] += (n + self.support_residents) * b * t
            self.stats["source_workspace_values"] += (n + self.support_residents) * b * t
            return self._fallback.evaluate_global(panel, active)
        self.stats["source_panel_reads"] += n * b * t
        reference, candidate = map(lambda loads: self._project(loads, active), self.canonical_loads(panel))
        lower, upper = np.zeros_like(reference), np.zeros_like(reference)
        for e, enabled in enumerate(active):
            if enabled:
                center = reference[:, e] - candidate[:, e]
                lower[:, e], upper[:, e] = np.nextafter(center, -np.inf), np.nextafter(center, np.inf)
                zero = reference[:, e] == candidate[:, e]
                lower[zero, e] = upper[zero, e] = 0.
                if e == 0:
                    lower[:, e] = upper[:, e] = center
        return lower, upper
