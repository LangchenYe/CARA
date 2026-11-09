"""Audited T=8 canonical scalar program with O(H*T) load scratch.

This is a strong reference control, not a guarantee for arbitrary NumPy
implementations. It reproduces the installed reference's balanced eight-time
reduction followed by its linear canonical-host fold for B>1. Every formal
input is byte-audited. Other horizons and B=1 use native-canonical replay.
"""
import ctypes

import numpy as np

from .action_query import ActionQueryPlan, canonical_fresh_losses
from .data import host_loads
from .native_aggregate import NativeAggregatePlan
from .native_canonical import NativeCanonicalPlan, _same_bits
from .native_runtime import load_kernel


_FUNCTION = None


def prepare_fused_canonical():
    global _FUNCTION
    library, metadata = load_kernel("fused_canonical")
    if _FUNCTION is None:
        f64_3 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=("C_CONTIGUOUS", "ALIGNED"))
        f64_2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED"))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        function = library.cara_fused_canonical
        function.argtypes = [f64_3, i64, i64, u8, i64, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_uint,
            f64_3, f64_2, f64_2, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        _FUNCTION = function
    return dict(metadata)


class FusedCanonicalPlan(ActionQueryPlan):
    """Canonical-order projection fused into the strict source scan."""
    def __init__(self, *args, aligned_assignment=None, **kwargs):
        kwargs["raw_error_bound"] = False
        super().__init__(*args, **kwargs)
        aligned, physical = self.physical_layout(aligned_assignment)
        self.native = NativeAggregatePlan(self.incumbent, aligned)
        self.candidate_order = np.ascontiguousarray(physical, dtype=np.int64)
        self.scratch = None
        self.fallback = None
        self.stats.update(source_workspace_values=0, aggregate_input_cells=0,
            source_panel_reads=0, peak_load_workspace_values=2 * self.hosts * self.horizon,
            materialized_load_table_values=0, canonical_shape_fallback_batches=0)
        if self.horizon == 8 and _FUNCTION is None:
            prepare_fused_canonical()

    def _reference_fallback(self):
        if self.fallback is None:
            self.fallback = NativeCanonicalPlan(self.incumbent, self.candidate,
                self.capacity, self.horizon, aligned_assignment=self.native.new,
                demand_upper=self._raw_parameters[1], tolerance=self.tolerance)
        return self.fallback

    def _run(self, panel, active, audit=False):
        panel = np.ascontiguousarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon) or not len(panel):
            raise ValueError("The fixed action and scene panel disagree")
        b, n, _ = panel.shape
        if self.scratch is None:
            self.scratch = np.empty((2, self.hosts, 8))
        lower, upper = np.empty((b, 4)), np.empty((b, 4))
        loads = (np.empty((2, self.hosts, b, 8)) if audit else None)
        losses = (np.empty((2, b, 4)) if audit else None)
        pointers = (loads[0].ctypes.data, loads[1].ctypes.data, losses.ctypes.data) if audit else (None,) * 3
        status = _FUNCTION(panel, self.native.old, self.native.new, self.native.changed,
            self.candidate_order, b, n, self.hosts, self.capacity,
            self.capacity + self.tolerance, sum(int(v) << e for e, v in enumerate(active)),
            self.scratch, lower, upper, *pointers)
        if status:
            raise RuntimeError("Fused canonical control failed")
        return lower, upper, loads, losses

    def audit(self, panel):
        if self.horizon != 8 or len(panel) == 1:
            result = self._reference_fallback().audit(panel)
            return {**result, "canonical_shape_fallback": True}
        lower, upper, loads, losses = self._run(panel, (True,) * 4, audit=True)
        for side, assignment in enumerate((self.incumbent, self.candidate)):
            if not _same_bits(loads[side], host_loads(panel, assignment)):
                raise AssertionError("Fused canonical control differs from canonical load bits")
            expected = canonical_fresh_losses(panel, assignment, self.capacity, self.tolerance)
            if not _same_bits(losses[side], expected):
                mismatch = np.argwhere(losses[side].view(np.uint64) != expected.view(np.uint64))[0]
                row, coordinate = map(int, mismatch)
                raise AssertionError(("Fused canonical loss bits differ", side, row, coordinate,
                    float(losses[side, row, coordinate]).hex(), float(expected[row, coordinate]).hex()))
        return {"rows": len(panel), "load_values": loads.size, "loss_values": losses.size,
                "all_load_and_loss_bits_equal": True, "canonical_shape_fallback": False}

    def evaluate_global(self, panel, active=(True, True, True, True)):
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags required")
        if self.horizon != 8 or len(panel) == 1:
            fallback = self._reference_fallback()
            before = {key: value.copy() if isinstance(value, list) else value
                      for key, value in fallback.stats.items()}
            lower, upper = fallback.evaluate_global(panel, active)
            for key, value in fallback.stats.items():
                previous = before[key]
                if isinstance(value, list):
                    self.stats[key] = [a + b - c for a, b, c in zip(self.stats.get(key, [0] * 4), value, previous)]
                else:
                    self.stats[key] = self.stats.get(key, 0) + value - previous
            self.stats["canonical_shape_fallback_batches"] += 1
            fallback_size = 2 * self.hosts * len(panel) * self.horizon
            self.stats["peak_load_workspace_values"] = max(self.stats["peak_load_workspace_values"], fallback_size)
            self.stats["materialized_load_table_values"] = max(self.stats["materialized_load_table_values"], fallback_size)
            return lower, upper
        lower, upper, _, _ = self._run(panel, active)
        b, n, t = np.shape(panel)
        self.stats["query_batches"] += 1
        self.stats["global_batches"] += 1
        work = (n + self.support_residents) * b * t
        self.stats["global_operand_cells"] += work
        self.stats["aggregate_input_cells"] += work
        self.stats["source_panel_reads"] += n * b * t
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        return lower, upper
