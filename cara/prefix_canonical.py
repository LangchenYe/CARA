"""Share identical canonical ordered prefixes without reassociating a sum.

Symbolic prefix states are compiled in one C pass before scene acquisition.
This control keeps the audited T=8 canonical raw tree and has the same shape
fallback as FusedCanonicalPlan. A shared addition is followed by an explicit
copy to the second root; copies are counted separately from additions.
"""
import ctypes

import numpy as np

from .fused_canonical import FusedCanonicalPlan
from .native_runtime import load_kernel


_FUNCTION = None
_COMPILE = None


def prepare_prefix_canonical():
    global _FUNCTION, _COMPILE
    library, metadata = load_kernel("prefix_canonical")
    if _FUNCTION is None:
        f64_3 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=("C_CONTIGUOUS", "ALIGNED"))
        f64_2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED"))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        u64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
        function = library.cara_prefix_canonical
        function.argtypes = [f64_3, i64, i64, u8, u8, i64, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_uint,
            f64_3, f64_2, f64_2, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int
        compile_flags = library.cara_compile_prefix
        compile_flags.argtypes = [i64, i64, u8, ctypes.c_size_t, ctypes.c_size_t, u8, u8, u64]
        compile_flags.restype = ctypes.c_size_t
        _FUNCTION, _COMPILE = function, compile_flags
    return dict(metadata)


class PrefixCanonicalPlan(FusedCanonicalPlan):
    """Exact ordered-prefix reuse, with no positive-subset reassociation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _FUNCTION is None:
            prepare_prefix_canonical()
        n = len(self.incumbent)
        self.share_flags = np.empty(n, dtype=np.uint8)
        self.actions = np.empty(n, dtype=np.uint8)
        states = np.empty(2 * self.hosts, dtype=np.uint64)
        self.saved_prefix_terms = int(_COMPILE(self.native.old, self.native.new,
            self.native.changed, n, self.hosts, self.share_flags, self.actions, states))
        self.share_flags.setflags(write=False)
        self.actions.setflags(write=False)
        self.common_hosts = int(np.count_nonzero(~self.native.changed.astype(bool)))
        self.stats.update(prefix_compiled_saved_terms=self.saved_prefix_terms,
            prefix_saved_additions=0, prefix_copy_cells=0, common_copy_cells=0)

    def _run(self, panel, active, audit=False):
        panel = np.ascontiguousarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon) or not len(panel):
            raise ValueError("The fixed action and scene panel disagree")
        b, n, _ = panel.shape
        if self.scratch is None:
            self.scratch = np.empty((2, self.hosts, 8))
        lower, upper = np.empty((b, 4)), np.empty((b, 4))
        loads = np.empty((2, self.hosts, b, 8)) if audit else None
        losses = np.empty((2, b, 4)) if audit else None
        pointers = (loads[0].ctypes.data, loads[1].ctypes.data, losses.ctypes.data) if audit else (None,) * 3
        status = _FUNCTION(panel, self.native.old, self.native.new, self.native.changed,
            self.actions, self.candidate_order, b, n, self.hosts, self.capacity,
            self.capacity + self.tolerance, sum(int(v) << e for e, v in enumerate(active)),
            self.scratch, lower, upper, *pointers)
        if status:
            raise RuntimeError("Canonical-prefix sharing failed")
        return lower, upper, loads, losses

    def evaluate_global(self, panel, active=(True, True, True, True)):
        result = super().evaluate_global(panel, active)
        if self.horizon == 8 and len(panel) > 1:
            cells = len(panel) * self.horizon
            saved = self.saved_prefix_terms * cells
            self.stats["aggregate_input_cells"] -= saved
            self.stats["global_operand_cells"] -= saved
            self.stats["prefix_saved_additions"] += saved
            self.stats["prefix_copy_cells"] += saved
        if self.horizon != 1 or len(panel) != 1:
            self.stats["common_copy_cells"] += self.common_hosts * len(panel) * self.horizon
        return result
