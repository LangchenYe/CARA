"""Strict binary64 source-scan aggregation, shared by full and edit plans.

Only source is shipped. ``prepare_native`` compiles/loads a cached library in
the temporary directory and must run once as common preparation before timed
plan comparisons. This module aggregates load centres, not canonical answers;
callers must use the certified load/projection allowances and exact refinement.
"""
import ctypes
from hashlib import blake2b
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
from time import perf_counter, process_time

import numpy as np


_FUNCTION = None
_PRODUCT_FUNCTION = None
_LIBRARY = None
_PREPARATION = None
_FLAGS = ("-std=c99", "-O3", "-ffp-contract=off", "-fno-fast-math")


def prepare_native():
    """Return common build/load metadata; no per-plan compiler work is hidden."""
    global _FUNCTION, _PRODUCT_FUNCTION, _LIBRARY, _PREPARATION
    if _FUNCTION is not None:
        return dict(_PREPARATION)
    source = Path(__file__).with_name("kernels") / "aggregate.c"
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError("A C99 compiler is required for the optional native aggregate backend")
    key = blake2b(source.read_bytes() + repr((_FLAGS, compiler, platform.machine(), platform.system())).encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / "cara-native-aggregate"
    cache.mkdir(exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = cache / (key + suffix)
    cached = library.exists()
    began_wall, began_cpu = perf_counter(), process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    if not cached:
        fd, temporary = tempfile.mkstemp(prefix=key + "-", suffix=suffix, dir=cache)
        os.close(fd)
        try:
            linking = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
            subprocess.run([compiler, *_FLAGS, "-fPIC", linking, str(source), "-lm", "-o", temporary],
                           check=True, capture_output=True, text=True)
            Path(temporary).replace(library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(library))
    function = loaded.cara_aggregate
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=("C_CONTIGUOUS", "ALIGNED"))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    function.argtypes = [f64, i64, i64, u8, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, f64, f64]
    function.restype = ctypes.c_int
    products = loaded.cara_extend_products
    matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED"))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    products.argtypes = [matrix, matrix, vector, ctypes.c_size_t, ctypes.c_size_t]
    products.restype = None
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    _FUNCTION, _PRODUCT_FUNCTION, _LIBRARY = function, products, loaded
    _PREPARATION = {"cache_hit": cached, "build_and_load_wall_seconds": perf_counter() - began_wall,
        "build_and_load_parent_cpu_seconds": process_time() - began_cpu,
        "compiler_child_cpu_seconds": children_after.ru_utime + children_after.ru_stime
            - children_before.ru_utime - children_before.ru_stime,
        "compiler": compiler, "flags": list(_FLAGS), "library": str(library)}
    return dict(_PREPARATION)


def extend_products(products, values, rates):
    """Extend every rate with the identical directed scalar operation order."""
    if _PRODUCT_FUNCTION is None:
        prepare_native()
    output = np.array(products, dtype=np.float64, order="C", copy=True)
    values = np.ascontiguousarray(values, dtype=np.float64)
    rates = np.ascontiguousarray(rates if hasattr(rates, "__len__") else tuple(rates), dtype=np.float64)
    if (output.ndim != 2 or output.shape[1] != 2 or values.ndim != 2 or values.shape[1] != 2
            or rates.shape != (len(output),)):
        raise ValueError("K,2 products, B,2 values and K rates are required")
    _PRODUCT_FUNCTION(output, values, rates, len(output), len(values))
    return output


class NativeAggregatePlan:
    """One B,N,T input scan; full candidate or signed edit in physical order."""
    def __init__(self, incumbent, aligned_candidate):
        self.old = np.array(incumbent, dtype=np.int64, order="C", copy=True)
        self.new = np.array(aligned_candidate, dtype=np.int64, order="C", copy=True)
        if self.old.ndim != 1 or not len(self.old) or self.old.shape != self.new.shape:
            raise ValueError("Complete incumbent and physically aligned candidate are required")
        if np.any(self.old < 0) or np.any(self.new < 0):
            raise ValueError("Physical host labels must be nonnegative")
        self.hosts = int(self.old.max()) + 1
        if (self.hosts > len(self.old) or self.new.max() >= self.hosts
                or not np.all(np.bincount(self.old, minlength=self.hosts))
                or not np.all(np.bincount(self.new, minlength=self.hosts))):
            raise ValueError("Both complete placements must use the same contiguous host labels")
        self.residents, self.migrations = len(self.old), int(np.count_nonzero(self.old != self.new))
        self.moved = np.flatnonzero(self.old != self.new).astype(np.int64)
        self.changed = np.zeros(self.hosts, dtype=np.uint8)
        self.changed[self.old[self.moved]] = self.changed[self.new[self.moved]] = 1
        self.support_residents = int(np.count_nonzero(self.changed[self.old]))
        self.old.setflags(write=False); self.new.setflags(write=False)
        self.moved.setflags(write=False); self.changed.setflags(write=False)
        prepare_native()

    def aggregate(self, panel, mode="full"):
        """Return H,B,T centres for a trusted finite nonnegative supported panel."""
        modes = {"full": 0, "edit": 1}
        if mode not in modes:
            raise ValueError("The native aggregate mode must be full or edit")
        panel = np.asarray(panel)
        if (panel.dtype != np.float64 or panel.ndim != 3 or panel.shape[1] != self.residents
                or min(panel.shape) < 1 or not panel.flags.c_contiguous or not panel.flags.aligned):
            raise ValueError("A nonempty aligned C-contiguous binary64 B,N,T panel is required")
        scenes, _, horizon = panel.shape
        old = np.empty((scenes, self.hosts, horizon), dtype=np.float64)
        second = np.empty_like(old)
        status = _FUNCTION(panel, self.old, self.new, self.changed, scenes, self.residents,
                           self.hosts, horizon, modes[mode], old, second)
        if status:
            raise RuntimeError("The native aggregate kernel rejected its mode")
        return old.transpose(1, 0, 2), second.transpose(1, 0, 2)
