"""Time-keyed paired queries over an immutable finite hourly demand table.

Each absolute-hour load and excess cell is computed once. Both plans share
symbolic canonical resident prefixes and exact integer count indexes. Only
WindowPrefixExact also shares exact balanced two/four/eight-hour expressions.
WindowPrefixIndexed replaces that exact time-expression DAG by a floating-point
prefix difference; a per-side allowance encloses the original stored target.
Fresh iid panels retain the existing scene reader and never use these keys.
"""
import ctypes
from fractions import Fraction
from hashlib import blake2b
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from time import perf_counter, process_time

import numpy as np

from .action_query import canonical_fresh_losses
from .data import canonical_assignment, host_loads
from .fused_canonical import FusedCanonicalPlan, prepare_fused_canonical
from .native_aggregate import prepare_native
from .native_canonical import _same_bits
from .prefix_canonical import PrefixCanonicalPlan, prepare_prefix_canonical
from .workload_views import host_projection, profile_from_projection


_FUNCTION = None
_DIRECT_FUNCTION = None
_METADATA = None
_FLAGS = ("-std=c99", "-O3", "-ffp-contract=off", "-fno-fast-math")


def prepare_window_queries():
    """Compile shipped source once, outside all matched execution timers."""
    global _FUNCTION, _DIRECT_FUNCTION, _METADATA
    if _FUNCTION is not None:
        return dict(_METADATA)
    native = prepare_native()
    fused = prepare_fused_canonical()
    prefix = prepare_prefix_canonical()
    source = Path(__file__).with_name("kernels") / "window.c"
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError("A C99 compiler is required for time-keyed queries")
    key = blake2b(source.read_bytes() + repr((_FLAGS, compiler, platform.machine())).encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / "cara-window-kernels"
    cache.mkdir(exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = cache / (key + suffix)
    tick_wall, tick_cpu = perf_counter(), process_time()
    hit = library.exists()
    if not hit:
        fd, temporary = tempfile.mkstemp(prefix=key, suffix=suffix, dir=cache)
        os.close(fd)
        try:
            linking = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
            subprocess.run([compiler, *_FLAGS, "-fPIC", linking, str(source), "-lm", "-o", temporary],
                           check=True, capture_output=True, text=True)
            Path(temporary).replace(library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(library))
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    f64_2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED"))
    f64_3 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags=("C_CONTIGUOUS", "ALIGNED"))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    u64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    function = loaded.cara_window_query
    argtypes = [f64_2, i64, i64, u8, u8, i64, i64,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, f64, u8,
        f64_3, f64, f64, u64, u64, f64_3, f64_2, f64_2,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    direct = loaded.cara_window_direct
    direct.argtypes, direct.restype = argtypes, ctypes.c_int
    _FUNCTION = function
    _DIRECT_FUNCTION = direct
    _METADATA = {"cache_hit": hit, "compiler": compiler, "flags": list(_FLAGS),
        "build_and_load_cpu_seconds": process_time() - tick_cpu,
        "build_and_load_wall_seconds": perf_counter() - tick_wall,
        "native_runtime": native, "fused_runtime": fused,
        "prefix_runtime": prefix, "library": str(library)}
    return dict(_METADATA)


def materialize_windows(matrix, starts, horizon=8):
    """Reference/audit utility; production time-keyed queries do not call it."""
    windows = np.lib.stride_tricks.sliding_window_view(matrix, horizon, axis=1)
    return np.ascontiguousarray(windows[:, starts, :].transpose(1, 0, 2))


def _historical_profile(columns):
    """Retain the original Boolean dtypes and canonical Python host fold."""
    projection = tuple(column.astype(bool, copy=False) if e < 3 else column
                       for e, column in enumerate(columns))
    return profile_from_projection(projection)


class WindowPrefixExact(PrefixCanonicalPlan):
    """Share time identities but retain each window's canonical raw time tree."""
    indexed = False

    def __init__(self, *args, share_prefix=True, reuse_raw_dag=True, reuse_hours=True, **kwargs):
        self.share_prefix, self.reuse_raw_dag = bool(share_prefix), bool(reuse_raw_dag)
        self.reuse_hours, self.side_count = bool(reuse_hours), 2
        if self.share_prefix:
            super().__init__(*args, **kwargs)
        else:
            # A matched no-prefix control must not pay for constructing
            # symbolic prefix states or shared flags that it never executes.
            FusedCanonicalPlan.__init__(self, *args, **kwargs)
            self.actions = np.where(self.native.changed[self.native.new], 0, 2).astype(np.uint8)
            self.actions.setflags(write=False)
            self.saved_prefix_terms = 0
            self.common_hosts = int(np.count_nonzero(~self.native.changed.astype(bool)))
            self.stats.update(prefix_compiled_saved_terms=0, prefix_saved_additions=0,
                prefix_copy_cells=0, common_copy_cells=0)
        if self.horizon != 8:
            raise ValueError("The time-keyed consumer supports the audited eight-hour program")
        if _FUNCTION is None:
            prepare_window_queries()
        self.window_scratch = None
        self._window_length = None
        self.raw_allowance2 = np.zeros(2, dtype=np.float64)
        self.raw_valid2 = np.zeros(2, dtype=np.uint8)
        self.stats.update(window_batches=0, source_key_hours=0, source_key_input_cells=0,
            window_input_cells=0, raw_cell_evaluations=0, exact_count_prefix_cells=0,
            raw_index_prefix_cells=0, window_source_workspace_values=0,
            window_shape_fallback_batches=0, window_raw_budget_compilations=0,
            exact_raw_dag_additions=0, raw_window_tree_additions=0, window_raw_lookups=0,
            historical_batches=0, historical_rows=0, historical_exact_delegations=0,
            share_prefix=self.share_prefix, reuse_raw_dag=self.reuse_raw_dag,
            reuse_hours=self.reuse_hours)

    def _source(self, matrix, starts):
        matrix = np.asarray(matrix)
        raw_starts = np.asarray(starts)
        if (matrix.dtype != np.float64 or matrix.ndim != 2 or matrix.shape[0] != len(self.incumbent)
                or matrix.shape[1] < 8 or not matrix.flags.c_contiguous or not matrix.flags.aligned
                or not np.isfinite(matrix).all() or np.any(matrix < 0)):
            raise ValueError("A finite nonnegative contiguous binary64 N,L source table is required")
        if (raw_starts.ndim != 1 or not len(raw_starts)
                or not np.issubdtype(raw_starts.dtype, np.integer)):
            raise ValueError("Nonempty integer absolute-hour window starts are required")
        starts = np.ascontiguousarray(raw_starts, dtype=np.int64)
        if np.any(starts < 0) or np.any(starts > matrix.shape[1] - 8):
            raise ValueError("Every complete window must lie inside the immutable source table")
        if self._raw_parameters[1] is not None and Fraction(float(matrix.max())) > self._raw_parameters[1]:
            raise ValueError("The finite source table exceeds its declared demand upper bound")
        return matrix, starts

    def _prepare_raw_index(self, length):
        self.raw_allowance2[:] = 0.
        self.raw_valid2[:] = 0

    def _run_windows(self, matrix, starts, active, audit=False, historical=False):
        matrix, starts = self._source(matrix, starts)
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        b, length = len(starts), matrix.shape[1]
        side_count = self.side_count
        use_indexed = self.indexed and self.reuse_hours and not historical
        if self._window_length != length:
            self.window_scratch = np.empty((side_count, self.hosts, length if self.reuse_hours else 8), dtype=np.float64)
            self._window_length = length
            self.raw_allowance2[:] = 0.
            self.raw_valid2[:] = 0
            self._raw_budget_length = None
        if use_indexed and active[3]:
            self._prepare_raw_index(length)
        lower, upper = np.empty((0 if historical else b, 4)), np.empty((0 if historical else b, 4))
        losses = np.empty((2, 0 if historical else b, 4))
        scratch_length = length if self.reuse_hours else 8
        cell = np.empty(scratch_length)
        raw_prefix = np.empty(scratch_length + 1)
        count_prefix = np.empty((3 if historical else 1) * (scratch_length + 1), dtype=np.uint64)
        count_work = np.empty(2 * b, dtype=np.uint64)
        loads = np.empty((2, self.hosts, b, 8)) if audit else None
        pointers = (loads[0].ctypes.data, loads[1].ctypes.data) if audit else (None, None)
        history = np.empty((side_count, 7, self.hosts, b)) if historical else None
        function = _FUNCTION if self.reuse_hours else _DIRECT_FUNCTION
        status = function(matrix, self.native.old, self.native.new, self.native.changed,
            self.actions, self.candidate_order, starts, len(self.incumbent), length,
            self.hosts, b, self.capacity, self.capacity + self.tolerance,
            sum(int(v) << e for e, v in enumerate(active)), int(use_indexed), int(self.reuse_raw_dag), side_count,
            self.raw_allowance2, self.raw_valid2, self.window_scratch, cell,
            raw_prefix, count_prefix, count_work, losses, lower, upper, *pointers,
            history.ctypes.data if historical else None)
        if status:
            raise OverflowError("The time-keyed finite-table consumer overflowed its supported range")
        cells = length if self.reuse_hours else b * 8
        additions = int(any(active)) * (len(self.incumbent) + self.support_residents - self.saved_prefix_terms) * cells
        self.stats["window_batches"] += 1
        self.stats["source_key_hours"] += length
        self.stats["source_key_input_cells"] += int(self.reuse_hours) * matrix.size
        self.stats["window_input_cells"] += b * len(self.incumbent) * 8
        self.stats["global_operand_cells"] += additions
        self.stats["aggregate_input_cells"] += additions
        self.stats["source_panel_reads"] += int(any(active)) * len(self.incumbent) * cells
        self.stats["prefix_saved_additions"] += int(any(active)) * self.saved_prefix_terms * cells
        self.stats["prefix_copy_cells"] += int(any(active)) * self.saved_prefix_terms * cells
        self.stats["common_copy_cells"] += int(any(active)) * self.common_hosts * cells
        self.stats["raw_cell_evaluations"] += int(active[3]) * side_count * self.hosts * cells
        self.stats["exact_count_prefix_cells"] += int(any(active[:3])) * side_count * self.hosts * cells
        self.stats["raw_index_prefix_cells"] += int(active[3]) * int(use_indexed) * int(self.raw_valid2.sum()) * self.hosts * length
        exact_sides = side_count - (int(self.raw_valid2.sum()) if use_indexed else 0)
        if self.reuse_raw_dag and self.reuse_hours:
            self.stats["exact_raw_dag_additions"] += int(active[3]) * exact_sides * self.hosts * (3 * length - 11)
        else:
            self.stats["raw_window_tree_additions"] += int(active[3]) * exact_sides * self.hosts * b * 7
        self.stats["window_raw_lookups"] += int(active[3]) * side_count * self.hosts * b
        self.stats["window_source_workspace_values"] = self.window_scratch.size
        self.stats["peak_load_workspace_values"] = max(self.stats["peak_load_workspace_values"], self.window_scratch.size)
        self.stats["materialized_load_table_values"] = max(self.stats["materialized_load_table_values"], self.window_scratch.size)
        self.stats["query_batches"] += 1
        self.stats["global_batches"] += 1
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        if historical:
            self.stats["historical_batches"] += 1
            self.stats["historical_rows"] += b
            self.stats["historical_exact_delegations"] += int(self.indexed)
            return history
        return lower, upper, loads, losses

    def historical_profiles(self, matrix, starts):
        """Exact historical L6 queries, including the original native means.

        Indexed callers explicitly use the exact consumer for this accessor;
        its floating raw prefix is not asserted to reproduce L6 stored bits.
        """
        if len(starts) == 1:
            matrix, starts = self._source(matrix, starts)
            panel = materialize_windows(matrix, starts)
            self.stats["window_shape_fallback_batches"] += 1
            return tuple(profile_from_projection(host_projection(host_loads(panel, assignment),
                self.capacity, self.tolerance)) for assignment in (self.incumbent, self.candidate))
        history = self._run_windows(matrix, starts, (True,) * 4, historical=True)
        return tuple(_historical_profile(side) for side in history)

    evaluate_native_windows = historical_profiles

    def evaluate_windows(self, matrix, starts, active=(True, True, True, True)):
        if len(starts) == 1:
            matrix, starts = self._source(matrix, starts)
            self.stats["window_shape_fallback_batches"] += 1
            return super().evaluate_global(materialize_windows(matrix, starts), active)
        lower, upper, _, _ = self._run_windows(matrix, starts, active)
        return lower, upper

    def audit_windows(self, matrix, starts):
        matrix, starts = self._source(matrix, starts)
        panel = materialize_windows(matrix, starts)
        if len(starts) == 1:
            result = super().audit(panel)
            return {**result, "all_boolean_loss_bits_equal": True,
                "all_raw_loss_enclosures_valid": True, "window_shape_fallback": True}
        lower, upper, loads, losses = self._run_windows(matrix, starts, (True,) * 4, audit=True)
        reference = []
        for side, assignment in enumerate((self.incumbent, self.candidate)):
            expected_loads = host_loads(panel, assignment)
            target = canonical_fresh_losses(panel, assignment, self.capacity, self.tolerance)
            reference.append(target)
            if not _same_bits(loads[side], expected_loads):
                raise AssertionError("Time-keyed loads differ from stored canonical load bits")
            if not _same_bits(losses[side, :, :3], target[:, :3]):
                raise AssertionError("Time-keyed Boolean losses differ from stored canonical bits")
            if not self.indexed or not self.raw_valid2[side]:
                if not _same_bits(losses[side, :, 3], target[:, 3]):
                    raise AssertionError("WindowExact raw losses differ from the canonical time tree")
            else:
                radius = Fraction(float(self.raw_allowance2[side]))
                for observed, exact in zip(losses[side, :, 3], target[:, 3]):
                    if abs(Fraction(float(observed)) - Fraction(float(exact))) > radius:
                        raise AssertionError("Indexed raw loss exceeds its declared canonical allowance")
        for row in range(len(starts)):
            for e in range(4):
                exact = Fraction(float(reference[0][row, e])) - Fraction(float(reference[1][row, e]))
                if not Fraction(float(lower[row, e])) <= exact <= Fraction(float(upper[row, e])):
                    raise AssertionError("Time-keyed query does not enclose the canonical stored paired gain")
        return {"rows": len(starts), "all_load_bits_equal": True,
            "all_boolean_loss_bits_equal": True, "all_raw_loss_enclosures_valid": True,
            "all_raw_loss_bits_equal": _same_bits(losses[:, :, 3], np.stack(reference)[:, :, 3]),
            "indexed_raw_sides": self.raw_valid2.tolist(), "raw_allowance2": self.raw_allowance2.tolist(),
            "window_shape_fallback": False}


class WindowPrefixIndexed(WindowPrefixExact):
    """Bound only floating raw prefix differences; Boolean counts stay exact."""
    indexed = True

    def _prepare_raw_index(self, length):
        if getattr(self, "_raw_budget_length", None) == length:
            return
        from .window_budget import window_raw_allowance_upper
        sizes = tuple(map(len, self.old)), tuple(map(len, self.new))
        for side in range(2):
            try:
                value = float(window_raw_allowance_upper(sizes[side], self.capacity,
                    self.horizon, length, self._raw_parameters[1]))
                if not np.isfinite(value) or value < 0:
                    raise OverflowError("Nonfinite window-index allowance")
                self.raw_allowance2[side] = value
                self.raw_valid2[side] = 1
            except OverflowError:
                self.raw_allowance2[side] = 0.
                self.raw_valid2[side] = 0
        self._raw_budget_length = length
        self.stats["window_raw_budget_compilations"] += 1


class DirectWindowPrefix(WindowPrefixExact):
    """Same source/metadata, direct scene-local reads, and 2*H*8 scratch."""
    def __init__(self, *args, **kwargs):
        kwargs["reuse_hours"] = False
        super().__init__(*args, **kwargs)


class KeyedProfilePlan:
    """One placement's original historical profile, without a paired F0 plan.

    This is the candidate-only Audit backend: the caller retains the already
    computed incumbent profile and applies the original exact census gate.
    No host matching, second placement, or symbolic shared-prefix plan is built.
    """
    indexed = False
    _source = WindowPrefixExact._source
    _run_windows = WindowPrefixExact._run_windows

    def __init__(self, assignment, capacity, horizon=8, *, reuse_hours=True,
                 reuse_raw_dag=True, demand_upper=None, tolerance=1e-12):
        self.incumbent = canonical_assignment(assignment)
        if (not self.incumbent or horizon != 8 or not np.isfinite(capacity) or capacity <= 0
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("A complete assignment, positive capacity and supported eight-hour program are required")
        self.hosts, self.horizon = max(self.incumbent) + 1, 8
        if self.hosts * 8 > 2**53 or len(self.incumbent) > 2**53:
            raise ValueError("Resident and host-time counts must not exceed 2**53")
        self.capacity, self.tolerance = float(capacity), float(tolerance)
        self.reuse_hours, self.reuse_raw_dag = bool(reuse_hours), bool(reuse_raw_dag)
        self.share_prefix, self.side_count = False, 1
        old = np.ascontiguousarray(self.incumbent, dtype=np.int64)
        old.setflags(write=False)
        empty_labels = np.empty(0, dtype=np.int64)
        empty_flags = np.empty(0, dtype=np.uint8)
        self.native = SimpleNamespace(old=old, new=empty_labels, changed=empty_flags)
        self.actions, self.candidate_order = empty_flags, empty_labels
        self.support_residents = self.saved_prefix_terms = self.common_hosts = 0
        bound = None if demand_upper is None else Fraction(demand_upper)
        if bound is not None and bound < 0:
            raise ValueError("A declared source bound must be nonnegative")
        self._raw_parameters = Fraction(self.capacity), bound
        self.window_scratch, self._window_length = None, None
        self.raw_allowance2 = np.zeros(2)
        self.raw_valid2 = np.zeros(2, dtype=np.uint8)
        self.stats = {key: 0 for key in ("window_batches", "source_key_hours", "source_key_input_cells",
            "window_input_cells", "global_operand_cells", "aggregate_input_cells", "source_panel_reads",
            "prefix_saved_additions", "prefix_copy_cells", "common_copy_cells", "raw_cell_evaluations",
            "exact_count_prefix_cells", "raw_index_prefix_cells", "exact_raw_dag_additions",
            "raw_window_tree_additions", "window_raw_lookups", "window_source_workspace_values",
            "peak_load_workspace_values", "materialized_load_table_values", "query_batches", "global_batches",
            "historical_batches", "historical_rows", "historical_exact_delegations",
            "window_shape_fallback_batches", "window_raw_budget_compilations", "source_workspace_values")}
        self.stats.update(active_coordinate_batches=[0] * 4, reuse_hours=self.reuse_hours,
                          reuse_raw_dag=self.reuse_raw_dag, share_prefix=False, single_assignment=True)
        if _FUNCTION is None:
            prepare_window_queries()

    def historical_profile(self, matrix, starts):
        if len(starts) == 1:
            matrix, starts = self._source(matrix, starts)
            panel = materialize_windows(matrix, starts)
            self.stats["window_shape_fallback_batches"] += 1
            return profile_from_projection(host_projection(host_loads(panel, self.incumbent),
                                           self.capacity, self.tolerance))
        history = self._run_windows(matrix, starts, (True,) * 4, historical=True)
        return _historical_profile(history[0])

    evaluate_native_windows = historical_profile
