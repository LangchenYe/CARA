"""Complete CloudCons decision cohorts, workload replay, and native metrics.

The bundled Arrow objects are read with PyArrow. CPU is the first channel in
two-channel objects. Each cohort has one common hourly start.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CAPACITIES = {"azure2019": 4.0, "borg2019_d": 1.0, "borg2019_e": 1.0, "huawei2025": 4.0}
COHORT_SIZES = {"azure2019": 150, "borg2019_d": 127, "borg2019_e": 68, "huawei2025": 575}


def _read_arrow(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all().to_pydict()


@dataclass(frozen=True)
class Dataset:
    name: str
    series: tuple[np.ndarray, ...]
    capacity: float
    item_ids: tuple[str, ...]

    @property
    def n_vms(self):
        return len(self.series)

    @property
    def min_length(self):
        return min(map(len, self.series))

    def panel(self, origin, horizon=24):
        """Return the complete VM-by-time interval [origin, origin + horizon)."""
        if origin < 0 or horizon <= 0 or origin + horizon > self.min_length:
            raise ValueError("The complete cohort does not cover the requested interval")
        return np.stack([values[origin:origin + horizon] for values in self.series])

    def forecast(self, origin, horizon=24, lookback=24):
        """A shared trailing-mean forecast, without access to the target interval."""
        history = self.panel(origin - lookback, lookback)
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        return np.repeat(history.mean(axis=1, keepdims=True), horizon, axis=1)

    def window_panel(self, origin, older, newer, horizon=8, stride=1, max_rows=None):
        """Windows wholly inside [origin - older, origin - newer), in time order.

        If capped, choose equally spaced start indices deterministically. Distinct
        experimental roles must use disjoint outer intervals; windows inside one
        role may overlap and must not be treated as independent observations.
        """
        if older <= newer or newer < 0 or horizon <= 0 or stride <= 0:
            raise ValueError("Invalid historical interval or window configuration")
        starts = np.arange(origin - older, origin - newer - horizon + 1, stride, dtype=int)
        if not len(starts) or starts[0] < 0 or starts[-1] + horizon > self.min_length:
            raise ValueError("The complete cohort does not cover the requested history")
        if max_rows is not None:
            if max_rows <= 0:
                raise ValueError("The row cap must be positive")
            if max_rows < len(starts):
                starts = starts[np.linspace(0, len(starts) - 1, max_rows, dtype=int)]
        return np.stack([self.panel(int(start), horizon) for start in starts])


def load_dataset(name, root=None):
    if name not in CAPACITIES:
        raise ValueError(f"Unknown decision cohort: {name}")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "raw" / "cloudcons"
    columns = _read_arrow(root / name / "data-00000-of-00001.arrow")
    targets = columns["target"]
    if isinstance(targets[0][0], list):
        targets = [target[0] for target in targets]
    series = tuple(np.asarray(target, dtype=np.float64) for target in targets)
    if len(series) != COHORT_SIZES[name] or len(set(columns["start"])) != 1 or Counter(columns["freq"]) != {"1H": len(series)}:
        raise ValueError("Unexpected decision-cohort size, start alignment, or frequency")
    if any(np.any(~np.isfinite(values)) or np.any(values < 0) for values in series):
        raise ValueError("Workloads must be finite and nonnegative")
    for values in series:
        values.setflags(write=False)
    return Dataset(name, series, CAPACITIES[name], tuple(columns["item_id"]))


class _CanonicalAssignment(tuple):
    """Immutable placement whose labels already follow first-occurrence order."""
    __slots__ = ()


def canonical_assignment(assignment):
    if isinstance(assignment, _CanonicalAssignment):
        return assignment
    labels, output = {}, []
    for label in assignment:
        label = int(label)
        if label not in labels:
            labels[label] = len(labels)
        output.append(labels[label])
    return _CanonicalAssignment(output)


def host_memberships(assignment):
    """Canonical resident sets in one ascending-VM pass, using O(N) work."""
    labels = canonical_assignment(assignment)
    groups = [[] for _ in range(max(labels) + 1)] if labels else []
    for vm, host in enumerate(labels):
        groups[host].append(vm)
    return tuple(tuple(group) for group in groups)


def host_loads(panel, assignment):
    """Canonical loads: (H,T) for (N,T), or (H,B,T) for (B,N,T).

    Each host is reduced over its residents in ascending VM order. For batched
    panels, reduction shape is preserved; the host dimension is stacked first.
    """
    panel = np.asarray(panel, dtype=np.float64)
    assignment = np.asarray(canonical_assignment(assignment), dtype=int)
    axis = 0 if panel.ndim == 2 else 1
    if panel.ndim not in (2, 3) or panel.shape[axis] != len(assignment) or not len(assignment):
        raise ValueError("Workload shape and complete assignment disagree")
    groups = host_memberships(assignment)
    if panel.ndim == 2:
        return np.stack([panel[list(group)].sum(axis=0) for group in groups])
    return np.stack([panel[:, list(group), :].sum(axis=1) for group in groups])


def native_metrics(demand, assignment, capacity):
    """CloudCons executable Util, VR, VS, and PAR, plus pooled raw excess.

    VS first conditions on violating times within each affected host, then on
    affected hosts. PAR is the largest host peak divided by mean host peak.
    """
    if capacity <= 0 or np.asarray(demand).ndim != 2:
        raise ValueError("Expected positive capacity and a VM-by-time target")
    loads = host_loads(demand, assignment)
    loads = loads[np.any(loads != 0, axis=1)]
    if not len(loads):
        return {"hosts": 0, "Util": -1., "VR": -1., "VS": 0., "PAR": -1., "raw": 0.}
    excess = np.maximum(loads / capacity - 1., 0.)
    overload = loads > capacity  # The native benchmark uses strict capacity.
    affected = overload.any(axis=1)
    severity = np.divide(excess.sum(axis=1), overload.sum(axis=1), out=np.zeros(len(loads)), where=affected)
    peaks = loads.max(axis=1)
    return {"hosts": int(len(loads)), "Util": float(loads.mean() / capacity), "VR": float(overload.mean(axis=1).mean()), "VS": float(severity[affected].mean()) if np.any(affected) else 0., "PAR": float(peaks.max() / peaks.mean()), "raw": float(excess.mean())}


def burdens(panel, assignment, capacity, h0=None, kappa=.01, tolerance=1e-12):
    """Return (B,4) stored burdens in J,V,R,S order for a (B,N,T) panel."""
    if np.asarray(panel).ndim != 3 or capacity <= 0 or kappa <= 0 or tolerance < 0:
        raise ValueError("Invalid burden panel or normalization")
    loads = host_loads(panel, assignment)
    return burdens_from_loads(loads, capacity, h0, kappa, tolerance)


def burdens_from_loads(loads, capacity, h0=None, kappa=.01, tolerance=1e-12):
    """Derive (B,4) burdens from canonical host-first (H,B,T) loads."""
    loads = np.asarray(loads, dtype=np.float64)
    if loads.ndim != 3 or capacity <= 0 or kappa <= 0 or tolerance < 0:
        raise ValueError("Invalid host loads or normalization")
    hosts, _, horizon = loads.shape
    h0 = hosts if h0 is None else h0
    if h0 < hosts:
        raise ValueError("The fixed reference host count cannot be smaller than the placement")
    overload = loads > capacity + tolerance
    raw = np.maximum(loads / capacity - 1., 0.).sum(axis=(0, 2)) / (h0 * horizon)
    return np.column_stack((overload.any(axis=(0, 2)), overload.any(axis=2).sum(axis=0) / h0, overload.sum(axis=(0, 2)) / (h0 * horizon), np.minimum(raw / kappa, 1.)))


def trace_intervals(origin, target_horizon=24):
    """Half-open intervals for the unchanged complete-cohort protocol."""
    if origin < 192 or target_horizon <= 0:
        raise ValueError("The protocol requires 192 historical hours")
    return {"audit": (origin - 192, origin - 128), "selection": (origin - 128, origin - 96), "build": (origin - 96, origin), "target": (origin, origin + target_horizon)}


def trace_panels(dataset, origin, build_rows=24):
    trace_intervals(origin)
    if origin + 24 > dataset.min_length:
        raise ValueError("The complete cohort does not cover the target interval")
    return {"build": dataset.window_panel(origin, 96, 0, max_rows=build_rows), "selection": dataset.window_panel(origin, 128, 96), "audit": dataset.window_panel(origin, 192, 128)}
