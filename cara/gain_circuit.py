"""The native relative-gain circuit with a reusable reference and batch layout.

The scalar oracle is ``refinement._relative_gains``.  Its advanced-index
subtraction produces a B-by-E Fortran-contiguous array, so each coordinate's
B operands form one contiguous reduction.  The batch path explicitly packs
Q-by-E-by-B C-contiguous operands and reduces the last axis.  This preserves
that operand order and reduction layout on the fixed NumPy backend; it does
not replace mean-of-rounded-differences by a difference of means.

The reference scale is computed on the supplied reference's original layout
before any packing.  The context then snapshots its selected operands and
scale.  It can be shared by exact evaluation and optimistic-bound evaluation.
Nonfinite arithmetic remains nonfinite; a certificate consumer must fall back
to exact refinement whenever its resulting key is not finite.
"""
from __future__ import annotations

import numpy as np


PROFILE_NAMES = ("J", "V", "VR", "raw", "VS", "PAR")


class GainsContext:
    """Relative gains for one fixed binary64 B-by-6 reference profile."""

    def __init__(self, reference, objective_names=PROFILE_NAMES):
        reference = np.asarray(reference, dtype=np.float64)
        self.objective_names = tuple(objective_names)
        if reference.ndim != 2 or reference.shape[0] < 1 or reference.shape[1] != 6:
            raise ValueError("Expected a nonempty B-by-6 native reference profile")
        if not self.objective_names or any(name not in PROFILE_NAMES
                                           for name in self.objective_names):
            raise ValueError("Objective names must be a nonempty subset of PROFILE_NAMES")
        self.scenes = reference.shape[0]
        self.indices = tuple(PROFILE_NAMES.index(name) for name in self.objective_names)
        # Keep this reduction on the original layout: transposing or copying
        # reference first could change the original mean's reduction circuit.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            means = reference.mean(axis=0)
            self.scale = np.maximum(np.abs(means[list(self.indices)]), 1e-8)
        self._reference_e_b = np.ascontiguousarray(reference[:, self.indices].T)
        self._reference_e_b.setflags(write=False)
        self.scale.setflags(write=False)

    def scalar(self, profile):
        """Run the original contiguous B-axis circuit for one B-by-6 row set."""
        profile = np.asarray(profile, dtype=np.float64)
        if profile.shape != (self.scenes, 6):
            raise ValueError("Expected a B-by-6 native candidate profile")
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            differences = self._reference_e_b.T - profile[:, self.indices]
            return differences.mean(axis=0) / self.scale

    def batch(self, profiles):
        """Return Q-by-E gains for Q-by-B-by-6 native candidate profiles."""
        profiles = np.asarray(profiles, dtype=np.float64)
        if profiles.ndim != 3 or profiles.shape[1:] != (self.scenes, 6):
            raise ValueError("Expected Q-by-B-by-6 native candidate profiles")
        if not len(profiles):
            return np.empty((0, len(self.indices)), dtype=np.float64)
        candidates = np.ascontiguousarray(
            np.take(profiles, self.indices, axis=2).transpose(0, 2, 1))
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            differences = self._reference_e_b[None, :, :] - candidates
            return differences.mean(axis=2) / self.scale
