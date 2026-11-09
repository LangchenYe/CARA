"""Batched positive-product filters for the unchanged canonical certificate.

Rates and endpoints are evaluated together. A block contains at most 32 factors
bounded away from zero, so its intermediate products are normal and finite.
The standard product error bound encloses its exact product; outward endpoint
multiplication then extends the stored interval. Threshold ambiguities retain
the original canonical-prefix/rational fallback.
"""
from fractions import Fraction

import numpy as np

from .action_query import _GainIntervalMixture, _outward


_UNIT = Fraction(1, 2**53)
_PRODUCT_ALLOWANCE = {}
for _count in range(1, 33):
    _gamma = _count * _UNIT / (1 - _count * _UNIT)
    _PRODUCT_ALLOWANCE[_count] = (
        _outward(1 / (1 + _gamma), False),
        _outward(1 / (1 - _gamma), True))


class BatchGainIntervalMixture(_GainIntervalMixture):
    """Vectorized materialized mixture, with the same exact comparison rule."""

    @staticmethod
    def _matrix_extend(products, values, rates):
        products = np.asarray(products, dtype=np.float64).copy()
        rates = np.asarray(tuple(rates), dtype=np.float64)[:, None]
        values = np.asarray(values, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            for start in range(0, len(values), 32):
                block = values[start:start + 32]
                factors_lo = np.nextafter(1. + np.nextafter(rates * block[:, 0], -np.inf), -np.inf)
                factors_hi = np.nextafter(1. + np.nextafter(rates * block[:, 1], np.inf), np.inf)
                c_lo, c_hi = _PRODUCT_ALLOWANCE[len(block)]
                # Each block factor lies in (2**-9, 3). At most 32 factors
                # keep EVERY intermediate product normal and finite; gamma
                # therefore applies independently of NumPy's product tree.
                block_lo = np.nextafter(np.prod(factors_lo, axis=1) * c_lo, -np.inf)
                block_hi = np.nextafter(np.prod(factors_hi, axis=1) * c_hi, np.inf)
                products[:, 0] = np.maximum(0., np.nextafter(products[:, 0] * block_lo, -np.inf))
                products[:, 1] = np.nextafter(products[:, 1] * block_hi, np.inf)
        return products

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        if (values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all()
                or np.any(values[:, 0] < -1) or np.any(values[:, 1] > 1)
                or np.any(values[:, 0] > values[:, 1])):
            raise ValueError("Ordered finite gain endpoints in [-1,1] are required")
        self.rows.extend(map(tuple, values))
        old_keys = tuple(self.small)
        products = self._matrix_extend(
            (self.high, *(self.small[j] for j in old_keys)), values,
            (255 / 256, *(2. ** -j for j in old_keys)))
        self.high = tuple(map(float, products[0]))
        for j, product in zip(old_keys, products[1:]):
            self.small[j] = tuple(map(float, product))
        count = max(self.minimum_components, len(self.rows).bit_length())
        new_keys = tuple(j for j in range(1, count + 1) if j not in self.small)
        if new_keys:
            products = self._matrix_extend(np.ones((len(new_keys), 2)), self.rows,
                                           (2. ** -j for j in new_keys))
            for j, product in zip(new_keys, products):
                self.small[j] = tuple(map(float, product))
