"""The original directed product fold, with rates executed in one native call."""
from .batch_evidence import BatchGainIntervalMixture
from .native_aggregate import extend_products


class NativeGainIntervalMixture(BatchGainIntervalMixture):
    # No new product approximation: the C loop has the same binary64 sequence
    # and nextafter operations as the scalar Python reference for every rate.
    _matrix_extend = staticmethod(extend_products)
