#include <stddef.h>
#include <stdint.h>

/* These are rounded certificate sums; callers retain the gamma-w enclosure. */
void cara_subset_sum_views(const double *first, const double *second,
                          const double *fixed, const int64_t *left,
                          const int64_t *right, size_t queries, size_t cells,
                          size_t split, int symmetric, double *output) {
    const uint64_t low_mask = (((uint64_t)1) << split) - 1;
    for (size_t q = 0; q < queries; ++q) {
        for (size_t side = 0; side < 2; ++side) {
            const uint64_t mask = (uint64_t)(side ? right[q] : left[q]);
            const double *a = first + (mask & low_mask) * cells;
            const double *b = second + (mask >> split) * cells;
            double *dest = output + (2 * q + side) * cells;
            if (symmetric && side == 0) {
                for (size_t j = 0; j < cells; ++j) {
                    const double value = a[j] + b[j];
                    dest[j] = value + fixed[j];
                }
            } else {
                for (size_t j = 0; j < cells; ++j)
                    dest[j] = a[j] + b[j];
            }
        }
    }
}
