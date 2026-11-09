#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Output buffers are B,H,T; the Python interface exposes H,B,T views.
   Each sum follows VM identity order. The edit output is a signed delta,
   not an assertion of numerical equality to a canonical candidate sum. */
int cara_aggregate(const double *restrict panel,
                   const int64_t *restrict incumbent,
                   const int64_t *restrict candidate,
                   const uint8_t *restrict changed,
                   size_t scenes, size_t residents, size_t hosts,
                   size_t horizon, int edit,
                   double *restrict old_load,
                   double *restrict second_load)
{
    if (edit != 0 && edit != 1) return -1;
    const size_t output_count = scenes * hosts * horizon;
    memset(old_load, 0, output_count * sizeof(double));
    memset(second_load, 0, output_count * sizeof(double));
    for (size_t b = 0; b < scenes; ++b) {
        const size_t output_base = b * hosts * horizon;
        const size_t input_base = b * residents * horizon;
        for (size_t i = 0; i < residents; ++i) {
            const size_t old_base = output_base + (size_t)incumbent[i] * horizon;
            const size_t new_base = output_base + (size_t)candidate[i] * horizon;
            const double *values = panel + input_base + i * horizon;
            if (!edit && changed[candidate[i]]) {
                for (size_t t = 0; t < horizon; ++t) {
                    const double x = values[t];
                    old_load[old_base + t] += x;
                    second_load[new_base + t] += x;
                }
            } else if (edit && old_base != new_base) {
                for (size_t t = 0; t < horizon; ++t) {
                    const double x = values[t];
                    old_load[old_base + t] += x;
                    second_load[old_base + t] -= x;
                    second_load[new_base + t] += x;
                }
            } else {
                for (size_t t = 0; t < horizon; ++t)
                    old_load[old_base + t] += values[t];
            }
        }
        if (!edit)
            for (size_t h = 0; h < hosts; ++h)
                if (!changed[h])
                    memcpy(second_load + output_base + h * horizon,
                           old_load + output_base + h * horizon, horizon * sizeof(double));
    }
    return 0;
}

/* The operation sequence matches the original directed scalar interval fold. */
void cara_extend_products(double *products, const double *values,
                          const double *rates, size_t components, size_t rows)
{
    for (size_t j = 0; j < components; ++j) {
        double lo = products[2 * j], hi = products[2 * j + 1];
        const double rate = rates[j];
        for (size_t i = 0; i < rows; ++i) {
            const double fl = nextafter(1.0 + nextafter(rate * values[2 * i], -INFINITY), -INFINITY);
            const double fh = nextafter(1.0 + nextafter(rate * values[2 * i + 1], INFINITY), INFINITY);
            lo = fmax(0.0, nextafter(lo * fl, -INFINITY));
            hi = nextafter(hi * fh, INFINITY);
        }
        products[2 * j] = lo;
        products[2 * j + 1] = hi;
    }
}
