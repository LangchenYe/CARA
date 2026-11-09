#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Strong canonical-layout control. Resident terms retain ascending VM order;
   output is contiguous H,B,T, matching the original NumPy projection layout. */
void cara_canonical(const double *restrict panel,
                    const int64_t *restrict incumbent,
                    const int64_t *restrict candidate,
                    const int64_t *restrict common_source,
                    size_t scenes, size_t residents, size_t hosts,
                    size_t horizon, double *restrict old_load,
                    double *restrict new_load)
{
    const size_t host_block = scenes * horizon;
    memset(old_load, 0, hosts * host_block * sizeof(double));
    memset(new_load, 0, hosts * host_block * sizeof(double));
    for (size_t b = 0; b < scenes; ++b) {
        const size_t input_base = b * residents * horizon;
        for (size_t i = 0; i < residents; ++i) {
            const size_t old_base = (size_t)incumbent[i] * host_block + b * horizon;
            const size_t target = (size_t)candidate[i];
            const size_t new_base = target * host_block + b * horizon;
            const double *values = panel + input_base + i * horizon;
            if (common_source[target] < 0) {
                for (size_t t = 0; t < horizon; ++t) {
                    const double x = values[t];
                    old_load[old_base + t] += x;
                    new_load[new_base + t] += x;
                }
            } else {
                for (size_t t = 0; t < horizon; ++t)
                    old_load[old_base + t] += values[t];
            }
        }
    }
    for (size_t h = 0; h < hosts; ++h)
        if (common_source[h] >= 0)
            memcpy(new_load + h * host_block,
                   old_load + (size_t)common_source[h] * host_block,
                   host_block * sizeof(double));
}
