#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Compile symbolic linear-fold states, without examining scene values.
   Action0: two additions, action1: shared add+copy, action2: common old-only. */
size_t cara_compile_prefix(const int64_t *restrict incumbent,
                            const int64_t *restrict candidate,
                            const uint8_t *restrict changed,
                            size_t residents, size_t hosts,
                            uint8_t *restrict shared_flags,
                            uint8_t *restrict actions,
                            uint64_t *restrict states)
{
    memset(states, 0, 2 * hosts * sizeof(uint64_t));
    uint64_t next_id = 1;
    size_t saved = 0;
    for (size_t i = 0; i < residents; ++i) {
        const size_t oh = (size_t)incumbent[i], nh = (size_t)candidate[i];
        const int share = states[oh] == states[hosts + nh];
        shared_flags[i] = (uint8_t)share;
        states[oh] = next_id++;
        states[hosts + nh] = share ? states[oh] : next_id++;
        actions[i] = !changed[nh] ? 2 : (uint8_t)share;
        saved += changed[nh] && share;
    }
    return saved;
}

/* Exact canonical-prefix sharing for the audited T=8 reduction program.
   Optional audit outputs materialize loads/losses only outside timing. */
int cara_prefix_canonical(const double *restrict panel,
                         const int64_t *restrict incumbent,
                         const int64_t *restrict candidate,
                         const uint8_t *restrict changed,
                         const uint8_t *restrict actions,
                         const int64_t *restrict candidate_order,
                         size_t scenes, size_t residents, size_t hosts,
                         double capacity, double threshold, unsigned int active,
                         double *restrict scratch, double *restrict lower,
                         double *restrict upper, double *restrict audit_old,
                         double *restrict audit_new, double *restrict audit_loss)
{
    const size_t cells = hosts * 8;
    const double denominator = (double)cells;
    for (size_t b = 0; b < scenes; ++b) {
        memset(scratch, 0, 2 * cells * sizeof(double));
        double *old = scratch, *next = scratch + cells;
        for (size_t i = 0; i < residents; ++i) {
            const size_t oh = (size_t)incumbent[i], nh = (size_t)candidate[i];
            const double *values = panel + (b * residents + i) * 8;
            if (actions[i] == 0) {
                for (size_t t = 0; t < 8; ++t) {
                    const double x = values[t];
                    old[oh * 8 + t] += x;
                    next[nh * 8 + t] += x;
                }
            } else if (actions[i] == 1) {
                for (size_t t = 0; t < 8; ++t) {
                    old[oh * 8 + t] += values[t];
                    next[nh * 8 + t] = old[oh * 8 + t];
                }
            } else {
                for (size_t t = 0; t < 8; ++t) old[oh * 8 + t] += values[t];
            }
        }
        for (size_t h = 0; h < hosts; ++h)
            if (!changed[h]) memcpy(next + h * 8, old + h * 8, 8 * sizeof(double));
        if (audit_old != NULL)
            for (size_t h = 0; h < hosts; ++h) {
                memcpy(audit_old + (h * scenes + b) * 8, old + h * 8, 8 * sizeof(double));
                memcpy(audit_new + (h * scenes + b) * 8,
                       next + (size_t)candidate_order[h] * 8, 8 * sizeof(double));
            }
        double loss[2][4] = {{0}};
        for (size_t side = 0; side < 2; ++side) {
            size_t affected = 0, duration = 0;
            double raw = 0.;
#define HOSTS(EXCESS) do { \
                for (size_t h = 0; h < hosts; ++h) { \
                    const size_t physical = side ? (size_t)candidate_order[h] : h; \
                    const double *values = scratch + side * cells + physical * 8; \
                    size_t count = 0; \
                    if (active & 7u) \
                        for (size_t t = 0; t < 8; ++t) count += values[t] > threshold; \
                    affected += count != 0; duration += count; \
                    if (active & 8u) { \
                        double e[8]; \
                        for (size_t t = 0; t < 8; ++t) e[t] = (EXCESS); \
                        const double first = (e[0] + e[1]) + (e[2] + e[3]); \
                        const double second = (e[4] + e[5]) + (e[6] + e[7]); \
                        raw += first + second; \
                    } \
                } \
            } while (0)
            if (capacity == 1.) HOSTS(fmax(values[t] - 1., 0.));
            else HOSTS(fmax(values[t] / capacity - 1., 0.));
#undef HOSTS
            if (active & 1u) loss[side][0] = affected != 0;
            if (active & 2u) loss[side][1] = (double)affected / (double)hosts;
            if (active & 4u) loss[side][2] = (double)duration / denominator;
            if (active & 8u) loss[side][3] = raw / denominator;
            if (audit_loss != NULL)
                memcpy(audit_loss + (side * scenes + b) * 4, loss[side], 4 * sizeof(double));
        }
        for (size_t e = 0; e < 4; ++e) {
            const size_t at = b * 4 + e;
            lower[at] = upper[at] = 0.;
            if (!(active & (1u << e))) continue;
            const double center = loss[0][e] - loss[1][e];
            if (e == 0 || loss[0][e] == loss[1][e]) lower[at] = upper[at] = center;
            else {
                lower[at] = nextafter(center, -INFINITY);
                upper[at] = nextafter(center, INFINITY);
            }
        }
    }
    return 0;
}
