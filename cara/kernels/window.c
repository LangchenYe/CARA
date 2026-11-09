#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static double tree8(const double *e)
{
    const double first = (e[0] + e[1]) + (e[2] + e[3]);
    const double second = (e[4] + e[5]) + (e[6] + e[7]);
    return first + second;
}

static void history_row(double *history, size_t side, size_t host, size_t row,
                        size_t hosts, size_t scenes, size_t nonzero,
                        size_t tolerant, size_t native, double raw, double peak)
{
    const size_t offset = side * 7 * hosts * scenes + host * scenes + row;
    const size_t step = hosts * scenes;
    history[offset] = nonzero != 0;
    history[offset + step] = tolerant != 0;
    history[offset + 2 * step] = native != 0;
    history[offset + 3 * step] = (double)native / 8.;
    history[offset + 4 * step] = raw / 8.;
    history[offset + 5 * step] = native ? raw / (double)native : 0.;
    history[offset + 6 * step] = peak;
}

static int pair_bounds(const double *losses, size_t scenes, unsigned int active,
                       int indexed, const double *allowance, const uint8_t *valid,
                       double *lower, double *upper)
{
    for (size_t b = 0; b < scenes; ++b)
        for (size_t e = 0; e < 4; ++e) {
            if (!(active & (1u << e))) continue;
            const double first = losses[b * 4 + e];
            const double second = losses[(scenes + b) * 4 + e];
            if (!isfinite(first) || !isfinite(second)) return 2;
            const double center = first - second;
            const size_t at = b * 4 + e;
            if (e == 3 && indexed && (valid[0] || valid[1])) {
                const double a = valid[0] ? allowance[0] : 0.;
                const double c = valid[1] ? allowance[1] : 0.;
                const double first_low = a ? fmax(0., nextafter(first - a, -INFINITY)) : first;
                const double first_high = a ? nextafter(first + a, INFINITY) : first;
                const double second_low = c ? fmax(0., nextafter(second - c, -INFINITY)) : second;
                const double second_high = c ? nextafter(second + c, INFINITY) : second;
                lower[at] = nextafter(first_low - second_high, -INFINITY);
                upper[at] = nextafter(first_high - second_low, INFINITY);
            } else if (e == 0 || first == second) {
                lower[at] = upper[at] = center;
            } else {
                lower[at] = nextafter(center, -INFINITY);
                upper[at] = nextafter(center, INFINITY);
            }
        }
    return 0;
}

/* All entry points use the same source table, frozen labels and strict
   consumer expressions. History outputs have (side,7,canonical-host,scene)
   order. side_count=1 is a single-placement historical query, not a pair. */
#define ARGUMENTS \
    const double *restrict source, const int64_t *restrict incumbent, \
    const int64_t *restrict candidate, const uint8_t *restrict changed, \
    const uint8_t *restrict actions, const int64_t *restrict candidate_order, \
    const int64_t *restrict starts, size_t residents, size_t length, size_t hosts, \
    size_t scenes, double capacity, double threshold, unsigned int active, \
    int indexed, int reuse_raw_dag, size_t side_count, \
    const double *restrict allowance, const uint8_t *restrict valid, \
    double *restrict loads, double *restrict cell, double *restrict raw_prefix, \
    uint64_t *restrict count_prefix, uint64_t *restrict count_work, \
    double *restrict losses, double *restrict lower, double *restrict upper, \
    double *restrict audit_old, double *restrict audit_new, double *restrict history

int cara_window_query(ARGUMENTS)
{
    if ((side_count != 1 && side_count != 2) || (side_count == 1 && history == NULL)) return 3;
    const size_t table_cells = hosts * length;
    const double denominator = (double)(hosts * 8);
    memset(loads, 0, side_count * table_cells * sizeof(double));
    if (history == NULL) {
        memset(losses, 0, 2 * scenes * 4 * sizeof(double));
        memset(lower, 0, scenes * 4 * sizeof(double));
        memset(upper, 0, scenes * 4 * sizeof(double));
    }
    if (!active) return 0;
    double *old = loads;
    double *next = side_count == 2 ? loads + table_cells : NULL;
    for (size_t i = 0; i < residents; ++i) {
        const size_t oh = (size_t)incumbent[i];
        const double *values = source + i * length;
        double *old_row = old + oh * length;
        if (side_count == 1) {
            for (size_t l = 0; l < length; ++l) old_row[l] += values[l];
            continue;
        }
        const size_t nh = (size_t)candidate[i];
        double *new_row = next + nh * length;
        if (actions[i] == 0) {
            for (size_t l = 0; l < length; ++l) {
                const double x = values[l];
                old_row[l] += x; new_row[l] += x;
            }
        } else if (actions[i] == 1) {
            for (size_t l = 0; l < length; ++l) {
                old_row[l] += values[l]; new_row[l] = old_row[l];
            }
        } else {
            for (size_t l = 0; l < length; ++l) old_row[l] += values[l];
        }
    }
    if (side_count == 2)
        for (size_t h = 0; h < hosts; ++h)
            if (!changed[h]) memcpy(next + h * length, old + h * length, length * sizeof(double));
    for (size_t side = 0; side < side_count; ++side) {
        uint64_t *affected = count_work, *duration = count_work + scenes;
        if (history == NULL) memset(count_work, 0, 2 * scenes * sizeof(uint64_t));
        double *side_losses = history == NULL ? losses + side * scenes * 4 : NULL;
        uint64_t *native_prefix = history != NULL ? count_prefix + length + 1 : NULL;
        uint64_t *active_prefix = history != NULL ? count_prefix + 2 * (length + 1) : NULL;
        for (size_t h = 0; h < hosts; ++h) {
            const size_t physical = side ? (size_t)candidate_order[h] : h;
            const double *values = loads + side * table_cells + physical * length;
            if (active & 7u) count_prefix[0] = 0;
            if (history != NULL) native_prefix[0] = active_prefix[0] = 0;
            if ((active & 8u) && indexed && valid[side]) raw_prefix[0] = 0.;
            for (size_t l = 0; l < length; ++l) {
                if (!isfinite(values[l])) return 1;
                if (active & 7u) count_prefix[l + 1] = count_prefix[l] + (values[l] > threshold);
                if (history != NULL) {
                    native_prefix[l + 1] = native_prefix[l] + (values[l] > capacity);
                    active_prefix[l + 1] = active_prefix[l] + (values[l] != 0.);
                }
                if (active & 8u) {
                    cell[l] = capacity == 1. ? fmax(values[l] - 1., 0.) : fmax(values[l] / capacity - 1., 0.);
                    if (indexed && valid[side]) raw_prefix[l + 1] = raw_prefix[l] + cell[l];
                }
            }
            /* Ascending in-place passes read future entries before any
               overwrite: the exact R2/R4/R8 expression DAG uses 3L-11 adds. */
            if ((active & 8u) && !(indexed && valid[side]) && reuse_raw_dag)
                for (size_t stride = 1; stride <= 4; stride *= 2)
                    for (size_t l = 0; l + 2 * stride <= length; ++l)
                        cell[l] = cell[l] + cell[l + stride];
            for (size_t b = 0; b < scenes; ++b) {
                const size_t start = (size_t)starts[b];
                const uint64_t count = active & 7u ? count_prefix[start + 8] - count_prefix[start] : 0;
                double total = 0.;
                if (active & 8u)
                    total = indexed && valid[side] ? raw_prefix[start + 8] - raw_prefix[start] :
                            reuse_raw_dag ? cell[start] : tree8(cell + start);
                if (history != NULL) {
                    if (!isfinite(total)) return 2;
                    double peak = values[start];
                    for (size_t t = 1; t < 8; ++t) if (values[start + t] > peak) peak = values[start + t];
                    history_row(history, side, h, b, hosts, scenes,
                        (size_t)(active_prefix[start + 8] - active_prefix[start]), (size_t)count,
                        (size_t)(native_prefix[start + 8] - native_prefix[start]), total, peak);
                } else {
                    affected[b] += count != 0; duration[b] += count;
                    if (active & 8u) side_losses[b * 4 + 3] += total;
                }
                if (audit_old != NULL) {
                    double *destination = side ? audit_new : audit_old;
                    memcpy(destination + (h * scenes + b) * 8, values + start, 8 * sizeof(double));
                }
            }
        }
        if (history == NULL)
            for (size_t b = 0; b < scenes; ++b) {
                if (active & 1u) side_losses[b * 4] = affected[b] != 0;
                if (active & 2u) side_losses[b * 4 + 1] = (double)affected[b] / (double)hosts;
                if (active & 4u) side_losses[b * 4 + 2] = (double)duration[b] / denominator;
                if (active & 8u) side_losses[b * 4 + 3] /= denominator;
            }
    }
    return history != NULL ? 0 : pair_bounds(losses, scenes, active, indexed, allowance, valid, lower, upper);
}

/* Strong no-time-reuse reference. It reads the same N,L source by (VM,start+t)
   and never packs B,N,T input; only one scene's H,8 loads are live. */
int cara_window_direct(ARGUMENTS)
{
    if ((side_count != 1 && side_count != 2) || (side_count == 1 && history == NULL)) return 3;
    const size_t cells = hosts * 8;
    const double denominator = (double)cells;
    if (history == NULL) {
        memset(losses, 0, 2 * scenes * 4 * sizeof(double));
        memset(lower, 0, scenes * 4 * sizeof(double));
        memset(upper, 0, scenes * 4 * sizeof(double));
    }
    if (!active) return 0;
    for (size_t b = 0; b < scenes; ++b) {
        memset(loads, 0, side_count * cells * sizeof(double));
        double *old = loads, *next = side_count == 2 ? loads + cells : NULL;
        const size_t start = (size_t)starts[b];
        for (size_t i = 0; i < residents; ++i) {
            const size_t oh = (size_t)incumbent[i];
            const double *values = source + i * length + start;
            double *old_row = old + oh * 8;
            if (side_count == 1) {
                for (size_t t = 0; t < 8; ++t) old_row[t] += values[t];
                continue;
            }
            const size_t nh = (size_t)candidate[i];
            double *new_row = next + nh * 8;
            if (actions[i] == 0) {
                for (size_t t = 0; t < 8; ++t) { const double x = values[t]; old_row[t] += x; new_row[t] += x; }
            } else if (actions[i] == 1) {
                for (size_t t = 0; t < 8; ++t) { old_row[t] += values[t]; new_row[t] = old_row[t]; }
            } else {
                for (size_t t = 0; t < 8; ++t) old_row[t] += values[t];
            }
        }
        if (side_count == 2)
            for (size_t h = 0; h < hosts; ++h)
                if (!changed[h]) memcpy(next + h * 8, old + h * 8, 8 * sizeof(double));
        for (size_t side = 0; side < side_count; ++side) {
            size_t affected = 0, duration = 0;
            double raw = 0.;
            for (size_t h = 0; h < hosts; ++h) {
                const size_t physical = side ? (size_t)candidate_order[h] : h;
                const double *values = loads + side * cells + physical * 8;
                uint64_t tolerant = 0, native = 0, nonzero = 0;
                double excess[8], peak = values[0];
                for (size_t t = 0; t < 8; ++t) {
                    if (!isfinite(values[t])) return 1;
                    if (active & 7u) tolerant += values[t] > threshold;
                    if (history != NULL) {
                        native += values[t] > capacity;
                        nonzero += values[t] != 0.;
                        if (values[t] > peak) peak = values[t];
                    }
                    if (active & 8u)
                        excess[t] = capacity == 1. ? fmax(values[t] - 1., 0.) : fmax(values[t] / capacity - 1., 0.);
                }
                const double total = active & 8u ? tree8(excess) : 0.;
                if (history != NULL) {
                    if (!isfinite(total)) return 2;
                    history_row(history, side, h, b, hosts, scenes,
                                nonzero, tolerant, native, total, peak);
                } else {
                    affected += tolerant != 0; duration += tolerant; raw += total;
                }
                if (audit_old != NULL) {
                    double *destination = side ? audit_new : audit_old;
                    memcpy(destination + (h * scenes + b) * 8, values, 8 * sizeof(double));
                }
            }
            if (history == NULL) {
                double *output = losses + (side * scenes + b) * 4;
                if (active & 1u) output[0] = affected != 0;
                if (active & 2u) output[1] = (double)affected / (double)hosts;
                if (active & 4u) output[2] = (double)duration / denominator;
                if (active & 8u) output[3] = raw / denominator;
            }
        }
    }
    return history != NULL ? 0 : pair_bounds(losses, scenes, active, 0, allowance, valid, lower, upper);
}
