/* Integer primal-dual assignment compilation and at-most-two-column replay.
 * This kernel exposes an optimum objective, not a physical tie mapping.
 * A nonzero return requests the ordinary exact matching fallback.
 */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>

#define INF (INT64_MAX / 4)

static int add_checked(int64_t a, int64_t b, int64_t *out) {
    if ((b > 0 && a > INT64_MAX - b) ||
        (b < 0 && a < INT64_MIN - b)) return 1;
    *out = a + b;
    return 0;
}

static int sub_checked(int64_t a, int64_t b, int64_t *out) {
    if ((b > 0 && a < INT64_MIN + b) ||
        (b < 0 && a > INT64_MAX + b)) return 1;
    *out = a - b;
    return 0;
}

static int dimension_ok(int64_t n) {
    return n > 0 && (uint64_t)n <= (uint64_t)SIZE_MAX / sizeof(int64_t) / 4 &&
        (uint64_t)n <= (uint64_t)SIZE_MAX / sizeof(int64_t) / (uint64_t)n;
}

static int weight_ok(int64_t n, int64_t w) {
    return w >= 0 && w <= INT64_MAX / 64 / n;
}

static int64_t member_weight(int64_t n, const int64_t *weights,
                            int64_t row, int64_t column,
                            int64_t changed_count, const int64_t *columns,
                            const int64_t *changed) {
    for (int64_t k = 0; k < changed_count; ++k)
        if (column == columns[k]) return changed[k * n + row];
    return weights[row * n + column];
}

/* state: u, v, row_to_column, column_to_row; work: slack, pred, reached_rows. */
static int augment(int64_t n, const int64_t *weights, int64_t *state,
                   int64_t root, int64_t changed_count,
                   const int64_t *columns, const int64_t *changed,
                   int64_t *work, uint8_t *used, int64_t *stats) {
    int64_t *u = state, *v = state + n;
    int64_t *r2c = state + 2*n, *c2r = state + 3*n;
    int64_t *slack = work, *pred = work + n, *reached = work + 2*n;
    for (int64_t j = 0; j < n; ++j) { slack[j] = INF; pred[j] = -1; }
    memset(used, 0, (size_t)n);
    int64_t row = root, previous = -1, reached_count = 0;
    while (1) {
        if (row < 0 || row >= n || reached_count >= n) return 3;
        reached[reached_count++] = row;
        int64_t column = -1, delta = INF;
        for (int64_t j = 0; j < n; ++j) {
            if (used[j]) continue;
            int64_t reduced;
            int64_t w = member_weight(n, weights, row, j,
                                      changed_count, columns, changed);
            if (sub_checked(-w, u[row], &reduced) ||
                sub_checked(reduced, v[j], &reduced)) return 2;
            if (reduced < 0) return 3;
            if (reduced < slack[j]) { slack[j] = reduced; pred[j] = previous; }
            if (slack[j] < delta) { delta = slack[j]; column = j; }
        }
        if (column < 0 || delta < 0 || delta >= INF) return 3;
        for (int64_t k = 0; k < reached_count; ++k)
            if (add_checked(u[reached[k]], delta, &u[reached[k]])) return 2;
        for (int64_t j = 0; j < n; ++j) {
            if (used[j]) {
                if (sub_checked(v[j], delta, &v[j])) return 2;
            } else if (sub_checked(slack[j], delta, &slack[j])) return 2;
        }
        ++stats[1];
        if (add_checked(stats[2], n, &stats[2])) return 2;
        if (c2r[column] < 0) {
            while (column >= 0) {
                previous = pred[column];
                row = previous < 0 ? root : c2r[previous];
                if (row < 0 || row >= n) return 3;
                r2c[row] = column;
                c2r[column] = row;
                column = previous;
            }
            break;
        }
        used[column] = 1;
        previous = column;
        row = c2r[column];
    }
    ++stats[0];
    return 0;
}

static int objective(int64_t n, const int64_t *state, int64_t *value) {
    int64_t sum = 0;
    for (int64_t i = 0; i < 2*n; ++i)
        if (add_checked(sum, state[i], &sum)) return 2;
    if (sum > 0 || sum == INT64_MIN) return 3;
    *value = -sum;
    return 0;
}

int cara_column_compile(int64_t n, const int64_t *weights, int64_t *state,
                        int64_t *work, uint8_t *used,
                        int64_t *stats, int64_t *value) {
    if (!dimension_ok(n)) return 1;
    int64_t *u = state, *v = state+n, *r2c = state+2*n, *c2r = state+3*n;
    memset(stats, 0, 4*sizeof(int64_t));
    for (int64_t i = 0; i < n; ++i) {
        int64_t maximum = 0;
        for (int64_t j = 0; j < n; ++j) {
            int64_t w = weights[i*n+j];
            if (!weight_ok(n, w)) return 1;
            if (w > maximum) maximum = w;
        }
        u[i] = -maximum;
        v[i] = 0;
        r2c[i] = c2r[i] = -1;
    }
    for (int64_t i = 0; i < n; ++i) {
        int64_t selected = -1;
        if (c2r[i] < 0 && -weights[i*n+i] == u[i]) selected = i;
        else for (int64_t j = 0; j < n; ++j)
            if (c2r[j] < 0 && -weights[i*n+j] == u[i]) { selected = j; break; }
        if (selected >= 0) {
            r2c[i] = selected; c2r[selected] = i; ++stats[3];
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        if (r2c[i] >= 0) continue;
        int rc = augment(n, weights, state, i, 0, NULL, NULL, work, used, stats);
        if (rc) return rc;
    }
    return objective(n, state, value);
}

int cara_column_replay(int64_t n, const int64_t *weights,
                       const int64_t *base_state, int64_t changed_count,
                       const int64_t *columns, const int64_t *changed,
                       int64_t *state, int64_t *work, uint8_t *used,
                       int64_t *stats, int64_t *value) {
    if (!dimension_ok(n) || changed_count < 1 || changed_count > 2) return 1;
    for (int64_t k = 0; k < changed_count; ++k) {
        if (columns[k] < 0 || columns[k] >= n) return 1;
        if (k && columns[k] == columns[0]) return 1;
        for (int64_t i = 0; i < n; ++i)
            if (!weight_ok(n, changed[k*n+i])) return 1;
    }
    memcpy(state, base_state, (size_t)(4*n)*sizeof(int64_t));
    memset(stats, 0, 4*sizeof(int64_t));
    int64_t *u = state, *v = state+n, *r2c = state+2*n, *c2r = state+3*n;
    int64_t roots[2];
    for (int64_t k = 0; k < changed_count; ++k) {
        int64_t j = columns[k];
        roots[k] = c2r[j];
        if (roots[k] < 0 || roots[k] >= n) return 3;
        r2c[roots[k]] = -1;
        c2r[j] = -1;
        int64_t minimum = INF;
        for (int64_t i = 0; i < n; ++i) {
            int64_t current;
            if (sub_checked(-changed[k*n+i], u[i], &current)) return 2;
            if (current < minimum) minimum = current;
        }
        v[j] = minimum;
    }
    for (int64_t k = 0; k < changed_count; ++k) {
        int rc = augment(n, weights, state, roots[k], changed_count,
                         columns, changed, work, used, stats);
        if (rc) return rc;
    }
    return objective(n, state, value);
}
