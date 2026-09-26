/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "gpu_lu.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::gpu_lu {

namespace {

constexpr int kNone = -1;
__host__ __device__ __forceinline__ int flip(int i) { return -i - 2; }

// Degree buckets: one doubly linked list per degree, keyed by row or column index.
struct buckets_t {
  int* head;  // degree -> first element, cap + 1
  int* next;
  int* prev;
  int* deg;

  __host__ __device__ void init(int cap, int count)
  {
    for (int d = 0; d <= cap; ++d) { head[d] = kNone; }
    for (int k = 0; k < count; ++k) { next[k] = prev[k] = kNone; }
  }
  __host__ __device__ void insert(int k, int d)
  {
    deg[k]  = d;
    prev[k] = kNone;
    next[k] = head[d];
    if (head[d] != kNone) { prev[head[d]] = k; }
    head[d] = k;
  }
  __host__ __device__ void remove(int k)
  {
    const int d = deg[k];
    if (prev[k] != kNone) {
      next[prev[k]] = next[k];
    } else {
      head[d] = next[k];
    }
    if (next[k] != kNone) { prev[next[k]] = prev[k]; }
  }
  __host__ __device__ void update(int k, int d)
  {
    if (deg[k] == d) { return; }
    remove(k);
    insert(k, d);
  }
};

// The nucleus as a trailing matrix, stored both by column (with values) and by row (indices),
// each column and row with a fixed stride of `cap` entries.
struct trailing_t {
  int n;    // nucleus dimension
  int cap;  // stride
  int* c_i;
  double* c_x;
  int* c_len;
  int* r_j;
  int* r_len;
  double* max_in_col;
  double* pivot_row_val;
  double* pivot_col_val;
  unsigned char* pivot_col_mark;
  int* pivot_col_index;
  buckets_t cols, rows;

  __host__ __device__ int* col_i(int j) { return c_i + size_t(j) * cap; }
  __host__ __device__ double* col_x(int j) { return c_x + size_t(j) * cap; }
  __host__ __device__ int* row_j(int i) { return r_j + size_t(i) * cap; }

  __host__ __device__ double lookup(int i, int j)
  {
    const int* ci = col_i(j);
    for (int q = 0; q < c_len[j]; ++q) {
      if (ci[q] == i) { return col_x(j)[q]; }
    }
    return 0.0;
  }

  // Markowitz search with threshold partial pivoting, as right_looking_lu's.
  __host__ __device__ void markowitz_search(
    double pivot_tol, double threshold_tol, int& pivot_i, int& pivot_j, double& pivot_val)
  {
    double markowitz = double(n) * double(n);
    int nz           = 1;
    while (nz <= n) {
      int lower_bound = (nz - 1) * (nz - 1);
      bool done       = false;
      for (int j = cols.head[nz]; j != kNone && !done; j = cols.next[j]) {
        const double mic = max_in_col[j];
        const int* ci    = col_i(j);
        const double* cx = col_x(j);
        for (int p = 0; p < c_len[j]; ++p) {
          const int i      = ci[p];
          const double val = cx[p];
          const int mij    = (rows.deg[i] - 1) * (nz - 1);
          if (mij < markowitz && fabs(val) >= threshold_tol * mic && fabs(val) >= pivot_tol) {
            markowitz = mij;
            pivot_i   = i;
            pivot_j   = j;
            pivot_val = val;
            if (markowitz <= lower_bound) {
              done = true;
              break;
            }
          }
        }
      }
      if (markowitz <= lower_bound) { break; }
      lower_bound = (nz - 1) * nz;
      for (int i = rows.head[nz]; i != kNone && !done; i = rows.next[i]) {
        const int* rj = row_j(i);
        for (int p = 0; p < r_len[i]; ++p) {
          const int j      = rj[p];
          const double val = lookup(i, j);
          const int mij    = (nz - 1) * (cols.deg[j] - 1);
          if (mij < markowitz && fabs(val) >= threshold_tol * max_in_col[j] &&
              fabs(val) >= pivot_tol) {
            markowitz = mij;
            pivot_i   = i;
            pivot_j   = j;
            pivot_val = val;
            if (markowitz <= lower_bound) {
              done = true;
              break;
            }
          }
        }
      }
      if (markowitz <= lower_bound) { break; }
      if (pivot_i != kNone && nz >= 2) { break; }
      ++nz;
    }
  }

  __host__ __device__ void cache_pivot_row(int pivot_i)
  {
    const int* rj = row_j(pivot_i);
    for (int p = 0; p < r_len[pivot_i]; ++p) {
      const int j      = rj[p];
      pivot_row_val[j] = lookup(pivot_i, j);
    }
  }

  __host__ __device__ void remove_from_row(int i, int j)
  {
    int* rj = row_j(i);
    for (int q = 0; q < r_len[i]; ++q) {
      if (rj[q] == j) {
        rj[q] = rj[r_len[i] - 1];
        --r_len[i];
        return;
      }
    }
  }

  // A22 <- A22 - l u^T. Returns false when a column or row outgrows its stride.
  __host__ __device__ bool schur_complement(int pivot_i, int pivot_j, double drop_tol, double pivot_val)
  {
    int pivot_col_count = 0;
    {
      const int* ci    = col_i(pivot_j);
      const double* cx = col_x(pivot_j);
      for (int p = 0; p < c_len[pivot_j]; ++p) {
        const int i = ci[p];
        if (i == pivot_i) { continue; }
        pivot_col_val[i]                     = cx[p] / pivot_val;
        pivot_col_mark[i]                    = 1;
        pivot_col_index[pivot_col_count++] = i;
      }
    }
    const int* prow = row_j(pivot_i);
    for (int p0 = 0; p0 < r_len[pivot_i]; ++p0) {
      const int j = prow[p0];
      if (j == pivot_j) { continue; }
      const double uj = pivot_row_val[j];
      int n_fill      = pivot_col_count;
      int n_cancel    = 0;
      int* ci         = col_i(j);
      double* cx      = col_x(j);
      for (int q = 0; q < c_len[j]; ++q) {
        const int i = ci[q];
        if (pivot_col_mark[i]) {
          pivot_col_mark[i] = 0;
          --n_fill;
          cx[q] -= pivot_col_val[i] * uj;
          const double a = fabs(cx[q]);
          if (a > max_in_col[j]) { max_in_col[j] = a; }
          if (a < drop_tol) {
            cx[q] = 0;
            ++n_cancel;
          }
        }
      }
      if (n_cancel > 0) {
        int new_end = 0;
        for (int q = 0; q < c_len[j]; ++q) {
          if (cx[q] != 0) {
            ci[new_end] = ci[q];
            cx[new_end] = cx[q];
            ++new_end;
          } else {
            const int dead = ci[q];
            remove_from_row(dead, j);
            rows.update(dead, rows.deg[dead] - 1);
          }
        }
        c_len[j] = new_end;
      }
      if (n_fill > 0) {
        if (c_len[j] + n_fill > cap) { return false; }
        for (int k = 0; k < pivot_col_count; ++k) {
          const int i = pivot_col_index[k];
          if (!pivot_col_mark[i]) { continue; }
          const double val = pivot_col_val[i] * uj;
          const double a   = fabs(val);
          if (a < drop_tol) { continue; }
          ci[c_len[j]] = i;
          cx[c_len[j]] = -val;
          ++c_len[j];
          if (a > max_in_col[j]) { max_in_col[j] = a; }
          if (r_len[i] + 1 > cap) { return false; }
          row_j(i)[r_len[i]++] = j;
          rows.update(i, rows.deg[i] + 1);
        }
      }
      cols.update(j, c_len[j] - 1);  // -1 for the pivot row, removed next
      for (int k = 0; k < pivot_col_count; ++k) { pivot_col_mark[pivot_col_index[k]] = 1; }
    }
    for (int k = 0; k < pivot_col_count; ++k) {
      const int i       = pivot_col_index[k];
      pivot_col_val[i]  = 0;
      pivot_col_mark[i] = 0;
    }
    return true;
  }

  __host__ __device__ void remove_pivot_row_and_column(int pivot_i, int pivot_j)
  {
    const int* prow = row_j(pivot_i);
    for (int p = 0; p < r_len[pivot_i]; ++p) {
      const int j      = prow[p];
      pivot_row_val[j] = 0;
      if (j == pivot_j) { continue; }
      int* ci    = col_i(j);
      double* cx = col_x(j);
      double mic = 0.0;
      for (int q = 0; q < c_len[j]; ++q) {
        if (ci[q] == pivot_i) {
          ci[q] = ci[c_len[j] - 1];
          cx[q] = cx[c_len[j] - 1];
          --c_len[j];
          --q;
          continue;
        }
        const double a = fabs(cx[q]);
        if (a > mic) { mic = a; }
      }
      max_in_col[j] = mic;
    }
    const int* pcol = col_i(pivot_j);
    for (int p = 0; p < c_len[pivot_j]; ++p) {
      const int i = pcol[p];
      if (i == pivot_i) { continue; }
      rows.update(i, rows.deg[i] - 1);
      remove_from_row(i, pivot_j);
    }
    c_len[pivot_j] = 0;
    cols.update(pivot_j, 0);
    r_len[pivot_i] = 0;
    rows.update(pivot_i, 0);
  }
};

// Singleton peeling (host order_singletons). X is the side being peeled, Y the other side;
// Xp/Xi give X's structure, Yp/Yi Y's. A column's structure is A's column basic_list[k].
struct side_t {
  int* deg;
  int* perm;
  bool is_col;  // X structure = basis columns (else the row representation Rp/Rj)
};

__host__ __device__ int order_singletons(const matrix_t& A,
                                const int* basic_list,
                                const int* Rp,
                                const int* Rj,
                                int* queue,
                                int q_head,
                                int q_tail,
                                int& found,
                                side_t X,
                                side_t Y)
{
  auto x_begin = [&](int x) { return X.is_col ? A.col_start[basic_list[x]] : Rp[x]; };
  auto x_end   = [&](int x) { return X.is_col ? A.col_start[basic_list[x] + 1] : Rp[x + 1]; };
  auto x_at    = [&](int p) { return X.is_col ? A.row_index[p] : Rj[p]; };
  auto y_begin = [&](int y) { return Y.is_col ? A.col_start[basic_list[y]] : Rp[y]; };
  auto y_end   = [&](int y) { return Y.is_col ? A.col_start[basic_list[y] + 1] : Rp[y + 1]; };
  auto y_at    = [&](int p) { return Y.is_col ? A.row_index[p] : Rj[p]; };
  while (q_head != q_tail) {
    const int xpivot = queue[q_head++];
    if (X.deg[xpivot] != 1) { continue; }
    int ypivot = kNone;
    for (int p = x_begin(xpivot); p < x_end(xpivot); ++p) {
      const int y = x_at(p);
      if (Y.deg[y] >= 0) {
        ypivot = y;
        break;
      }
    }
    for (int p = y_begin(ypivot); p < y_end(ypivot); ++p) {
      const int x = y_at(p);
      if (X.deg[x] < 0 || x == xpivot) { continue; }
      if (--X.deg[x] == 1) { queue[q_tail++] = x; }
    }
    X.deg[xpivot]  = flip(1);
    Y.deg[ypivot]  = flip(Y.deg[ypivot]);
    X.perm[found]  = xpivot;
    Y.perm[found]  = ypivot;
    ++found;
  }
  return found;
}

__host__ __device__ void complete_permutation(int singletons, int n, int* deg, int* perm)
{
  int num_empty = 0;
  int start     = singletons;
  for (int k = 0; k < n; ++k) {
    const int d = deg[k];
    if (d == 0) {
      ++num_empty;
      perm[n - num_empty] = k;
    } else if (d > 0) {
      perm[start++] = k;
    } else {
      deg[k] = flip(d);
    }
  }
}

__host__ __device__ void transpose(int m, const int* cs, const int* ci, const double* cx, int* ts, int* ti,
                          double* tx)
{
  for (int i = 0; i <= m; ++i) { ts[i] = 0; }
  for (int p = 0; p < cs[m]; ++p) { ++ts[ci[p] + 1]; }
  for (int i = 0; i < m; ++i) { ts[i + 1] += ts[i]; }
  // ts[i] is now the start of row i; use a cursor copy in place by walking columns in order.
  for (int j = 0; j < m; ++j) {
    for (int p = cs[j]; p < cs[j + 1]; ++p) {
      const int q = ts[ci[p]]++;
      ti[q]       = j;
      tx[q]       = cx[p];
    }
  }
  for (int i = m; i > 0; --i) { ts[i] = ts[i - 1]; }
  ts[0] = 0;
}

}  // namespace

// Singleton peeling of B = A(:, basic_list) into w.col_perm / w.row_perm / w.row_perm_inv:
// column singletons first, then row singletons, then the nucleus (natural order), as the host.
__host__ __device__ int singleton_stage(const matrix_t& A,
                                        const int* basic_list,
                                        ws_ptrs_t& w,
                                        const layout_t& layout,
                                        int& col_singletons,
                                        int& row_singletons)
{
  const int m = A.m;
  // ---- degrees and the row representation of B = A(:, basic_list) ----
  long long Bnz = 0;
  for (int i = 0; i < m; ++i) { w.Rdeg[i] = 0; }
  for (int k = 0; k < m; ++k) {
    const int j = basic_list[k];
    w.Cdeg[k]   = A.col_start[j + 1] - A.col_start[j];
    Bnz += w.Cdeg[k];
    for (int p = A.col_start[j]; p < A.col_start[j + 1]; ++p) { ++w.Rdeg[A.row_index[p]]; }
  }
  if (Bnz > layout.b_cap) { return kBasisTooBig; }
  w.Rp[0] = 0;
  for (int i = 0; i < m; ++i) { w.Rp[i + 1] = w.Rp[i] + w.Rdeg[i]; }
  for (int i = 0; i < m; ++i) { w.queue[i] = w.Rp[i]; }  // cursor
  for (int k = 0; k < m; ++k) {
    const int j = basic_list[k];
    for (int p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
      w.Rj[w.queue[A.row_index[p]]++] = k;
    }
  }

  // ---- singletons: columns, then rows ----
  int found          = 0;
  col_singletons     = 0;
  int q_tail         = 0;
  for (int k = m - 1; k >= 0; --k) {
    if (w.Cdeg[k] == 1) { w.queue[q_tail++] = k; }
  }
  if (q_tail > 0) {
    col_singletons = order_singletons(A, basic_list, w.Rp, w.Rj, w.queue, 0, q_tail, found,
                                      side_t{w.Cdeg, w.col_perm, true},
                                      side_t{w.Rdeg, w.row_perm, false});
  }
  q_tail = 0;
  for (int i = m - 1; i >= 0; --i) {
    if (w.Rdeg[i] == 1) { w.queue[q_tail++] = i; }
  }
  row_singletons = 0;
  if (q_tail > 0) {
    const int last = found;
    order_singletons(A, basic_list, w.Rp, w.Rj, w.queue, 0, q_tail, found,
                     side_t{w.Rdeg, w.row_perm, false}, side_t{w.Cdeg, w.col_perm, true});
    row_singletons = found - last;
  }
  complete_permutation(found, m, w.Cdeg, w.col_perm);
  complete_permutation(found, m, w.Rdeg, w.row_perm);
  for (int k = 0; k < m; ++k) { w.row_perm_inv[w.row_perm[k]] = k; }

  return kOk;
}

__host__ __device__ int factorize(const matrix_t& A,
                         int* basic_list,
                         double threshold_tol,
                         unsigned char* ws_base,
                         const layout_t& layout,
                         const factors_t& out,
                         int* L_nz,
                         int* U_nz)
{
  const int m = A.m;
  ws_ptrs_t w;
  carve_workspace(ws_base, m, layout.nucleus_cap, layout.b_cap, layout.urow_cap, w);

  int col_singletons = 0, row_singletons = 0;
  {
    const int st = singleton_stage(A, basic_list, w, layout, col_singletons, row_singletons);
    if (st != kOk) { return st; }
  }
  const int ns   = col_singletons + row_singletons;
  const int Sdim = m - ns;
  if (Sdim > layout.nucleus_cap) { return kNucleusTooBig; }
  const double tol = ns > 0 ? threshold_tol : 1e-12;

  // ---- L and U of the singleton part ----
  int Lnz = 0, Unz = 0;
  auto col_begin = [&](int k) { return A.col_start[basic_list[w.col_perm[k]]]; };
  auto col_end   = [&](int k) { return A.col_start[basic_list[w.col_perm[k]] + 1]; };
  for (int k = 0; k < col_singletons; ++k) {
    if (Lnz + 1 > out.L_cap) { return kFactorOverflow; }
    out.L_col_start[k] = Lnz;
    out.L_i[Lnz]       = k;
    out.L_x[Lnz]       = 1.0;
    ++Lnz;
  }
  for (int k = 0; k < col_singletons; ++k) {
    out.U_col_start[k] = Unz;
    if (Unz + (col_end(k) - col_begin(k)) > out.U_cap) { return kFactorOverflow; }
    for (int p = col_begin(k); p < col_end(k); ++p) {
      out.U_i[Unz] = w.row_perm_inv[A.row_index[p]];
      out.U_x[Unz] = A.values[p];
      ++Unz;
    }
    // U(k, k) last in its column.
    const int cs = out.U_col_start[k], last = Unz - 1;
    for (int p = cs; p < last; ++p) {
      if (out.U_i[p] == k) {
        const double t = out.U_x[p];
        out.U_i[p]     = out.U_i[last];
        out.U_x[p]     = out.U_x[last];
        out.U_i[last]  = k;
        out.U_x[last]  = t;
        break;
      }
    }
  }
  for (int k = col_singletons; k < ns; ++k) {
    out.L_col_start[k] = Lnz;
    out.U_col_start[k] = Unz;
    if (Lnz + (col_end(k) - col_begin(k)) > out.L_cap ||
        Unz + (col_end(k) - col_begin(k)) + 1 > out.U_cap) {
      return kFactorOverflow;
    }
    for (int p = col_begin(k); p < col_end(k); ++p) {
      const int i = w.row_perm_inv[A.row_index[p]];
      if (i >= col_singletons) {
        out.L_i[Lnz] = i;
        out.L_x[Lnz] = A.values[p];
        ++Lnz;
      } else {
        out.U_i[Unz] = i;
        out.U_x[Unz] = A.values[p];
        ++Unz;
      }
    }
    out.U_i[Unz] = k;
    out.U_x[Unz] = 1.0;
    ++Unz;
  }
  // L(k, k) first in its column for the row-singleton columns.
  for (int k = col_singletons; k < ns; ++k) {
    const int cs = out.L_col_start[k];
    if (out.L_i[cs] == k) { continue; }
    for (int p = cs + 1; p < (k + 1 < ns ? out.L_col_start[k + 1] : Lnz); ++p) {
      if (out.L_i[p] == k) {
        const int ti   = out.L_i[cs];
        const double t = out.L_x[cs];
        out.L_i[cs]    = k;
        out.L_x[cs]    = out.L_x[p];
        out.L_i[p]     = ti;
        out.L_x[p]     = t;
        break;
      }
    }
  }

  // ---- the nucleus S ----
  const int C = layout.nucleus_cap;
  trailing_t T{Sdim, C, w.c_i, w.c_x, w.c_len, w.r_j, w.r_len, w.max_in_col, w.pivot_row_val,
               w.pivot_col_val, w.pivot_col_mark, w.pivot_col_index,
               buckets_t{w.col_head, w.col_next, w.col_prev, w.col_deg},
               buckets_t{w.row_head, w.row_next, w.row_prev, w.row_deg}};
  for (int t = 0; t < Sdim; ++t) {
    w.c_len[t] = w.r_len[t] = 0;
    w.max_in_col[t] = w.pivot_row_val[t] = w.pivot_col_val[t] = 0.0;
    w.pivot_col_mark[t] = 0;
    w.pinv_S[t] = w.q_S[t] = w.qinv_S[t] = kNone;
  }
  for (int k = ns; k < m; ++k) {
    const int s = k - ns;
    for (int p = col_begin(k); p < col_end(k); ++p) {
      const int i = w.row_perm_inv[A.row_index[p]];
      if (i < ns) { continue; }
      const int r = i - ns;
      if (T.c_len[s] >= C || T.r_len[r] >= C) { return kNucleusTooBig; }
      T.col_i(s)[T.c_len[s]] = r;
      T.col_x(s)[T.c_len[s]] = A.values[p];
      ++T.c_len[s];
      T.row_j(r)[T.r_len[r]++] = s;
      const double a = fabs(A.values[p]);
      if (a > T.max_in_col[s]) { T.max_in_col[s] = a; }
    }
  }
  T.cols.init(C, Sdim);
  T.rows.init(C, Sdim);
  for (int t = 0; t < Sdim; ++t) {
    T.cols.insert(t, T.c_len[t]);
    T.rows.insert(t, T.r_len[t]);
  }

  constexpr double pivot_tol = 1e-11;
  const double drop_tol      = tol == 1.0 ? 0.0 : 1e-13;
  int pivots                 = 0;
  int urow_nz                = 0;
  const int L33_start        = Lnz;
  for (int k = 0; k < Sdim; ++k) {
    int pivot_i = kNone, pivot_j = kNone;
    double pivot_val = 0.0;
    T.markowitz_search(pivot_tol, tol, pivot_i, pivot_j, pivot_val);
    if (pivot_i == kNone || pivot_j == kNone) { break; }
    w.pinv_S[pivot_i] = k;
    w.q_S[k]          = pivot_j;
    w.qinv_S[pivot_j] = k;
    ++pivots;
    T.cache_pivot_row(pivot_i);
    // U row k: the pivot, then the rest of the pivot row.
    if (urow_nz + T.r_len[pivot_i] > layout.urow_cap) { return kFactorOverflow; }
    w.urow_start[k]    = urow_nz;
    w.urow_j[urow_nz]  = pivot_j;
    w.urow_x[urow_nz]  = pivot_val;
    ++urow_nz;
    for (int p = 0; p < T.r_len[pivot_i]; ++p) {
      const int j = T.row_j(pivot_i)[p];
      if (j == pivot_j) { continue; }
      w.urow_j[urow_nz] = j;
      w.urow_x[urow_nz] = T.pivot_row_val[j];
      ++urow_nz;
    }
    // L column k (nucleus row indices; remapped after the loop).
    if (Lnz + T.c_len[pivot_j] > out.L_cap) { return kFactorOverflow; }
    out.L_col_start[ns + k] = Lnz;
    out.L_i[Lnz]            = pivot_i;
    out.L_x[Lnz]            = 1.0;
    ++Lnz;
    for (int p = 0; p < T.c_len[pivot_j]; ++p) {
      const int i = T.col_i(pivot_j)[p];
      if (i == pivot_i) { continue; }
      out.L_i[Lnz] = i;
      out.L_x[Lnz] = T.col_x(pivot_j)[p] / pivot_val;
      ++Lnz;
    }
    if (!T.schur_complement(pivot_i, pivot_j, drop_tol, pivot_val)) { return kNucleusTooBig; }
    T.remove_pivot_row_and_column(pivot_i, pivot_j);
  }
  if (pivots < Sdim) { return kRankDeficient; }
  w.urow_start[Sdim] = urow_nz;
  out.L_col_start[m] = Lnz;

  // S_perm[k] = nucleus row pivoted at step k (reuse queue).
  int* S_perm = w.queue;
  for (int r = 0; r < Sdim; ++r) { S_perm[w.pinv_S[r]] = r; }
  // L_33 rows to final positions, L_32 rows likewise.
  for (int p = L33_start; p < Lnz; ++p) { out.L_i[p] = ns + w.pinv_S[out.L_i[p]]; }
  for (int k = col_singletons; k < ns; ++k) {
    const int ce = (k + 1 < m ? out.L_col_start[k + 1] : Lnz);
    for (int p = out.L_col_start[k]; p < ce; ++p) {
      if (out.L_i[p] >= ns) { out.L_i[p] = ns + w.pinv_S[out.L_i[p] - ns]; }
    }
  }

  // Column order of the nucleus part: col_perm[ns + k] = old col_perm[ns + q_S[k]].
  int* saved = w.Cdeg;  // no longer needed
  for (int k = 0; k < Sdim; ++k) { saved[k] = w.col_perm[ns + k]; }
  for (int k = 0; k < Sdim; ++k) { w.col_perm[ns + k] = saved[w.q_S[k]]; }

  // U_33 by column (nucleus column q_S[k] -> final column ns + k), rows in pivot order.
  for (int k = 0; k <= Sdim; ++k) { w.u33_start[k] = 0; }
  for (int p = 0; p < urow_nz; ++p) { ++w.u33_start[w.qinv_S[w.urow_j[p]] + 1]; }
  for (int k = 0; k < Sdim; ++k) { w.u33_start[k + 1] += w.u33_start[k]; }
  // Final U columns ns..m-1: U_13 from B, then U_33.
  for (int k = ns; k < m; ++k) {
    out.U_col_start[k] = Unz;
    for (int p = col_begin(k); p < col_end(k); ++p) {
      const int i = w.row_perm_inv[A.row_index[p]];
      if (i < ns) {
        if (Unz + 1 > out.U_cap) { return kFactorOverflow; }
        out.U_i[Unz] = i;
        out.U_x[Unz] = A.values[p];
        ++Unz;
      }
    }
    const int s   = k - ns;
    const int cnt = w.u33_start[s + 1] - w.u33_start[s];
    if (Unz + cnt > out.U_cap) { return kFactorOverflow; }
    Unz += cnt;
  }
  out.U_col_start[m] = Unz;
  // Scatter U_33 rows (ascending pivot order, so each column's diagonal lands last).
  for (int k = 0; k < Sdim; ++k) { w.pivot_col_index[k] = 0; }  // per-column fill cursor
  for (int r = 0; r < Sdim; ++r) {
    for (int p = w.urow_start[r]; p < w.urow_start[r + 1]; ++p) {
      const int s   = w.qinv_S[w.urow_j[p]];
      const int end = out.U_col_start[ns + s + 1];
      const int cnt = w.u33_start[s + 1] - w.u33_start[s];
      const int q   = end - cnt + w.pivot_col_index[s]++;
      out.U_i[q]    = ns + r;
      out.U_x[q]    = w.urow_x[p];
    }
  }

  // Row permutation fix-up: row_perm[ns + k] = old row_perm[ns + S_perm[k]].
  int* last_perm = w.Rdeg;  // no longer needed
  for (int k = 0; k < Sdim; ++k) { last_perm[k] = w.row_perm[ns + k]; }
  for (int k = 0; k < Sdim; ++k) { w.row_perm[ns + k] = last_perm[S_perm[k]]; }
  for (int k = 0; k < m; ++k) {
    out.row_perm[k]                 = w.row_perm[k];
    out.inv_row_perm[w.row_perm[k]] = k;
  }
  // Basic list in pivot order.
  int* tmp = w.queue;
  for (int k = 0; k < m; ++k) { tmp[k] = basic_list[w.col_perm[k]]; }
  for (int k = 0; k < m; ++k) { basic_list[k] = tmp[k]; }

  transpose(m, out.L_col_start, out.L_i, out.L_x, out.Lt_col_start, out.Lt_i, out.Lt_x);
  transpose(m, out.U_col_start, out.U_i, out.U_x, out.Ut_col_start, out.Ut_i, out.Ut_x);
  *L_nz = Lnz;
  *U_nz = Unz;
  return kOk;
}

__host__ __device__ int singleton_order(const matrix_t& A,
                                        const int* basic_list,
                                        unsigned char* ws_base,
                                        const layout_t& layout,
                                        int* col_order,
                                        int* row_order,
                                        int* col_singletons,
                                        int* row_singletons)
{
  ws_ptrs_t w;
  carve_workspace(ws_base, A.m, layout.nucleus_cap, layout.b_cap, layout.urow_cap, w);
  const int st = singleton_stage(A, basic_list, w, layout, *col_singletons, *row_singletons);
  if (st != kOk) { return st; }
  for (int k = 0; k < A.m; ++k) {
    col_order[k] = w.col_perm[k];
    row_order[k] = w.row_perm[k];
  }
  return kOk;
}

namespace {
__global__ void batched_factorize_kernel(matrix_t A,
                                         int* basic_lists,
                                         int count,
                                         double threshold_tol,
                                         unsigned char* workspace,
                                         layout_t layout,
                                         const factors_t* factors,
                                         int* status,
                                         int* L_nz,
                                         int* U_nz)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= count) { return; }
  status[b] = factorize(A, basic_lists + size_t(b) * A.m, threshold_tol,
                        workspace + size_t(b) * layout.bytes, layout, factors[b], L_nz + b,
                        U_nz + b);
}
}  // namespace

void launch_batched_factorize(const matrix_t& A,
                              int* basic_lists,
                              int count,
                              double threshold_tol,
                              unsigned char* workspace,
                              const layout_t& layout,
                              const factors_t* factors,
                              int* status,
                              int* L_nz,
                              int* U_nz,
                              cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 32;
  batched_factorize_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
    A, basic_lists, count, threshold_tol, workspace, layout, factors, status, L_nz, U_nz);
}

namespace {
// One block per seat: zero the dense basis, scatter its columns, record its pointer.
__global__ void dense_form_kernel(matrix_t A, const dense_seat_t* seats, double* dense, double** ptrs)
{
  const int b     = blockIdx.x;
  const int m     = A.m;
  double* M       = dense + size_t(b) * m * m;
  if (threadIdx.x == 0) { ptrs[b] = M; }
  for (size_t t = threadIdx.x; t < size_t(m) * m; t += blockDim.x) { M[t] = 0.0; }
  __syncthreads();
  const int* basic = seats[b].basic_list;
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const int j = basic[k];
    for (int p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
      M[size_t(k) * m + A.row_index[p]] = A.values[p];
    }
  }
}

// One warp per seat: status, row permutation, compacted L (unit diagonal first) and U (diagonal
// last), then the transposes.
__global__ void dense_extract_kernel(int m,
                                     const dense_seat_t* seats,
                                     const double* dense,
                                     const int* piv,
                                     const int* cublas_info,
                                     double pivot_tol)
{
  const int b          = blockIdx.x;
  const int lane       = threadIdx.x;
  const dense_seat_t S = seats[b];
  const double* M      = dense + size_t(b) * m * m;
  const unsigned full  = 0xffffffffu;
  // Status: a zero or tiny pivot sends the node to the host.
  bool small = false;
  for (int k = lane; k < m; k += 32) { small |= !(fabs(M[size_t(k) * m + k]) >= pivot_tol); }
  small = __any_sync(full, small) || cublas_info[b] != 0;
  if (small) {
    if (lane == 0) { S.info[0] = kRankDeficient; }
    return;
  }
  if (lane == 0) {
    int* perm = S.inv_row_perm;  // scratch for the forward permutation
    for (int i = 0; i < m; ++i) { perm[i] = i; }
    const int* ip = piv + size_t(b) * m;
    for (int k = 0; k < m; ++k) {
      const int r = ip[k] - 1;
      const int t = perm[k];
      perm[k]     = perm[r];
      perm[r]     = t;
    }
    for (int k = 0; k < m; ++k) { S.row_perm[k] = perm[k]; }
    for (int k = 0; k < m; ++k) { S.inv_row_perm[S.row_perm[k]] = k; }
  }
  // L: column k = (k, 1), then the nonzeros below the diagonal. U: column k = the nonzeros above
  // the diagonal, then (k, U(k,k)). Warp-cooperative compaction, one column at a time.
  int Lnz = 0, Unz = 0;
  bool overflow = false;
  for (int k = 0; k < m && !overflow; ++k) {
    const double* col = M + size_t(k) * m;
    if (lane == 0) {
      S.L_col_start[k] = Lnz;
      S.U_col_start[k] = Unz;
    }
    if (Lnz + 1 > S.L_cap) {
      overflow = true;
      break;
    }
    if (lane == 0) {
      S.L_i[Lnz] = k;
      S.L_x[Lnz] = 1.0;
    }
    ++Lnz;
    for (int base = k + 1; base < m; base += 32) {
      const int i       = base + lane;
      const bool nz     = i < m && col[i] != 0.0;
      const unsigned bm = __ballot_sync(full, nz);
      const int cnt     = __popc(bm);
      if (Lnz + cnt > S.L_cap) {
        overflow = true;
        break;
      }
      if (nz) {
        const int q = Lnz + __popc(bm & ((1u << lane) - 1));
        S.L_i[q]    = i;
        S.L_x[q]    = col[i];
      }
      Lnz += cnt;
    }
    if (overflow) { break; }
    for (int base = 0; base < k; base += 32) {
      const int i       = base + lane;
      const bool nz     = i < k && col[i] != 0.0;
      const unsigned bm = __ballot_sync(full, nz);
      const int cnt     = __popc(bm);
      if (Unz + cnt > S.U_cap) {
        overflow = true;
        break;
      }
      if (nz) {
        const int q = Unz + __popc(bm & ((1u << lane) - 1));
        S.U_i[q]    = i;
        S.U_x[q]    = col[i];
      }
      Unz += cnt;
    }
    if (overflow || Unz + 1 > S.U_cap) {
      overflow = true;
      break;
    }
    if (lane == 0) {
      S.U_i[Unz] = k;
      S.U_x[Unz] = col[k];
    }
    ++Unz;
  }
  if (overflow) {
    if (lane == 0) { S.info[0] = kFactorOverflow; }
    return;
  }
  if (lane == 0) {
    S.L_col_start[m] = Lnz;
    S.U_col_start[m] = Unz;
    transpose(m, S.L_col_start, S.L_i, S.L_x, S.Lt_col_start, S.Lt_i, S.Lt_x);
    transpose(m, S.U_col_start, S.U_i, S.U_x, S.Ut_col_start, S.Ut_i, S.Ut_x);
    S.info[0] = kOk;
    S.info[1] = Lnz;
    S.info[2] = Unz;
  }
}
}  // namespace

void run_dense_batch(void* cublas_handle,
                     const matrix_t& A,
                     const dense_seat_t* seats,
                     int count,
                     double* dense,
                     double** ptrs,
                     int* piv,
                     int* cublas_info,
                     double pivot_tol,
                     cudaStream_t stream)
{
  if (count <= 0) { return; }
  auto h = static_cast<cublasHandle_t>(cublas_handle);
  dense_form_kernel<<<count, 256, 0, stream>>>(A, seats, dense, ptrs);
  cublasSetStream(h, stream);
  cublasDgetrfBatched(h, A.m, ptrs, A.m, piv, cublas_info, count);
  dense_extract_kernel<<<count, 32, 0, stream>>>(A.m, seats, dense, piv, cublas_info, pivot_tol);
}

void* create_blas_handle()
{
  cublasHandle_t h = nullptr;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { return nullptr; }
  return h;
}

void destroy_blas_handle(void* handle)
{
  if (handle != nullptr) { cublasDestroy(static_cast<cublasHandle_t>(handle)); }
}

long long basis_nnz_bound(int m, int n, const int* host_col_start)
{
  std::vector<int> len(n);
  for (int j = 0; j < n; ++j) { len[j] = host_col_start[j + 1] - host_col_start[j]; }
  std::sort(len.begin(), len.end(), std::greater<int>());
  long long s = 0;
  for (int k = 0; k < m && k < n; ++k) { s += len[k]; }
  return s;
}

// ============================================================================================
// Batch LU of fresh seats without cuBLAS (run_batch_lu). One block per basis:
//   1. structural singleton peeling in parallel rounds (column singletons: one entry in the
//      unassigned rows, then row singletons: one entry in the unassigned columns); a round's
//      singletons are ordered by index (deterministic); two claiming the same row / column means a
//      structurally singular basis;
//   2. the nucleus (what is left)
//        m <= kBatchDenseMaxM: densely in shared memory, partial pivoting, one pivot published per
//                              step through a shared flag (no block barrier inside the loop);
//        m >  kBatchDenseMaxM: sparsely in a per-basis global arena, in rounds of pairwise
//                              independent Markowitz pivots under the host's threshold rule;
//   3. factors in the seat slot format (L unit diagonal first, U diagonal last, Lt diagonal last,
//      Ut diagonal first; other entries in any order: the solve reads only the diagonal position).
// The factor columns are reordered (singletons first): the basic list is rewritten in that order.
namespace {
namespace bl {

constexpr unsigned kFull    = 0xffffffffu;
constexpr int kDenseRows    = 4;   // rows per lane in the dense update: nucleus <= 32 * kDenseRows
constexpr int kArenaCap     = 128;  // entries per nucleus column / row list in the sparse arena
constexpr double kThreshold = 0.1;

__host__ __device__ inline size_t align16(size_t v) { return (v + 15) & ~size_t{15}; }

__device__ inline int rank_below(unsigned mask, int lane) { return __popc(mask & ((1u << lane) - 1)); }

// Exclusive prefix sum of a[0..n) in shared memory, in place; returns the total. All threads.
template <int T>
__device__ int block_scan(int* a, int n)
{
  __shared__ int warp_sums[T / 32];
  __shared__ int total;
  const int per  = (n + T - 1) / T;
  const int lo   = min(n, int(threadIdx.x) * per);
  const int hi   = min(n, lo + per);
  const int lane = threadIdx.x & 31;
  const int w    = threadIdx.x >> 5;
  int s          = 0;
  for (int i = lo; i < hi; ++i) { s += a[i]; }
  int x = s;
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(kFull, x, o);
    if (lane >= o) { x += y; }
  }
  if (lane == 31) { warp_sums[w] = x; }
  __syncthreads();
  if (w == 0) {
    int v = lane < T / 32 ? warp_sums[lane] : 0;
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(kFull, v, o);
      if (lane >= o) { v += y; }
    }
    if (lane < T / 32) { warp_sums[lane] = v; }
    if (lane == T / 32 - 1) { total = v; }
  }
  __syncthreads();
  int excl = x - s + (w > 0 ? warp_sums[w - 1] : 0);
  for (int i = lo; i < hi; ++i) {
    const int t = a[i];
    a[i]        = excl;
    excl += t;
  }
  __syncthreads();
  return total;
}

// Per-block shared state common to both kernels: basis copy, positions, four scratch arrays.
struct common_t {
  int* basic;    // the seat's basic list (copied in; rewritten at the end)
  int* col_pos;  // basis column -> factor position (-1: nucleus)
  int* row_pos;  // row -> factor position (-1: nucleus)
  int *w0, *w1, *w2, *w3;
};

__device__ inline common_t carve_common(unsigned char* smem, int m)
{
  int* b = reinterpret_cast<int*>(smem);
  return common_t{b, b + m, b + 2 * m, b + 3 * m, b + 4 * m, b + 5 * m, b + 6 * m};
}
__host__ __device__ inline size_t common_bytes(int m) { return align16(7 * sizeof(int) * size_t(m)); }

// Peel the singletons. Returns with s_pos = number of singletons, *cs = column singletons, s_bad set
// on a structurally singular basis.
template <int T>
__device__ void peel(const matrix_t& A, const common_t& c, int* s_bad, int* s_pos, int* cs)
{
  const int m = A.m, tid = threadIdx.x;
  const int *Acs = A.col_start, *Ari = A.row_index;
  while (true) {
    for (int i = tid; i < m; i += T) { c.w2[i] = -1; }  // row claims
    for (int k = tid; k < m; k += T) {
      int cnt = 0, r = -1;
      if (c.col_pos[k] < 0) {
        const int j = c.basic[k];
        for (int p = Acs[j]; p < Acs[j + 1]; ++p) {
          const int i = Ari[p];
          if (c.row_pos[i] < 0) {
            r = i;
            if (++cnt > 1) { break; }
          }
        }
        if (cnt == 0) { *s_bad = kRankDeficient; }
      }
      const bool cand = c.col_pos[k] < 0 && cnt == 1;
      c.w0[k]         = cand;
      c.w1[k]         = cand ? r : -1;
    }
    __syncthreads();
    for (int k = tid; k < m; k += T) {
      if (c.w1[k] >= 0 && atomicCAS(&c.w2[c.w1[k]], -1, k) != -1) { *s_bad = kRankDeficient; }
    }
    __syncthreads();
    if (*s_bad) { return; }
    const int total = block_scan<T>(c.w0, m);
    for (int k = tid; k < m; k += T) {
      if (c.w1[k] >= 0) {
        const int p        = *s_pos + c.w0[k];
        c.col_pos[k]       = p;
        c.row_pos[c.w1[k]] = p;
      }
    }
    __syncthreads();
    if (tid == 0) { *s_pos += total; }
    __syncthreads();
    if (total == 0) { break; }
  }
  *cs = *s_pos;
  while (true) {
    for (int i = tid; i < m; i += T) {
      c.w0[i] = 0;   // degree in the unassigned columns
      c.w1[i] = -1;  // that column
      c.w2[i] = -1;  // column claims
    }
    __syncthreads();
    for (int k = tid; k < m; k += T) {
      if (c.col_pos[k] >= 0) { continue; }
      const int j = c.basic[k];
      for (int p = Acs[j]; p < Acs[j + 1]; ++p) {
        const int i = Ari[p];
        if (c.row_pos[i] < 0) {
          atomicAdd(&c.w0[i], 1);
          c.w1[i] = k;
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < m; i += T) {
      int flag = 0;
      if (c.row_pos[i] < 0) {
        if (c.w0[i] == 0) { *s_bad = kRankDeficient; }
        if (c.w0[i] == 1) {
          flag = 1;
          if (atomicCAS(&c.w2[c.w1[i]], -1, i) != -1) { *s_bad = kRankDeficient; }
        }
      }
      c.w3[i] = flag;
    }
    __syncthreads();
    if (*s_bad) { return; }
    const int total = block_scan<T>(c.w3, m);
    for (int i = tid; i < m; i += T) {
      if (c.row_pos[i] < 0 && c.w0[i] == 1) {
        const int p          = *s_pos + c.w3[i];
        c.row_pos[i]         = p;
        c.col_pos[c.w1[i]]   = p;
      }
    }
    __syncthreads();
    if (tid == 0) { *s_pos += total; }
    __syncthreads();
    if (total == 0) { break; }
  }
}

__device__ inline void write_info(const dense_seat_t& S, int status, int L_nz, int U_nz)
{
  if (threadIdx.x == 0) {
    S.info[0] = status;
    S.info[1] = L_nz;
    S.info[2] = U_nz;
  }
}

// Factors from the final positions. `nuc` reads a nucleus column: len(c) entries, entry(c, q, pos, x)
// with pos its row's factor position; local(k) is basis column k's nucleus column.
template <int T, class Nuc>
__device__ void emit(const matrix_t& A, const dense_seat_t& S, const common_t& c, int cs, int ns,
                     const Nuc& nuc)
{
  constexpr int W  = T / 32;
  const int m      = A.m;
  const int tid    = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int *Acs   = A.col_start, *Ari = A.row_index;
  const double* Ax = A.values;
  int* pos_col     = c.w1;  // position -> basis column
  for (int k = tid; k < m; k += T) {
    pos_col[c.col_pos[k]]      = k;
    S.row_perm[c.row_pos[k]]   = k;
    S.inv_row_perm[k]          = c.row_pos[k];
  }
  __syncthreads();
  // sizes (warp per position)
  for (int p = warp; p < m; p += W) {
    const int k = pos_col[p], j = c.basic[k], p0 = Acs[j], p1 = Acs[j + 1];
    int Lc = 1, Uc = 0;
    if (p < cs) {
      Uc = p1 - p0;
    } else if (p < ns) {
      Uc = 1;
      for (int base = p0; base < p1; base += 32) {
        const int q = base + lane;
        const int r = q < p1 ? c.row_pos[Ari[q]] : p;
        Uc += __popc(__ballot_sync(kFull, r < cs));
        Lc += __popc(__ballot_sync(kFull, r > p));
      }
    } else {
      const int cl  = nuc.local(k, p);
      const int len = nuc.len(cl);
      for (int base = p0; base < p1; base += 32) {
        const int q = base + lane;
        Uc += __popc(__ballot_sync(kFull, q < p1 && c.row_pos[Ari[q]] < ns));
      }
      for (int base = 0; base < len; base += 32) {
        const int q = base + lane;
        int rp      = -1;
        double x    = 0.0;
        if (q < len) { nuc.entry(cl, q, rp, x); }
        const bool keep = rp == p || (rp >= 0 && x != 0.0);
        Uc += __popc(__ballot_sync(kFull, keep && rp <= p));
        Lc += __popc(__ballot_sync(kFull, keep && rp > p));
      }
    }
    if (lane == 0) {
      c.w2[p] = Lc;
      c.w3[p] = Uc;
    }
  }
  __syncthreads();
  const int L_nz = block_scan<T>(c.w2, m);
  const int U_nz = block_scan<T>(c.w3, m);
  if (L_nz > S.L_cap || U_nz > S.U_cap) {
    write_info(S, kFactorOverflow, L_nz, U_nz);
    return;
  }
  // L (unit diagonal first) and U (diagonal last), warp per column
  for (int p = warp; p < m; p += W) {
    const int k = pos_col[p], j = c.basic[k], p0 = Acs[j], p1 = Acs[j + 1];
    int lq = c.w2[p] + 1, uq = c.w3[p];
    if (lane == 0) {
      S.L_col_start[p] = c.w2[p];
      S.U_col_start[p] = uq;
      S.L_i[c.w2[p]]   = p;
      S.L_x[c.w2[p]]   = 1.0;
    }
    double piv = 0.0;
    if (p < ns) {
      // U: every entry but the pivot for a column singleton, the rows < cs for a row singleton
      for (int base = p0; base < p1; base += 32) {
        const int q       = base + lane;
        const int r       = q < p1 ? c.row_pos[Ari[q]] : -1;
        const double x    = q < p1 ? Ax[q] : 0.0;
        const unsigned pm = __ballot_sync(kFull, r == p);
        if (pm != 0) { piv = __shfl_sync(kFull, x, __ffs(pm) - 1); }
        const bool sel    = r >= 0 && r != p && (r < cs || p < cs);
        const unsigned bm = __ballot_sync(kFull, sel);
        if (sel) {
          const int at = uq + rank_below(bm, lane);
          S.U_i[at]    = r;
          S.U_x[at]    = x;
        }
        uq += __popc(bm);
      }
      if (p >= cs) {  // row singleton: L below the pivot
        const double rp = 1.0 / piv;
        for (int base = p0; base < p1; base += 32) {
          const int q       = base + lane;
          const int r       = q < p1 ? c.row_pos[Ari[q]] : -1;
          const bool sel    = r > p;
          const unsigned bm = __ballot_sync(kFull, sel);
          if (sel) {
            const int at = lq + rank_below(bm, lane);
            S.L_i[at]    = r;
            S.L_x[at]    = Ax[q] * rp;
          }
          lq += __popc(bm);
        }
      }
    } else {
      for (int base = p0; base < p1; base += 32) {  // U13: the singleton rows
        const int q       = base + lane;
        const int r       = q < p1 ? c.row_pos[Ari[q]] : ns;
        const bool sel    = r < ns;
        const unsigned bm = __ballot_sync(kFull, sel);
        if (sel) {
          const int at = uq + rank_below(bm, lane);
          S.U_i[at]    = r;
          S.U_x[at]    = Ax[q];
        }
        uq += __popc(bm);
      }
      const int cl  = nuc.local(k, p);
      const int len = nuc.len(cl);
      for (int base = 0; base < len; base += 32) {
        const int q = base + lane;
        int rp      = -1;
        double x    = 0.0;
        if (q < len) { nuc.entry(cl, q, rp, x); }
        const unsigned dm = __ballot_sync(kFull, rp == p);
        if (dm != 0) { piv = __shfl_sync(kFull, x, __ffs(dm) - 1); }
        const bool su     = rp >= 0 && rp < p && x != 0.0;
        const bool sl     = rp > p && x != 0.0;
        const unsigned bu = __ballot_sync(kFull, su);
        const unsigned bl = __ballot_sync(kFull, sl);
        if (su) {
          const int at = uq + rank_below(bu, lane);
          S.U_i[at]    = rp;
          S.U_x[at]    = x;
        }
        if (sl) {
          const int at = lq + rank_below(bl, lane);
          S.L_i[at]    = rp;
          S.L_x[at]    = x;
        }
        uq += __popc(bu);
        lq += __popc(bl);
      }
    }
    if (lane == 0) {
      S.U_i[uq] = p;
      S.U_x[uq] = piv;
    }
  }
  if (tid == 0) {
    S.L_col_start[m] = L_nz;
    S.U_col_start[m] = U_nz;
  }
  __syncthreads();
  // transposes: Lt row r with its diagonal last, Ut row r with its diagonal first
  int* basic_out = S.basic_list;
  for (int p = tid; p < m; p += T) { basic_out[p] = c.basic[pos_col[p]]; }
  __syncthreads();  // pos_col (w1) is reused below
  for (int r = tid; r < m; r += T) {
    c.w0[r] = 0;
    c.w1[r] = 0;
    c.w2[r] = 0;
    c.w3[r] = 0;
  }
  __syncthreads();
  for (int col = warp; col < m; col += W) {
    for (int q = S.L_col_start[col] + lane; q < S.L_col_start[col + 1]; q += 32) {
      atomicAdd(&c.w2[S.L_i[q]], 1);
    }
    for (int q = S.U_col_start[col] + lane; q < S.U_col_start[col + 1]; q += 32) {
      atomicAdd(&c.w3[S.U_i[q]], 1);
    }
  }
  __syncthreads();
  block_scan<T>(c.w2, m);
  block_scan<T>(c.w3, m);
  for (int r = tid; r < m; r += T) {
    S.Lt_col_start[r] = c.w2[r];
    S.Ut_col_start[r] = c.w3[r];
  }
  for (int col = warp; col < m; col += W) {
    for (int q = S.L_col_start[col] + lane; q < S.L_col_start[col + 1]; q += 32) {
      const int r   = S.L_i[q];
      const int end = r + 1 < m ? c.w2[r + 1] : L_nz;
      const int at  = r == col ? end - 1 : c.w2[r] + atomicAdd(&c.w0[r], 1);
      S.Lt_i[at]    = col;
      S.Lt_x[at]    = S.L_x[q];
    }
    for (int q = S.U_col_start[col] + lane; q < S.U_col_start[col + 1]; q += 32) {
      const int r  = S.U_i[q];
      const int at = r == col ? c.w3[r] : c.w3[r] + 1 + atomicAdd(&c.w1[r], 1);
      S.Ut_i[at]   = col;
      S.Ut_x[at]   = S.U_x[q];
    }
  }
  if (tid == 0) {
    S.Lt_col_start[m] = L_nz;
    S.Ut_col_start[m] = U_nz;
  }
  write_info(S, kOk, L_nz, U_nz);
}

// ---- dense nucleus (m <= kBatchDenseMaxM) -------------------------------------------------

__host__ __device__ inline size_t dense_shared_bytes(int m)
{
  // common, nucleus maps (w2 / w3 are reused: local column / row), pivot rows, reciprocals, D
  return common_bytes(m) + align16(sizeof(int) * size_t(m)) + size_t(m) * sizeof(double) +
         size_t(m) * m * sizeof(double);
}

struct dense_nuc_t {
  const double* D;
  int Sd, ns;
  __device__ int local(int, int p) const { return p - ns; }
  __device__ int len(int) const { return Sd; }
  __device__ void entry(int cl, int q, int& pos, double& x) const
  {
    pos = ns + q;
    x   = D[size_t(cl) * Sd + q];
  }
};

template <int T>
__global__ void __launch_bounds__(T) batch_dense_kernel(matrix_t A, const dense_seat_t* seats,
                                                        double pivot_tol)
{
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int W           = T / 32;
  const int m               = A.m;
  const int tid             = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const dense_seat_t S      = seats[blockIdx.x];
  const common_t c          = carve_common(smem, m);
  int* piv_rows             = reinterpret_cast<int*>(smem + common_bytes(m));
  double* piv_rcp           = reinterpret_cast<double*>(smem + common_bytes(m) +
                                                        align16(sizeof(int) * size_t(m)));
  double* D                 = piv_rcp + m;
  const int *Acs            = A.col_start, *Ari = A.row_index;
  const double* Ax          = A.values;
  __shared__ int s_bad, s_pos, s_cs;
  for (int i = tid; i < m; i += T) {
    c.basic[i]   = S.basic_list[i];
    c.col_pos[i] = -1;
    c.row_pos[i] = -1;
  }
  if (tid == 0) {
    s_bad = 0;
    s_pos = 0;
  }
  __syncthreads();
  peel<T>(A, c, &s_bad, &s_pos, &s_cs);
  __syncthreads();
  const int cs = s_cs, ns = s_pos;
  if (s_bad) {
    write_info(S, s_bad, 0, 0);
    return;
  }
  const int Sd = m - ns;
  // nucleus numbering: w0 column -> local, w1 row -> local; w2 local -> column, w3 local -> row
  for (int k = tid; k < m; k += T) {
    c.w0[k] = c.col_pos[k] < 0;
    c.w1[k] = c.row_pos[k] < 0;
  }
  __syncthreads();
  block_scan<T>(c.w0, m);
  block_scan<T>(c.w1, m);
  for (int k = tid; k < m; k += T) {
    if (c.col_pos[k] < 0) { c.w2[c.w0[k]] = k; }
    if (c.row_pos[k] < 0) { c.w3[c.w1[k]] = k; }
  }
  for (int t = tid; t < Sd * Sd; t += T) { D[t] = 0.0; }
  __syncthreads();
  for (int jl = warp; jl < Sd; jl += W) {
    const int j = c.basic[c.w2[jl]], p1 = Acs[j + 1];
    for (int p = Acs[j] + lane; p < p1; p += 32) {
      const int i = Ari[p];
      if (c.row_pos[i] < 0) { D[size_t(jl) * Sd + c.w1[i]] = Ax[p]; }
    }
  }
  for (int l = tid; l < Sd; l += T) {
    c.w0[l]     = l;   // pivot order of the local rows
    piv_rows[l] = -1;  // pivot row of each step; -1 = not yet published
  }
  __syncthreads();

  // Elimination. Column j belongs to warp j % W. Step kk reads the pivot column kk in place and
  // never writes it: its row swap and the L scaling are applied to every L column after the loop.
  // A warp applies each step's row swap to its own columns while updating them. The owner of
  // column kk+1 updates it first, searches it and publishes step kk+1's pivot row and reciprocal,
  // then the flag; a warp waits only for the pivot of the step it is on. Columns whose pivot-row
  // entry is zero only swap two entries.
  volatile int* vp = piv_rows;
  auto publish     = [&](int kk) {
    // argmax |D(i, kk)|, i >= kk, as one redux over |value| as float (high bits) and 127 - i
    // (low 7 bits): ties at float resolution go to the smaller row
    unsigned key = 0;
    for (int i = kk + lane; i < Sd; i += 32) {
      const float a    = __double2float_rz(fabs(D[size_t(kk) * Sd + i]));
      const unsigned k = (__float_as_uint(a) & ~0x7Fu) | unsigned(127 - i);
      key              = max(key, k);
    }
    key           = __reduce_max_sync(kFull, key);
    const int idx = 127 - int(key & 0x7Fu);
    if (lane == 0) {
      const double pv = key != 0 ? D[size_t(kk) * Sd + idx] : 0.0;
      if (!(fabs(pv) >= pivot_tol)) {
        s_bad = kRankDeficient;
        __threadfence_block();
        vp[kk] = kk;  // release the waiters; s_bad stops them
      } else {
        double r;  // reciprocal: approximation + two Newton steps
        asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(pv));
        double e    = fma(-pv, r, 1.0);
        r           = fma(r, e, r);
        e           = fma(-pv, r, 1.0);
        r           = fma(r, e, r);
        piv_rcp[kk] = r;
        __threadfence_block();
        vp[kk] = idx;
      }
    }
    __syncwarp();
  };
  if (Sd > 0 && warp == 0) { publish(0); }
  for (int kk = 0; kk < Sd; ++kk) {
    const int first = kk + 1 + ((warp - (kk + 1) % W) + W) % W;  // this warp's first column > kk
    if (first >= Sd) { break; }
    while (vp[kk] < 0) {}
    __threadfence_block();
    if (*reinterpret_cast<volatile int*>(&s_bad)) { break; }
    const int bi      = vp[kk];
    const double rpiv = piv_rcp[kk];
    const double* pc  = D + size_t(kk) * Sd;
    double lr[kDenseRows];  // multipliers of this lane's rows (row bi now holds the old row kk)
    bool any_l = false;
#pragma unroll
    for (int r = 0; r < kDenseRows; ++r) {
      const int ii = kk + 1 + lane + 32 * r;
      lr[r]        = ii < Sd ? pc[ii == bi ? kk : ii] * rpiv : 0.0;
      any_l |= lr[r] != 0.0;
    }
    any_l = __any_sync(kFull, any_l);
    for (int jj = first; jj < Sd; jj += W) {
      double* col     = D + size_t(jj) * Sd;
      const double ak = col[bi];  // the pivot row's entry
      const double vk = col[kk];  // moves to row bi
      if (ak == 0.0 || !any_l) {
        if (lane == 0 && bi != kk) {
          col[kk] = ak;
          col[bi] = vk;
        }
      } else {
        double v[kDenseRows];
#pragma unroll
        for (int r = 0; r < kDenseRows; ++r) {
          const int ii = kk + 1 + lane + 32 * r;
          v[r]         = ii < Sd ? (ii == bi ? vk : col[ii]) : 0.0;
        }
        __syncwarp();
#pragma unroll
        for (int r = 0; r < kDenseRows; ++r) {
          const int ii = kk + 1 + lane + 32 * r;
          if (ii < Sd) { col[ii] = v[r] - lr[r] * ak; }
        }
        if (lane == 0) { col[kk] = ak; }
      }
      __syncwarp();
      if (jj == kk + 1) { publish(kk + 1); }
    }
  }
  __syncthreads();
  if (s_bad) {
    write_info(S, s_bad, 0, 0);
    return;
  }
  // deferred: row swaps of each L column (steps j..Sd-1), then its scaling below the diagonal
  for (int j = tid; j < Sd; j += T) {
    double* col = D + size_t(j) * Sd;
    for (int kk = j; kk < Sd; ++kk) {
      const int r = piv_rows[kk];
      if (r != kk) {
        const double t = col[kk];
        col[kk]        = col[r];
        col[r]         = t;
      }
    }
  }
  if (tid == 0) {
    for (int kk = 0; kk < Sd; ++kk) {
      const int r = piv_rows[kk];
      const int t = c.w0[kk];
      c.w0[kk]    = c.w0[r];
      c.w0[r]     = t;
    }
  }
  __syncthreads();
  for (int j = warp; j < Sd; j += W) {
    double* col    = D + size_t(j) * Sd;
    const double r = 1.0 / col[j];
    for (int i = j + 1 + lane; i < Sd; i += 32) { col[i] *= r; }
  }
  for (int jl = tid; jl < Sd; jl += T) { c.col_pos[c.w2[jl]] = ns + jl; }
  for (int l = tid; l < Sd; l += T) { c.row_pos[c.w3[c.w0[l]]] = ns + l; }
  __syncthreads();
  emit<T>(A, S, c, cs, ns, dense_nuc_t{D, Sd, ns});
}

// ---- sparse nucleus (m > kBatchDenseMaxM) --------------------------------------------------

__host__ __device__ inline size_t sparse_arena_bytes(int m)
{
  return align16(size_t(m) * 2 * kArenaCap * sizeof(int)) + align16(size_t(m) * kArenaCap * sizeof(double));
}

template <int W>
__host__ __device__ inline size_t sparse_shared_bytes(int m)
{
  // common, 14 m ints of nucleus state + one m-int row -> slot map per warp, 3 m claim keys,
  // m column maxima
  return common_bytes(m) + align16((14 + size_t(W)) * sizeof(int) * size_t(m)) +
         align16(3 * sizeof(unsigned) * size_t(m)) + size_t(m) * sizeof(double);
}

struct sparse_nuc_t {
  const int* cloc;
  const int* col_len;
  const int* col_i;
  const double* col_x;
  const int* rpiv;
  __device__ int local(int k, int) const { return cloc[k]; }
  __device__ int len(int cl) const { return col_len[cl]; }
  __device__ void entry(int cl, int q, int& pos, double& x) const
  {
    pos = rpiv[col_i[size_t(cl) * kArenaCap + q]];
    x   = col_x[size_t(cl) * kArenaCap + q];
  }
};

template <int T>
__global__ void __launch_bounds__(T) batch_sparse_kernel(matrix_t A, const dense_seat_t* seats,
                                                         void* arena, double pivot_tol)
{
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int W      = T / 32;
  constexpr int cc     = kArenaCap, cr = kArenaCap;
  const int m          = A.m;
  const int tid        = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const dense_seat_t S = seats[blockIdx.x];
  const common_t c     = carve_common(smem, m);
  int* st              = reinterpret_cast<int*>(smem + common_bytes(m));
  int* ncol            = st;              // local column -> basis column
  int* nrow            = ncol + m;        // local row -> row
  int* rloc            = nrow + m;        // row -> local row (-1: singleton row)
  int* cloc            = rloc + m;        // basis column -> local column (-1)
  int* col_len         = cloc + m;
  int* row_len         = col_len + m;
  int* ccnt            = row_len + m;     // active entries per local column
  int* rcnt            = ccnt + m;        // active entries per local row
  int* cpiv            = rcnt + m;        // factor position of a local column (-1: active)
  int* rpiv            = cpiv + m;        // factor position of a local row (-1: active)
  int* cand_row        = rpiv + m;        // this round's candidate row per local column (-1: none)
  int* winf            = cand_row + m;    // winner flags, then positions within the round
  int* wmap            = winf + 2 * m + warp * m;  // this warp's row -> slot map (-1: absent)
  unsigned* row_claim  = reinterpret_cast<unsigned*>(
    smem + common_bytes(m) + align16((14 + size_t(W)) * sizeof(int) * size_t(m)));
  unsigned* col_claim  = row_claim + m;
  unsigned* cand_key   = col_claim + m;   // cost (20 bits, clamped) << 12 | local column
  double* cmax         = reinterpret_cast<double*>(
    reinterpret_cast<unsigned char*>(row_claim) + align16(3 * sizeof(unsigned) * size_t(m)));
  auto* abase          = static_cast<unsigned char*>(arena) + size_t(blockIdx.x) * sparse_arena_bytes(m);
  int* col_i           = reinterpret_cast<int*>(abase);
  int* row_j           = col_i + size_t(m) * cc;
  double* col_x        = reinterpret_cast<double*>(abase + align16(size_t(m) * (cc + cr) * sizeof(int)));
  const int *Acs       = A.col_start, *Ari = A.row_index;
  const double* Ax     = A.values;
  __shared__ int s_bad, s_pos, s_cs, s_done;
  __shared__ unsigned s_mincost;
  for (int i = tid; i < m; i += T) {
    c.basic[i]   = S.basic_list[i];
    c.col_pos[i] = -1;
    c.row_pos[i] = -1;
  }
  if (tid == 0) {
    s_bad  = 0;
    s_pos  = 0;
    s_done = 0;
  }
  __syncthreads();
  peel<T>(A, c, &s_bad, &s_pos, &s_cs);
  __syncthreads();
  const int cs = s_cs, ns = s_pos;
  if (s_bad) {
    write_info(S, s_bad, 0, 0);
    return;
  }
  const int Sd = m - ns;
  // ---- nucleus into the arena ----
  for (int k = tid; k < m; k += T) {
    c.w0[k] = c.col_pos[k] < 0;
    c.w1[k] = c.row_pos[k] < 0;
  }
  __syncthreads();
  block_scan<T>(c.w0, m);
  block_scan<T>(c.w1, m);
  for (int k = tid; k < m; k += T) {
    cloc[k] = c.col_pos[k] < 0 ? c.w0[k] : -1;
    rloc[k] = c.row_pos[k] < 0 ? c.w1[k] : -1;
    if (c.col_pos[k] < 0) { ncol[c.w0[k]] = k; }
    if (c.row_pos[k] < 0) { nrow[c.w1[k]] = k; }
  }
  for (int l = tid; l < Sd; l += T) {
    col_len[l] = 0;
    row_len[l] = 0;
    cpiv[l]    = -1;
    rpiv[l]    = -1;
  }
  for (int l = lane; l < m; l += 32) { wmap[l] = -1; }
  __syncthreads();
  for (int cl = warp; cl < Sd; cl += W) {
    const int j = c.basic[ncol[cl]], p1 = Acs[j + 1];
    int len     = 0;
    for (int base = Acs[j]; base < p1; base += 32) {
      const int p       = base + lane;
      const int r       = p < p1 ? rloc[Ari[p]] : -1;
      const unsigned bm = __ballot_sync(kFull, r >= 0);
      if (r >= 0) {
        const int at = len + rank_below(bm, lane);
        if (at < cc) {
          col_i[size_t(cl) * cc + at] = r;
          col_x[size_t(cl) * cc + at] = Ax[p];
        }
        const int rs = atomicAdd(&row_len[r], 1);
        if (rs < cr) { row_j[size_t(r) * cr + rs] = cl; }
      }
      len += __popc(bm);
    }
    if (lane == 0) {
      col_len[cl] = len;
      if (len > cc) { s_bad = kFactorOverflow; }
    }
  }
  __syncthreads();
  for (int r = tid; r < Sd; r += T) {
    if (row_len[r] > cr) { s_bad = kFactorOverflow; }
  }
  __syncthreads();

  // ---- elimination rounds ----
  while (!s_bad && s_done < Sd) {
    // counts and column maxima of the active part; round state reset (thread per column / row)
    for (int cl = tid; cl < Sd; cl += T) {
      row_claim[cl] = 0xffffffffu;
      col_claim[cl] = 0xffffffffu;
      cand_key[cl]  = 0xffffffffu;
      cand_row[cl]  = -1;
      winf[cl]      = 0;
      if (cpiv[cl] >= 0) { continue; }
      int cnt       = 0;
      double mx     = 0.0;
      const int len = col_len[cl];
      for (int q = 0; q < len; ++q) {
        if (rpiv[col_i[size_t(cl) * cc + q]] < 0) {
          ++cnt;
          mx = fmax(mx, fabs(col_x[size_t(cl) * cc + q]));
        }
      }
      ccnt[cl] = cnt;
      cmax[cl] = mx;
    }
    for (int r = tid; r < Sd; r += T) {
      if (rpiv[r] >= 0) { continue; }
      int cnt       = 0;
      const int len = row_len[r];
      for (int q = 0; q < len; ++q) { cnt += cpiv[row_j[size_t(r) * cr + q]] < 0; }
      rcnt[r] = cnt;
    }
    if (tid == 0) { s_mincost = 0xffffffffu; }
    __syncthreads();
    // candidates: per active column, the cheapest acceptable entry (larger |x|, then smaller row)
    for (int cl = tid; cl < Sd; cl += T) {
      if (cpiv[cl] >= 0) { continue; }
      const int len           = col_len[cl];
      const double acc        = fmax(kThreshold * cmax[cl], pivot_tol);
      unsigned long long best = ~0ull;
      int best_r              = -1;
      for (int q = 0; q < len; ++q) {
        const int r    = col_i[size_t(cl) * cc + q];
        const double a = fabs(col_x[size_t(cl) * cc + q]);
        if (rpiv[r] < 0 && a >= acc) {
          const unsigned cost          = unsigned(rcnt[r] - 1) * unsigned(ccnt[cl] - 1);
          const unsigned long long key = (static_cast<unsigned long long>(cost) << 32) |
                                         (~__float_as_uint(float(a)) & 0x7fffffffu);
          if (key < best || (key == best && r < best_r)) {
            best   = key;
            best_r = r;
          }
        }
      }
      if (best_r < 0) {
        s_bad = kRankDeficient;  // no acceptable pivot left in an active column
      } else {
        cand_row[cl]        = best_r;
        const unsigned cost = min(unsigned(best >> 32), 0xfffffu);
        cand_key[cl]        = (cost << 12) | unsigned(cl);
        atomicMin(&s_mincost, cost);
      }
    }
    __syncthreads();
    if (s_bad) { break; }
    const unsigned bound = s_mincost;  // this round: the cheapest pivots only
    // claims: rows of the candidate's column, columns of its row (thread per candidate)
    for (int cl = tid; cl < Sd; cl += T) {
      if (cand_row[cl] < 0 || (cand_key[cl] >> 12) > bound) { continue; }
      const unsigned key = cand_key[cl];
      const int r        = cand_row[cl];
      for (int q = 0; q < col_len[cl]; ++q) {
        const int i = col_i[size_t(cl) * cc + q];
        if (rpiv[i] < 0 && row_claim[i] > key) { atomicMin(&row_claim[i], key); }
      }
      for (int q = 0; q < row_len[r]; ++q) {
        const int j = row_j[size_t(r) * cr + q];
        if (cpiv[j] < 0 && col_claim[j] > key) { atomicMin(&col_claim[j], key); }
      }
    }
    __syncthreads();
    for (int cl = tid; cl < Sd; cl += T) {  // winners hold every claim: pairwise independent
      if (cand_row[cl] < 0 || (cand_key[cl] >> 12) > bound) { continue; }
      const unsigned key = cand_key[cl];
      const int r        = cand_row[cl];
      bool mine          = true;
      for (int q = 0; q < col_len[cl] && mine; ++q) {
        const int i = col_i[size_t(cl) * cc + q];
        if (rpiv[i] < 0) { mine = row_claim[i] == key; }
      }
      for (int q = 0; q < row_len[r] && mine; ++q) {
        const int j = row_j[size_t(r) * cr + q];
        if (cpiv[j] < 0) { mine = col_claim[j] == key; }
      }
      winf[cl] = mine;
    }
    __syncthreads();
    int* wpos = winf + m;  // positions of this round's pivots, by column index
    for (int l = tid; l < Sd; l += T) { wpos[l] = winf[l]; }
    __syncthreads();
    const int won = block_scan<T>(wpos, Sd);
    if (won == 0) {
      if (tid == 0) { s_bad = kRankDeficient; }
      break;
    }
    // eliminate each winner: warp per pivot, lanes over the pivot column's rows
    for (int cl = warp; cl < Sd; cl += W) {
      if (!winf[cl]) { continue; }
      const int r    = cand_row[cl];
      int* ci        = col_i + size_t(cl) * cc;
      double* cx     = col_x + size_t(cl) * cc;
      const int clen = col_len[cl];
      double v       = 0.0;
      for (int q = lane; q < clen; q += 32) {
        if (ci[q] == r) { v = cx[q]; }
      }
      for (int o = 16; o > 0; o >>= 1) {
        const double ov = __shfl_down_sync(kFull, v, o);
        if (ov != 0.0) { v = ov; }
      }
      v               = __shfl_sync(kFull, v, 0);
      const double rv = 1.0 / v;
      for (int q = lane; q < clen; q += 32) {
        if (ci[q] != r && rpiv[ci[q]] < 0) { cx[q] *= rv; }  // L multipliers
      }
      __syncwarp();
      const int rlen = row_len[r];
      for (int t = 0; t < rlen; ++t) {
        const int j = row_j[size_t(r) * cr + t];
        if (j == cl || cpiv[j] >= 0) { continue; }
        int* ji        = col_i + size_t(j) * cc;
        double* jx     = col_x + size_t(j) * cc;
        const int len0 = col_len[j];
        for (int q = lane; q < len0; q += 32) { wmap[ji[q]] = q; }
        __syncwarp();
        const int ur   = wmap[r];
        const double u = ur >= 0 ? jx[ur] : 0.0;
        if (u != 0.0) {
          for (int q = lane; q < clen; q += 32) {
            const int i = ci[q];
            if (i == r || rpiv[i] >= 0) { continue; }
            const double d = cx[q] * u;
            const int at   = wmap[i];
            if (at >= 0) {
              jx[at] -= d;
            } else {  // fill
              const int slot = atomicAdd(&col_len[j], 1);
              const int rs   = atomicAdd(&row_len[i], 1);
              if (slot < cc && rs < cr) {
                ji[slot]                   = i;
                jx[slot]                   = -d;
                row_j[size_t(i) * cr + rs] = j;
              } else {
                s_bad = kFactorOverflow;
              }
            }
          }
        }
        __syncwarp();
        for (int q = lane; q < len0; q += 32) { wmap[ji[q]] = -1; }
        __syncwarp();
      }
    }
    __syncthreads();
    for (int cl = tid; cl < Sd; cl += T) {
      if (winf[cl]) {
        const int p        = ns + s_done + wpos[cl];
        cpiv[cl]           = p;
        rpiv[cand_row[cl]] = p;
      }
    }
    __syncthreads();
    if (tid == 0) { s_done += won; }
    __syncthreads();
  }
  __syncthreads();
  if (s_bad) {
    write_info(S, s_bad, 0, 0);
    return;
  }
  for (int l = tid; l < Sd; l += T) {
    c.col_pos[ncol[l]] = cpiv[l];
    c.row_pos[nrow[l]] = rpiv[l];
  }
  __syncthreads();
  emit<T>(A, S, c, cs, ns, sparse_nuc_t{cloc, col_len, col_i, col_x, rpiv});
}

template <class K>
void set_shared(K kernel, size_t bytes)
{
  if (bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  }
}

}  // namespace bl
}  // namespace

size_t batch_lu_workspace_bytes(int m, int count)
{
  return m > kBatchDenseMaxM ? size_t(count) * bl::sparse_arena_bytes(m) : 0;
}

void run_batch_lu(const matrix_t& A,
                  const dense_seat_t* seats,
                  int count,
                  void* workspace,
                  double pivot_tol,
                  cudaStream_t stream)
{
  if (count <= 0) { return; }
  const int m = A.m;
  if (m <= kBatchDenseMaxM) {
    // 128 threads for small bases, 512 above (one basis per SM, most warps per basis)
    if (m <= 48) {
      const size_t bytes = bl::dense_shared_bytes(m);
      bl::set_shared(bl::batch_dense_kernel<128>, bytes);
      bl::batch_dense_kernel<128><<<count, 128, bytes, stream>>>(A, seats, pivot_tol);
    } else {
      const size_t bytes = bl::dense_shared_bytes(m);
      bl::set_shared(bl::batch_dense_kernel<512>, bytes);
      bl::batch_dense_kernel<512><<<count, 512, bytes, stream>>>(A, seats, pivot_tol);
    }
  } else {
    const size_t bytes = bl::sparse_shared_bytes<8>(m);
    bl::set_shared(bl::batch_sparse_kernel<256>, bytes);
    bl::batch_sparse_kernel<256><<<count, 256, bytes, stream>>>(A, seats, workspace, pivot_tol);
  }
}

namespace {
__global__ void expand_fresh_kernel(expand_seats_t e)
{
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (warp >= e.count) { return; }
  const int s                = e.seats[warp];
  const std::uint8_t* packed = e.packed + static_cast<size_t>(s) * e.packed_stride;
  std::int8_t* vstatus       = e.vstatus + static_cast<size_t>(s) * e.n;
  int* basic                 = e.basic_list + static_cast<size_t>(s) * e.m;
  int* nonbasic              = e.nonbasic_list + static_cast<size_t>(s) * (e.n - e.m);
  double* dy_se              = e.dy_se + static_cast<size_t>(s) * e.n;
  // variable_status_t: BASIC 0, NONBASIC_LOWER -1, NONBASIC_UPPER 1, NONBASIC_FREE 2.
  constexpr std::int8_t decode[4] = {0, -1, 1, 2};
  int basics = 0, nonbasics = 0;
  for (int base = 0; base < e.n; base += 32) {
    const int j     = base + lane;
    const bool live = j < e.n;
    const int code  = live ? (packed[j >> 2] >> (2 * (j & 3))) & 3 : 0;
    if (live) {
      vstatus[j] = decode[code];
      dy_se[j]   = e.root_dy_se[j];
    }
    const unsigned is_basic    = __ballot_sync(0xffffffffu, live && code == 0);
    const unsigned is_nonbasic = __ballot_sync(0xffffffffu, live && code != 0);
    const unsigned below       = (1u << lane) - 1u;
    if (live && code == 0) {
      const int k = basics + __popc(is_basic & below);
      if (k < e.m) { basic[k] = j; }
    } else if (live) {
      const int k = nonbasics + __popc(is_nonbasic & below);
      if (k < e.n - e.m) { nonbasic[k] = j; }
    }
    basics += __popc(is_basic);
    nonbasics += __popc(is_nonbasic);
  }
}
}  // namespace

void expand_fresh_seats(const expand_seats_t& e, cudaStream_t stream)
{
  if (e.count <= 0) { return; }
  constexpr int threads = 128;
  const int blocks      = (e.count * 32 + threads - 1) / threads;
  expand_fresh_kernel<<<blocks, threads, 0, stream>>>(e);
}

}  // namespace cuopt::linear_programming::dual_simplex::gpu_lu
