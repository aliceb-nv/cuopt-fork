/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Explicit inverses of the base LU factors for the fissioned GPU dual simplex (P B = L0 U0):
//   Linv = L0^{-1} P, stored with column index = ORIGINAL row (P folded in), rows in L-space;
//   Uinv = U0^{-1},   rows = basis position, columns in L-space.
// Each is kept row-wise (CSR) and column-wise (CSC): FTRAN scatters columns of both, BTRAN reads a
// row of Uinv and scatters rows of Linv. Column k of an inverse solves T y = e_k touching only the
// reach of k (depth-first search on T's graph, then numeric in topological order).

#include <cstdint>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::explicit_inverse {

struct sparse_t {
  std::vector<int> p, i;  // p: m+1 pointers; i: indices (CSR: columns; CSC: rows)
  std::vector<double> x;
};

struct inverses_t {
  sparse_t Lr, Lc, Ur, Uc;  // r: CSR, c: CSC
};

struct workspace_t {
  std::vector<double> y;
  std::vector<int> stack, next, order;
  std::vector<char> mark;
};

// Column-wise inverse of a triangular CSC factor. lower: diagonal stored first in each column;
// upper: diagonal stored last. col_map (optional) renames output columns.
inline void inverse_columns(int m,
                            const int* cp,
                            const int* ri,
                            const double* cx,
                            bool lower,
                            const int* col_map,
                            workspace_t& w,
                            sparse_t& out)
{
  w.y.assign(m, 0.0);
  w.mark.assign(m, 0);
  w.stack.resize(m);
  w.next.resize(m);
  w.order.resize(m);
  // Columns with renamed indices are emitted in renamed order so the CSC stays sorted by column.
  std::vector<int> by_out(m);
  for (int k = 0; k < m; ++k) { by_out[col_map != nullptr ? col_map[k] : k] = k; }
  out.p.assign(m + 1, 0);
  out.i.clear();
  out.x.clear();
  const auto off_lo = [&](int j) { return lower ? cp[j] + 1 : cp[j]; };
  const auto off_hi = [&](int j) { return lower ? cp[j + 1] : cp[j + 1] - 1; };
  const auto diag   = [&](int j) { return lower ? cx[cp[j]] : cx[cp[j + 1] - 1]; };
  for (int c = 0; c < m; ++c) {
    const int k = by_out[c];
    // Reach of k in reverse topological order (iterative DFS); order[top..m) is topological.
    int top = m, head = 0;
    w.stack[0] = k;
    w.next[0]  = off_lo(k);
    w.mark[k]  = 1;
    while (head >= 0) {
      const int j = w.stack[head];
      int p       = w.next[head];
      const int e = off_hi(j);
      while (p < e && w.mark[ri[p]]) { ++p; }
      if (p < e) {
        w.next[head] = p + 1;
        const int r  = ri[p];
        w.mark[r]    = 1;
        ++head;
        w.stack[head] = r;
        w.next[head]  = off_lo(r);
      } else {
        w.order[--top] = j;
        --head;
      }
    }
    w.y[k] = 1.0;
    for (int t = top; t < m; ++t) {
      const int j = w.order[t];
      const double v = w.y[j] / diag(j);
      w.y[j] = v;
      if (v != 0.0) {
        for (int p = off_lo(j); p < off_hi(j); ++p) { w.y[ri[p]] -= cx[p] * v; }
      }
    }
    for (int t = top; t < m; ++t) {
      const int j = w.order[t];
      if (w.y[j] != 0.0) {
        out.i.push_back(j);
        out.x.push_back(w.y[j]);
      }
      w.y[j] = 0.0, w.mark[j] = 0;
    }
    out.p[c + 1] = static_cast<int>(out.i.size());
  }
}

// CSC <-> CSR of an m x m sparse matrix.
inline void transpose(int m, const sparse_t& a, sparse_t& t)
{
  t.p.assign(m + 1, 0);
  for (int r : a.i) { ++t.p[r + 1]; }
  for (int k = 0; k < m; ++k) { t.p[k + 1] += t.p[k]; }
  t.i.resize(a.i.size());
  t.x.resize(a.i.size());
  std::vector<int> pos(t.p.begin(), t.p.end() - 1);
  for (int c = 0; c < m; ++c) {
    for (int q = a.p[c]; q < a.p[c + 1]; ++q) {
      const int d = pos[a.i[q]]++;
      t.i[d]      = c;
      t.x[d]      = a.x[q];
    }
  }
}

// perm: P b = b[perm] (the factorization's row permutation).
inline void build(int m,
                  const int* Lp, const int* Li, const double* Lx,
                  const int* Up, const int* Ui, const double* Ux,
                  const int* perm,
                  workspace_t& w,
                  inverses_t& inv)
{
  inverse_columns(m, Lp, Li, Lx, true, perm, w, inv.Lc);
  transpose(m, inv.Lc, inv.Lr);
  inverse_columns(m, Up, Ui, Ux, false, nullptr, w, inv.Uc);
  transpose(m, inv.Uc, inv.Ur);
}

}  // namespace cuopt::linear_programming::dual_simplex::explicit_inverse
