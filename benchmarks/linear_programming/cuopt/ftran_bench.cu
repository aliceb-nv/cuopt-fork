/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Batched FTRAN micro-benchmark (test_netlib --ftran-bench <pool snapshot>).
// Base factors only: x = U0^{-1} L0^{-1} P b for two right-hand sides per seat (an entering column
// of A and rho = B^{-T} e_r), dense route, no product-form updates. Each stored node's host LU is
// replicated into `replicas` separate allocations so the working set matches a full GPU; every
// kernel variant is checked against the host b_solve and timed against a DRAM-bandwidth roofline.

#include "ftran_bench.hpp"

#include <branch_and_bound/gpu_node_pool.hpp>
#include <branch_and_bound/pool_snapshot.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/gpu_phase2/explicit_inverse.hpp>
#include <math_optimization/tic_toc.hpp>

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

namespace ds  = cuopt::mathematical_optimization::simplex;
namespace mip = cuopt::mathematical_optimization::mip;

namespace {

#define FB_CHECK(x)                                                                         \
  do {                                                                                      \
    cudaError_t e_ = (x);                                                                   \
    if (e_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "ftran-bench: %s failed: %s\n", #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

// One triangular factor, row-wise without the diagonal, rows ordered by level.
struct tri_dev {
  const int* rp;     // m+1, row pointers of the off-diagonal entries (original row order)
  const int* ci;     // column indices
  const double* cx;  // values
  const double* d;   // diagonal, m
  const int* order;  // rows sorted by level, m
  const int* lptr;   // level offsets into order, nlev+1
  int nlev;
};

// Level-ordered layout: position q holds row row[q]; entries of q are [rp[q], rp[q+1]). Within a
// level, rows with more than kShort entries come last, starting at lsplit[l].
constexpr int kShort = 8;
constexpr int kStageRows    = 384;
constexpr int kStageEntries = 512;
struct tri2_dev {
  const int* rp;
  const int* ci;
  const double* cx;
  const double* d;
  const int* row;
  const int* lptr;
  const int* lsplit;
  int nlev;
};

struct seat_dev {
  tri_dev L, U;
  tri2_dev L2, U2;
  tri2_dev L3, U3;  // no-op rows dropped
  const int4* chunks;  // v6: {factor (0 L3, 1 U3), q0, q1, unused}
  int nchunks;
  // v8: explicit inverses, row-wise. Linv's columns already index b (P folded in).
  const int *Lirp, *Lici, *Uirp, *Uici;
  const double *Licx, *Uicx;
  double *z0, *z1;  // intermediate L^-1 P b
  // v9: column-wise inverses (CSC) and sparse right-hand sides.
  const int *Lccp, *Lcri, *Uccp, *Ucri;
  const double *Lccx, *Uccx;
  const int *bi0, *bi1;
  const double *bx0, *bx1;
  int bn0, bn1;
  const int* perm;   // x[k] = b[perm[k]]
  const double* b0;  // m
  const double* b1;
  double* x0;        // m, output
  double* x1;
};

struct tri_host {
  std::vector<int> rp, ci, order, lptr;
  std::vector<double> cx, d;
};

struct tri2_host {
  std::vector<int> rp, ci, row, lptr, lsplit;
  std::vector<double> cx, d;
};

tri2_host level_ordered(const tri_host& t, bool drop_noop = false)
{
  tri2_host o;
  const int m    = static_cast<int>(t.d.size());
  const int nlev = static_cast<int>(t.lptr.size()) - 1;
  o.lptr         = t.lptr;
  o.lsplit.resize(nlev);
  o.rp.push_back(0);
  for (int l = 0; l < nlev; ++l) {
    std::vector<int> rows;
    for (int q = t.lptr[l]; q < t.lptr[l + 1]; ++q) {
      const int i = t.order[q];
      if (drop_noop && t.rp[i + 1] == t.rp[i] && t.d[i] == 1.0) { continue; }
      rows.push_back(i);
    }
    const int base = static_cast<int>(o.row.size());
    o.lptr[l]      = base;
    std::stable_sort(rows.begin(), rows.end(), [&](int a, int b) {
      return (t.rp[a + 1] - t.rp[a] > kShort) < (t.rp[b + 1] - t.rp[b] > kShort);
    });
    const int end = base + static_cast<int>(rows.size());
    o.lsplit[l]   = end;
    for (size_t k = 0; k < rows.size(); ++k) {
      const int i = rows[k];
      if (t.rp[i + 1] - t.rp[i] > kShort && o.lsplit[l] == end) {
        o.lsplit[l] = base + static_cast<int>(k);
      }
      o.row.push_back(i);
      o.d.push_back(t.d[i]);
      for (int p = t.rp[i]; p < t.rp[i + 1]; ++p) {
        o.ci.push_back(t.ci[p]);
        o.cx.push_back(t.cx[p]);
      }
      o.rp.push_back(static_cast<int>(o.ci.size()));
    }
  }
  o.lptr[nlev] = static_cast<int>(o.row.size());
  (void)m;
  return o;
}

// CSC (diag first for L, diag last for U) -> off-diagonal CSR + diagonal + level schedule.
tri_host to_rows(const cuopt::mathematical_optimization::csc_matrix_t<int, double>& A, bool lower)
{
  const int m = A.n;
  tri_host t;
  t.d.assign(m, 0.0);
  std::vector<int> cnt(m + 1, 0);
  for (int j = 0; j < m; ++j) {
    for (int p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
      const int i = A.i[p];
      if (i == j) {
        t.d[j] = A.x[p];
      } else {
        ++cnt[i + 1];
      }
    }
  }
  for (int i = 0; i < m; ++i) { cnt[i + 1] += cnt[i]; }
  t.rp = cnt;
  t.ci.resize(cnt[m]);
  t.cx.resize(cnt[m]);
  std::vector<int> pos(cnt.begin(), cnt.end() - 1);
  for (int j = 0; j < m; ++j) {
    for (int p = A.col_start[j]; p < A.col_start[j + 1]; ++p) {
      const int i = A.i[p];
      if (i == j) { continue; }
      t.ci[pos[i]] = j;
      t.cx[pos[i]] = A.x[p];
      ++pos[i];
    }
  }
  std::vector<int> lev(m, 0);
  int nlev = 0;
  for (int s = 0; s < m; ++s) {
    const int i = lower ? s : m - 1 - s;
    int l       = 0;
    for (int p = t.rp[i]; p < t.rp[i + 1]; ++p) { l = std::max(l, lev[t.ci[p]] + 1); }
    lev[i] = l;
    nlev   = std::max(nlev, l + 1);
  }
  t.lptr.assign(nlev + 1, 0);
  for (int i = 0; i < m; ++i) { ++t.lptr[lev[i] + 1]; }
  for (int l = 0; l < nlev; ++l) { t.lptr[l + 1] += t.lptr[l]; }
  t.order.resize(m);
  std::vector<int> cur(t.lptr.begin(), t.lptr.end() - 1);
  for (int i = 0; i < m; ++i) { t.order[cur[lev[i]]++] = i; }
  return t;
}

// Explicit inverse of a triangular CSC factor (L: diag first; U: diag last), returned row-wise
// (CSR) for gather SpMV. Column k solves T y = e_k touching only the reachable part.
struct csr_host {
  std::vector<int> rp, ci;
  std::vector<double> cx;
};
csr_host inverse_rows(const cuopt::mathematical_optimization::csc_matrix_t<int, double>& T, bool lower,
                      const std::vector<int>* col_map)
{
  const int m = T.n;
  std::vector<double> y(m, 0.0);
  std::vector<int> touched;
  std::vector<char> mark(m, 0);
  std::vector<std::vector<std::pair<int, double>>> rows(m);
  for (int k = 0; k < m; ++k) {
    touched.clear();
    y[k] = 1.0, mark[k] = 1, touched.push_back(k);
    if (lower) {
      for (int j = k; j < m; ++j) {
        if (y[j] == 0.0) { continue; }
        const int c0 = T.col_start[j], c1 = T.col_start[j + 1];
        y[j] /= T.x[c0];
        for (int p = c0 + 1; p < c1; ++p) {
          const int i = T.i[p];
          if (!mark[i]) { mark[i] = 1, touched.push_back(i); }
          y[i] -= T.x[p] * y[j];
        }
      }
    } else {
      for (int j = k; j >= 0; --j) {
        if (y[j] == 0.0) { continue; }
        const int c0 = T.col_start[j], c1 = T.col_start[j + 1] - 1;
        y[j] /= T.x[c1];
        for (int p = c0; p < c1; ++p) {
          const int i = T.i[p];
          if (!mark[i]) { mark[i] = 1, touched.push_back(i); }
          y[i] -= T.x[p] * y[j];
        }
      }
    }
    const int col = col_map != nullptr ? (*col_map)[k] : k;
    for (int i : touched) {
      if (y[i] != 0.0) { rows[i].push_back({col, y[i]}); }
      y[i] = 0.0, mark[i] = 0;
    }
  }
  csr_host r;
  r.rp.push_back(0);
  for (int i = 0; i < m; ++i) {
    for (const auto& [c, v] : rows[i]) { r.ci.push_back(c), r.cx.push_back(v); }
    r.rp.push_back(static_cast<int>(r.ci.size()));
  }
  return r;
}

// ---- kernels ----------------------------------------------------------------------------------

__device__ __forceinline__ void level_rows(const tri_dev& T, int l, int tid, int nthreads,
                                           double* x0, double* x1)
{
  for (int q = T.lptr[l] + tid; q < T.lptr[l + 1]; q += nthreads) {
    const int i = T.order[q];
    double s0 = 0.0, s1 = 0.0;
    for (int p = T.rp[i]; p < T.rp[i + 1]; ++p) {
      const int j    = T.ci[p];
      const double a = T.cx[p];
      s0 += a * x0[j];
      s1 += a * x1[j];
    }
    const double d = T.d[i];
    x0[i]          = (x0[i] - s0) / d;
    x1[i]          = (x1[i] - s1) / d;
  }
}

// Baseline shape: one warp per seat, vectors in global memory (as the production kernel).
__global__ void __launch_bounds__(32) ftran_warp(const seat_dev* seats, int nseats, int m)
{
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  for (int k = threadIdx.x; k < m; k += 32) {
    S.x0[k] = S.b0[S.perm[k]];
    S.x1[k] = S.b1[S.perm[k]];
  }
  __syncwarp();
  for (int l = 0; l < S.L.nlev; ++l) {
    level_rows(S.L, l, threadIdx.x, 32, S.x0, S.x1);
    __syncwarp();
  }
  for (int l = 0; l < S.U.nlev; ++l) {
    level_rows(S.U, l, threadIdx.x, 32, S.x0, S.x1);
    __syncwarp();
  }
}

// W warps per seat; vectors in shared memory (Smem) or global memory.
template <bool Smem>
__global__ void ftran_block(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = Smem ? sh : S.x0;
  double* x1       = Smem ? sh + m : S.x1;
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  __syncthreads();
  for (int l = 0; l < S.L.nlev; ++l) {
    level_rows(S.L, l, threadIdx.x, blockDim.x, x0, x1);
    __syncthreads();
  }
  for (int l = 0; l < S.U.nlev; ++l) {
    level_rows(S.U, l, threadIdx.x, blockDim.x, x0, x1);
    __syncthreads();
  }
  if constexpr (Smem) {
    for (int k = threadIdx.x; k < m; k += blockDim.x) {
      S.x0[k] = x0[k];
      S.x1[k] = x1[k];
    }
  }
}

__device__ __forceinline__ void level2(const tri2_dev& T, int l, double* x0, double* x1)
{
  const int tid = threadIdx.x, nth = blockDim.x;
  const int b = T.lptr[l], sp = T.lsplit[l], e = T.lptr[l + 1];
  // Short rows: one thread each; all entry loads issued before the gathers.
  for (int q = b + tid; q < sp; q += nth) {
    const int p0 = T.rp[q], len = T.rp[q + 1] - p0;
    int c[kShort];
    double a[kShort];
#pragma unroll
    for (int k = 0; k < kShort; ++k) {
      c[k] = k < len ? T.ci[p0 + k] : 0;
      a[k] = k < len ? T.cx[p0 + k] : 0.0;
    }
    double s0 = 0.0, s1 = 0.0;
#pragma unroll
    for (int k = 0; k < kShort; ++k) {
      if (k < len) {
        s0 += a[k] * x0[c[k]];
        s1 += a[k] * x1[c[k]];
      }
    }
    const int i = T.row[q];
    const double d = T.d[q];
    x0[i] = (x0[i] - s0) / d;
    x1[i] = (x1[i] - s1) / d;
  }
  // Long rows: one warp each.
  const int lane = tid & 31, warp = tid >> 5, nw = nth >> 5;
  for (int q = sp + warp; q < e; q += nw) {
    const int p0 = T.rp[q], p1 = T.rp[q + 1];
    double s0 = 0.0, s1 = 0.0;
    for (int p = p0 + lane; p < p1; p += 32) {
      const int j    = T.ci[p];
      const double a = T.cx[p];
      s0 += a * x0[j];
      s1 += a * x1[j];
    }
    for (int off = 16; off > 0; off >>= 1) {
      s0 += __shfl_xor_sync(0xffffffffu, s0, off);
      s1 += __shfl_xor_sync(0xffffffffu, s1, off);
    }
    if (lane == 0) {
      const int i    = T.row[q];
      const double d = T.d[q];
      x0[i]          = (x0[i] - s0) / d;
      x1[i]          = (x1[i] - s1) / d;
    }
  }
}

// L2 prefetch of level l's contiguous row and entry ranges.
__device__ __forceinline__ void prefetch_level(const tri2_dev& T, int l)
{
  if (l >= T.nlev) { return; }
  const int q0 = T.lptr[l], q1 = T.lptr[l + 1];
  const int p0 = T.rp[q0], p1 = T.rp[q1];
  const auto pf = [](const void* b, const void* e) {
    for (const char* c = static_cast<const char*>(b) + 128 * threadIdx.x; c < e; c += 128 * blockDim.x) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(c));
    }
  };
  pf(T.ci + p0, T.ci + p1);
  pf(T.cx + p0, T.cx + p1);
  pf(T.rp + q0, T.rp + q1 + 1);
  pf(T.row + q0, T.row + q1);
  pf(T.d + q0, T.d + q1);
}

template <bool Smem, int Ahead>
__global__ void ftran_block4(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = Smem ? sh : S.x0;
  double* x1       = Smem ? sh + m : S.x1;
  const int nL = S.L2.nlev, total = nL + S.U2.nlev;
  const auto lev = [&](int g, const tri2_dev*& T) {
    T = g < nL ? &S.L2 : &S.U2;
    return g < nL ? g : g - nL;
  };
  for (int g = 0; g < Ahead && g < total; ++g) {
    const tri2_dev* T;
    const int l = lev(g, T);
    prefetch_level(*T, l);
  }
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  __syncthreads();
  for (int g = 0; g < total; ++g) {
    if (g + Ahead < total) {
      const tri2_dev* T;
      const int l = lev(g + Ahead, T);
      prefetch_level(*T, l);
    }
    const tri2_dev* T;
    const int l = lev(g, T);
    level2(*T, l, x0, x1);
    __syncthreads();
  }
  if constexpr (Smem) {
    for (int k = threadIdx.x; k < m; k += blockDim.x) {
      S.x0[k] = x0[k];
      S.x1[k] = x1[k];
    }
  }
}

// v2 with clock64 at every level barrier: lvl_cyc[s * maxlev + g] = cycles of level g.
__global__ void ftran_block2_timed(const seat_dev* seats, int nseats, int m, long long* lvl_cyc,
                                   int maxlev)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = sh;
  double* x1       = sh + m;
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  __syncthreads();
  long long t = clock64();
  int g       = 0;
  for (int l = 0; l < S.L2.nlev; ++l, ++g) {
    level2(S.L2, l, x0, x1);
    __syncthreads();
    if (threadIdx.x == 0 && g < maxlev) {
      const long long t1 = clock64();
      lvl_cyc[size_t(s) * maxlev + g] = t1 - t;
      t = t1;
    }
  }
  for (int l = 0; l < S.U2.nlev; ++l, ++g) {
    level2(S.U2, l, x0, x1);
    __syncthreads();
    if (threadIdx.x == 0 && g < maxlev) {
      const long long t1 = clock64();
      lvl_cyc[size_t(s) * maxlev + g] = t1 - t;
      t = t1;
    }
  }
}

// v2 + no-op rows dropped (L3/U3) + level bounds in shared memory + each thread's next-level short
// row header (row pointers, row, diagonal) loaded before the barrier.
constexpr int kMaxLev = 128;
template <int Threads>
__global__ void __launch_bounds__(Threads)
  ftran_v7(const seat_dev* seats, int nseats, int m, long long* lvl_cyc = nullptr)
{
  extern __shared__ double sh[];
  __shared__ int lb[2][3][kMaxLev + 1];  // [factor][lptr, lsplit, -][level]
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = sh;
  double* x1       = sh + m;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = Threads >> 5;
  const int nL = S.L3.nlev, nU = S.U3.nlev, total = nL + nU;
  for (int k = tid; k <= nL; k += Threads) {
    lb[0][0][k] = S.L3.lptr[k];
    if (k < nL) { lb[0][1][k] = S.L3.lsplit[k]; }
  }
  for (int k = tid; k <= nU; k += Threads) {
    lb[1][0][k] = S.U3.lptr[k];
    if (k < nU) { lb[1][1][k] = S.U3.lsplit[k]; }
  }
  for (int k = tid; k < m; k += Threads) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  __syncthreads();
  int h0 = 0, h1 = 0, hrow = 0;
  double hd = 1.0;
  bool have = false;
  auto header = [&](int g) {
    have = false;
    if (g >= total) { return; }
    const int f = g < nL ? 0 : 1, l = g < nL ? g : g - nL;
    const tri2_dev& T = f == 0 ? S.L3 : S.U3;
    const int q = lb[f][0][l] + tid;
    if (q < lb[f][1][l]) {
      h0 = T.rp[q], h1 = T.rp[q + 1], hrow = T.row[q], hd = T.d[q];
      have = true;
    }
  };
  header(0);
  long long tc = clock64();
  if (lvl_cyc != nullptr && tid == 0) { lvl_cyc[size_t(s) * kMaxLev * 2] = 0; }
  for (int g = 0; g < total; ++g) {
    const int f = g < nL ? 0 : 1, l = g < nL ? g : g - nL;
    const tri2_dev& T = f == 0 ? S.L3 : S.U3;
    const int c0 = h0, c1 = h1, crow = hrow;
    const double cd = hd;
    const bool chave = have;
    header(g + 1);
    if (chave) {
      const int len = c1 - c0;
      int c[kShort];
      double a[kShort];
#pragma unroll
      for (int k = 0; k < kShort; ++k) {
        c[k] = k < len ? T.ci[c0 + k] : 0;
        a[k] = k < len ? T.cx[c0 + k] : 0.0;
      }
      double s0 = 0.0, s1 = 0.0;
#pragma unroll
      for (int k = 0; k < kShort; ++k) {
        if (k < len) {
          s0 += a[k] * x0[c[k]];
          s1 += a[k] * x1[c[k]];
        }
      }
      x0[crow] = (x0[crow] - s0) / cd;
      x1[crow] = (x1[crow] - s1) / cd;
    }
    const int b = lb[f][0][l], sp = lb[f][1][l], e = lb[f][0][l + 1];
    for (int q = b + Threads + tid; q < sp; q += Threads) {
      const int p0 = T.rp[q], len = T.rp[q + 1] - p0;
      double s0 = 0.0, s1 = 0.0;
      for (int k = 0; k < len; ++k) {
        const int j    = T.ci[p0 + k];
        const double v = T.cx[p0 + k];
        s0 += v * x0[j];
        s1 += v * x1[j];
      }
      const int i = T.row[q];
      x0[i]       = (x0[i] - s0) / T.d[q];
      x1[i]       = (x1[i] - s1) / T.d[q];
    }
    for (int q = sp + warp; q < e; q += nw) {
      const int p0 = T.rp[q], p1 = T.rp[q + 1];
      double s0 = 0.0, s1 = 0.0;
      for (int p = p0 + lane; p < p1; p += 32) {
        const int j    = T.ci[p];
        const double v = T.cx[p];
        s0 += v * x0[j];
        s1 += v * x1[j];
      }
      for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      if (lane == 0) {
        const int i = T.row[q];
        x0[i]       = (x0[i] - s0) / T.d[q];
        x1[i]       = (x1[i] - s1) / T.d[q];
      }
    }
    __syncthreads();
    if (lvl_cyc != nullptr && tid == 0) {
      const long long t1 = clock64();
      lvl_cyc[size_t(s) * kMaxLev * 2 + 1 + g] = t1 - tc;
      tc = t1;
    }
  }
  for (int k = tid; k < m; k += Threads) {
    S.x0[k] = x0[k];
    S.x1[k] = x1[k];
  }
}

template <bool Smem>
__global__ void ftran_block2(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = Smem ? sh : S.x0;
  double* x1       = Smem ? sh + m : S.x1;
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  __syncthreads();
  for (int l = 0; l < S.L2.nlev; ++l) {
    level2(S.L2, l, x0, x1);
    __syncthreads();
  }
  for (int l = 0; l < S.U2.nlev; ++l) {
    level2(S.U2, l, x0, x1);
    __syncthreads();
  }
  if constexpr (Smem) {
    for (int k = threadIdx.x; k < m; k += blockDim.x) {
      S.x0[k] = x0[k];
      S.x1[k] = x1[k];
    }
  }
}

struct short_row_t {
  int i, len;
  int c[kShort];
  double a[kShort], d;
};

__device__ __forceinline__ void load_short(const tri2_dev& T, int q, short_row_t& r)
{
  const int p0 = T.rp[q];
  r.len        = T.rp[q + 1] - p0;
  r.i          = T.row[q];
  r.d          = T.d[q];
#pragma unroll
  for (int k = 0; k < kShort; ++k) {
    r.c[k] = k < r.len ? T.ci[p0 + k] : 0;
    r.a[k] = k < r.len ? T.cx[p0 + k] : 0.0;
  }
}

__device__ __forceinline__ void apply_short(const short_row_t& r, double* x0, double* x1)
{
  double s0 = 0.0, s1 = 0.0;
#pragma unroll
  for (int k = 0; k < kShort; ++k) {
    if (k < r.len) {
      s0 += r.a[k] * x0[r.c[k]];
      s1 += r.a[k] * x1[r.c[k]];
    }
  }
  x0[r.i] = (x0[r.i] - s0) / r.d;
  x1[r.i] = (x1[r.i] - s1) / r.d;
}

// v2 plus: no-op rows dropped on the host, and each thread's first short row of the NEXT level
// loaded into registers before the current level's barrier.
template <bool Smem, int Threads, int MinBlocks>
__global__ void __launch_bounds__(Threads, MinBlocks)
  ftran_block3(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = Smem ? sh : S.x0;
  double* x1       = Smem ? sh + m : S.x1;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = Threads >> 5;
  for (int k = tid; k < m; k += Threads) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  const tri2_dev* T = &S.L3;
  const int total   = S.L3.nlev + S.U3.nlev;
  short_row_t nxt;
  bool have = false;
  {
    const int q = T->lptr[0] + tid;
    if (q < T->lsplit[0]) { load_short(*T, q, nxt), have = true; }
  }
  __syncthreads();
  for (int g = 0; g < total; ++g) {
    const bool upper  = g >= S.L3.nlev;
    const int l       = upper ? g - S.L3.nlev : g;
    const tri2_dev& C = upper ? S.U3 : S.L3;
    const short_row_t cur = nxt;
    const bool had        = have;
    have                  = false;
    if (g + 1 < total) {
      const bool up2    = g + 1 >= S.L3.nlev;
      const int l2      = up2 ? g + 1 - S.L3.nlev : g + 1;
      const tri2_dev& N = up2 ? S.U3 : S.L3;
      const int q       = N.lptr[l2] + tid;
      if (q < N.lsplit[l2]) { load_short(N, q, nxt), have = true; }
    }
    if (had) { apply_short(cur, x0, x1); }
    for (int q = C.lptr[l] + Threads + tid; q < C.lsplit[l]; q += Threads) {
      short_row_t r;
      load_short(C, q, r);
      apply_short(r, x0, x1);
    }
    for (int q = C.lsplit[l] + warp; q < C.lptr[l + 1]; q += nw) {
      const int p0 = C.rp[q], p1 = C.rp[q + 1];
      double s0 = 0.0, s1 = 0.0;
      for (int p = p0 + lane; p < p1; p += 32) {
        const int j    = C.ci[p];
        const double a = C.cx[p];
        s0 += a * x0[j];
        s1 += a * x1[j];
      }
      for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      if (lane == 0) {
        const int i = C.row[q];
        x0[i]       = (x0[i] - s0) / C.d[q];
        x1[i]       = (x1[i] - s1) / C.d[q];
      }
    }
    __syncthreads();
  }
  (void)T;
  if constexpr (Smem) {
    for (int k = tid; k < m; k += Threads) {
      S.x0[k] = x0[k];
      S.x1[k] = x1[k];
    }
  }
}

// Sync-free solve over one factor: rows in level (topological) order, one thread per row, each
// entry waits on its column's ready flag in shared memory; no barriers between levels.
__device__ __forceinline__ void syncfree_tri(const tri2_dev& T, double* x0, double* x1,
                                             volatile unsigned char* flag)
{
  const int nrows = T.lptr[T.nlev];
  for (int q = threadIdx.x; q < nrows; q += blockDim.x) {
    const int p0 = T.rp[q], p1 = T.rp[q + 1];
    double s0 = 0.0, s1 = 0.0;
    for (int p = p0; p < p1; ++p) {
      const int j    = T.ci[p];
      const double a = T.cx[p];
      while (flag[j] == 0) {}
      s0 += a * x0[j];
      s1 += a * x1[j];
    }
    const int i = T.row[q];
    const double d = T.d[q];
    x0[i] = (x0[i] - s0) / d;
    x1[i] = (x1[i] - s1) / d;
    __threadfence_block();
    flag[i] = 1;
  }
}

__global__ void ftran_syncfree(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0 = sh;
  double* x1 = sh + m;
  volatile unsigned char* flag = reinterpret_cast<unsigned char*>(sh + 2 * m);
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    x0[k]   = S.b0[S.perm[k]];
    x1[k]   = S.b1[S.perm[k]];
    flag[k] = 0;
  }
  __syncthreads();
  syncfree_tri(S.L2, x0, x1, flag);
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) { flag[k] = 0; }
  __syncthreads();
  syncfree_tri(S.U2, x0, x1, flag);
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    S.x0[k] = x0[k];
    S.x1[k] = x1[k];
  }
}

struct stage_t {
  int row[kStageRows];
  int rp[kStageRows + 1];
  int ci[kStageEntries];
  double d[kStageRows];
  double cx[kStageEntries];
};

__device__ __forceinline__ void cp4(void* dst, const void* src)
{
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(
                 static_cast<unsigned>(__cvta_generic_to_shared(dst))),
               "l"(src));
}
__device__ __forceinline__ void cp8(void* dst, const void* src)
{
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(
                 static_cast<unsigned>(__cvta_generic_to_shared(dst))),
               "l"(src));
}

__device__ __forceinline__ void stage_load(const seat_dev& S, int c, stage_t& st)
{
  const int4 ch     = S.chunks[c];
  const tri2_dev& T = ch.x == 0 ? S.L3 : S.U3;
  const int q0 = ch.y, nr = ch.z - ch.y;
  const int p0 = T.rp[q0], ne = T.rp[ch.z] - p0;
  for (int k = threadIdx.x; k < nr; k += blockDim.x) {
    cp4(&st.row[k], T.row + q0 + k);
    cp8(&st.d[k], T.d + q0 + k);
  }
  for (int k = threadIdx.x; k <= nr; k += blockDim.x) { cp4(&st.rp[k], T.rp + q0 + k); }
  for (int k = threadIdx.x; k < ne; k += blockDim.x) {
    cp4(&st.ci[k], T.ci + p0 + k);
    cp8(&st.cx[k], T.cx + p0 + k);
  }
  asm volatile("cp.async.commit_group;");
}

// Factor data streamed chunk by chunk into a double-buffered shared stage (cp.async), so the
// solve itself touches only shared memory.
__global__ void ftran_stream(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev S = seats[s];
  double* x0       = sh;
  double* x1       = sh + m;
  stage_t* stg     = reinterpret_cast<stage_t*>(sh + 2 * m);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  if (S.nchunks > 0) { stage_load(S, 0, stg[0]); }
  for (int k = tid; k < m; k += blockDim.x) {
    x0[k] = S.b0[S.perm[k]];
    x1[k] = S.b1[S.perm[k]];
  }
  for (int c = 0; c < S.nchunks; ++c) {
    if (c + 1 < S.nchunks) {
      stage_load(S, c + 1, stg[(c + 1) & 1]);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
    const stage_t& st = stg[c & 1];
    const int4 ch     = S.chunks[c];
    const int nr = ch.z - ch.y, base = st.rp[0];
    for (int k = tid; k < nr; k += blockDim.x) {
      const int a0 = st.rp[k] - base, a1 = st.rp[k + 1] - base;
      if (a1 - a0 > kShort) { continue; }
      double s0 = 0.0, s1 = 0.0;
      for (int p = a0; p < a1; ++p) {
        const int j = st.ci[p];
        s0 += st.cx[p] * x0[j];
        s1 += st.cx[p] * x1[j];
      }
      const int i = st.row[k];
      x0[i]       = (x0[i] - s0) / st.d[k];
      x1[i]       = (x1[i] - s1) / st.d[k];
    }
    for (int k = warp; k < nr; k += nw) {
      const int a0 = st.rp[k] - base, a1 = st.rp[k + 1] - base;
      if (a1 - a0 <= kShort) { continue; }
      double s0 = 0.0, s1 = 0.0;
      for (int p = a0 + lane; p < a1; p += 32) {
        const int j = st.ci[p];
        s0 += st.cx[p] * x0[j];
        s1 += st.cx[p] * x1[j];
      }
      for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      if (lane == 0) {
        const int i = st.row[k];
        x0[i]       = (x0[i] - s0) / st.d[k];
        x1[i]       = (x1[i] - s1) / st.d[k];
      }
    }
    __syncthreads();
  }
  for (int k = tid; k < m; k += blockDim.x) {
    S.x0[k] = x0[k];
    S.x1[k] = x1[k];
  }
}

// v8: out = A * in for every seat, G lanes per row (G divides 32), rows of all seats flattened.
template <int G, bool Lower>
__global__ void spmv_all(const seat_dev* seats, int nseats, int m)
{
  const long long tidg = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long item = tidg / G;
  const int sub        = static_cast<int>(tidg % G);
  if (item >= static_cast<long long>(nseats) * m) { return; }
  const int s = static_cast<int>(item / m), i = static_cast<int>(item % m);
  const seat_dev& S = seats[s];
  const int* rp     = Lower ? S.Lirp : S.Uirp;
  const int* ci     = Lower ? S.Lici : S.Uici;
  const double* cx  = Lower ? S.Licx : S.Uicx;
  const double* in0 = Lower ? S.b0 : S.z0;
  const double* in1 = Lower ? S.b1 : S.z1;
  double* out0      = Lower ? S.z0 : S.x0;
  double* out1      = Lower ? S.z1 : S.x1;
  double s0 = 0.0, s1 = 0.0;
  for (int p = rp[i] + sub; p < rp[i + 1]; p += G) {
    const int j    = ci[p];
    const double a = cx[p];
    s0 += a * in0[j];
    s1 += a * in1[j];
  }
#pragma unroll
  for (int off = G / 2; off > 0; off >>= 1) {
    s0 += __shfl_down_sync(0xffffffffu, s0, off, G);
    s1 += __shfl_down_sync(0xffffffffu, s1, off, G);
  }
  if (sub == 0) {
    out0[i] = s0;
    out1[i] = s1;
  }
}

// v9: sparse right-hand sides through explicit inverses, column by column, accumulated in shared
// memory with atomics: z = L0^-1 P b (scatter columns of L0^-1 P), compact z, x = U0^-1 z.
constexpr int kZCap = 4096;
__device__ __forceinline__ void scatter_cols(const int* cp, const int* ri, const double* cx,
                                             const int* idx, const double* val, int n, double* acc,
                                             int lane, int warp, int nw)
{
  // One warp per input nonzero; lanes stride the column.
  for (int t = warp; t < n; t += nw) {
    const int c = idx[t];
    const double v = val[t];
    for (int p = cp[c] + lane; p < cp[c + 1]; p += 32) { atomicAdd(&acc[ri[p]], cx[p] * v); }
  }
}

template <int Threads>
__global__ void __launch_bounds__(Threads) ftran_inv_sparse(const seat_dev* seats, int nseats, int m)
{
  extern __shared__ double sh[];
  __shared__ int zcount[2];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev& S = seats[s];
  double* acc0      = sh;
  double* acc1      = sh + m;
  int* zi0          = reinterpret_cast<int*>(sh + 2 * m);
  int* zi1          = zi0 + kZCap;
  double* zx0       = reinterpret_cast<double*>(zi1 + kZCap);
  double* zx1       = zx0 + kZCap;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = Threads >> 5;
  for (int k = tid; k < m; k += Threads) { acc0[k] = 0.0, acc1[k] = 0.0; }
  if (tid < 2) { zcount[tid] = 0; }
  __syncthreads();
  scatter_cols(S.Lccp, S.Lcri, S.Lccx, S.bi0, S.bx0, S.bn0, acc0, lane, warp, nw);
  scatter_cols(S.Lccp, S.Lcri, S.Lccx, S.bi1, S.bx1, S.bn1, acc1, lane, warp, nw);
  __syncthreads();
  // Compact z (order irrelevant: the next stage sums with atomics) and clear the accumulators.
  for (int k = tid; k < m; k += Threads) {
    const double v0 = acc0[k], v1 = acc1[k];
    if (v0 != 0.0) {
      const int q = atomicAdd(&zcount[0], 1);
      if (q < kZCap) { zi0[q] = k, zx0[q] = v0; }
    }
    if (v1 != 0.0) {
      const int q = atomicAdd(&zcount[1], 1);
      if (q < kZCap) { zi1[q] = k, zx1[q] = v1; }
    }
    acc0[k] = 0.0, acc1[k] = 0.0;
  }
  __syncthreads();
  scatter_cols(S.Uccp, S.Ucri, S.Uccx, zi0, zx0, min(zcount[0], kZCap), acc0, lane, warp, nw);
  scatter_cols(S.Uccp, S.Ucri, S.Uccx, zi1, zx1, min(zcount[1], kZCap), acc1, lane, warp, nw);
  __syncthreads();
  for (int k = tid; k < m; k += Threads) {
    S.x0[k] = acc0[k];
    S.x1[k] = acc1[k];
  }
}

template <int G>
__device__ __forceinline__ void scatter_cols_g(const int* cp, const int* ri, const double* cx,
                                               const int* idx, const double* val, int n, double* acc,
                                               int tid, int nth)
{
  const int grp = tid / G, sub = tid % G, ngrp = nth / G;
  for (int t = grp; t < n; t += ngrp) {
    const int c    = idx[t];
    const double v = val[t];
    for (int p = cp[c] + sub; p < cp[c + 1]; p += G) { atomicAdd(&acc[ri[p]], cx[p] * v); }
  }
}

template <int Threads, int G, int ZCap>
__global__ void __launch_bounds__(Threads)
  ftran_inv_sparse2(const seat_dev* seats, int nseats, int m, long long* ph = nullptr)
{
  long long t = clock64();
  auto mark = [&](int k) {
    if (ph != nullptr && threadIdx.x == 0) {
      const long long t1 = clock64();
      ph[size_t(blockIdx.x) * 6 + k] = t1 - t;
      t = t1;
    }
  };
  extern __shared__ double sh[];
  __shared__ int zcount[2];
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev& S = seats[s];
  double* acc0      = sh;
  double* acc1      = sh + m;
  double* zx0       = sh + 2 * m;
  double* zx1       = zx0 + ZCap;
  int* zi0          = reinterpret_cast<int*>(zx1 + ZCap);
  int* zi1          = zi0 + ZCap;
  const int tid = threadIdx.x;
  for (int k = tid; k < m; k += Threads) { acc0[k] = 0.0, acc1[k] = 0.0; }
  if (tid < 2) { zcount[tid] = 0; }
  __syncthreads();
  mark(0);
  scatter_cols_g<G>(S.Lccp, S.Lcri, S.Lccx, S.bi0, S.bx0, S.bn0, acc0, tid, Threads);
  scatter_cols_g<G>(S.Lccp, S.Lcri, S.Lccx, S.bi1, S.bx1, S.bn1, acc1, tid, Threads);
  __syncthreads();
  mark(1);
  for (int k = tid; k < m; k += Threads) {
    const double v0 = acc0[k], v1 = acc1[k];
    if (v0 != 0.0) {
      const int q = atomicAdd(&zcount[0], 1);
      if (q < ZCap) { zi0[q] = k, zx0[q] = v0; }
    }
    if (v1 != 0.0) {
      const int q = atomicAdd(&zcount[1], 1);
      if (q < ZCap) { zi1[q] = k, zx1[q] = v1; }
    }
    acc0[k] = 0.0, acc1[k] = 0.0;
  }
  __syncthreads();
  mark(2);
  if (zcount[0] > ZCap || zcount[1] > ZCap) { __trap(); }
  scatter_cols_g<G>(S.Uccp, S.Ucri, S.Uccx, zi0, zx0, zcount[0], acc0, tid, Threads);
  scatter_cols_g<G>(S.Uccp, S.Ucri, S.Uccx, zi1, zx1, zcount[1], acc1, tid, Threads);
  __syncthreads();
  mark(3);
  for (int k = tid; k < m; k += Threads) {
    S.x0[k] = acc0[k];
    S.x1[k] = acc1[k];
  }
  __syncthreads();
  mark(4);
}

// v11: entry-balanced scatter. Items are (column, value, which rhs); a block-wide exclusive scan of
// the column lengths gives each thread an even share of entries, located by binary search.
template <int Threads, int Cap>
__device__ __forceinline__ void balanced_scatter(const int* cp, const int* ri, const double* cx,
                                                 const int* col, const double* val,
                                                 const unsigned char* which, int n, int* pre,
                                                 double* acc0, double* acc1)
{
  using scan_t = cub::BlockScan<int, Threads>;
  __shared__ typename scan_t::TempStorage tmp;
  __shared__ int total;
  constexpr int kItems = (Cap + Threads - 1) / Threads;
  int len[kItems];
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int t = threadIdx.x * kItems + k;
    len[k]      = t < n ? cp[col[t] + 1] - cp[col[t]] : 0;
  }
  int agg = 0;
  scan_t(tmp).ExclusiveSum(len, len, agg);
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int t = threadIdx.x * kItems + k;
    if (t < n) { pre[t] = len[k]; }
  }
  if (threadIdx.x == 0) { total = agg; }
  __syncthreads();
  const int E = total;
  for (int e = threadIdx.x; e < E; e += Threads) {
    int lo = 0, hi = n - 1;  // last t with pre[t] <= e
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (pre[mid] <= e) { lo = mid; } else { hi = mid - 1; }
    }
    const int p = cp[col[lo]] + (e - pre[lo]);
    atomicAdd(&(which[lo] ? acc1 : acc0)[ri[p]], cx[p] * val[lo]);
  }
  __syncthreads();
}

template <int Threads, int Cap, int MinBlocks = 1>
__global__ void __launch_bounds__(Threads, MinBlocks)
  ftran_inv_bal(const seat_dev* seats, int nseats, int m, long long* ph = nullptr)
{
  extern __shared__ double sh[];
  __shared__ int zc;
  long long t = clock64();
  auto mark = [&](int k) {
    if (ph != nullptr && threadIdx.x == 0) {
      const long long t1 = clock64();
      ph[size_t(blockIdx.x) * 6 + k] = t1 - t;
      t = t1;
    }
  };
  const int s = blockIdx.x;
  if (s >= nseats) { return; }
  const seat_dev& S = seats[s];
  double* acc0      = sh;
  double* acc1      = sh + m;
  double* zx        = sh + 2 * m;
  int* zi           = reinterpret_cast<int*>(zx + Cap);
  int* pre          = zi + Cap;
  unsigned char* zw = reinterpret_cast<unsigned char*>(pre + Cap);
  const int tid = threadIdx.x;
  for (int k = tid; k < m; k += Threads) { acc0[k] = 0.0, acc1[k] = 0.0; }
  // Stage 1 items: b's nonzeros of both right-hand sides.
  const int nb = S.bn0 + S.bn1;
  if (nb > Cap) { __trap(); }
  for (int k = tid; k < nb; k += Threads) {
    const bool r1 = k >= S.bn0;
    zi[k] = r1 ? S.bi1[k - S.bn0] : S.bi0[k];
    zx[k] = r1 ? S.bx1[k - S.bn0] : S.bx0[k];
    zw[k] = r1 ? 1 : 0;
  }
  if (tid == 0) { zc = 0; }
  __syncthreads();
  mark(0);
  balanced_scatter<Threads, Cap>(S.Lccp, S.Lcri, S.Lccx, zi, zx, zw, nb, pre, acc0, acc1);
  mark(1);
  for (int k = tid; k < m; k += Threads) {
    const double v0 = acc0[k], v1 = acc1[k];
    if (v0 != 0.0) {
      const int q = atomicAdd(&zc, 1);
      if (q < Cap) { zi[q] = k, zx[q] = v0, zw[q] = 0; }
    }
    if (v1 != 0.0) {
      const int q = atomicAdd(&zc, 1);
      if (q < Cap) { zi[q] = k, zx[q] = v1, zw[q] = 1; }
    }
    acc0[k] = 0.0, acc1[k] = 0.0;
  }
  __syncthreads();
  if (zc > Cap) { __trap(); }
  mark(2);
  balanced_scatter<Threads, Cap>(S.Uccp, S.Ucri, S.Uccx, zi, zx, zw, zc, pre, acc0, acc1);
  mark(3);
  for (int k = tid; k < m; k += Threads) {
    S.x0[k] = acc0[k];
    S.x1[k] = acc1[k];
  }
  __syncthreads();
  mark(4);
}

__global__ void copy_kernel(const double4* __restrict__ a, double4* __restrict__ b, size_t n)
{
  for (size_t k = blockIdx.x * size_t(blockDim.x) + threadIdx.x; k < n;
       k += size_t(gridDim.x) * blockDim.x) {
    b[k] = a[k];
  }
}

// ---- host ---------------------------------------------------------------------------------------

// Seat data comes from one contiguous arena (FTRAN_BENCH_ARENA=0: one cudaMalloc per array).
struct arena_t {
  char* base{nullptr};
  size_t used{0}, cap{0};
} g_arena;

template <typename T>
T* upload(const std::vector<T>& v)
{
  T* p = nullptr;
  const size_t nb = std::max<size_t>(1, v.size()) * sizeof(T);
  if (g_arena.base != nullptr) {
    const size_t off = (g_arena.used + 255) & ~size_t(255);
    if (off + nb > g_arena.cap) {
      std::fprintf(stderr, "ftran-bench: arena exhausted\n");
      std::exit(1);
    }
    p            = reinterpret_cast<T*>(g_arena.base + off);
    g_arena.used = off + nb;
  } else {
    FB_CHECK(cudaMalloc(&p, nb));
  }
  if (!v.empty()) { FB_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  return p;
}

tri2_dev upload_tri2(const tri2_host& t)
{
  return tri2_dev{upload(t.rp), upload(t.ci), upload(t.cx), upload(t.d), upload(t.row),
                  upload(t.lptr), upload(t.lsplit), static_cast<int>(t.lptr.size()) - 1};
}

tri_dev upload_tri(const tri_host& t)
{
  return tri_dev{upload(t.rp), upload(t.ci), upload(t.cx), upload(t.d),
                 upload(t.order), upload(t.lptr), static_cast<int>(t.lptr.size()) - 1};
}

}  // namespace

int ftran_bench_main(const raft::handle_t& handle,
                     const std::string& file,
                     int replicas,
                     int repeats,
                     int prepare_threads)
{
  namespace ps = mip::pool_snapshot;
  ps::pool_t<int, double> snap(&handle);
  std::string why;
  if (!ps::read<int, double>(file, snap, why)) {
    std::fprintf(stderr, "ftran-bench: cannot load %s (%s)\n", file.c_str(), why.c_str());
    return 1;
  }
  const auto& lp = snap.lp;
  const int m = snap.h.m, n = snap.h.n;

  std::vector<mip::gpu_batch_node_t<int, double>> nodes;
  for (const auto& s : snap.seats) {
    if (s.rec.command == ps::PARKED) { continue; }
    nodes.emplace_back(m, n);
    auto& node = nodes.back();
    node.lower = lp.lower;
    node.upper = lp.upper;
    for (size_t q = 0; q < s.cols.size(); ++q) {
      node.lower[s.cols[q]] = s.lo[q];
      node.upper[s.cols[q]] = s.up[q];
    }
    node.vstatus.resize(n);
    for (int j = 0; j < n; ++j) { node.vstatus[j] = static_cast<ds::variable_status_t>(s.vstatus[j]); }
    node.dy_se = s.dy_se;
    ps::apply_settings<int, double>(s.rec, node.settings);
    node.settings.time_limit = std::numeric_limits<double>::infinity();
  }
  std::vector<mip::batch_seat_state_t<int, double>> prepared;
  const auto ok = mip::gpu_batch_factor_fresh<int, double>(lp, nodes, prepared, prepare_threads);

  // Per stored node: factors in row form, both right-hand sides, host reference solutions.
  struct node_data {
    tri_host L, U;
    tri2_host L2, U2, L3, U3;
    std::vector<int4> chunks;
    csr_host Linv, Uinv, LinvC, UinvC;  // C: column-wise (rp = column pointers, ci = rows)
    std::vector<int> bi0, bi1;
    std::vector<double> bx0, bx1;
    std::vector<int> perm;
    std::vector<double> b0, b1, ref0, ref1;
  };
  std::vector<node_data> nd;
  size_t nnzL = 0, nnzU = 0, levL = 0, levU = 0;
  for (size_t k = 0; k < nodes.size(); ++k) {
    if (!ok[k]) { continue; }
    const auto& ft = prepared[k].ft;
    node_data d;
    d.L    = to_rows(ft.L0(), true);
    d.U    = to_rows(ft.U0(), false);
    d.perm = ft.row_permutation();
    d.L2   = level_ordered(d.L);
    d.U2   = level_ordered(d.U);
    d.L3   = level_ordered(d.L, true);
    d.U3   = level_ordered(d.U, true);
    // Chunks never cross a level and fit the shared-memory stage (rows and entries).
    for (int f = 0; f < 2; ++f) {
      const tri2_host& t = f == 0 ? d.L3 : d.U3;
      for (size_t l = 0; l + 1 < t.lptr.size(); ++l) {
        int q = t.lptr[l];
        while (q < t.lptr[l + 1]) {
          int e = q;
          while (e < t.lptr[l + 1] && e - q < kStageRows && t.rp[e + 1] - t.rp[q] <= kStageEntries) { ++e; }
          d.chunks.push_back(make_int4(f, q, e, 0));
          q = e;
        }
      }
    }
    d.b0.assign(m, 0.0);
    // Entering column: a nonbasic structural column with entries.
    const auto& nb = prepared[k].nonbasic_list;
    for (size_t t = 0; t < nb.size(); ++t) {
      const int j = nb[(k * 7919 + t) % nb.size()];
      if (lp.A.col_start[j + 1] > lp.A.col_start[j]) {
        for (int p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) { d.b0[lp.A.i[p]] = lp.A.x[p]; }
        break;
      }
    }
    std::vector<double> er(m, 0.0);
    er[(k * 104729) % m] = 1.0;
    d.b1.assign(m, 0.0);
    ft.b_transpose_solve(er, d.b1);
    d.ref0.assign(m, 0.0);
    d.ref1.assign(m, 0.0);
    ft.b_solve(d.b0, d.ref0);
    ft.b_solve(d.b1, d.ref1);
    nnzL += d.L.ci.size(), nnzU += d.U.ci.size();
    levL += d.L.lptr.size() - 1, levU += d.U.lptr.size() - 1;
    nd.push_back(std::move(d));
  }
  const int U = static_cast<int>(nd.size());
  if (U == 0) { return 1; }
  {
    // Explicit inverses, in parallel over nodes. L0^-1 P: column k of L0^-1 multiplies (P b)[k] =
    // b[perm[k]], so its column index becomes perm[k].
    std::vector<int> ids;
    for (size_t k = 0; k < nodes.size(); ++k) {
      if (ok[k]) { ids.push_back(static_cast<int>(k)); }
    }
    const double t0 = cuopt::mathematical_optimization::tic();
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int w = 0; w < std::max(1, prepare_threads); ++w) {
      th.emplace_back([&] {
        for (int u = next++; u < U; u = next++) {
          const auto& ft = prepared[ids[u]].ft;
          nd[u].Linv     = inverse_rows(ft.L0(), true, &nd[u].perm);
          nd[u].Uinv     = inverse_rows(ft.U0(), false, nullptr);
          for (int f = 0; f < 2; ++f) {
            const csr_host& R = f == 0 ? nd[u].Linv : nd[u].Uinv;
            csr_host& C       = f == 0 ? nd[u].LinvC : nd[u].UinvC;
            C.rp.assign(m + 1, 0);
            for (int c : R.ci) { ++C.rp[c + 1]; }
            for (int c = 0; c < m; ++c) { C.rp[c + 1] += C.rp[c]; }
            C.ci.resize(R.ci.size());
            C.cx.resize(R.ci.size());
            std::vector<int> pos(C.rp.begin(), C.rp.end() - 1);
            for (int i = 0; i < m; ++i) {
              for (int q = R.rp[i]; q < R.rp[i + 1]; ++q) {
                C.ci[pos[R.ci[q]]] = i;
                C.cx[pos[R.ci[q]]++] = R.cx[q];
              }
            }
          }
          for (int j = 0; j < m; ++j) {
            if (nd[u].b0[j] != 0.0) { nd[u].bi0.push_back(j), nd[u].bx0.push_back(nd[u].b0[j]); }
            if (nd[u].b1[j] != 0.0) { nd[u].bi1.push_back(j), nd[u].bx1.push_back(nd[u].b1[j]); }
          }
        }
      });
    }
    for (auto& t : th) { t.join(); }
    {
      // Sparse builder (explicit_inverse.hpp) against the O(m^2) one above: same entries, time per LU.
      namespace ei = cuopt::linear_programming::dual_simplex::explicit_inverse;
      ei::workspace_t w;
      double t_build = 0, maxdiff = 0;
      size_t mism = 0;
      for (int u = 0; u < U; ++u) {
        const auto& ft = prepared[ids[u]].ft;
        const auto& L  = ft.L0();
        const auto& Uf = ft.U0();
        ei::inverses_t inv;
        const double tb = cuopt::mathematical_optimization::tic();
        ei::build(m, L.col_start.data(), L.i.data(), L.x.data(), Uf.col_start.data(), Uf.i.data(),
                  Uf.x.data(), nd[u].perm.data(), w, inv);
        t_build += cuopt::mathematical_optimization::toc(tb);
        for (int f = 0; f < 2; ++f) {
          const auto& R  = f == 0 ? nd[u].Linv : nd[u].Uinv;
          const auto& N  = f == 0 ? inv.Lr : inv.Ur;
          if (N.i.size() != R.ci.size()) { ++mism; continue; }
          for (int i = 0; i < m; ++i) {
            std::map<int, double> a;
            for (int q = R.rp[i]; q < R.rp[i + 1]; ++q) { a[R.ci[q]] = R.cx[q]; }
            for (int q = N.p[i]; q < N.p[i + 1]; ++q) {
              auto it = a.find(N.i[q]);
              if (it == a.end()) { ++mism; continue; }
              maxdiff = std::max(maxdiff, std::abs(it->second - N.x[q]) / (1.0 + std::abs(it->second)));
            }
          }
        }
      }
      std::printf("sparse inverse builder: %.3f ms per LU (single thread), entry mismatches %zu, max rel diff %.1e\n",
                  1e3 * t_build / U, mism, maxdiff);
    }
    double li = 0, ui = 0;
    size_t lmax = 0, umax = 0, l0max = 0, u0max = 0;
    for (size_t k = 0; k < nd.size(); ++k) {
      const auto& d = nd[k];
      li += d.Linv.ci.size(), ui += d.Uinv.ci.size();
      lmax = std::max(lmax, d.Linv.ci.size()), umax = std::max(umax, d.Uinv.ci.size());
      l0max = std::max(l0max, d.L.ci.size() + m), u0max = std::max(u0max, d.U.ci.size() + m);
    }
    std::printf("explicit inverses for %d nodes in %.2fs: L0^-1 nnz %.0f (max %zu; max L0 %zu), U0^-1 nnz %.0f (max %zu; max U0 %zu) per node\n", U,
                cuopt::mathematical_optimization::toc(t0), li / U, lmax, l0max, ui / U, umax, u0max);
    // Column-oriented work for sparse right-hand sides: z = sum over nz(b) of columns of L0^-1 P,
    // x = sum over nz(z) of columns of U0^-1. Column lengths from the row-wise inverses.
    double nb[2] = {0, 0}, nz[2] = {0, 0}, nx[2] = {0, 0}, wl[2] = {0, 0}, wu[2] = {0, 0};
    for (int u = 0; u < U; ++u) {
      const auto& d = nd[u];
      std::vector<int> lcol(m, 0), ucol(m, 0);
      for (int c : d.Linv.ci) { ++lcol[c]; }
      for (int c : d.Uinv.ci) { ++ucol[c]; }
      for (int r = 0; r < 2; ++r) {
        const auto& b = r == 0 ? d.b0 : d.b1;
        const auto& x = r == 0 ? d.ref0 : d.ref1;
        std::vector<double> z(m, 0.0);
        for (int i = 0; i < m; ++i) {
          for (int p2 = d.Linv.rp[i]; p2 < d.Linv.rp[i + 1]; ++p2) { z[i] += d.Linv.cx[p2] * b[d.Linv.ci[p2]]; }
        }
        for (int j = 0; j < m; ++j) {
          if (b[j] != 0.0) { nb[r] += 1, wl[r] += lcol[j]; }
          if (z[j] != 0.0) { nz[r] += 1, wu[r] += ucol[j]; }
          if (x[j] != 0.0) { nx[r] += 1; }
        }
      }
    }
    for (int r = 0; r < 2; ++r) {
      std::printf("  rhs%d (%s): nnz b=%.0f z=%.0f x=%.0f | column work L0^-1=%.0f U0^-1=%.0f entries "
                  "(vs %.0f + %.0f row-wise)\n",
                  r, r == 0 ? "a_q" : "rho", nb[r] / U, nz[r] / U, nx[r] / U, wl[r] / U, wu[r] / U, li / U,
                  ui / U);
    }
  }
  {
    // Critical path in entries: per level the longest row, summed over levels.
    double crit = 0.0, total = 0.0, rows_over[4] = {0, 0, 0, 0};
    int maxrow = 0;
    for (const auto& d : nd) {
      for (const tri_host* t : {&d.L, &d.U}) {
        for (size_t l = 0; l + 1 < t->lptr.size(); ++l) {
          int mx = 0;
          for (int q = t->lptr[l]; q < t->lptr[l + 1]; ++q) {
            const int i   = t->order[q];
            const int len = t->rp[i + 1] - t->rp[i];
            mx            = std::max(mx, len);
            total += len;
            rows_over[0] += len > 8, rows_over[1] += len > 32, rows_over[2] += len > 128,
              rows_over[3] += len > 512;
          }
          crit += mx;
          maxrow = std::max(maxrow, mx);
        }
      }
    }
    std::printf("row lengths per node (L+U): entries=%.0f, sum over levels of the longest row=%.0f, "
                "longest row=%d, rows longer than 8/32/128/512: %.1f/%.1f/%.1f/%.1f\n",
                total / U, crit / U, maxrow, rows_over[0] / U, rows_over[1] / U, rows_over[2] / U,
                rows_over[3] / U);
  }
  size_t b1nz = 0;
  for (const auto& d : nd) {
    for (double v : d.b1) { b1nz += v != 0.0; }
  }
  std::printf("ftran-bench: m=%d nodes=%d replicas=%d -> seats=%d | per node avg: L offdiag nnz=%.0f "
              "levels=%.1f, U offdiag nnz=%.0f levels=%.1f, rho nnz=%.0f\n",
              m, U, replicas, U * replicas, double(nnzL) / U, double(levL) / U, double(nnzU) / U,
              double(levU) / U, double(b1nz) / U);

  // Separate copies per seat so the working set is replicas x nodes.
  const int S = U * replicas;
  if (const char* e = std::getenv("FTRAN_BENCH_ARENA"); e == nullptr || std::string(e) != "0") {
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    g_arena.cap = fr - (size_t(6) << 30);  // leave room for the copy-bandwidth buffers
    FB_CHECK(cudaMalloc(&g_arena.base, g_arena.cap));
  }
  std::vector<seat_dev> hs(S);
  for (int r = 0; r < replicas; ++r) {
    for (int u = 0; u < U; ++u) {
      auto& z = hs[r * U + u];
      z.L     = upload_tri(nd[u].L);
      z.U     = upload_tri(nd[u].U);
      z.L2    = upload_tri2(nd[u].L2);
      z.U2    = upload_tri2(nd[u].U2);
      z.L3    = upload_tri2(nd[u].L3);
      z.U3    = upload_tri2(nd[u].U3);
      z.chunks  = upload(nd[u].chunks);
      z.Lirp = upload(nd[u].Linv.rp), z.Lici = upload(nd[u].Linv.ci), z.Licx = upload(nd[u].Linv.cx);
      z.Uirp = upload(nd[u].Uinv.rp), z.Uici = upload(nd[u].Uinv.ci), z.Uicx = upload(nd[u].Uinv.cx);
      {
        std::vector<double> zero(m, 0.0);
        z.z0 = upload(zero);
        z.z1 = upload(zero);
      }
      z.Lccp = upload(nd[u].LinvC.rp), z.Lcri = upload(nd[u].LinvC.ci), z.Lccx = upload(nd[u].LinvC.cx);
      z.Uccp = upload(nd[u].UinvC.rp), z.Ucri = upload(nd[u].UinvC.ci), z.Uccx = upload(nd[u].UinvC.cx);
      z.bi0 = upload(nd[u].bi0), z.bx0 = upload(nd[u].bx0), z.bn0 = static_cast<int>(nd[u].bi0.size());
      z.bi1 = upload(nd[u].bi1), z.bx1 = upload(nd[u].bx1), z.bn1 = static_cast<int>(nd[u].bi1.size());
      z.nchunks = static_cast<int>(nd[u].chunks.size());
      z.perm  = upload(nd[u].perm);
      z.b0    = upload(nd[u].b0);
      z.b1    = upload(nd[u].b1);
      std::vector<double> zero(m, 0.0);
      z.x0 = upload(zero);
      z.x1 = upload(zero);
    }
  }
  seat_dev* dseats = upload(hs);

  // Compulsory bytes per FTRAN (both right-hand sides): factor entries (index + value), row
  // pointers, diagonals, level order, permutation, the two inputs read and two outputs written.
  const double bytes_v2 =
    (double(nnzL + nnzU) * 12.0 + 2.0 * U * (m + 1) * 4.0 + 2.0 * U * m * 8.0 + 2.0 * U * m * 4.0 +
     U * m * 4.0 + 4.0 * U * m * 8.0) /
    U;

  // Roofline reference: device-to-device copy bandwidth.
  double copy_gbs = 0.0;
  {
    const size_t nb = size_t(2) << 30;
    double4 *a = nullptr, *b = nullptr;
    FB_CHECK(cudaMalloc(&a, nb));
    FB_CHECK(cudaMalloc(&b, nb));
    FB_CHECK(cudaMemset(a, 0, nb));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    copy_kernel<<<4096, 256>>>(a, b, nb / sizeof(double4));
    cudaEventRecord(e0);
    for (int k = 0; k < 5; ++k) { copy_kernel<<<4096, 256>>>(a, b, nb / sizeof(double4)); }
    cudaEventRecord(e1);
    FB_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    copy_gbs = 5.0 * 2.0 * nb / (ms / 1e3) / 1e9;
    cudaFree(a), cudaFree(b);
  }
  std::printf("roofline: device copy bandwidth %.0f GB/s; compulsory bytes per FTRAN (2 rhs) %.0f KB "
              "-> bound %.2fM FTRAN/s\n",
              copy_gbs, bytes_v2 / 1e3, copy_gbs * 1e9 / bytes_v2 / 1e6);

  int dev = 0, sms = 0, clock_khz = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);

  double bytes_v9 = 0;
  auto run = [&](const char* name, auto launch, int threads, size_t smem, const void* fn) {
    const double bytes = std::strncmp(name, "v9", 2) == 0 || std::strncmp(name, "v1", 2) == 0 ? bytes_v9 : bytes_v2;
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, fn, threads, smem);
    cudaFuncAttributes fa{};
    cudaFuncGetAttributes(&fa, fn);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    launch();
    FB_CHECK(cudaGetLastError());
    FB_CHECK(cudaDeviceSynchronize());
    // Check every node's first replica against the host solve.
    double err = 0.0;
    std::vector<double> x0(m), x1(m);
    for (int u = 0; u < U; ++u) {
      FB_CHECK(cudaMemcpy(x0.data(), hs[u].x0, m * sizeof(double), cudaMemcpyDeviceToHost));
      FB_CHECK(cudaMemcpy(x1.data(), hs[u].x1, m * sizeof(double), cudaMemcpyDeviceToHost));
      for (int i = 0; i < m; ++i) {
        err = std::max(err, std::abs(x0[i] - nd[u].ref0[i]) / (1.0 + std::abs(nd[u].ref0[i])));
        err = std::max(err, std::abs(x1[i] - nd[u].ref1[i]) / (1.0 + std::abs(nd[u].ref1[i])));
      }
    }
    std::vector<float> t;
    for (int k = 0; k < repeats; ++k) {
      cudaEventRecord(e0);
      launch();
      cudaEventRecord(e1);
      FB_CHECK(cudaEventSynchronize(e1));
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    const double s     = t[t.size() / 2] / 1e3;
    const double rate  = S / s;
    const double gbs   = rate * bytes / 1e9;
    std::printf("%-26s regs=%3d threads/seat=%4d smem/seat=%6zuKB seats/SM=%2d warps/SM=%3d | "
                "%.2f ms for %d seats -> %.3fM FTRAN/s, %.0f GB/s = %.1f%% of copy bw | max rel err %.1e\n",
                name, fa.numRegs, threads, smem / 1024, bps, bps * threads / 32, s * 1e3, S,
                rate / 1e6, gbs, 100.0 * gbs / copy_gbs, err);
  };

  run("warp/seat global", [&] { ftran_warp<<<S, 32>>>(dseats, S, m); }, 32, 0,
      reinterpret_cast<const void*>(&ftran_warp));
  const size_t sm2 = 2 * size_t(m) * sizeof(double);
  FB_CHECK(cudaFuncSetAttribute(ftran_block<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(sm2)));
  FB_CHECK(cudaFuncSetAttribute(ftran_block2<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(sm2)));
  {
    // Per-level cycles of v2 (16 warps, shared memory) against each level's shape.
    const int maxlev = 128;
    long long* dc    = nullptr;
    FB_CHECK(cudaMalloc(&dc, size_t(S) * maxlev * sizeof(long long)));
    FB_CHECK(cudaMemset(dc, 0, size_t(S) * maxlev * sizeof(long long)));
    FB_CHECK(cudaFuncSetAttribute(ftran_block2_timed, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(sm2)));
    ftran_block2_timed<<<S, 512, sm2>>>(dseats, S, m, dc, maxlev);
    FB_CHECK(cudaDeviceSynchronize());
    std::vector<long long> hc(size_t(S) * maxlev);
    FB_CHECK(cudaMemcpy(hc.data(), dc, hc.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    cudaFree(dc);
    // Bucket levels by (rows, longest row) and report mean cycles.
    struct agg_t { double cyc = 0, n = 0, rows = 0, ent = 0, mx = 0; };
    std::map<std::string, agg_t> by;
    double tot = 0, nlv = 0;
    for (int u = 0; u < U; ++u) {
      int g = 0;
      for (const tri2_host* t : {&nd[u].L2, &nd[u].U2}) {
        for (size_t l = 0; l + 1 < t->lptr.size(); ++l, ++g) {
          if (g >= maxlev) { continue; }
          const int q0 = t->lptr[l], q1 = t->lptr[l + 1];
          int mx = 0;
          for (int q = q0; q < q1; ++q) { mx = std::max(mx, t->rp[q + 1] - t->rp[q]); }
          const int ent = t->rp[q1] - t->rp[q0];
          const char* rb = (q1 - q0) <= 64 ? "rows<=64" : (q1 - q0) <= 512 ? "rows<=512" : "rows>512";
          const char* mb = mx == 0 ? "max=0" : mx <= kShort ? "max<=8" : mx <= 32 ? "max<=32" : "max>32";
          auto& a = by[std::string(t == &nd[u].L2 ? "L " : "U ") + rb + " " + mb];
          const double c = static_cast<double>(hc[size_t(u) * maxlev + g]);
          a.cyc += c, a.n += 1, a.rows += q1 - q0, a.ent += ent, a.mx += mx;
          tot += c, nlv += 1;
        }
      }
    }
    std::printf("v2 timed: mean levels/seat=%.1f mean cycles/level=%.0f mean cycles/seat=%.0f\n",
                nlv / U, tot / nlv, tot / U);
    for (const auto& [k, a] : by) {
      std::printf("  %-26s levels/seat=%5.2f cycles/level=%7.0f rows=%7.1f entries=%7.1f max_row=%5.1f | "
                  "share of seat cycles=%.1f%%\n",
                  k.c_str(), a.n / U, a.cyc / a.n, a.rows / a.n, a.ent / a.n, a.mx / a.n,
                  100.0 * a.cyc / tot);
    }
  }
  {
    double li = 0, ui = 0;
    for (const auto& d : nd) { li += d.Linv.ci.size(), ui += d.Uinv.ci.size(); }
    // Compulsory bytes of the two-SpMV form: inverse entries, row pointers, b read, z written and
    // read, x written (both right-hand sides).
    const double b8 = ((li + ui) * 12.0) / U + 2.0 * (m + 1) * 4.0 + 2.0 * 4.0 * m * 8.0;
    char nm[64];
    auto v8 = [&](auto kl, auto ku, int g) {
      const long long work = static_cast<long long>(S) * m * g;
      const int th = 256;
      const int nb = static_cast<int>((work + th - 1) / th);
      std::snprintf(nm, sizeof(nm), "v8 2xSpMV %d lanes/row", g);
      run(nm, [&] {
        kl<<<nb, th>>>(dseats, S, m);
        ku<<<nb, th>>>(dseats, S, m);
      }, th, 0, reinterpret_cast<const void*>(ku));
    };
    std::printf("v8 compulsory bytes per FTRAN (2 rhs) %.0f KB -> bound %.2fM FTRAN/s\n", b8 / 1e3,
                copy_gbs * 1e9 / b8 / 1e6);
    {
      double cw = 0;
      int zmax  = 0;
      for (const auto& d : nd) {
        for (int r = 0; r < 2; ++r) {
          const auto& bi = r == 0 ? d.bi0 : d.bi1;
          std::vector<double> z(m, 0.0);
          for (size_t t = 0; t < bi.size(); ++t) {
            const int c = bi[t];
            cw += d.LinvC.rp[c + 1] - d.LinvC.rp[c];
            for (int q = d.LinvC.rp[c]; q < d.LinvC.rp[c + 1]; ++q) { z[d.LinvC.ci[q]] = 1.0; }
          }
          int zc = 0;
          for (int k = 0; k < m; ++k) {
            if (z[k] != 0.0) { cw += d.UinvC.rp[k + 1] - d.UinvC.rp[k], ++zc; }
          }
          zmax = std::max(zmax, zc);
        }
      }
      // Compulsory bytes: touched inverse entries (row index + value), column pointers of the touched
      // columns, sparse b, x written dense.
      const double b9 = cw * 12.0 / U + 2.0 * m * 8.0;
      std::printf("v9 touched entries per FTRAN (2 rhs) %.0f, compulsory bytes %.0f KB -> bound %.2fM FTRAN/s; "
                  "max nnz(z) %d\n",
                  cw / U, b9 / 1e3, copy_gbs * 1e9 / b9 / 1e6, zmax);
      bytes_v9 = b9;
      const size_t sm9 = 2 * size_t(m) * 8 + 2 * size_t(kZCap) * (4 + 8);
      auto v9 = [&](auto kern, int th) {
        FB_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm9)));
        std::snprintf(nm, sizeof(nm), "v9 inv-sparse %d warps/seat", th / 32);
        run(nm, [&] { kern<<<S, th, sm9>>>(dseats, S, m); }, th, sm9, reinterpret_cast<const void*>(kern));
      };
      auto v10 = [&](auto kern, int th, int g, int cap) {
        const size_t sm = 2 * size_t(m) * 8 + 2 * size_t(cap) * (4 + 8);
        FB_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm)));
        std::snprintf(nm, sizeof(nm), "v10 %dw g%d cap%d", th / 32, g, cap);
        run(nm, [&] { kern<<<S, th, sm>>>(dseats, S, m, nullptr); }, th, sm, reinterpret_cast<const void*>(kern));
        long long* dph = nullptr;
        FB_CHECK(cudaMalloc(&dph, size_t(S) * 6 * sizeof(long long)));
        kern<<<S, th, sm>>>(dseats, S, m, dph);
        FB_CHECK(cudaDeviceSynchronize());
        std::vector<long long> hph(size_t(S) * 6);
        FB_CHECK(cudaMemcpy(hph.data(), dph, hph.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        cudaFree(dph);
        double a[5] = {0, 0, 0, 0, 0};
        for (int q = 0; q < S; ++q) {
          for (int k = 0; k < 5; ++k) { a[k] += double(hph[size_t(q) * 6 + k]); }
        }
        std::printf("    cycles per seat: clear=%.0f L^-1 scatter=%.0f compact=%.0f U^-1 scatter=%.0f write=%.0f\n",
                    a[0] / S, a[1] / S, a[2] / S, a[3] / S, a[4] / S);
      };
      auto v11 = [&](auto kern, int th, int cap) {
        const size_t sm = 2 * size_t(m) * 8 + size_t(cap) * (8 + 4 + 4 + 1) + 16;
        FB_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm)));
        std::snprintf(nm, sizeof(nm), "v11 %dw cap%d", th / 32, cap);
        run(nm, [&] { kern<<<S, th, sm>>>(dseats, S, m, nullptr); }, th, sm, reinterpret_cast<const void*>(kern));
        long long* dph = nullptr;
        FB_CHECK(cudaMalloc(&dph, size_t(S) * 6 * sizeof(long long)));
        kern<<<S, th, sm>>>(dseats, S, m, dph);
        FB_CHECK(cudaDeviceSynchronize());
        std::vector<long long> hph(size_t(S) * 6);
        FB_CHECK(cudaMemcpy(hph.data(), dph, hph.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        cudaFree(dph);
        double a[5] = {0, 0, 0, 0, 0};
        for (int q = 0; q < S; ++q) {
          for (int k = 0; k < 5; ++k) { a[k] += double(hph[size_t(q) * 6 + k]); }
        }
        std::printf("    cycles per seat: load b=%.0f L^-1 scatter=%.0f compact=%.0f U^-1 scatter=%.0f write=%.0f\n",
                    a[0] / S, a[1] / S, a[2] / S, a[3] / S, a[4] / S);
      };
      if (2 * zmax <= 1024) {
        v11(ftran_inv_bal<256, 1024>, 256, 1024);
        v11(ftran_inv_bal<512, 1024>, 512, 1024);
        v11(ftran_inv_bal<1024, 1024>, 1024, 1024);
        v11(ftran_inv_bal<1024, 1024, 2>, 1024, 1024);
        v11(ftran_inv_bal<512, 1024, 4>, 512, 1024);
      }
      if (zmax <= 1024) {
        v10(ftran_inv_sparse2<256, 4, 1024>, 256, 4, 1024);
        v10(ftran_inv_sparse2<256, 8, 1024>, 256, 8, 1024);
        v10(ftran_inv_sparse2<512, 4, 1024>, 512, 4, 1024);
        v10(ftran_inv_sparse2<512, 8, 1024>, 512, 8, 1024);
        v10(ftran_inv_sparse2<1024, 8, 1024>, 1024, 8, 1024);
      }
      v9(ftran_inv_sparse<128>, 128);
      v9(ftran_inv_sparse<256>, 256);
      v9(ftran_inv_sparse<512>, 512);
    }
    v8(spmv_all<1, true>, spmv_all<1, false>, 1);
    v8(spmv_all<2, true>, spmv_all<2, false>, 2);
    v8(spmv_all<4, true>, spmv_all<4, false>, 4);
    v8(spmv_all<8, true>, spmv_all<8, false>, 8);
  }
  {
    int maxl = 0;
    for (const auto& d : nd) { maxl = std::max<int>(maxl, std::max(d.L3.lptr.size(), d.U3.lptr.size()) - 1); }
    if (maxl > kMaxLev) {
      std::fprintf(stderr, "ftran-bench: %d levels exceed kMaxLev\n", maxl);
      return 1;
    }
    char nm[64];
    auto v7 = [&](auto kern, int th) {
      FB_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(sm2)));
      std::snprintf(nm, sizeof(nm), "v7 %d warps/seat", th / 32);
      run(nm, [&] { kern<<<S, th, sm2>>>(dseats, S, m, nullptr); }, th, sm2, reinterpret_cast<const void*>(kern));
    };
    {
      long long* dc = nullptr;
      const size_t stride = kMaxLev * 2;
      FB_CHECK(cudaMalloc(&dc, size_t(S) * stride * sizeof(long long)));
      FB_CHECK(cudaMemset(dc, 0, size_t(S) * stride * sizeof(long long)));
      FB_CHECK(cudaFuncSetAttribute(ftran_v7<512>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(sm2)));
      ftran_v7<512><<<S, 512, sm2>>>(dseats, S, m, dc);
      FB_CHECK(cudaDeviceSynchronize());
      std::vector<long long> hc(size_t(S) * stride);
      FB_CHECK(cudaMemcpy(hc.data(), dc, hc.size() * sizeof(long long), cudaMemcpyDeviceToHost));
      cudaFree(dc);
      double tot = 0, nlv = 0, small = 0, nsmall = 0, big = 0, nbig = 0;
      for (int u = 0; u < U; ++u) {
        int g = 0;
        for (const tri2_host* t : {&nd[u].L3, &nd[u].U3}) {
          for (size_t l = 0; l + 1 < t->lptr.size(); ++l, ++g) {
            const double c = static_cast<double>(hc[size_t(u) * stride + 1 + g]);
            const int rows = t->lptr[l + 1] - t->lptr[l];
            tot += c, nlv += 1;
            if (rows <= 64) { small += c, nsmall += 1; } else { big += c, nbig += 1; }
          }
        }
      }
      std::printf("v7 timed (16 warps): levels/seat=%.1f cycles/seat=%.0f | levels<=64 rows: %.1f/seat at "
                  "%.0f cycles | larger: %.1f/seat at %.0f cycles\n",
                  nlv / U, tot / U, nsmall / U, small / std::max(1.0, nsmall), nbig / U,
                  big / std::max(1.0, nbig));
    }
    v7(ftran_v7<256>, 256);
    v7(ftran_v7<512>, 512);
    v7(ftran_v7<1024>, 1024);
  }
  {
    char nm[64];
    auto v3 = [&](auto kern, int th, bool smem, int mb) {
      if (smem) {
        FB_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sm2)));
      }
      std::snprintf(nm, sizeof(nm), "v3 %d warps/seat %s mb%d", th / 32, smem ? "smem" : "global", mb);
      run(nm, [&] { kern<<<S, th, smem ? sm2 : 0>>>(dseats, S, m); }, th, smem ? sm2 : 0,
          reinterpret_cast<const void*>(kern));
    };
    auto v4 = [&](auto kern, int th, bool smem, int ahead) {
      if (smem) {
        FB_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sm2)));
      }
      std::snprintf(nm, sizeof(nm), "v4 %d warps/seat %s ahead%d", th / 32, smem ? "smem" : "global", ahead);
      run(nm, [&] { kern<<<S, th, smem ? sm2 : 0>>>(dseats, S, m); }, th, smem ? sm2 : 0,
          reinterpret_cast<const void*>(kern));
    };
    {
      const size_t smf = sm2 + size_t(m);
      FB_CHECK(cudaFuncSetAttribute(ftran_syncfree, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smf)));
      for (int th : {128, 256, 512, 1024}) {
        std::snprintf(nm, sizeof(nm), "v5 syncfree %d warps/seat", th / 32);
        run(nm, [&] { ftran_syncfree<<<S, th, smf>>>(dseats, S, m); }, th, smf,
            reinterpret_cast<const void*>(&ftran_syncfree));
      }
    }
    {
      const size_t sms = sm2 + 2 * sizeof(stage_t);
      FB_CHECK(cudaFuncSetAttribute(ftran_stream, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(sms)));
      for (int th : {256, 384, 512, 1024}) {
        std::snprintf(nm, sizeof(nm), "v6 stream %d warps/seat", th / 32);
        run(nm, [&] { ftran_stream<<<S, th, sms>>>(dseats, S, m); }, th, sms,
            reinterpret_cast<const void*>(&ftran_stream));
      }
    }
    v4(ftran_block4<true, 1>, 512, true, 1);
    v4(ftran_block4<true, 2>, 512, true, 2);
    v4(ftran_block4<true, 4>, 512, true, 4);
    v4(ftran_block4<true, 2>, 1024, true, 2);
    v4(ftran_block4<false, 2>, 256, false, 2);
    v4(ftran_block4<false, 2>, 512, false, 2);
    v3(ftran_block3<true, 512, 1>, 512, true, 1);
    v3(ftran_block3<true, 1024, 1>, 1024, true, 1);
    v3(ftran_block3<false, 256, 4>, 256, false, 4);
    v3(ftran_block3<false, 256, 8>, 256, false, 8);
    v3(ftran_block3<false, 128, 16>, 128, false, 16);
    v3(ftran_block3<false, 512, 4>, 512, false, 4);
  }
  for (int w : {4, 8, 16, 32}) {
    const int th = 32 * w;
    char nm[64];
    std::snprintf(nm, sizeof(nm), "v2 %d warps/seat global", w);
    run(nm, [&] { ftran_block2<false><<<S, th>>>(dseats, S, m); }, th, 0,
        reinterpret_cast<const void*>(&ftran_block2<false>));
    std::snprintf(nm, sizeof(nm), "v2 %d warps/seat smem", w);
    run(nm, [&] { ftran_block2<true><<<S, th, sm2>>>(dseats, S, m); }, th, sm2,
        reinterpret_cast<const void*>(&ftran_block2<true>));
  }
  for (int w : {2, 4, 8, 16}) {
    const int th = 32 * w;
    char nm[64];
    std::snprintf(nm, sizeof(nm), "%d warps/seat global", w);
    run(nm, [&] { ftran_block<false><<<S, th>>>(dseats, S, m); }, th, 0,
        reinterpret_cast<const void*>(&ftran_block<false>));
    std::snprintf(nm, sizeof(nm), "%d warps/seat smem", w);
    run(nm, [&] { ftran_block<true><<<S, th, sm2>>>(dseats, S, m); }, th, sm2,
        reinterpret_cast<const void*>(&ftran_block<true>));
  }
  (void)sms, (void)clock_khz;
  return 0;
}
