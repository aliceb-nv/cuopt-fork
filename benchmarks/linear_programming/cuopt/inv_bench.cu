/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// --inv-bench: explicit inverses L0^{-1} P and U0^{-1} of every stored node's base factors, in column
// (CSC) and row (CSR) form, exactly as explicit_inverse::build produces them.
//   host baseline: single-thread build of every node, then one pinned upload (single-thread pack);
//   device: trivial columns + task list, warp-per-column one-pass solves (rows in index order, which
//   is a topological order for a triangular factor), per-seat CSC packing, per-seat CSR transpose.
// Capacities are the per-seat maxima of this snapshot; exceeding one fails loudly.

#include "inv_bench.hpp"

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
#include <string>
#include <vector>

namespace ds  = cuopt::mathematical_optimization::simplex;
namespace mip = cuopt::mathematical_optimization::mip;
namespace mo  = cuopt::mathematical_optimization;
namespace ei  = cuopt::linear_programming::dual_simplex::explicit_inverse;

namespace {

#define IB_CHECK(x)                                                                              \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "inv-bench: %s failed: %s\n", #x, cudaGetErrorString(e_));            \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

template <class T>
T* dalloc(size_t n)
{
  T* p = nullptr;
  IB_CHECK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
  return p;
}
template <class T>
T* dupload(const std::vector<T>& v)
{
  T* p = dalloc<T>(v.size());
  IB_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
template <class T>
std::vector<T> ddownload(const T* p, size_t n)
{
  std::vector<T> v(n);
  IB_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

// Factor f (0: L0, diagonal first in each column; 1: U0, diagonal last) of seat s.
struct dev_in {
  int N, m;
  const int* cp[2];      // N * (m + 1), per-seat column pointers (seat-local entry offsets)
  const long long* eo[2];  // N, entry offset of seat s
  const int* ri[2];
  const double* cx[2];
  const int* perm;       // N * m
};
struct dev_out {
  long long cap[2];      // entries per seat
  int* cur[2];           // N: scratch cursor
  int* cstart[2];        // N * m, by output column
  int* clen[2];
  int* si[2];            // N * cap scratch
  double* sx[2];
  int* cp[2];            // N * (m + 1) packed CSC
  int* ci[2];
  double* cv[2];
  int* rp[2];            // N * (m + 1) CSR
  int* rj[2];
  double* rv[2];
  int* tasks;            // non-trivial (seat, factor, column), encoded (s * 2 + f) * m + k
  int* ntasks;
  int* next;
  int* err;
};

__device__ __forceinline__ int out_col(const dev_in& in, int s, int f, int k)
{
  return f == 0 ? in.perm[static_cast<size_t>(s) * in.m + k] : k;
}

// Thread per (seat, factor, column): trivial columns (no off-diagonal entries) written directly.
__global__ void k_trivial(dev_in in, dev_out out)
{
  const long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long T = 2LL * in.N * in.m;
  if (t >= T) { return; }
  const int k = static_cast<int>(t % in.m), f = static_cast<int>((t / in.m) % 2), s = static_cast<int>(t / (2LL * in.m));
  const int* cp = in.cp[f] + static_cast<size_t>(s) * (in.m + 1);
  if (cp[k + 1] - cp[k] != 1) {
    out.tasks[atomicAdd(out.ntasks, 1)] = (s * 2 + f) * in.m + k;
    return;
  }
  const long long e = in.eo[f][s] + cp[k];
  const int pos     = atomicAdd(&out.cur[f][s], 1);
  if (pos >= out.cap[f]) {
    atomicExch(out.err, 1);
    return;
  }
  const size_t base = static_cast<size_t>(s) * out.cap[f];
  out.si[f][base + pos] = k;
  out.sx[f][base + pos] = 1.0 / in.cx[f][e];
  const size_t c        = static_cast<size_t>(s) * in.m + out_col(in, s, f, k);
  out.cstart[f][c]      = pos;
  out.clen[f][c]        = 1;
}

// Warp per non-trivial column, dynamic task fetch. Solves T y = e_k with rows finalized in index
// order (ascending for L0, descending for U0), discovered rows in a bitset, values dense; both per
// warp in global workspace, left zero after each task.
__global__ void __launch_bounds__(256) k_columns(dev_in in, dev_out out, double* ws_y,
                                                 unsigned long long* ws_b, int* ws_l)
{
  const int lane = threadIdx.x & 31;
  const int gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int m    = in.m;
  const int W64  = (m + 63) / 64;
  double* y      = ws_y + static_cast<size_t>(gw) * m;
  unsigned long long* bits = ws_b + static_cast<size_t>(gw) * W64;
  int* lst       = ws_l + static_cast<size_t>(gw) * m;
  const int nt   = *out.ntasks;
  for (;;) {
    int t = 0;
    if (lane == 0) { t = atomicAdd(out.next, 1); }
    t = __shfl_sync(0xffffffffu, t, 0);
    if (t >= nt) { return; }
    const int code = out.tasks[t];
    const int k = code % m, f = (code / m) % 2, s = code / (2 * m);
    const bool asc   = f == 0;
    const int* cp    = in.cp[f] + static_cast<size_t>(s) * (m + 1);
    const int* ri    = in.ri[f] + in.eo[f][s];
    const double* cx = in.cx[f] + in.eo[f][s];
    if (lane == 0) {
      y[k] = 1.0;
      bits[k >> 6] |= 1ull << (k & 63);
    }
    __syncwarp();
    int cnt = 0, cur = k;
    for (;;) {
      // Next discovered row at or past `cur` in solve order.
      int r = -1;
      const int w0 = cur >> 6;
      for (int base = 0; r < 0; base += 32) {
        const int w = asc ? w0 + base + lane : w0 - base - lane;
        unsigned long long word = 0;
        if (w >= 0 && w < W64) {
          word = bits[w];
          if (w == w0) {
            const int b = cur & 63;
            word &= asc ? (~0ull << b) : (b == 63 ? ~0ull : ((2ull << b) - 1));
          }
        }
        const unsigned hit = __ballot_sync(0xffffffffu, word != 0);
        if (hit != 0) {
          const int src = __ffs(hit) - 1;
          const unsigned long long ww = __shfl_sync(0xffffffffu, word, src);
          const int wi                = asc ? w0 + base + src : w0 - base - src;
          r = wi * 64 + (asc ? __ffsll(static_cast<long long>(ww)) - 1 : 63 - __clzll(static_cast<long long>(ww)));
        } else if ((asc && w0 + base + 31 >= W64 - 1) || (!asc && w0 - base - 31 <= 0)) {
          break;
        }
      }
      if (r < 0) { break; }
      const double d  = asc ? cx[cp[r]] : cx[cp[r + 1] - 1];
      const double yr = y[r] / d;
      __syncwarp();
      if (lane == 0) {
        y[r] = yr;
        lst[cnt] = r;
        bits[r >> 6] &= ~(1ull << (r & 63));
      }
      ++cnt;
      __syncwarp();
      const int p0 = asc ? cp[r] + 1 : cp[r];
      const int p1 = asc ? cp[r + 1] : cp[r + 1] - 1;
      for (int p = p0 + lane; p < p1; p += 32) {
        const int i = ri[p];
        y[i] -= cx[p] * yr;
        atomicOr(&bits[i >> 6], 1ull << (i & 63));
      }
      __syncwarp();
      cur = r;
    }
    // Exact zeros are dropped, as in the host build.
    int nz = 0;
    for (int q = lane; q < cnt; q += 32) { nz += y[lst[q]] != 0.0 ? 1 : 0; }
    for (int o = 16; o > 0; o >>= 1) { nz += __shfl_xor_sync(0xffffffffu, nz, o); }
    int pos = 0;
    if (lane == 0) { pos = atomicAdd(&out.cur[f][s], nz); }
    pos            = __shfl_sync(0xffffffffu, pos, 0);
    const bool fit = pos + nz <= out.cap[f];
    if (!fit && lane == 0) { atomicExch(out.err, 1); }
    const size_t base = static_cast<size_t>(s) * out.cap[f];
    int w             = 0;
    for (int q0 = 0; q0 < cnt; q0 += 32) {
      const int q     = q0 + lane;
      const int row   = q < cnt ? lst[q] : 0;
      const double v  = q < cnt ? y[row] : 0.0;
      const bool keep = q < cnt && v != 0.0;
      const unsigned b = __ballot_sync(0xffffffffu, keep);
      if (keep && fit) {
        const int o          = pos + w + __popc(b & ((1u << lane) - 1u));
        out.si[f][base + o] = row;
        out.sx[f][base + o] = v;
      }
      w += __popc(b);
      if (q < cnt) { y[row] = 0.0; }
    }
    __syncwarp();
    if (lane == 0) {
      const size_t c   = static_cast<size_t>(s) * m + out_col(in, s, f, k);
      out.cstart[f][c] = pos;
      out.clen[f][c]   = nz;
    }
  }
}

// Block per (seat, factor): column pointers by scan over the output columns, entries copied in.
constexpr int kPackThreads = 1024;
__global__ void __launch_bounds__(kPackThreads) k_pack(dev_in in, dev_out out)
{
  using scan_t = cub::BlockScan<int, kPackThreads>;
  __shared__ typename scan_t::TempStorage tmp;
  const int s = blockIdx.x / 2, f = blockIdx.x % 2, m = in.m;
  const int per = (m + kPackThreads - 1) / kPackThreads;
  const int c0 = threadIdx.x * per, c1 = min(m, c0 + per);
  const int* clen = out.clen[f] + static_cast<size_t>(s) * m;
  const int* cst  = out.cstart[f] + static_cast<size_t>(s) * m;
  int* cp         = out.cp[f] + static_cast<size_t>(s) * (m + 1);
  int sum = 0;
  for (int c = c0; c < c1; ++c) { sum += clen[c]; }
  int off = 0, tot = 0;
  scan_t(tmp).ExclusiveSum(sum, off, tot);
  for (int c = c0; c < c1; ++c) {
    cp[c] = off;
    off += clen[c];
  }
  if (threadIdx.x == 0) { cp[m] = tot; }
  __syncthreads();
  const size_t base = static_cast<size_t>(s) * out.cap[f];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int c = warp; c < m; c += kPackThreads / 32) {
    const int src = cst[c], dst = cp[c], len = clen[c];
    for (int q = lane; q < len; q += 32) {
      out.ci[f][base + dst + q] = out.si[f][base + src + q];
      out.cv[f][base + dst + q] = out.sx[f][base + src + q];
    }
  }
}

// Block per (seat, factor): CSR by counting, scan, then a scatter. Ordered: one warp walks the
// columns in order (each row's entries in column order, deterministic). Otherwise: warps take
// columns in parallel with shared atomic row cursors (order within a row varies run to run).
constexpr int kTrThreads = 256;
template <bool Ordered>
__global__ void __launch_bounds__(kTrThreads) k_transpose(dev_in in, dev_out out)
{
  extern __shared__ int rc[];  // m: counts, then cursors
  using scan_t = cub::BlockScan<int, kTrThreads>;
  __shared__ typename scan_t::TempStorage tmp;
  const int s = blockIdx.x / 2, f = blockIdx.x % 2, m = in.m;
  const size_t base = static_cast<size_t>(s) * out.cap[f];
  const int* cp     = out.cp[f] + static_cast<size_t>(s) * (m + 1);
  const int* ci     = out.ci[f] + base;
  const double* cv  = out.cv[f] + base;
  int* rp           = out.rp[f] + static_cast<size_t>(s) * (m + 1);
  for (int r = threadIdx.x; r < m; r += kTrThreads) { rc[r] = 0; }
  __syncthreads();
  const int nnz = cp[m];
  for (int q = threadIdx.x; q < nnz; q += kTrThreads) { atomicAdd(&rc[ci[q]], 1); }
  __syncthreads();
  const int per = (m + kTrThreads - 1) / kTrThreads;
  const int r0 = threadIdx.x * per, r1 = min(m, r0 + per);
  int sum = 0;
  for (int r = r0; r < r1; ++r) { sum += rc[r]; }
  int off = 0, tot = 0;
  scan_t(tmp).ExclusiveSum(sum, off, tot);
  __syncthreads();
  for (int r = r0; r < r1; ++r) {
    const int c = rc[r];
    rp[r] = off, rc[r] = off;
    off += c;
  }
  if (threadIdx.x == 0) { rp[m] = tot; }
  __syncthreads();
  if constexpr (Ordered) {
    if (threadIdx.x < 32) {
      const int lane = threadIdx.x;
      for (int c = 0; c < m; ++c) {
        for (int q = cp[c] + lane; q < cp[c + 1]; q += 32) {
          const int r = ci[q];
          const int o = rc[r]++;
          out.rj[f][base + o] = c;
          out.rv[f][base + o] = cv[q];
        }
        __syncwarp();
      }
    }
  } else {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int c = warp; c < m; c += kTrThreads / 32) {
      for (int q = cp[c] + lane; q < cp[c + 1]; q += 32) {
        const int o = atomicAdd(&rc[ci[q]], 1);
        out.rj[f][base + o] = c;
        out.rv[f][base + o] = cv[q];
      }
    }
  }
}

double rel(double a, double b) { return std::fabs(a - b) / std::max(1.0, std::fabs(b)); }

// Device vs host sparse form (same orientation), patterns merged with missing entries as zero.
// Returns the max relative difference of entries present on both sides; one-sided entries are counted
// and their largest |value| relative to the column's largest is kept in `one_sided`.
struct cmp_t {
  double both = 0.0, one_sided = 0.0;
  long long n_one = 0, cols_diff = 0;
};
void compare(int m, const int* dp, const int* di, const double* dv, const ei::sparse_t& h, cmp_t& r)
{
  std::vector<std::pair<int, double>> a, b;
  for (int c = 0; c < m; ++c) {
    a.clear(), b.clear();
    for (int q = dp[c]; q < dp[c + 1]; ++q) { a.emplace_back(di[q], dv[q]); }
    for (int q = h.p[c]; q < h.p[c + 1]; ++q) { b.emplace_back(h.i[q], h.x[q]); }
    std::sort(a.begin(), a.end()), std::sort(b.begin(), b.end());
    double cmax = 0.0;
    for (auto& e : b) { cmax = std::max(cmax, std::fabs(e.second)); }
    for (auto& e : a) { cmax = std::max(cmax, std::fabs(e.second)); }
    size_t x = 0, y = 0;
    bool diff = false;
    while (x < a.size() || y < b.size()) {
      if (y >= b.size() || (x < a.size() && a[x].first < b[y].first)) {
        r.one_sided = std::max(r.one_sided, std::fabs(a[x].second) / std::max(cmax, 1e-300)), ++r.n_one, diff = true, ++x;
      } else if (x >= a.size() || b[y].first < a[x].first) {
        r.one_sided = std::max(r.one_sided, std::fabs(b[y].second) / std::max(cmax, 1e-300)), ++r.n_one, diff = true, ++y;
      } else {
        r.both = std::max(r.both, rel(a[x].second, b[y].second)), ++x, ++y;
      }
    }
    r.cols_diff += diff ? 1 : 0;
  }
}

}  // namespace

int inv_bench_main(const raft::handle_t& handle, const std::string& file, int repeats, int prepare_threads)
{
  namespace ps = mip::pool_snapshot;
  ps::pool_t<int, double> snap(&handle);
  std::string why;
  if (!ps::read<int, double>(file, snap, why)) {
    std::fprintf(stderr, "inv-bench: cannot load %s (%s)\n", file.c_str(), why.c_str());
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
  // The stored nodes' LUs (the replay's up-front factorization; not part of the comparison).
  std::vector<mip::batch_seat_state_t<int, double>> prepared;
  const auto ok = mip::gpu_batch_factor_fresh<int, double>(lp, nodes, prepared, prepare_threads);
  std::vector<int> seats;
  for (size_t k = 0; k < nodes.size(); ++k) {
    if (ok[k]) { seats.push_back(static_cast<int>(k)); }
  }
  const int N = static_cast<int>(seats.size());

  // ---- Host baseline: single-thread build, single-thread pack into pinned memory, one upload.
  std::vector<ei::inverses_t> hinv(N);
  ei::workspace_t ws;
  const double tb0 = mo::tic();
  for (int s = 0; s < N; ++s) {
    const auto& ft = prepared[seats[s]].ft;
    ei::build(m, ft.L0().col_start.data(), ft.L0().i.data(), ft.L0().x.data(), ft.U0().col_start.data(),
              ft.U0().i.data(), ft.U0().x.data(), ft.row_permutation().data(), ws, hinv[s]);
  }
  const double t_build = mo::toc(tb0);
  size_t up_bytes = 0;
  for (const auto& h : hinv) {
    for (const ei::sparse_t* sp : {&h.Lc, &h.Lr, &h.Uc, &h.Ur}) {
      up_bytes += sp->p.size() * sizeof(int) + sp->i.size() * sizeof(int) + sp->x.size() * sizeof(double);
    }
  }
  unsigned char* pinned = nullptr;
  IB_CHECK(cudaMallocHost(&pinned, up_bytes));
  unsigned char* dup = dalloc<unsigned char>(up_bytes);
  std::vector<double> t_pack, t_h2d;
  cudaEvent_t e0, e1, e2, e3, e4, e5;
  for (cudaEvent_t* e : {&e0, &e1, &e2, &e3, &e4, &e5}) { IB_CHECK(cudaEventCreate(e)); }
  for (int rep = 0; rep < repeats; ++rep) {
    const double tp0 = mo::tic();
    size_t o = 0;
    auto put = [&](const void* src, size_t nb) {
      std::memcpy(pinned + o, src, nb);
      o += nb;
    };
    for (const auto& h : hinv) {
      for (const ei::sparse_t* sp : {&h.Lc, &h.Lr, &h.Uc, &h.Ur}) {
        put(sp->p.data(), sp->p.size() * sizeof(int));
        put(sp->i.data(), sp->i.size() * sizeof(int));
        put(sp->x.data(), sp->x.size() * sizeof(double));
      }
    }
    t_pack.push_back(mo::toc(tp0));
    IB_CHECK(cudaEventRecord(e0));
    IB_CHECK(cudaMemcpyAsync(dup, pinned, up_bytes, cudaMemcpyHostToDevice));
    IB_CHECK(cudaEventRecord(e1));
    IB_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    IB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    t_h2d.push_back(ms / 1e3);
  }

  // ---- Device inputs: the base factors as staged (not timed: the pool uploads them already).
  long long cap[2] = {0, 0}, tot_nnz[2] = {0, 0};
  for (const auto& h : hinv) {
    cap[0] = std::max<long long>(cap[0], static_cast<long long>(h.Lc.i.size()));
    cap[1] = std::max<long long>(cap[1], static_cast<long long>(h.Uc.i.size()));
    tot_nnz[0] += h.Lc.i.size(), tot_nnz[1] += h.Uc.i.size();
  }
  dev_in in{};
  in.N = N, in.m = m;
  std::vector<int> perm(static_cast<size_t>(N) * m);
  for (int f = 0; f < 2; ++f) {
    std::vector<int> cp(static_cast<size_t>(N) * (m + 1)), ri;
    std::vector<long long> eo(N);
    std::vector<double> cx;
    for (int s = 0; s < N; ++s) {
      const auto& ft = prepared[seats[s]].ft;
      const auto& F  = f == 0 ? ft.L0() : ft.U0();
      eo[s]          = static_cast<long long>(ri.size());
      std::copy(F.col_start.begin(), F.col_start.end(), cp.begin() + static_cast<size_t>(s) * (m + 1));
      ri.insert(ri.end(), F.i.begin(), F.i.begin() + F.col_start[m]);
      cx.insert(cx.end(), F.x.begin(), F.x.begin() + F.col_start[m]);
      if (f == 0) { std::copy(ft.row_permutation().begin(), ft.row_permutation().end(), perm.begin() + static_cast<size_t>(s) * m); }
    }
    in.cp[f] = dupload(cp), in.eo[f] = dupload(eo), in.ri[f] = dupload(ri), in.cx[f] = dupload(cx);
  }
  in.perm = dupload(perm);
  dev_out out{};
  for (int f = 0; f < 2; ++f) {
    const size_t ne = static_cast<size_t>(N) * cap[f], nc = static_cast<size_t>(N) * m;
    out.cap[f] = cap[f];
    out.cur[f] = dalloc<int>(N), out.cstart[f] = dalloc<int>(nc), out.clen[f] = dalloc<int>(nc);
    out.si[f] = dalloc<int>(ne), out.sx[f] = dalloc<double>(ne);
    out.cp[f] = dalloc<int>(nc + N), out.ci[f] = dalloc<int>(ne), out.cv[f] = dalloc<double>(ne);
    out.rp[f] = dalloc<int>(nc + N), out.rj[f] = dalloc<int>(ne), out.rv[f] = dalloc<double>(ne);
  }
  out.tasks = dalloc<int>(2ull * N * m), out.ntasks = dalloc<int>(1), out.next = dalloc<int>(1),
  out.err = dalloc<int>(1);
  int dev = 0, sms = 0, bps = 0;
  IB_CHECK(cudaGetDevice(&dev));
  IB_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  IB_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_columns, 256, 0));
  const int grid_c = sms * bps, warps = grid_c * 256 / 32, W64 = (m + 63) / 64;
  double* ws_y = dalloc<double>(static_cast<size_t>(warps) * m);
  unsigned long long* ws_b = dalloc<unsigned long long>(static_cast<size_t>(warps) * W64);
  int* ws_l = dalloc<int>(static_cast<size_t>(warps) * m);
  IB_CHECK(cudaMemset(ws_y, 0, static_cast<size_t>(warps) * m * sizeof(double)));
  IB_CHECK(cudaMemset(ws_b, 0, static_cast<size_t>(warps) * W64 * sizeof(unsigned long long)));
  IB_CHECK(cudaFuncSetAttribute(k_transpose<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, m * static_cast<int>(sizeof(int))));
  IB_CHECK(cudaFuncSetAttribute(k_transpose<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, m * static_cast<int>(sizeof(int))));
  const bool ordered = std::getenv("INV_BENCH_ORDERED") != nullptr;
  std::vector<double> t_dev[4];
  int ntasks = 0;
  for (int rep = 0; rep < repeats; ++rep) {
    for (int f = 0; f < 2; ++f) { IB_CHECK(cudaMemsetAsync(out.cur[f], 0, N * sizeof(int))); }
    IB_CHECK(cudaMemsetAsync(out.ntasks, 0, sizeof(int)));
    IB_CHECK(cudaMemsetAsync(out.next, 0, sizeof(int)));
    IB_CHECK(cudaMemsetAsync(out.err, 0, sizeof(int)));
    const long long T = 2LL * N * m;
    IB_CHECK(cudaEventRecord(e0));
    k_trivial<<<static_cast<unsigned>((T + 255) / 256), 256>>>(in, out);
    IB_CHECK(cudaEventRecord(e1));
    k_columns<<<grid_c, 256>>>(in, out, ws_y, ws_b, ws_l);
    IB_CHECK(cudaEventRecord(e2));
    k_pack<<<2 * N, kPackThreads>>>(in, out);
    IB_CHECK(cudaEventRecord(e3));
    if (ordered) {
      k_transpose<true><<<2 * N, kTrThreads, m * sizeof(int)>>>(in, out);
    } else {
      k_transpose<false><<<2 * N, kTrThreads, m * sizeof(int)>>>(in, out);
    }
    IB_CHECK(cudaEventRecord(e4));
    IB_CHECK(cudaEventSynchronize(e4));
    IB_CHECK(cudaGetLastError());
    int err = 0;
    IB_CHECK(cudaMemcpy(&err, out.err, sizeof(int), cudaMemcpyDeviceToHost));
    IB_CHECK(cudaMemcpy(&ntasks, out.ntasks, sizeof(int), cudaMemcpyDeviceToHost));
    if (err != 0) {
      std::fprintf(stderr, "inv-bench: a seat's inverse exceeded its capacity (L %lld, U %lld entries)\n", cap[0], cap[1]);
      return 1;
    }
    float a = 0, b = 0, c = 0, d = 0;
    IB_CHECK(cudaEventElapsedTime(&a, e0, e1));
    IB_CHECK(cudaEventElapsedTime(&b, e1, e2));
    IB_CHECK(cudaEventElapsedTime(&c, e2, e3));
    IB_CHECK(cudaEventElapsedTime(&d, e3, e4));
    t_dev[0].push_back(a / 1e3), t_dev[1].push_back(b / 1e3), t_dev[2].push_back(c / 1e3), t_dev[3].push_back(d / 1e3);
  }

  // ---- Correctness against the host build, every seat, both factors, both forms.
  cmp_t res[4];
  for (int f = 0; f < 2; ++f) {
    const auto cp = ddownload(out.cp[f], static_cast<size_t>(N) * (m + 1));
    const auto ci = ddownload(out.ci[f], static_cast<size_t>(N) * cap[f]);
    const auto cv = ddownload(out.cv[f], static_cast<size_t>(N) * cap[f]);
    const auto rp = ddownload(out.rp[f], static_cast<size_t>(N) * (m + 1));
    const auto rj = ddownload(out.rj[f], static_cast<size_t>(N) * cap[f]);
    const auto rv = ddownload(out.rv[f], static_cast<size_t>(N) * cap[f]);
    for (int s = 0; s < N; ++s) {
      const size_t pb = static_cast<size_t>(s) * (m + 1), eb = static_cast<size_t>(s) * cap[f];
      const auto& h   = hinv[s];
      compare(m, cp.data() + pb, ci.data() + eb, cv.data() + eb, f == 0 ? h.Lc : h.Uc, res[2 * f]);
      compare(m, rp.data() + pb, rj.data() + eb, rv.data() + eb, f == 0 ? h.Lr : h.Ur, res[2 * f + 1]);
    }
  }

  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  auto mx = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
  int devid = 0;
  cudaDeviceProp prop{};
  IB_CHECK(cudaGetDevice(&devid));
  IB_CHECK(cudaGetDeviceProperties(&prop, devid));
  std::printf("inv-bench: %s, seats=%d m=%d | inverse nnz per seat: L mean %.0f max %lld, U mean %.0f max %lld | non-trivial columns=%d\n",
              prop.name, N, m, double(tot_nnz[0]) / N, cap[0], double(tot_nnz[1]) / N, cap[1], ntasks);
  std::printf("host baseline (single thread): build %.1f ms + pack into pinned %.1f ms (median of %d) + upload %.1f ms (median; %.1f MB, %.1f GB/s) = %.1f ms\n",
              1e3 * t_build, 1e3 * med(t_pack), repeats, 1e3 * med(t_h2d), up_bytes / 1e6, up_bytes / 1e9 / med(t_h2d),
              1e3 * (t_build + med(t_pack) + med(t_h2d)));
  const double dev_tot = med(t_dev[0]) + med(t_dev[1]) + med(t_dev[2]) + med(t_dev[3]);
  std::printf("CSR transpose: %s\n", ordered ? "ordered (deterministic)" : "atomic row cursors");
  std::printf("device (median of %d, max in brackets): trivial+tasks %.2f (%.2f) ms, columns %.2f (%.2f) ms, pack CSC %.2f (%.2f) ms, CSR %.2f (%.2f) ms = %.2f ms | %d column warps\n",
              repeats, 1e3 * med(t_dev[0]), 1e3 * mx(t_dev[0]), 1e3 * med(t_dev[1]), 1e3 * mx(t_dev[1]),
              1e3 * med(t_dev[2]), 1e3 * mx(t_dev[2]), 1e3 * med(t_dev[3]), 1e3 * mx(t_dev[3]), 1e3 * dev_tot, warps);
  const char* nm[4] = {"Lc", "Lr", "Uc", "Ur"};
  for (int q = 0; q < 4; ++q) {
    std::printf("correctness %s: max rel diff (both sides) %.2e | one-sided entries %lld in %lld columns, largest %.2e of the column max\n",
                nm[q], res[q].both, res[q].n_one, res[q].cols_diff, res[q].one_sided);
  }
  return 0;
}
