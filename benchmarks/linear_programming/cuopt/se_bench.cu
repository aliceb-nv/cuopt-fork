/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// --se-bench: exact steepest-edge weights w_i = ||e_i' B^{-1}||^2 of every basic position i of every
// stored node, on the device from the explicit inverses: row i of B^{-1} = U0^{-1}(i,:) L0^{-1} P, and P
// does not change the norm, so w_i = || sum_c U0^{-1}(i,c) L0^{-1}(c,:) ||^2.
//   device: the inverses are built first (untimed, inv_build.cuh); then warp per (seat, row) with a
//   dense per-warp accumulator and a touched-column list (timed).
//   host: phase2::initialize_steepest_edge_norms (one B^T solve per row through the host LU), timed
//   single-threaded on the first seats; the rest are computed on worker threads for verification only.
// Also reported, for information: how the snapshot's stored weights compare with the exact ones.

#include "se_bench.hpp"
#include "inv_build.cuh"

#include <dual_simplex/phase2.hpp>

#include <atomic>
#include <thread>

namespace {

constexpr long long kSeCapL = 19926, kSeCapU = 329296;  // per-seat inverse maxima measured by --inv-bench
constexpr int kSeChunk      = 8;                          // CHOSEN: rows taken per task fetch

// Warp per (seat, basic position). Lanes take entries (c, u) of row i of U0^{-1} and add u * L0^{-1}(c,:)
// into the warp's dense accumulator (double atomics: two entries can reach the same column), recording
// first touches; the squares are then summed over the touched columns, which are cleared.
__global__ void __launch_bounds__(256) k_se(dev_out o, int N, int m, int* next, double* w_out,
                                            double* ws_acc, unsigned long long* ws_bits, int* ws_list,
                                            int* ws_cnt)
{
  const int lane = threadIdx.x & 31;
  const int gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int W64  = (m + 63) / 64;
  double* acc    = ws_acc + static_cast<size_t>(gw) * m;
  unsigned long long* bits = ws_bits + static_cast<size_t>(gw) * W64;
  int* lst       = ws_list + static_cast<size_t>(gw) * m;
  int* cnt       = ws_cnt + gw;
  const long long T = static_cast<long long>(N) * m;
  for (;;) {
    long long t0 = 0;
    if (lane == 0) { t0 = atomicAdd(reinterpret_cast<unsigned long long*>(next), kSeChunk); }
    t0 = __shfl_sync(0xffffffffu, t0, 0);
    if (t0 >= T) { return; }
    for (long long t = t0; t < min(T, t0 + kSeChunk); ++t) {
      const int s = static_cast<int>(t / m), i = static_cast<int>(t % m);
      const size_t pb = static_cast<size_t>(s) * (m + 1);
      const int* urp = o.rp[1] + pb;
      const int* urj = o.rj[1] + static_cast<size_t>(s) * o.cap[1];
      const double* urv = o.rv[1] + static_cast<size_t>(s) * o.cap[1];
      const int* lrp = o.rp[0] + pb;
      const int* lrj = o.rj[0] + static_cast<size_t>(s) * o.cap[0];
      const double* lrv = o.rv[0] + static_cast<size_t>(s) * o.cap[0];
      for (int q = urp[i] + lane; q < urp[i + 1]; q += 32) {
        const int c    = urj[q];
        const double u = urv[q];
        for (int p = lrp[c]; p < lrp[c + 1]; ++p) {
          const int j = lrj[p];
          atomicAdd(&acc[j], u * lrv[p]);
          const unsigned long long bit = 1ull << (j & 63);
          if ((atomicOr(&bits[j >> 6], bit) & bit) == 0) { lst[atomicAdd(cnt, 1)] = j; }
        }
      }
      __syncwarp();
      const int nt = *reinterpret_cast<volatile int*>(cnt);
      double sum = 0.0;
      for (int q = lane; q < nt; q += 32) {
        const int j = lst[q];
        const double v = acc[j];
        sum += v * v;
        acc[j] = 0.0;
        bits[j >> 6] = 0;
      }
      for (int off = 16; off > 0; off >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, off); }
      __syncwarp();
      if (lane == 0) {
        w_out[t] = sum;
        *cnt = 0;
      }
      __syncwarp();
    }
  }
}

}  // namespace

int se_bench_main(const raft::handle_t& handle, const std::string& file, int repeats, int prepare_threads,
                  int host_timed_seats)
{
  auto data = load_bench_data(handle, file, prepare_threads, "se-bench");
  if (!data) { return 1; }
  const int m = data->m, n = data->n;
  const int N = static_cast<int>(data->seats.size());
  const auto& lp = data->snap.lp;

  // ---- Inverses on the device (not timed here; --inv-bench times them).
  const long long cap[2] = {kSeCapL, kSeCapU};
  dev_build db           = make_dev_build(*data, cap);
  if (run_dev_build(db, false) != 0) {
    std::fprintf(stderr, "se-bench: a seat's inverse exceeded its capacity (L %lld, U %lld entries)\n", cap[0], cap[1]);
    return 1;
  }

  // ---- Device weights.
  int dev = 0, sms = 0, bps = 0;
  IB_CHECK(cudaGetDevice(&dev));
  IB_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  IB_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_se, 256, 0));
  const int grid = sms * bps, warps = grid * 8, W64 = (m + 63) / 64;
  double* ws_acc = dalloc<double>(static_cast<size_t>(warps) * m);
  auto* ws_bits  = dalloc<unsigned long long>(static_cast<size_t>(warps) * W64);
  int* ws_list   = dalloc<int>(static_cast<size_t>(warps) * m);
  int* ws_cnt    = dalloc<int>(warps);
  IB_CHECK(cudaMemset(ws_acc, 0, static_cast<size_t>(warps) * m * sizeof(double)));
  IB_CHECK(cudaMemset(ws_bits, 0, static_cast<size_t>(warps) * W64 * sizeof(unsigned long long)));
  IB_CHECK(cudaMemset(ws_cnt, 0, warps * sizeof(int)));
  long long* d_next = dalloc<long long>(1);
  double* d_w       = dalloc<double>(static_cast<size_t>(N) * m);
  cudaEvent_t e0, e1;
  IB_CHECK(cudaEventCreate(&e0));
  IB_CHECK(cudaEventCreate(&e1));
  std::vector<double> t_dev;
  for (int rep = 0; rep < repeats; ++rep) {
    IB_CHECK(cudaMemset(d_next, 0, sizeof(long long)));
    IB_CHECK(cudaEventRecord(e0));
    k_se<<<grid, 256>>>(db.out, N, m, reinterpret_cast<int*>(d_next), d_w, ws_acc, ws_bits, ws_list, ws_cnt);
    IB_CHECK(cudaEventRecord(e1));
    IB_CHECK(cudaEventSynchronize(e1));
    IB_CHECK(cudaGetLastError());
    float ms = 0;
    IB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    t_dev.push_back(ms / 1e3);
  }
  const auto w_dev = ddownload(d_w, static_cast<size_t>(N) * m);

  // ---- Host reference (exact weights, indexed by variable).
  std::vector<std::vector<double>> w_host(N);
  auto host_one = [&](int s) {
    auto& prep     = data->prepared[data->seats[s]];
    auto settings  = data->nodes[data->seats[s]].settings;
    settings.log.log = false;
    std::vector<double> dy(n, -1.0);
    double work = 0.0;
    ds::phase2::initialize_steepest_edge_norms<int, double>(lp, settings, mo::tic(), prep.basic_list, prep.ft, dy, work);
    w_host[s] = std::move(dy);
  };
  const int timed = std::min(N, std::max(1, host_timed_seats));
  const double th0 = mo::tic();
  for (int s = 0; s < timed; ++s) { host_one(s); }
  const double t_host = mo::toc(th0);
  {
    std::atomic<int> next{timed};
    std::vector<std::thread> th;
    for (int t = 0; t < std::max(1, prepare_threads); ++t) {
      th.emplace_back([&] {
        for (int s = next++; s < N; s = next++) { host_one(s); }
      });
    }
    for (auto& t : th) { t.join(); }
  }

  // ---- Correctness, and the stored weights' staleness (information only).
  std::vector<double> rels;
  rels.reserve(static_cast<size_t>(N) * m);
  long long n_stored = 0, below_half = 0, above_two = 0, nonpos = 0, repaired = 0;
  std::vector<double> ratios;
  for (int s = 0; s < N; ++s) {
    const auto& prep   = data->prepared[data->seats[s]];
    const auto& stored = data->nodes[data->seats[s]].dy_se;
    for (int i = 0; i < m; ++i) {
      const int j      = prep.basic_list[i];
      const double ref = w_host[s][j];
      const double got = w_dev[static_cast<size_t>(s) * m + i];
      rels.push_back(std::fabs(got - ref) / std::max(std::fabs(ref), 1e-300));
      if (static_cast<size_t>(j) < stored.size()) {
        const double st = stored[j];
        ++n_stored;
        if (st <= 0.0) { ++nonpos; continue; }
        if (st == 1e-4) { ++repaired; }
        const double r = st / ref;
        ratios.push_back(r);
        below_half += r < 0.5 ? 1 : 0;
        above_two += r > 2.0 ? 1 : 0;
      }
    }
  }
  auto pct = [](std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1)))];
  };
  cudaDeviceProp prop{};
  IB_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("se-bench: %s, seats=%d m=%d, %d warps\n", prop.name, N, m, warps);
  std::printf("device weights (median of %d, max in brackets): %.3f (%.3f) ms for %d seats\n", repeats,
              1e3 * pct(t_dev, 0.5), 1e3 * pct(t_dev, 1.0), N);
  std::printf("host initialize_steepest_edge_norms (single thread): %.1f ms for %d seats = %.2f ms per seat -> %.0f ms for %d seats\n",
              1e3 * t_host, timed, 1e3 * t_host / timed, 1e3 * t_host / timed * N, N);
  std::printf("correctness vs host (all %d seats, %zu weights): rel diff p50 %.2e p99 %.2e p99.9 %.2e max %.2e\n", N,
              rels.size(), pct(rels, 0.5), pct(rels, 0.99), pct(rels, 0.999), pct(rels, 1.0));
  std::printf("stored weights vs exact (information only): %lld basic weights | non-positive %lld | exactly 1e-4 %lld | "
              "stored/exact p1 %.3f p10 %.3f p50 %.3f p90 %.3f p99 %.3f | below 0.5x (a reset on first pick) %lld = %.1f%% | above 2x %lld = %.1f%%\n",
              n_stored, nonpos, repaired, pct(ratios, 0.01), pct(ratios, 0.1), pct(ratios, 0.5), pct(ratios, 0.9),
              pct(ratios, 0.99), below_half, 100.0 * below_half / std::max<long long>(1, n_stored), above_two,
              100.0 * above_two / std::max<long long>(1, n_stored));
  return 0;
}
