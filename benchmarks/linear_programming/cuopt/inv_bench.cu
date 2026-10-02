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
#include "inv_build.cuh"

#include <dual_simplex/gpu_phase2/explicit_inverse.hpp>

namespace ei = cuopt::linear_programming::dual_simplex::explicit_inverse;

namespace {

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
  auto data = load_bench_data(handle, file, prepare_threads, "inv-bench");
  if (!data) { return 1; }
  const int m = data->m;
  const int N = static_cast<int>(data->seats.size());
  const auto& prepared = data->prepared;
  const auto& seats    = data->seats;
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
  cudaEvent_t e0, e1;
  for (cudaEvent_t* e : {&e0, &e1}) { IB_CHECK(cudaEventCreate(e)); }
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

  // ---- Device build (inputs not timed: the pool uploads the factors already).
  long long cap[2] = {0, 0}, tot_nnz[2] = {0, 0};
  for (const auto& h : hinv) {
    cap[0] = std::max<long long>(cap[0], static_cast<long long>(h.Lc.i.size()));
    cap[1] = std::max<long long>(cap[1], static_cast<long long>(h.Uc.i.size()));
    tot_nnz[0] += h.Lc.i.size(), tot_nnz[1] += h.Uc.i.size();
  }
  dev_build db = make_dev_build(*data, cap);
  dev_out& out = db.out;
  const int warps = db.warps;
  const bool ordered = std::getenv("INV_BENCH_ORDERED") != nullptr;
  std::vector<double> t_dev[4];
  int ntasks = 0;
  cudaEvent_t ev[5];
  for (auto& e : ev) { IB_CHECK(cudaEventCreate(&e)); }
  for (int rep = 0; rep < repeats; ++rep) {
    if (run_dev_build(db, ordered, ev) != 0) {
      std::fprintf(stderr, "inv-bench: a seat's inverse exceeded its capacity (L %lld, U %lld entries)\n", cap[0], cap[1]);
      return 1;
    }
    IB_CHECK(cudaMemcpy(&ntasks, out.ntasks, sizeof(int), cudaMemcpyDeviceToHost));
    for (int q = 0; q < 4; ++q) {
      float ms = 0;
      IB_CHECK(cudaEventElapsedTime(&ms, ev[q], ev[q + 1]));
      t_dev[q].push_back(ms / 1e3);
    }
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
