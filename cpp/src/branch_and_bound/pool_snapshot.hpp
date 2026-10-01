/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// One GPU pool's seats at one relaunch, for offline replay of the batch DS kernel
// (CUOPT_BNB_GPU_SNAPSHOT=<file>). Every seat is stored as a FRESH start: bounds as a sparse diff
// against the LP bounds, plus the basis and steepest-edge norms it starts (or continues) from.
// CHILD_CONTINUE / REFACTOR_RESUME seats carry the device state of the seat at capture time.

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip::pool_snapshot {

constexpr char kMagic[8] = {'C', 'U', 'O', 'P', 'T', 'P', 'S', '1'};
constexpr int kVersion   = 1;

enum command_t : int32_t { FRESH = 0, REFACTOR = 1, CHILD = 2, PARKED = 3 };

struct header_t {
  char magic[8];
  int32_t version;
  int32_t m, n;
  int32_t seats;
  int64_t nnz;
  int32_t device;
  int32_t pools_on_device;  // replay default for the clone count
  int32_t worker;
  int32_t n_var_types;      // 0: no device strengthening
  int32_t n_pc;             // 0: no pseudocost snapshot this generation
  int32_t n_root_x;
  int32_t dive_rounds;      // 0: dive not set this generation
  int32_t reliable_threshold;
  int32_t objective_is_integral;
  int32_t count[4];         // seats per command_t
  int32_t pad;
  double elapsed;
  double obj_constant, obj_scale;
  double pc_avg_down, pc_avg_up, integer_tol;
};

struct seat_t {
  int32_t command;
  int32_t nbounds;
  int32_t refactor_frequency, use_harris_ratio, use_bound_flip_ratio, random_seed, iteration_limit;
  int32_t pad;
  double cut_off, primal_tol, dual_tol, pivot_tol, tight_tol, fixed_tol, zero_tol;
  double steepest_edge_ratio, steepest_edge_primal_tol, threshold_partial_pivoting_tol;
};

template <typename i_t, typename f_t>
inline seat_t seat_from_settings(int32_t command,
                                 const simplex::simplex_solver_settings_t<i_t, f_t>& s)
{
  seat_t r{};
  r.command                        = command;
  r.refactor_frequency             = static_cast<int32_t>(s.refactor_frequency);
  r.use_harris_ratio               = s.use_harris_ratio ? 1 : 0;
  r.use_bound_flip_ratio           = s.use_bound_flip_ratio ? 1 : 0;
  r.random_seed                    = static_cast<int32_t>(s.random_seed);
  r.iteration_limit                = static_cast<int32_t>(s.iteration_limit);
  r.cut_off                        = s.cut_off;
  r.primal_tol                     = s.primal_tol;
  r.dual_tol                       = s.dual_tol;
  r.pivot_tol                      = s.pivot_tol;
  r.tight_tol                      = s.tight_tol;
  r.fixed_tol                      = s.fixed_tol;
  r.zero_tol                       = s.zero_tol;
  r.steepest_edge_ratio            = s.steepest_edge_ratio;
  r.steepest_edge_primal_tol       = s.steepest_edge_primal_tol;
  r.threshold_partial_pivoting_tol = s.threshold_partial_pivoting_tol;
  return r;
}

template <typename i_t, typename f_t>
inline void apply_settings(const seat_t& r, simplex::simplex_solver_settings_t<i_t, f_t>& s)
{
  s.refactor_frequency             = static_cast<i_t>(r.refactor_frequency);
  s.use_harris_ratio               = r.use_harris_ratio != 0;
  s.use_bound_flip_ratio           = r.use_bound_flip_ratio != 0;
  s.random_seed                    = static_cast<i_t>(r.random_seed);
  s.iteration_limit                = static_cast<i_t>(r.iteration_limit);
  s.cut_off                        = r.cut_off;
  s.primal_tol                     = r.primal_tol;
  s.dual_tol                       = r.dual_tol;
  s.pivot_tol                      = r.pivot_tol;
  s.tight_tol                      = r.tight_tol;
  s.fixed_tol                      = r.fixed_tol;
  s.zero_tol                       = r.zero_tol;
  s.steepest_edge_ratio            = r.steepest_edge_ratio;
  s.steepest_edge_primal_tol       = r.steepest_edge_primal_tol;
  s.threshold_partial_pivoting_tol = r.threshold_partial_pivoting_tol;
}

template <typename T>
inline bool put(std::FILE* f, const T* p, size_t count)
{
  return count == 0 || std::fwrite(p, sizeof(T), count, f) == count;
}

template <typename T>
inline bool get(std::FILE* f, std::vector<T>& v, size_t count)
{
  v.resize(count);
  return count == 0 || std::fread(v.data(), sizeof(T), count, f) == count;
}

// One seat's starting state; command PARKED carries no vectors.
template <typename f_t>
struct seat_state_t {
  seat_t rec{};
  std::vector<int32_t> cols;
  std::vector<f_t> lo, up;
  std::vector<int8_t> vstatus;
  std::vector<f_t> dy_se;
};

template <typename i_t, typename f_t>
struct pool_t {
  header_t h{};
  simplex::lp_problem_t<i_t, f_t> lp;
  std::vector<int8_t> var_types;
  std::vector<f_t> pc_sum_down, pc_sum_up, root_x;
  std::vector<int32_t> pc_num_down, pc_num_up;
  std::vector<seat_state_t<f_t>> seats;
  explicit pool_t(raft::handle_t const* handle) : lp(handle, 1, 1, 1) {}
};

// Writer: header, LP, pool-level vectors, then seats (written by append_seat).
template <typename i_t, typename f_t>
inline bool write_head(std::FILE* f,
                       const header_t& h,
                       const simplex::lp_problem_t<i_t, f_t>& lp,
                       const std::vector<int8_t>& var_types,
                       const std::vector<f_t>& pc_sum_down,
                       const std::vector<f_t>& pc_sum_up,
                       const std::vector<i_t>& pc_num_down,
                       const std::vector<i_t>& pc_num_up,
                       const f_t* root_x)
{
  bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
  ok      = ok && put(f, lp.A.col_start.data(), static_cast<size_t>(h.n) + 1);
  ok      = ok && put(f, lp.A.i.data(), static_cast<size_t>(h.nnz));
  ok      = ok && put(f, lp.A.x.data(), static_cast<size_t>(h.nnz));
  ok      = ok && put(f, lp.rhs.data(), static_cast<size_t>(h.m));
  ok      = ok && put(f, lp.objective.data(), static_cast<size_t>(h.n));
  ok      = ok && put(f, lp.lower.data(), static_cast<size_t>(h.n));
  ok      = ok && put(f, lp.upper.data(), static_cast<size_t>(h.n));
  ok      = ok && put(f, var_types.data(), static_cast<size_t>(h.n_var_types));
  ok      = ok && put(f, pc_sum_down.data(), static_cast<size_t>(h.n_pc));
  ok      = ok && put(f, pc_sum_up.data(), static_cast<size_t>(h.n_pc));
  ok      = ok && put(f, pc_num_down.data(), static_cast<size_t>(h.n_pc));
  ok      = ok && put(f, pc_num_up.data(), static_cast<size_t>(h.n_pc));
  ok      = ok && put(f, root_x, static_cast<size_t>(h.n_root_x));
  return ok;
}

// Bounds are diffed against the LP's; vstatus is written as int8.
template <typename f_t>
inline bool append_seat(std::FILE* f,
                        seat_t rec,
                        const std::vector<f_t>& lp_lower,
                        const std::vector<f_t>& lp_upper,
                        const std::vector<f_t>& lower,
                        const std::vector<f_t>& upper,
                        const std::vector<simplex::variable_status_t>& vstatus,
                        const std::vector<f_t>& dy_se)
{
  if (rec.command == PARKED) {
    rec.nbounds = 0;
    return std::fwrite(&rec, sizeof(rec), 1, f) == 1;
  }
  const size_t n = lp_lower.size();
  std::vector<int32_t> cols;
  std::vector<f_t> lo, up;
  for (size_t j = 0; j < n; ++j) {
    if (lower[j] != lp_lower[j] || upper[j] != lp_upper[j]) {
      cols.push_back(static_cast<int32_t>(j));
      lo.push_back(lower[j]);
      up.push_back(upper[j]);
    }
  }
  std::vector<int8_t> vs(n);
  for (size_t j = 0; j < n; ++j) { vs[j] = static_cast<int8_t>(vstatus[j]); }
  rec.nbounds = static_cast<int32_t>(cols.size());
  bool ok     = std::fwrite(&rec, sizeof(rec), 1, f) == 1;
  ok          = ok && put(f, cols.data(), cols.size());
  ok          = ok && put(f, lo.data(), lo.size());
  ok          = ok && put(f, up.data(), up.size());
  ok          = ok && put(f, vs.data(), n);
  ok          = ok && put(f, dy_se.data(), n);
  return ok;
}

template <typename i_t, typename f_t>
inline bool read(const std::string& path, pool_t<i_t, f_t>& p, std::string& why)
{
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    why = "cannot open";
    return false;
  }
  auto& h  = p.h;
  bool ok  = std::fread(&h, sizeof(h), 1, f) == 1;
  for (int k = 0; ok && k < 8; ++k) { ok = h.magic[k] == kMagic[k]; }
  if (!ok || h.version != kVersion) {
    why = "bad magic or version";
    std::fclose(f);
    return false;
  }
  const size_t m = h.m, n = h.n, nnz = h.nnz;
  auto& lp       = p.lp;
  lp.num_rows = lp.A.m = h.m;
  lp.num_cols = lp.A.n = h.n;
  ok = get(f, lp.A.col_start, n + 1) && get(f, lp.A.i, nnz) && get(f, lp.A.x, nnz) &&
       get(f, lp.rhs, m) && get(f, lp.objective, n) && get(f, lp.lower, n) &&
       get(f, lp.upper, n) && get(f, p.var_types, h.n_var_types) &&
       get(f, p.pc_sum_down, h.n_pc) && get(f, p.pc_sum_up, h.n_pc) &&
       get(f, p.pc_num_down, h.n_pc) && get(f, p.pc_num_up, h.n_pc) &&
       get(f, p.root_x, h.n_root_x);
  lp.obj_constant          = h.obj_constant;
  lp.obj_scale             = h.obj_scale;
  lp.objective_is_integral = h.objective_is_integral != 0;
  p.seats.resize(h.seats);
  for (auto& s : p.seats) {
    if (!ok) { break; }
    ok = std::fread(&s.rec, sizeof(s.rec), 1, f) == 1;
    if (!ok || s.rec.command == PARKED) { continue; }
    const size_t nb = s.rec.nbounds;
    ok = get(f, s.cols, nb) && get(f, s.lo, nb) && get(f, s.up, nb) && get(f, s.vstatus, n) &&
         get(f, s.dy_se, n);
  }
  std::fclose(f);
  if (!ok) { why = "short body"; }
  return ok;
}

}  // namespace cuopt::mathematical_optimization::mip::pool_snapshot
