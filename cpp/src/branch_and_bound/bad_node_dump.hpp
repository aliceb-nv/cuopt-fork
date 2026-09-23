/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Capture a branch-and-bound node on which the GPU driver made a WRONG DECISION, with everything
// needed to re-solve it in exactly the same state, and replay it offline through all three drivers.
//
// WHY THIS EXISTS. The defects that matter here are rare and non-deterministic: the branch-and-bound
// tree differs run to run, so a node that mis-decides once is usually not revisited. Repeatedly the
// only honest thing to say about a failure was "it did not reproduce". A dump turns one observed
// failure into a permanent, replayable artifact -- solvable in isolation, comparable across drivers,
// and stable while a fix is developed.
//
// FIDELITY IS THE WHOLE POINT. This is a raw binary image of the exact arrays, not an MPS
// round-trip: the failures under investigation turn on the last bits of the objective and the
// bounds, so a text format that re-parses doubles would not reproduce the state it is meant to
// capture. The state recorded is the state AS UPLOADED to the device -- after `refactor_basis` (which
// may repair `vstatus`) and after bound strengthening -- because that, not the node's entry state,
// is what the drivers actually consumed.
//
// Writer: `gpu_solve_node_serial` under CUOPT_BNB_DUMP_BAD_NODES=<dir>.
// Reader: `test_netlib --replay-node <file>`, which runs the full CPU driver, the simplified CPU
//         driver and the GPU driver from this identical starting basis.

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace bad_node {

// Bumped whenever the layout changes, and checked on read: a silently mismatched layout would
// produce a plausible-looking but wrong LP, which is worse than refusing to load.
constexpr char kMagic[8]   = {'C', 'U', 'O', 'P', 'T', 'B', 'N', '1'};
// v3 adds the SOLVE STATE (x, y, z) at launch. v2 captured only the basis, the bounds and the
// steepest-edge norms, which is enough to replay a node that starts cold -- but NOT one that was
// mid-solve. A seat resumed after REFACTOR_NEEDED carries x/y/z across the leg boundary, and
// without them a replay starts from a different point and follows a different trajectory.
// Measured on the one wrong-objective node we caught: the live run stopped at iteration 428 with
// obj 20829.75, the replay ran 477 iterations to 20807.62 (deterministic, 3/3) -- the same inputs
// as far as v2 could record them, and a different answer. That gap is why the occurrence could not
// be diagnosed, so the format has to close it.
constexpr int  kVersion    = 3;

struct header_t {
  char     magic[8];
  int32_t  version;
  int32_t  m;
  int32_t  n;
  int64_t  nnz;
  int32_t  has_dy_se;
  // v3: non-zero when x/y/z follow dy_se in the body.
  int32_t  has_solve_state;

  // Every settings field the device reads. A replay that guessed these would not be the same solve.
  double   cut_off;
  double   primal_tol;
  double   dual_tol;
  double   pivot_tol;
  double   tight_tol;
  double   fixed_tol;
  double   zero_tol;
  double   steepest_edge_ratio;
  double   steepest_edge_primal_tol;
  double   threshold_partial_pivoting_tol;
  int32_t  refactor_frequency;
  // The ratio-test switches. Leaving these to the replay's defaults would run a DIFFERENT pivot
  // rule than the device ran, which is exactly the part under suspicion -- the replay would then
  // disagree with the capture for a reason that has nothing to do with the defect.
  int32_t  use_harris_ratio;
  int32_t  use_bound_flip_ratio;
  int32_t  random_seed;
  int32_t  iteration_limit;

  // Objective-space fields. `cut_off` is compared against the objective, so a replay that dropped
  // the constant or the min/max scale would test a different question entirely.
  double   obj_constant;
  double   obj_scale;
  int32_t  objective_is_integral;

  // What was observed, so the artifact carries its own verdict and can be triaged without the log.
  int32_t  gpu_status;
  int32_t  cpu_status;
  double   gpu_objective;
  double   cpu_objective;
  int32_t  gpu_iters;
  int32_t  cpu_iters;
};

template <typename T>
static inline bool write_vec(std::FILE* f, const std::vector<T>& v)
{
  return v.empty() || std::fwrite(v.data(), sizeof(T), v.size(), f) == v.size();
}

template <typename T>
static inline bool read_vec(std::FILE* f, std::vector<T>& v, size_t count)
{
  v.resize(count);
  return count == 0 || std::fread(v.data(), sizeof(T), count, f) == count;
}

// Returns false on any I/O failure -- the caller should report it rather than assume a dump exists.
template <typename i_t, typename f_t>
static inline bool write(const std::string& path,
                         const simplex::lp_problem_t<i_t, f_t>& lp,
                         const std::vector<f_t>& lower,
                         const std::vector<f_t>& upper,
                         const std::vector<simplex::variable_status_t>& vstatus,
                         const std::vector<f_t>& dy_se,
                         const std::vector<f_t>& x,
                         const std::vector<f_t>& y,
                         const std::vector<f_t>& z,
                         const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                         int gpu_status, int cpu_status,
                         f_t gpu_objective, f_t cpu_objective,
                         int gpu_iters, int cpu_iters)
{
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) { return false; }

  header_t h{};
  for (int k = 0; k < 8; ++k) { h.magic[k] = kMagic[k]; }
  h.version             = kVersion;
  h.m                   = static_cast<int32_t>(lp.num_rows);
  h.n                   = static_cast<int32_t>(lp.num_cols);
  h.nnz                 = static_cast<int64_t>(lp.A.x.size());
  h.has_dy_se           = dy_se.empty() ? 0 : 1;
  h.has_solve_state     = (x.size() == static_cast<size_t>(lp.num_cols) &&
                       y.size() == static_cast<size_t>(lp.num_rows) &&
                       z.size() == static_cast<size_t>(lp.num_cols))
                        ? 1
                        : 0;
  h.cut_off             = settings.cut_off;
  h.primal_tol          = settings.primal_tol;
  h.dual_tol            = settings.dual_tol;
  h.pivot_tol           = settings.pivot_tol;
  h.tight_tol           = settings.tight_tol;
  h.fixed_tol           = settings.fixed_tol;
  h.zero_tol            = settings.zero_tol;
  h.steepest_edge_ratio = settings.steepest_edge_ratio;
  h.steepest_edge_primal_tol       = settings.steepest_edge_primal_tol;
  h.threshold_partial_pivoting_tol = settings.threshold_partial_pivoting_tol;
  h.refactor_frequency  = static_cast<int32_t>(settings.refactor_frequency);
  h.use_harris_ratio    = settings.use_harris_ratio ? 1 : 0;
  h.use_bound_flip_ratio = settings.use_bound_flip_ratio ? 1 : 0;
  h.random_seed         = static_cast<int32_t>(settings.random_seed);
  h.iteration_limit     = static_cast<int32_t>(settings.iteration_limit);
  h.obj_constant        = lp.obj_constant;
  h.obj_scale           = lp.obj_scale;
  h.objective_is_integral = lp.objective_is_integral ? 1 : 0;
  h.gpu_status          = gpu_status;
  h.cpu_status          = cpu_status;
  h.gpu_objective       = gpu_objective;
  h.cpu_objective       = cpu_objective;
  h.gpu_iters           = gpu_iters;
  h.cpu_iters           = cpu_iters;

  bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
  ok = ok && write_vec(f, lp.A.col_start);
  ok = ok && write_vec(f, lp.A.i);
  ok = ok && write_vec(f, lp.A.x);
  ok = ok && write_vec(f, lp.rhs);
  ok = ok && write_vec(f, lp.objective);
  ok = ok && write_vec(f, lower);
  ok = ok && write_vec(f, upper);
  // vstatus is an enum over int8; write it as raw bytes so the format does not depend on the
  // enum's underlying type staying put.
  {
    std::vector<int8_t> vs(vstatus.size());
    for (size_t k = 0; k < vstatus.size(); ++k) { vs[k] = static_cast<int8_t>(vstatus[k]); }
    ok = ok && write_vec(f, vs);
  }
  if (h.has_dy_se) { ok = ok && write_vec(f, dy_se); }
  if (h.has_solve_state) {
    ok = ok && write_vec(f, x);
    ok = ok && write_vec(f, y);
    ok = ok && write_vec(f, z);
  }

  std::fclose(f);
  return ok;
}

template <typename i_t, typename f_t>
static inline bool read(const std::string& path,
                        header_t& h,
                        simplex::lp_problem_t<i_t, f_t>& lp,
                        std::vector<f_t>& lower,
                        std::vector<f_t>& upper,
                        std::vector<simplex::variable_status_t>& vstatus,
                        std::vector<f_t>& dy_se,
                        std::vector<f_t>& x,
                        std::vector<f_t>& y,
                        std::vector<f_t>& z,
                        std::string& why)
{
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) { why = "cannot open"; return false; }
  if (std::fread(&h, sizeof(h), 1, f) != 1) { why = "short header"; std::fclose(f); return false; }
  for (int k = 0; k < 8; ++k) {
    if (h.magic[k] != kMagic[k]) { why = "bad magic"; std::fclose(f); return false; }
  }
  // ACCEPT v2 AS WELL AS v3. v3 only APPENDS: has_solve_state lands in the 4 bytes that were
  // padding after has_dy_se (the writer zero-initialises the header, so a v2 file reads it as 0),
  // and x/y/z are appended after dy_se. Refusing v2 would orphan every dump taken before the change
  // -- including the three nasty nodes the replay kit ships.
  if (h.version != kVersion && h.version != 2) {
    why = "version mismatch";
    std::fclose(f);
    return false;
  }

  lp.num_rows = h.m;
  lp.num_cols = h.n;
  lp.A.m      = h.m;
  lp.A.n      = h.n;

  bool ok = read_vec(f, lp.A.col_start, static_cast<size_t>(h.n) + 1);
  ok = ok && read_vec(f, lp.A.i, static_cast<size_t>(h.nnz));
  ok = ok && read_vec(f, lp.A.x, static_cast<size_t>(h.nnz));
  ok = ok && read_vec(f, lp.rhs, static_cast<size_t>(h.m));
  ok = ok && read_vec(f, lp.objective, static_cast<size_t>(h.n));
  ok = ok && read_vec(f, lower, static_cast<size_t>(h.n));
  ok = ok && read_vec(f, upper, static_cast<size_t>(h.n));
  {
    std::vector<int8_t> vs;
    ok = ok && read_vec(f, vs, static_cast<size_t>(h.n));
    vstatus.resize(vs.size());
    for (size_t k = 0; k < vs.size(); ++k) {
      vstatus[k] = static_cast<simplex::variable_status_t>(vs[k]);
    }
  }
  dy_se.clear();
  if (ok && h.has_dy_se) { ok = read_vec(f, dy_se, static_cast<size_t>(h.n)); }
  x.clear();
  y.clear();
  z.clear();
  if (ok && h.has_solve_state) {
    ok = ok && read_vec(f, x, static_cast<size_t>(h.n));
    ok = ok && read_vec(f, y, static_cast<size_t>(h.m));
    ok = ok && read_vec(f, z, static_cast<size_t>(h.n));
  }

  // The node's bounds ARE the LP's bounds -- they were strengthened before the solve, and replaying
  // against the unstrengthened ones would be a different problem.
  lp.lower                = lower;
  lp.upper                = upper;
  lp.obj_constant         = h.obj_constant;
  lp.obj_scale            = h.obj_scale;
  lp.objective_is_integral = h.objective_is_integral != 0;

  std::fclose(f);
  if (!ok) { why = "short body"; }
  return ok;
}

// Put the captured settings back on a settings object. A single place, so a replay cannot silently
// miss a field that the capture recorded.
template <typename i_t, typename f_t>
static inline void apply_settings(const header_t& h,
                                  simplex::simplex_solver_settings_t<i_t, f_t>& s)
{
  s.cut_off                        = h.cut_off;
  s.primal_tol                     = h.primal_tol;
  s.dual_tol                       = h.dual_tol;
  s.pivot_tol                      = h.pivot_tol;
  s.tight_tol                      = h.tight_tol;
  s.fixed_tol                      = h.fixed_tol;
  s.zero_tol                       = h.zero_tol;
  s.steepest_edge_ratio            = h.steepest_edge_ratio;
  s.steepest_edge_primal_tol       = h.steepest_edge_primal_tol;
  s.threshold_partial_pivoting_tol = h.threshold_partial_pivoting_tol;
  s.refactor_frequency             = static_cast<i_t>(h.refactor_frequency);
  s.use_harris_ratio               = h.use_harris_ratio != 0;
  s.use_bound_flip_ratio           = h.use_bound_flip_ratio != 0;
  s.random_seed                    = static_cast<i_t>(h.random_seed);
  s.iteration_limit                = static_cast<i_t>(h.iteration_limit);
}

}  // namespace bad_node

}  // namespace cuopt::mathematical_optimization::mip
