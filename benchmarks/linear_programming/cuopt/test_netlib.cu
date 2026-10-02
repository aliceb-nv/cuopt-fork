/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// test_netlib: run the CPU dual simplex solver over a directory of NETLIB MPS
// instances and compare the full dual phase 2 against the minimal
// `simplified_dual_phase2` driver, started from the same post-phase-1
// dual-feasible factorized basis.

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <pdlp/translate.hpp>

#include <branch_and_bound/bad_node_dump.hpp>
#include <branch_and_bound/gpu_node_pool.hpp>
#include <branch_and_bound/pool_snapshot.hpp>
#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/gpu_phase2/gpu_phase2.cuh>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/scaling.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>

#include <argparse/argparse.hpp>

#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <omp.h>

#include <unordered_map>   // collapsing duplicate [TRUE MISMATCH] lines across replicas

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "benchmark_helper.hpp"
#include "ftran_bench.hpp"
#include "inv_bench.hpp"
#include "se_bench.hpp"

// `NAME=0` must mean OFF. The older TEST_NETLIB_* switches test only for presence, so `=0` turns
// them ON; the switches added here use this helper instead, matching the CUOPT_* convention.
static inline bool env_on(const char* name)
{
  const char* e = std::getenv(name);
  return e != nullptr && std::string(e) != "0";
}

// The shared simplex primitives were promoted to mathematical_optimization::simplex;
// tic/toc and the linear-algebra types (csc_matrix_t) live in the parent
// mathematical_optimization; the GPU driver stays in linear_programming::dual_simplex.
namespace ds  = cuopt::mathematical_optimization::simplex;
namespace mo  = cuopt::mathematical_optimization;
namespace gds = cuopt::linear_programming::dual_simplex;
namespace mip = cuopt::mathematical_optimization::mip;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct instance_result_t {
  std::string name;
  bool parsed = false;
  int m = 0;  // rows of the presolved+scaled LP fed to phase 2
  int n = 0;  // cols

  // Baseline: full dual simplex solve_linear_program.
  ds::lp_status_t base_status = ds::lp_status_t::UNSET;
  double base_obj             = kNaN;
  int base_iters              = 0;
  double base_time            = 0.0;

  // Setup of the post-phase-1 dual-feasible factorized basis.
  bool setup_ok    = false;
  double phase1_obj = kNaN;
  double dual_inf   = kNaN;  // dual infeasibility of the basis handed to phase 2

  // Advanced phase 2 started from that exact basis (initialize_basis = false).
  bool adv_ran                = false;
  ds::dual::status_t adv_status = ds::dual::status_t::UNSET;
  double adv_obj              = kNaN;
  int adv_iters              = 0;
  double adv_time            = 0.0;  // wall-clock seconds of the advanced phase-2 driver call

  // Minimal driver started from the same basis.
  bool simpl_ran                = false;
  ds::dual::status_t simpl_status = ds::dual::status_t::UNSET;
  double simpl_obj              = kNaN;
  int simpl_iters              = 0;
  int simpl_num_updates        = 0;  // basis updates completed before stopping
  double simpl_time            = 0.0;  // wall-clock seconds of the minimal driver call

  // GPU port of the minimal driver from the same basis.
  bool gpu_ran                = false;
  ds::dual::status_t gpu_status = ds::dual::status_t::UNSET;
  double gpu_obj              = kNaN;
  int gpu_iters              = 0;

  std::string note;
};

// Load an MPS file into a dual_simplex user_problem_t using the existing,
// tested conversion chain (parse -> GPU optimization_problem -> user_problem).
ds::user_problem_t<int, double> load_user_problem(const raft::handle_t& handle,
                                                  const std::string& path)
{
  auto model = cuopt::mathematical_optimization::io::read_mps<int, double>(path);
  auto opt =
    cuopt::mathematical_optimization::mps_data_model_to_optimization_problem<int, double>(&handle, model);
  return cuopt::mathematical_optimization::cuopt_optimization_problem_to_user_problem<int, double>(&handle,
                                                                                           opt);
}

// y solves B' y = c_B ; z_N = c_N - N' y ; z_B = 0.  Thin wrapper over the library's
// phase2::compute_dual_solution_from_basis (declared in phase2.hpp); this used to be a private
// copy of it here.
void compute_duals(const ds::lp_problem_t<int, double>& lp,
                   ds::basis_update_mpf_t<int, double>& ft,
                   const std::vector<int>& basic_list,
                   const std::vector<int>& nonbasic_list,
                   std::vector<double>& y,
                   std::vector<double>& z)
{
  double work = 0.0;
  ds::phase2::compute_dual_solution_from_basis(lp, ft, basic_list, nonbasic_list, y, z, work);
}

// Sum of dual-feasibility sign violations of the reduced costs against the
// nonbasic statuses.  ~0 means the basis is dual feasible (minimization form).
// Max violation of x against the node's own bounds, and the max residual of Ax = b. A dual simplex
// stops when its PRIMAL infeasibility list empties; that list is maintained incrementally, so if it
// goes stale the driver can stop at a point that is not actually feasible -- which is exactly the
// case a suboptimal "OPTIMAL" has to be tested against.
static void primal_residuals(const ds::lp_problem_t<int, double>& lp,
                             const std::vector<double>& x,
                             double& bound_viol,
                             double& row_resid)
{
  bound_viol = 0.0;
  for (size_t j = 0; j < x.size(); ++j) {
    bound_viol = std::max(bound_viol, lp.lower[j] - x[j]);
    bound_viol = std::max(bound_viol, x[j] - lp.upper[j]);
  }
  std::vector<double> ax(static_cast<size_t>(lp.num_rows), 0.0);
  for (int j = 0; j < lp.num_cols; ++j) {
    for (int p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
      ax[lp.A.i[p]] += lp.A.x[p] * x[j];
    }
  }
  row_resid = 0.0;
  for (int i = 0; i < lp.num_rows; ++i) {
    row_resid = std::max(row_resid, std::fabs(ax[i] - lp.rhs[i]));
  }
}

double dual_infeasibility(const ds::lp_problem_t<int, double>& lp,
                          const std::vector<ds::variable_status_t>& vstatus,
                          const std::vector<double>& z,
                          double dual_tol)
{
  double inf = 0.0;
  for (int j = 0; j < lp.num_cols; ++j) {
    switch (vstatus[j]) {
      case ds::variable_status_t::NONBASIC_LOWER:
        if (z[j] < -dual_tol) inf += -z[j];  // need z >= 0
        break;
      case ds::variable_status_t::NONBASIC_UPPER:
        if (z[j] > dual_tol) inf += z[j];  // need z <= 0
        break;
      case ds::variable_status_t::NONBASIC_FREE:
        inf += std::abs(z[j]);  // need z == 0
        break;
      default: break;  // BASIC (z==0), NONBASIC_FIXED (no sign condition)
    }
  }
  return inf;
}

// Reproduce the production front-end (convert -> presolve -> scale -> phase 1)
// to obtain the presolved+scaled LP `lp` and a dual-feasible `vstatus`.
bool setup_dual_feasible_basis(const ds::user_problem_t<int, double>& up,
                               ds::simplex_solver_settings_t<int, double>& settings,
                               ds::lp_problem_t<int, double>& lp_out,
                               std::vector<ds::variable_status_t>& vstatus_out,
                               double& phase1_obj_out,
                               std::string& note)
{
  const raft::handle_t* h = up.handle_ptr;

  ds::lp_problem_t<int, double> original_lp(h, 1, 1, 1);
  std::vector<int> new_slacks;
  ds::dualize_info_t<int, double> dualize_info;
  ds::convert_user_problem(up, settings, original_lp, new_slacks, dualize_info);

  ds::lp_problem_t<int, double> presolved_lp(h, 1, 1, 1);
  ds::presolve_info_t<int, double> presolve_info;
  const int ok = ds::presolve(original_lp, settings, presolved_lp, presolve_info);
  if (ok == -1) {
    note = "presolve: infeasible";
    return false;
  }
  if (ok != 0) {
    note = "presolve: code " + std::to_string(ok);
    return false;
  }

  ds::lp_problem_t<int, double> lp(
    h, presolved_lp.num_rows, presolved_lp.num_cols, presolved_lp.A.col_start[presolved_lp.num_cols]);
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  ds::scaling(presolved_lp, settings, lp, column_scales, row_scales);

  if (lp.num_rows == 0) {
    note = "presolved to 0 rows";
    return false;
  }

  ds::lp_problem_t<int, double> phase1_problem(h, 1, 1, 1);
  ds::create_phase1_problem(lp, phase1_problem);

  std::vector<ds::variable_status_t> phase1_vstatus(phase1_problem.num_cols,
                                                    ds::variable_status_t::NONBASIC_LOWER);
  int num_basic = 0;
  for (int j = phase1_problem.num_cols - 1; j >= 0; --j) {
    const int cs = phase1_problem.A.col_start[j];
    const int ce = phase1_problem.A.col_start[j + 1];
    if (ce - cs == 1 && std::abs(phase1_problem.A.x[cs]) == 1.0) {
      phase1_vstatus[j] = ds::variable_status_t::BASIC;
      num_basic++;
    }
    if (num_basic == phase1_problem.num_rows) break;
  }

  int iter = 0;
  ds::lp_solution_t<int, double> phase1_solution(phase1_problem.num_rows, phase1_problem.num_cols);
  std::vector<double> edge_norms;
  const double start_time = mo::tic();
  const ds::dual::status_t st = ds::dual_phase2(
    1, 1, start_time, phase1_problem, settings, phase1_vstatus, phase1_solution, iter, edge_norms,
    nullptr);

  if (st == ds::dual::status_t::NUMERICAL) {
    note = "phase1: numerical";
    return false;
  }
  if (st == ds::dual::status_t::DUAL_UNBOUNDED) {
    note = "phase1: dual unbounded";
    return false;
  }
  if (st == ds::dual::status_t::TIME_LIMIT) {
    note = "phase1: time limit";
    return false;
  }
  phase1_obj_out = phase1_solution.objective;
  if (!(phase1_solution.objective > -settings.primal_tol)) {
    note = "dual infeasible (primal unbounded/infeasible)";
    return false;
  }

  lp_out      = std::move(lp);
  vstatus_out = std::move(phase1_vstatus);
  return true;
}

// [SESTAT] TEMP debug (remove after diagnosis): SE-context b_solve profile for one driver run.
struct se_dbg_t {
  long calls = 0, hyper_l = 0, hyper_u = 0, nz_in = 0, nz_l = 0, nz_u = 0, numupd = 0;
};

// Factorize the basis described by `vstatus`, compute its reduced costs into
// `sol.z`, and run either the minimal driver or the advanced phase 2 from it.
// Returns the phase-2 status; fills `out_obj`, `out_iters`, `out_dual_inf`.
// mode: 0 = advanced phase 2, 1 = CPU simplified_dual_phase2, 2 = GPU
// gpu_simplified_dual_phase2. Each call rebuilds a fresh factorized basis from
// `vstatus_in`, so the three drivers start from the identical basis.
ds::dual::status_t run_phase2_from_basis(const ds::lp_problem_t<int, double>& lp,
                                         const std::vector<ds::variable_status_t>& vstatus_in,
                                         ds::simplex_solver_settings_t<int, double>& settings,
                                         int mode,
                                         double& out_obj,
                                         int& out_iters,
                                         double& out_dual_inf,
                                         int& out_num_updates,
                                         bool initialize_basis                     = false,
                                         const std::vector<double>& dy_se_override = {},
                                         double* out_seconds                       = nullptr,
                                         se_dbg_t* out_se                          = nullptr)
{
  const int m = lp.num_rows;
  const int n = lp.num_cols;

  std::vector<ds::variable_status_t> vstatus = vstatus_in;
  std::vector<int> basic_list(m);
  std::vector<int> nonbasic_list;
  std::vector<int> superbasic_list;
  ds::get_basis_from_vstatus<int>(m, vstatus, basic_list, nonbasic_list, superbasic_list);

  ds::basis_update_mpf_t<int, double> ft(m, settings.refactor_frequency);
  const double start_time = mo::tic();
  ft.refactor_basis(
    lp.A, settings, lp.lower, lp.upper, start_time, basic_list, nonbasic_list, vstatus);

  ds::lp_solution_t<int, double> sol(m, n);
  compute_duals(lp, ft, basic_list, nonbasic_list, sol.y, sol.z);
  out_dual_inf = dual_infeasibility(lp, vstatus, sol.z, settings.dual_tol);

  std::vector<double> dy_se;  // empty -> cold-start steepest-edge weights
  // Exact replay: feed the B&B's carried steepest-edge norms verbatim (the driver
  // uses a non-empty dy_se as-is instead of reinitializing).
  if (!dy_se_override.empty()) { dy_se = dy_se_override; }
  ds::dual::status_t status;
  int out_num_updates_adv = -1;  // set by mode 0 (full driver on its own ft)
  // [SESTAT] SE debug counters were removed from basis_update_mpf_t; reset is a no-op now.

  // NOTE: this tree is MPF/LU ONLY. The explicit-inverse + Sherman-Morrison basis update
  // (`basis_update_inverse_add_t`, driven by CUOPT_SIMPLIFIED_INVERSE / CUOPT_FULL_MPF=0) does not
  // exist here — `basis_updates.hpp` never declares it — because the kernel-fission effort targets
  // the LU path only. Both the minimal and the advanced driver therefore run on the MPF `ft`
  // refactored above, outside the driver timer.

  // Time JUST the phase-2 driver call (the refactor_basis setup above is identical for both
  // the advanced and the minimal driver, so isolating the driver is the fair time comparison).
  const double driver_t0 = mo::tic();
  if (mode == 1) {
    status = ds::simplified_dual_phase2(
      lp, settings, vstatus, ft, basic_list, nonbasic_list, sol, dy_se);
    out_iters = sol.iterations;
  } else if (mode == 2) {
    status = gds::gpu_simplified_dual_phase2(
      lp, settings, vstatus, ft, basic_list, nonbasic_list, sol, dy_se);
    out_iters = sol.iterations;
  } else {
    int iter = 0;
    // MPF/LU only in this tree (see the note above): the advanced driver reuses the mpf `ft`
    // already refactored outside the driver timer. CUOPT_FULL_MPF / CUOPT_FULL_SPARSE selected the
    // inverse-add form and are inert here.
    status = ds::dual_phase2_with_advanced_basis(2, 0, initialize_basis, start_time, lp, settings,
                                                 vstatus, ft, basic_list, nonbasic_list, sol, iter,
                                                 dy_se, nullptr);
    out_num_updates_adv = ft.num_updates();
    out_iters = iter;
  }
  if (out_seconds != nullptr) { *out_seconds = mo::toc(driver_t0); }
  // [SESTAT] temp: for CPU drivers (mode 0/1) the SE counters live on this ft; for the GPU (mode 2)
  // they live in the device g_se global and are emitted separately as [SEPROF].
  if (out_se != nullptr) {
    // [SESTAT] SE debug counters were removed from basis_update_mpf_t; zero them out.
    out_se->calls   = 0;
    out_se->hyper_l = 0;
    out_se->hyper_u = 0;
    out_se->nz_in   = 0;
    out_se->nz_l    = 0;
    out_se->nz_u    = 0;
    out_se->numupd  = 0;
  }
  out_obj         = ds::compute_user_objective<int, double>(lp, sol.x);
  out_num_updates = (mode == 0) ? out_num_updates_adv : ft.num_updates();
  return status;
}

ds::simplex_solver_settings_t<int, double> make_settings(double time_limit)
{
  ds::simplex_solver_settings_t<int, double> settings;
  settings.log.log  = false;  // silence per-instance solver chatter
  settings.time_limit = time_limit;
  return settings;
}

// Turn on a full per-iteration trace (both the advanced driver's built-in log
// and the minimal driver's [MIN] log).
void enable_verbose(ds::simplex_solver_settings_t<int, double>& s)
{
  s.log.log                 = true;
  s.log.log_to_console      = true;
  s.iteration_log_frequency = 1;
  s.first_iteration_log     = 1000000000;
}

void process_instance(const raft::handle_t& handle,
                      const std::string& name,
                      const std::string& path,
                      double time_limit,
                      double dual_inf_gate,
                      bool verbose,
                      instance_result_t& r)
{
  r.name = name;

  ds::user_problem_t<int, double> up = load_user_problem(handle, path);
  r.parsed = true;

  // --- Step 2: baseline full dual simplex solve ---
  {
    auto settings = make_settings(time_limit);
    ds::lp_solution_t<int, double> sol(up.num_rows, up.num_cols);
    const double t0 = mo::tic();
    r.base_status   = ds::solve_linear_program<int, double>(up, settings, sol);
    r.base_time     = mo::toc(t0);
    r.base_obj      = sol.user_objective;
    r.base_iters    = sol.iterations;
  }

  // --- Step 4: build the post-phase-1 dual-feasible factorized basis ---
  auto settings = make_settings(time_limit);
  ds::lp_problem_t<int, double> lp(&handle, 1, 1, 1);
  std::vector<ds::variable_status_t> vstatus;
  r.setup_ok = setup_dual_feasible_basis(up, settings, lp, vstatus, r.phase1_obj, r.note);
  if (!r.setup_ok) return;
  r.m = lp.num_rows;
  r.n = lp.num_cols;

  // Advanced phase 2 from that exact basis (sanity / iteration-count reference).
  {
    auto s2 = make_settings(time_limit);
    if (verbose) {
      enable_verbose(s2);
      std::printf("\n==== FULL  dual_phase2_with_advanced_basis : %s ====\n", name.c_str());
      std::fflush(stdout);
    }
    int adv_updates = 0;
    r.adv_status    = run_phase2_from_basis(lp, vstatus, s2, /*mode advanced*/ 0, r.adv_obj,
                                         r.adv_iters, r.dual_inf, adv_updates,
                                         /*initialize_basis*/ false, /*dy_se_override*/ {},
                                         &r.adv_time);
    r.adv_ran = true;
  }

  // Gate: only feed the minimal driver a genuinely dual-feasible basis.
  if (!(r.dual_inf <= dual_inf_gate)) {
    r.note = "gate: dual_inf=" + std::to_string(r.dual_inf);
    return;
  }

  // --- Step 5 input: minimal driver from the same basis ---
  {
    auto s3 = make_settings(time_limit);
    // Isolated cutoff smoke test: TEST_NETLIB_CUTOFF=<v> feeds the minimal driver
    // a finite objective cutoff so its new CUTOFF path can be exercised on any
    // NETLIB instance (e.g. set just below the known optimum -> expect CUTOFF).
    if (const char* c = std::getenv("TEST_NETLIB_CUTOFF")) { s3.cut_off = std::atof(c); }
    if (verbose) {
      enable_verbose(s3);
      std::printf("\n==== MINIMAL simplified_dual_phase2 : %s ====\n", name.c_str());
      std::fflush(stdout);
    }
    double tmp_dual_inf = 0.0;
    r.simpl_status      = run_phase2_from_basis(lp,
                                           vstatus,
                                           s3,
                                           /*mode CPU simplified*/ 1,
                                           r.simpl_obj,
                                           r.simpl_iters,
                                           tmp_dual_inf,
                                           r.simpl_num_updates,
                                           /*initialize_basis*/ false,
                                           /*dy_se_override*/ {},
                                           &r.simpl_time);
    r.simpl_ran = true;
  }

  // --- Step 5 (GPU): GPU port of the minimal driver from the same basis ---
  // Opt-in (TEST_NETLIB_GPU=1). Off by default so a pure CPU full-vs-minimal
  // comparison doesn't pay the (single-thread, slow) GPU cost on big instances.
  if (std::getenv("TEST_NETLIB_GPU") != nullptr) {
    auto s4 = make_settings(time_limit);
    // Same isolated cutoff smoke test as the CPU minimal above (TEST_NETLIB_CUTOFF),
    // so the GPU's ported CUTOFF path can be exercised and bitwise-compared to the CPU.
    if (const char* c = std::getenv("TEST_NETLIB_CUTOFF")) { s4.cut_off = std::atof(c); }
    if (verbose) {
      enable_verbose(s4);
      std::printf("\n==== GPU     gpu_simplified_dual_phase2 : %s ====\n", name.c_str());
      std::fflush(stdout);
    }
    double tmp_dual_inf2 = 0.0;
    int gpu_updates      = 0;
    r.gpu_status         = run_phase2_from_basis(
      lp, vstatus, s4, /*mode GPU simplified*/ 2, r.gpu_obj, r.gpu_iters, tmp_dual_inf2, gpu_updates);
    r.gpu_ran = true;
  }
}

bool obj_matches(double a, double b)
{
  if (std::isnan(a) || std::isnan(b)) return false;
  return std::abs(a - b) <= 1e-6 * (1.0 + std::abs(b));
}

// ===========================================================================
// --verify-nodes mode: replay B&B node samples dumped by cuopt_cli
// (CUOPT_DUMP_NODES). Each <dir>/node_*/ holds parent.mps, child.mps, verify.log.
// We solve the parent to its optimal basis, then solve the child FROM that basis
// with both the advanced and the simplified drivers (with the recorded objective
// cutoff), and check they reproduce the node's recorded status/objective.
// ===========================================================================

// Load an all-equality MPS produced by lp_problem_t::write_mps DIRECTLY into an
// lp_problem_t, preserving column/row indices 1:1. It deliberately BYPASSES
// convert_user_problem (which re-adds one artificial per equality row, breaking
// the vstatus<->column mapping) — it only uses the tested parse ->
// optimization_problem -> user_problem front half (whose A is already CSC) and
// copies fields straight across. Assumes every constraint row is 'E'.
ds::lp_problem_t<int, double> load_lp_problem_direct(const raft::handle_t& handle,
                                                     const std::string& path)
{
  ds::user_problem_t<int, double> up = load_user_problem(handle, path);
  for (int i = 0; i < up.num_rows; ++i) {
    if (up.row_sense[i] != 'E') {
      throw std::runtime_error("load_lp_problem_direct: non-equality row in " + path +
                               " (expected an lp_problem_t::write_mps dump)");
    }
  }
  const int m  = up.num_rows;
  const int n  = up.num_cols;
  const int nz = up.A.col_start[n];
  ds::lp_problem_t<int, double> lp(&handle, m, n, nz);
  lp.A                     = up.A;  // already CSC (to_compressed_col in translate.hpp)
  lp.objective             = up.objective;
  lp.rhs                   = up.rhs;
  lp.lower                 = up.lower;
  lp.upper                 = up.upper;
  lp.obj_scale             = 1.0;  // write_mps drops obj_scale/obj_constant, so raw c'x
  lp.obj_constant          = 0.0;  // round-trips: compute_objective == the verify.log value
  lp.objective_is_integral = false;
  return lp;
}

// Solve an already-converted lp_problem_t to OPTIMAL and return its optimal
// vstatus IN lp's OWN column space (no presolve/scale — those would move the
// basis into a different space than the child). Phase 1 (slack basis ->
// dual_phase2 phase 1) builds a dual-feasible basis, then dual_phase2 phase 2
// drives it to the optimum with no cutoff.
bool solve_lp_to_optimal_vstatus(const ds::lp_problem_t<int, double>& lp,
                                 const ds::simplex_solver_settings_t<int, double>& base_settings,
                                 std::vector<ds::variable_status_t>& vstatus_out,
                                 double& obj_out,
                                 std::string& note)
{
  const raft::handle_t* h = lp.handle_ptr;
  const int m             = lp.num_rows;
  const int n             = lp.num_cols;

  ds::simplex_solver_settings_t<int, double> s = base_settings;
  s.cut_off                                     = std::numeric_limits<double>::infinity();
  s.scale_columns                               = false;

  // Phase 1: dual-feasible basis for `lp` (create_phase1_problem keeps num_cols).
  ds::lp_problem_t<int, double> phase1_problem(h, 1, 1, 1);
  ds::create_phase1_problem(lp, phase1_problem);
  std::vector<ds::variable_status_t> vstatus(phase1_problem.num_cols,
                                             ds::variable_status_t::NONBASIC_LOWER);
  int num_basic = 0;
  for (int j = phase1_problem.num_cols - 1; j >= 0; --j) {
    const int cs = phase1_problem.A.col_start[j];
    const int ce = phase1_problem.A.col_start[j + 1];
    if (ce - cs == 1 && std::abs(phase1_problem.A.x[cs]) == 1.0) {
      vstatus[j] = ds::variable_status_t::BASIC;
      num_basic++;
    }
    if (num_basic == phase1_problem.num_rows) break;
  }

  int iter1 = 0;
  ds::lp_solution_t<int, double> p1sol(phase1_problem.num_rows, phase1_problem.num_cols);
  std::vector<double> en1;
  const ds::dual::status_t st1 =
    ds::dual_phase2(1, 1, mo::tic(), phase1_problem, s, vstatus, p1sol, iter1, en1, nullptr);
  if (st1 != ds::dual::status_t::OPTIMAL) {
    note = "parent phase1: " + ds::dual_status_to_string(st1);
    return false;
  }

  // Phase 2: solve `lp` to optimality from the dual-feasible basis (no cutoff).
  int iter2 = 0;
  ds::lp_solution_t<int, double> sol(m, n);
  std::vector<double> en2;
  const ds::dual::status_t st2 =
    ds::dual_phase2(2, 0, mo::tic(), lp, s, vstatus, sol, iter2, en2, nullptr);
  if (st2 != ds::dual::status_t::OPTIMAL) {
    note = "parent phase2: " + ds::dual_status_to_string(st2);
    return false;
  }

  vstatus_out = std::move(vstatus);
  obj_out     = ds::compute_objective<int, double>(lp, sol.x);
  return true;
}

double parse_maybe_inf(const std::string& s)
{
  if (s == "inf" || s == "+inf" || s == "infinity" || s == "Inf") {
    return std::numeric_limits<double>::infinity();
  }
  if (s == "-inf" || s == "-infinity") { return -std::numeric_limits<double>::infinity(); }
  try {
    return std::stod(s);
  } catch (...) {
    return kNaN;
  }
}

struct node_verify_info_t {
  int node_id     = -1;
  int depth       = -1;
  int branch_var  = -1;
  int iterations  = 0;
  std::string branch_dir;
  std::string status;  // "OPTIMAL" | "CUTOFF"
  double incumbent = kNaN;
  double cut_off   = std::numeric_limits<double>::infinity();
  double objective = kNaN;
  bool ok          = false;
  // Warm-start / settings state (new dump format; defaults match simplex_solver_settings_t).
  bool has_state               = false;  // true if the dump carried recompute_basis + settings
  int recompute_basis          = 1;
  int refactor_frequency       = 100;
  double dual_tol              = 1e-6;
  double primal_tol            = 1e-6;
  double pivot_tol             = 1e-7;
  double tight_tol             = 1e-9;
  double zero_tol              = 1e-12;
  double steepest_edge_ratio   = 0.5;
  double steepest_edge_primal_tol = 1e-9;
  int use_bound_flip_ratio     = 1;
  int use_harris_ratio         = 0;
  int random_seed              = 0;
};

node_verify_info_t parse_verify_log(const std::string& path)
{
  node_verify_info_t v;
  std::ifstream f(path);
  if (!f.is_open()) return v;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    std::string key;
    if (!(iss >> key)) continue;
    std::string val;
    std::getline(iss, val);
    const std::size_t p = val.find_first_not_of(" \t");
    val                 = (p == std::string::npos) ? "" : val.substr(p);
    if (key == "node_id") {
      v.node_id = std::atoi(val.c_str());
    } else if (key == "depth") {
      v.depth = std::atoi(val.c_str());
    } else if (key == "branch_var") {
      v.branch_var = std::atoi(val.c_str());
    } else if (key == "branch_dir") {
      v.branch_dir = val;
    } else if (key == "incumbent") {
      v.incumbent = parse_maybe_inf(val);
    } else if (key == "cut_off") {
      v.cut_off = parse_maybe_inf(val);
    } else if (key == "status") {
      v.status = val;
    } else if (key == "objective") {
      v.objective = parse_maybe_inf(val);
    } else if (key == "iterations") {
      v.iterations = std::atoi(val.c_str());
    } else if (key == "recompute_basis") {
      v.recompute_basis = std::atoi(val.c_str());
      v.has_state       = true;
    } else if (key == "refactor_frequency") {
      v.refactor_frequency = std::atoi(val.c_str());
    } else if (key == "dual_tol") {
      v.dual_tol = parse_maybe_inf(val);
    } else if (key == "primal_tol") {
      v.primal_tol = parse_maybe_inf(val);
    } else if (key == "pivot_tol") {
      v.pivot_tol = parse_maybe_inf(val);
    } else if (key == "tight_tol") {
      v.tight_tol = parse_maybe_inf(val);
    } else if (key == "zero_tol") {
      v.zero_tol = parse_maybe_inf(val);
    } else if (key == "steepest_edge_ratio") {
      v.steepest_edge_ratio = parse_maybe_inf(val);
    } else if (key == "steepest_edge_primal_tol") {
      v.steepest_edge_primal_tol = parse_maybe_inf(val);
    } else if (key == "use_bound_flip_ratio") {
      v.use_bound_flip_ratio = std::atoi(val.c_str());
    } else if (key == "use_harris_ratio") {
      v.use_harris_ratio = std::atoi(val.c_str());
    } else if (key == "random_seed") {
      v.random_seed = std::atoi(val.c_str());
    }
  }
  v.ok = !v.status.empty();
  return v;
}

// Load a warm-start vstatus (one int per line) saved by the B&B dump hook.
bool load_vstatus_file(const std::string& path, std::vector<ds::variable_status_t>& out)
{
  std::ifstream f(path);
  if (!f) return false;
  out.clear();
  int v;
  while (f >> v) { out.push_back(static_cast<ds::variable_status_t>(v)); }
  return !out.empty();
}

// Load a saved vector of doubles (one per line) — e.g. the carried edge norms.
bool load_doubles_file(const std::string& path, std::vector<double>& out)
{
  std::ifstream f(path);
  if (!f) return false;
  out.clear();
  double v;
  while (f >> v) { out.push_back(v); }
  return !out.empty();
}

// Apply the phase-2 settings recorded in the B&B dump, so the replay uses the
// identical driver configuration (defaults are left in place for old dumps).
void apply_dumped_settings(ds::simplex_solver_settings_t<int, double>& s,
                           const node_verify_info_t& info)
{
  // TEST_NETLIB_NO_CUTOFF=1 drives every node to its own optimum instead of stopping at the
  // recorded bound. That is what isolates "does the driver get the right ANSWER" from "does it
  // fathom at the right moment" -- with the cutoff on, a node can look right for the wrong reason.
  static const bool no_cutoff = [] {
    const char* e = std::getenv("TEST_NETLIB_NO_CUTOFF");
    return e != nullptr && std::string(e) != "0";
  }();
  s.cut_off = no_cutoff ? std::numeric_limits<double>::infinity() : info.cut_off;
  if (!info.has_state) return;
  s.refactor_frequency       = info.refactor_frequency;
  s.dual_tol                 = info.dual_tol;
  s.primal_tol               = info.primal_tol;
  s.pivot_tol                = info.pivot_tol;
  s.tight_tol                = info.tight_tol;
  s.zero_tol                 = info.zero_tol;
  s.steepest_edge_ratio      = info.steepest_edge_ratio;
  s.steepest_edge_primal_tol = info.steepest_edge_primal_tol;
  s.use_bound_flip_ratio     = (info.use_bound_flip_ratio != 0);
  s.use_harris_ratio         = (info.use_harris_ratio != 0);
  s.random_seed              = info.random_seed;
}

void verify_nodes_mode(const raft::handle_t& handle, const std::string& dir, double time_limit)
{
  namespace fs = std::filesystem;
  std::vector<std::string> node_dirs;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_directory() && e.path().filename().string().rfind("node_", 0) == 0) {
      node_dirs.push_back(e.path().string());
    }
  }
  // COLLECTION ROOT: a directory of INSTANCE directories, each holding node_* + its own
  // edge_norms.txt. Walking them in ONE process instead of one process per instance removes a CUDA
  // init and a multi-GB malloc-arena allocation per instance -- 199 of them on the V7 collection.
  if (node_dirs.empty()) {
    for (const auto& e : fs::directory_iterator(dir)) {
      if (!e.is_directory()) { continue; }
      for (const auto& g : fs::directory_iterator(e.path())) {
        if (g.is_directory() && g.path().filename().string().rfind("node_", 0) == 0) {
          node_dirs.push_back(g.path().string());
        }
      }
    }
  }
  std::sort(node_dirs.begin(), node_dirs.end());

  std::printf("Found %zu node folders in %s\n\n", node_dirs.size(), dir.c_str());
  std::printf("%-22s %-8s %10s | %-16s %10s %8s %8s | %-16s %10s %8s %8s | %6s | %s\n",
              "node",
              "expect",
              "exp_obj",
              "adv",
              "adv_obj",
              "adv_it",
              "adv_s",
              "simpl",
              "simpl_obj",
              "sim_it",
              "sim_s",
              "bb_it",
              "result");
  std::fflush(stdout);

  int n_pass = 0, n_fail = 0, n_opt = 0, n_cut = 0, n_adv_ok = 0, n_sim_ok = 0;
  int n_gpu_ran = 0, n_gpu_mismatch = 0;
  for (const auto& nd : node_dirs) {
    // "<instance>/<node>" -- node folder names like node_d74_id148 REPEAT across instances, so the
    // bare name is not a key. A scorer that uses it silently merges different nodes.
    const std::string name = fs::path(nd).parent_path().filename().string() + "/" +
                             fs::path(nd).filename().string();
    node_verify_info_t info = parse_verify_log(nd + "/verify.log");
    if (!info.ok) {
      std::printf("%-22s  (missing/invalid verify.log)\n", name.c_str());
      n_fail++;
      continue;
    }
    const ds::dual::status_t expect = (info.status == "CUTOFF") ? ds::dual::status_t::CUTOFF
                                                                : ds::dual::status_t::OPTIMAL;
    if (expect == ds::dual::status_t::CUTOFF) n_cut++; else n_opt++;

    try {
      ds::lp_problem_t<int, double> parent_lp = load_lp_problem_direct(handle, nd + "/parent.mps");
      ds::lp_problem_t<int, double> child_lp  = load_lp_problem_direct(handle, nd + "/child.mps");
      if (child_lp.num_cols != parent_lp.num_cols || child_lp.num_rows != parent_lp.num_rows) {
        std::printf("%-22s  (parent/child dimension mismatch)\n", name.c_str());
        n_fail++;
        continue;
      }

      auto base = make_settings(time_limit);
      // Prefer the B&B's EXACT warm-start basis (child_vstatus.txt, saved by the
      // dump hook) — replaying from it reproduces the recorded outcome. If it is
      // absent (older dumps), fall back to re-solving the parent to an optimal
      // basis, which may be a DIFFERENT optimal vertex (degenerate LPs).
      std::vector<ds::variable_status_t> vstatus_warm;
      std::vector<double> edge_norms;   // carried steepest-edge norms (exact replay)
      bool init_basis       = false;
      const char* basis_src = "saved-basis";
      const bool have_saved = load_vstatus_file(nd + "/child_vstatus.txt", vstatus_warm) &&
                              static_cast<int>(vstatus_warm.size()) == child_lp.num_cols;
      if (have_saved) {
        // Exact replay: also feed the carried edge norms + the recompute_basis flag
        // so dual_phase2_with_advanced_basis gets the identical warm-start state.
        // The steepest-edge seed is a single shared vector per instance. In collection-root mode
        // `dir` is the directory ABOVE all instances, so derive the file from this node's parent;
        // otherwise every collection replay silently falls back to `saved(no-en)`. Fall back to
        // the legacy per-node file for older dumps.
        const std::string instance_edge_norms =
          (fs::path(nd).parent_path() / "edge_norms.txt").string();
        if ((!load_doubles_file(instance_edge_norms, edge_norms) &&
             !load_doubles_file(nd + "/child_edge_norms.txt", edge_norms)) ||
            static_cast<int>(edge_norms.size()) != child_lp.num_cols) {
          edge_norms.clear();
        }
        init_basis = info.has_state ? (info.recompute_basis != 0) : false;
        basis_src  = edge_norms.empty() ? "saved(no-en)"
                                        : (init_basis ? "saved+en,refac" : "saved+en,reuse");
      } else {
        vstatus_warm.clear();
        double parent_obj = kNaN;
        std::string note;
        if (!solve_lp_to_optimal_vstatus(parent_lp, base, vstatus_warm, parent_obj, note)) {
          std::printf("%-22s  (parent solve failed: %s)\n", name.c_str(), note.c_str());
          n_fail++;
          continue;
        }
        basis_src = "parent-resolve";
      }

      // Run advanced (mode 0) and simplified (mode 1) on the child from the
      // warm-start basis, with the recorded objective cutoff + settings + edge norms.
      double adv_obj = kNaN, sim_obj = kNaN, dinf = 0.0;
      int adv_it = 0, sim_it = 0, upd = 0;
      double adv_sec = 0.0, sim_sec = 0.0;
      se_dbg_t se_adv, se_sim;  // [SESTAT] temp
      // [SESTAT] TEMP debug (remove after diagnosis): per-node marker so the interleaved CPU
      // [SESTAT] lines (full/min, below) and the GPU device [SEPROF] line (emitted during the
      // mode-2 kernel when CUOPT_GPU_PROFILE=1) can be grouped by node.
      // TEST_NETLIB_VERBOSE=1 turns on both drivers' per-iteration bit traces
      // ([ADV] from advanced, [CMP] from simplified) so the exact iteration where the
      // two pivot sequences diverge can be located. Use on a single-node dir.
      const bool verbose_trace = std::getenv("TEST_NETLIB_VERBOSE") != nullptr;
      // TEST_NETLIB_GPU_ONLY=1: skip BOTH CPU legs. The full driver's answer is the golden
      // reference and does not change between GPU experiments, so re-solving it on every run is
      // pure cost -- freeze it once (TEST_NETLIB_REF_DUMP) and compare against that file instead.
      static const bool gpu_only = env_on("TEST_NETLIB_GPU_ONLY");
      auto s_adv = make_settings(time_limit);
      apply_dumped_settings(s_adv, info);
      if (verbose_trace) { enable_verbose(s_adv); std::printf(">>> ADV %s\n", name.c_str()); }
      const ds::dual::status_t adv_st = gpu_only ? ds::dual::status_t::UNSET : run_phase2_from_basis(
        child_lp, vstatus_warm, s_adv, /*advanced*/ 0, adv_obj, adv_it, dinf, upd, init_basis,
        edge_norms, &adv_sec, &se_adv);
      auto s_sim = make_settings(time_limit);
      apply_dumped_settings(s_sim, info);
      if (verbose_trace) { enable_verbose(s_sim); std::printf(">>> SIM %s\n", name.c_str()); }
      const ds::dual::status_t sim_st = gpu_only ? ds::dual::status_t::UNSET : run_phase2_from_basis(
        child_lp, vstatus_warm, s_sim, /*simplified*/ 1, sim_obj, sim_it, dinf, upd, false,
        edge_norms, &sim_sec, &se_sim);
      // GPU parity check (TEST_NETLIB_GPU=1): run the GPU simplified driver (mode 2) from
      // the identical warm start and compare status/iters/objective against the CPU
      // simplified driver. With every CPU opt-in feature OFF (no CUOPT_SIMPLIFIED_* flags)
      // the CPU and GPU share the same feature set and must match exactly.
      const bool run_gpu       = std::getenv("TEST_NETLIB_GPU") != nullptr;
      double gpu_obj           = kNaN;
      int gpu_it               = 0;
      double gpu_sec           = 0.0;
      ds::dual::status_t gpu_st = ds::dual::status_t::OPTIMAL;
      bool gpu_match           = true;
      if (run_gpu) {
        auto s_gpu = make_settings(time_limit);
        apply_dumped_settings(s_gpu, info);
        if (verbose_trace) { enable_verbose(s_gpu); std::printf(">>> GPU %s\n", name.c_str()); }
        double gdinf = 0.0;
        int gupd     = 0;
        gpu_st       = run_phase2_from_basis(child_lp,
                                       vstatus_warm,
                                       s_gpu,
                                       /*gpu*/ 2,
                                       gpu_obj,
                                       gpu_it,
                                       gdinf,
                                       gupd,
                                       false,
                                       edge_norms,
                                       &gpu_sec);
        gpu_match = (gpu_st == sim_st) && (gpu_it == sim_it) && (gpu_obj == sim_obj);
        n_gpu_ran++;
        if (!gpu_match) n_gpu_mismatch++;
      }

      // Did a driver reproduce the B&B decision for this node?
      //  - expected OPTIMAL (node kept): require OPTIMAL at the recorded optimum.
      //  - expected CUTOFF (node pruned): the node's LP bound can't beat the
      //    incumbent, so it is pruned. Accept CUTOFF (early prune) OR OPTIMAL with
      //    obj >= cut_off (solved through but still fathomed-by-bound) — both are
      //    the SAME prune decision. The exact objective at cutoff detection is
      //    path-dependent (long dual steps), so it is not required to match.
      auto driver_reproduced = [&](ds::dual::status_t st, double obj) -> bool {
        if (expect == ds::dual::status_t::OPTIMAL) {
          return st == ds::dual::status_t::OPTIMAL && obj_matches(obj, info.objective);
        }
        if (st == ds::dual::status_t::CUTOFF) return true;
        if (st == ds::dual::status_t::OPTIMAL) {
          return obj >= info.cut_off - 1e-6 * (1.0 + std::abs(info.cut_off));
        }
        return false;
      };
      const bool adv_ok = driver_reproduced(adv_st, adv_obj);
      const bool sim_ok = driver_reproduced(sim_st, sim_obj);
      const bool pass   = adv_ok && sim_ok;
      if (adv_ok) n_adv_ok++;
      if (sim_ok) n_sim_ok++;

      std::printf("%-22s %-8s %10.6g | %-16s %10.6g %8d %8.4f | %-16s %10.6g %8d %8.4f | %6d | %s [%s]",
                  name.c_str(),
                  info.status.c_str(),
                  info.objective,
                  ds::dual_status_to_string(adv_st).c_str(),
                  adv_obj,
                  adv_it,
                  adv_sec,
                  ds::dual_status_to_string(sim_st).c_str(),
                  sim_obj,
                  sim_it,
                  sim_sec,
                  info.iterations,
                  pass ? "PASS" : "FAIL",
                  basis_src);
      if (run_gpu) {
        std::printf(" || GPU %-10s %10.6g %8d %8.4f [%s]",
                    ds::dual_status_to_string(gpu_st).c_str(),
                    gpu_obj,
                    gpu_it,
                    gpu_sec,
                    gpu_match ? "cpu==gpu" : "CPU!=GPU");
      }
      // Full-precision per-node GPU result, for comparing the SEQUENTIAL <<<1,32>>> path against
      // the BATCH <<<N,32>>> path node-for-node. The table above prints %10.6g, far too coarse to
      // detect a batching-induced divergence. %.17g round-trips a double exactly, so a text diff
      // of two [NODEDUMP] streams IS a bitwise comparison. TEST_NETLIB_NODE_DUMP=1.
      if (run_gpu && env_on("TEST_NETLIB_NODE_DUMP")) {
        std::printf("\n[NODEDUMP] %-44s %-16s it=%-6d obj=%.17g",
                    name.c_str(), ds::dual_status_to_string(gpu_st).c_str(), gpu_it, gpu_obj);
      }
      // Full-precision CPU results. The summary table above prints %10.6g, which is far too coarse
      // to build a golden reference from -- an LP optimum of -278045.4060865763 lands as -278045.
      // Emitted for BOTH CPU drivers so a golden can be derived from the full driver, independently
      // of the simplified driver the GPU is a port of.
      if (env_on("TEST_NETLIB_REF_DUMP")) {
        std::printf("\n[REFDUMP] %-44s adv %-16s it=%-6d obj=%.17g | sim %-16s it=%-6d obj=%.17g",
                    name.c_str(),
                    ds::dual_status_to_string(adv_st).c_str(), adv_it, adv_obj,
                    ds::dual_status_to_string(sim_st).c_str(), sim_it, sim_obj);
      }
      std::printf("\n");
      std::fflush(stdout);
      if (pass) n_pass++; else n_fail++;
    } catch (const std::exception& ex) {
      std::printf("%-22s  (exception: %s)\n", name.c_str(), ex.what());
      n_fail++;
    }
  }

  std::printf("\n==== verify-nodes summary ====\n");
  std::printf("node folders                         : %zu\n", node_dirs.size());
  std::printf("  expected OPTIMAL                    : %d\n", n_opt);
  std::printf("  expected CUTOFF                     : %d\n", n_cut);
  std::printf("Milestone 1 - advanced reproduced    : %d / %zu\n", n_adv_ok, node_dirs.size());
  std::printf("Milestone 2 - simplified reproduced  : %d / %zu\n", n_sim_ok, node_dirs.size());
  std::printf("PASS (both)                          : %d\n", n_pass);
  std::printf("FAIL                                 : %d\n", n_fail);
  if (n_gpu_ran > 0) {
    std::printf("GPU parity (CPU-simpl == GPU: status+iters+obj) : %d / %d  (%d mismatch)\n",
                n_gpu_ran - n_gpu_mismatch, n_gpu_ran, n_gpu_mismatch);
  }
  if (n_cut == 0) {
    std::printf("WARNING: no CUTOFF nodes in the sample — the cutoff path was not exercised.\n");
  }
  if (n_opt == 0) {
    std::printf("WARNING: no OPTIMAL nodes in the sample.\n");
  }
}

// GPU-only correctness sweep through the SAME resumable batch state machine used by branch and
// bound. Nodes are grouped by instance, then split whenever their immutable A/rhs/objective block
// differs (cut-expanded B&B nodes from one instance can have different shapes). A chunk is one
// multi-block launch sequence: seats independently stop for host refactorization, carry their
// complete state back, and relaunch until terminal.
//
// CPU-full references are intentionally not recomputed here. Generate them once with
// --verify-nodes + TEST_NETLIB_REF_DUMP=1, then concatenate that frozen log with this mode's
// [NODEDUMP] stream for score_goal_verify.py. That makes repeated GPU experiments cheap without
// weakening the golden reference.
void verify_nodes_batch_mode(const raft::handle_t& handle,
                             const std::string& dir,
                             double time_limit,
                             int requested_seats)
{
  namespace fs = std::filesystem;
  using batch_node_t = mip::gpu_batch_node_t<int, double>;

  const int seats = std::max(1, requested_seats);
  std::map<std::string, std::vector<std::string>> by_instance;
  if (fs::path(dir).filename().string().rfind("node_", 0) == 0) {
    by_instance[fs::path(dir).parent_path().string()].push_back(dir);
  } else {
    for (const auto& e : fs::directory_iterator(dir)) {
      if (!e.is_directory()) { continue; }
      if (e.path().filename().string().rfind("node_", 0) == 0) {
        by_instance[dir].push_back(e.path().string());
        continue;
      }
      for (const auto& n : fs::directory_iterator(e.path())) {
        if (n.is_directory() && n.path().filename().string().rfind("node_", 0) == 0) {
          by_instance[e.path().string()].push_back(n.path().string());
        }
      }
    }
  }
  for (auto& [instance, nodes] : by_instance) { std::sort(nodes.begin(), nodes.end()); }

  rmm::cuda_stream stream;
  int total = 0, direct = 0, fallback = 0, skipped = 0, cold_edge_norms = 0;
  double kernel_s = 0.0;

  for (const auto& [instance, node_dirs] : by_instance) {
    std::vector<double> instance_edge_norms;
    load_doubles_file(instance + "/edge_norms.txt", instance_edge_norms);

    size_t begin = 0;
    while (begin < node_dirs.size()) {
      std::vector<batch_node_t> batch;
      std::vector<std::string> names;
      batch.reserve(static_cast<size_t>(seats));
      names.reserve(static_cast<size_t>(seats));
      std::unique_ptr<ds::lp_problem_t<int, double>> shared_lp;

      size_t p = begin;
      while (p < node_dirs.size() && batch.size() < static_cast<size_t>(seats)) {
        const std::string& nd = node_dirs[p];
        const std::string name = fs::path(instance).filename().string() + "/" +
                                 fs::path(nd).filename().string();
        const node_verify_info_t info = parse_verify_log(nd + "/verify.log");
        if (!info.ok) {
          std::printf("[BATCHSKIP] %s invalid verify.log\n", name.c_str());
          ++skipped;
          ++p;
          continue;
        }

        try {
          ds::lp_problem_t<int, double> child =
            load_lp_problem_direct(handle, nd + "/child.mps");
          std::vector<ds::variable_status_t> vstatus;
          if (!load_vstatus_file(nd + "/child_vstatus.txt", vstatus) ||
              static_cast<int>(vstatus.size()) != child.num_cols) {
            std::printf("[BATCHSKIP] %s missing/invalid saved basis\n", name.c_str());
            ++skipped;
            ++p;
            continue;
          }

          if (shared_lp == nullptr) {
            shared_lp = std::make_unique<ds::lp_problem_t<int, double>>(child);
          } else {
            // Per-instance batching is valid only when the immutable LP block is genuinely shared.
            // Refuse a malformed collection instead of solving a node against another node's A.
            const bool same_immutable =
              child.num_rows == shared_lp->num_rows && child.num_cols == shared_lp->num_cols &&
              child.A.col_start == shared_lp->A.col_start && child.A.i == shared_lp->A.i &&
              child.A.x == shared_lp->A.x && child.rhs == shared_lp->rhs &&
              child.objective == shared_lp->objective;
            if (!same_immutable) {
              // Finish the current homogeneous chunk. Do not consume this node: it becomes the
              // first seat of the next batch, with its own shared immutable block.
              break;
            }
          }

          std::vector<double> node_edge_norms = instance_edge_norms;
          if (static_cast<int>(node_edge_norms.size()) != child.num_cols) {
            node_edge_norms.clear();
            load_doubles_file(nd + "/child_edge_norms.txt", node_edge_norms);
          }
          if (static_cast<int>(node_edge_norms.size()) != child.num_cols) {
            node_edge_norms.clear();
            ++cold_edge_norms;
            std::printf("[BATCHCOLD] %s no valid edge norms for n=%d; initialize on device\n",
                        name.c_str(), child.num_cols);
          }

          batch.emplace_back(child.num_rows, child.num_cols);
          auto& node   = batch.back();
          node.lower   = std::move(child.lower);
          node.upper   = std::move(child.upper);
          node.settings = make_settings(time_limit);
          apply_dumped_settings(node.settings, info);
          node.vstatus = std::move(vstatus);
          node.dy_se   = std::move(node_edge_norms);
          names.push_back(name);
          ++p;
        } catch (const std::exception& ex) {
          std::printf("[BATCHSKIP] %s exception: %s\n", name.c_str(), ex.what());
          ++skipped;
          ++p;
        }
      }

      begin = p;

      if (batch.empty()) { continue; }
      auto* ro_cache = gds::gpu_ro_cache_create<int, double>();
      const double t0 = mo::tic();
      mip::gpu_solve_nodes_batch<int, double>(*shared_lp, batch, stream.view(), ro_cache);
      kernel_s += mo::toc(t0);
      gds::gpu_ro_cache_destroy<int, double>(ro_cache);

      for (size_t k = 0; k < batch.size(); ++k) {
        auto& node = batch[k];
        ++total;
        ds::dual::status_t status = node.status;
        double obj = kNaN;
        if (node.solved) {
          ++direct;
          obj = ds::compute_user_objective<int, double>(*shared_lp, node.sol.x);
        } else {
          // No device answer is a safe CPU handoff, never a terminal claim about the subtree.
          status = ds::dual::status_t::NUMERICAL;
          ++fallback;
          std::printf("[BATCHDROP] %s reason=%d\n", names[k].c_str(),
                      static_cast<int>(node.drop));
        }
        std::printf("[NODEDUMP] %-44s %-16s it=%-6d obj=%.17g\n",
                    names[k].c_str(), ds::dual_status_to_string(status).c_str(),
                    node.sol.iterations, obj);
      }
      std::fflush(stdout);
    }
  }

  std::printf("\n==== verify-nodes-batch summary ====\n");
  std::printf("nodes=%d direct=%d fallback=%d skipped=%d cold_edge_norms=%d seats=%d wall_s=%.6f\n",
              total, direct, fallback, skipped, cold_edge_norms, seats, kernel_s);
}

// ===========================================================================
// --benchmark-parallel mode: throughput of the GPU multi-block batch vs the CPU
// OpenMP-across-problems path. Replicate the dumped B&B nodes R times (default 10)
// to act as many independent problems, solve them all on the CPU (OpenMP, one
// thread per problem) and on the GPU (one block per problem, single launch),
// measure total runtime for each, and verify every result is bit-for-bit equal to
// a single-thread sequential reference (embarrassingly parallel => parallelism
// must not change status / iterations / objective).
// ===========================================================================

// True iff two doubles are bit-identical (the hard equality the benchmark asserts).
bool bits_equal(double a, double b)
{
  std::uint64_t x, y;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

// True iff a solve reproduces the sequential reference bit-for-bit. Status and objective must
// always match exactly; iteration count is only asserted for OPTIMAL/CUTOFF because the driver
// leaves sol.iterations UNSET on a mid-loop NUMERICAL / unbounded / limit terminal return (so on
// those the CPU reports it=0 while the GPU reports the real count — same solve, reporting artifact).
bool reproduces_ref(ds::dual::status_t st, int it, double obj, ds::dual::status_t rst, int rit,
                    double robj)
{
  if (st != rst) return false;
  if (!bits_equal(obj, robj)) return false;
  const bool clean_exit =
    (rst == ds::dual::status_t::OPTIMAL || rst == ds::dual::status_t::CUTOFF);
  if (clean_exit && it != rit) return false;
  return true;
}

// Everything a single solve needs, prepared from a warm-start basis: the same setup
// run_phase2_from_basis performs (get_basis_from_vstatus + refactor_basis +
// compute_duals) plus the carried steepest-edge norms as dy_se. Held on the heap so a
// vector of N of them never has to move a basis_update_mpf_t.
struct prepared_problem_t {
  std::vector<ds::variable_status_t> vstatus;
  std::vector<int> basic_list;
  std::vector<int> nonbasic_list;
  ds::basis_update_mpf_t<int, double> ft;
  ds::lp_solution_t<int, double> sol;
  std::vector<double> dy_se;
  prepared_problem_t(int m, int n, int refactor_frequency)
    : basic_list(m), ft(m, refactor_frequency), sol(m, n)
  {
  }
};

std::unique_ptr<prepared_problem_t> prepare_from_basis(
  const ds::lp_problem_t<int, double>& lp,
  const std::vector<ds::variable_status_t>& vstatus_in,
  const ds::simplex_solver_settings_t<int, double>& settings,
  const std::vector<double>& edge_norms)
{
  const int m = lp.num_rows;
  const int n = lp.num_cols;
  auto p     = std::make_unique<prepared_problem_t>(m, n, settings.refactor_frequency);
  p->vstatus = vstatus_in;
  std::vector<int> superbasic_list;
  ds::get_basis_from_vstatus<int>(m, p->vstatus, p->basic_list, p->nonbasic_list, superbasic_list);
  const double start_time = mo::tic();
  p->ft.refactor_basis(lp.A, settings, lp.lower, lp.upper, start_time, p->basic_list,
                       p->nonbasic_list, p->vstatus);
  compute_duals(lp, p->ft, p->basic_list, p->nonbasic_list, p->sol.y, p->sol.z);
  if (!edge_norms.empty()) { p->dy_se = edge_norms; }
  return p;
}

// Device static-buffer bytes the batch will cudaMalloc for this problem (read-only
// factors/matrix + mutable state), EXCLUDING the in-kernel S_ heap. Mirrors
// gpu_phase2_detail::problem_static_bytes so the up-front reject estimate matches what
// the batch wrapper actually allocates (the wrapper re-checks authoritatively).
size_t estimate_static_bytes(const ds::lp_problem_t<int, double>& lp,
                             const ds::basis_update_mpf_t<int, double>& ft)
{
  const size_t m  = lp.num_rows;
  const size_t n  = lp.num_cols;
  const auto& L0  = ft.L0();
  const auto& U0  = ft.U0();
  const auto& L0t = ft.L0_transpose();
  const auto& U0t = ft.U0_transpose();
  auto bi = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(int); };
  auto bd = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(double); };
  size_t b = 0;
  b += bi(lp.A.col_start.size()) + bi(lp.A.nnz()) + bd(lp.A.nnz());
  b += bd(m) + 3 * bd(n);
  b += bi(m + 1) + bi(lp.A.nnz()) + bd(lp.A.nnz());  // Arow CSR: always uploaded now
  b += bi(L0.col_start.size())  + bi(L0.nnz())  + bd(L0.nnz());
  b += bi(U0.col_start.size())  + bi(U0.nnz())  + bd(U0.nnz());
  b += bi(L0t.col_start.size()) + bi(L0t.nnz()) + bd(L0t.nnz());
  b += bi(U0t.col_start.size()) + bi(U0t.nnz()) + bd(U0t.nnz());
  b += bi(m) + bi(m) + bi(2 * m) + bd(m) + m * sizeof(unsigned char);  // reach marks
  b += n * sizeof(ds::variable_status_t) + bi(m) + bi(n) + bd(n) + bd(m) + bd(n) + bd(n);
  return b;
}

// Available host RAM in bytes (Linux /proc/meminfo MemAvailable). 0 if unreadable.
size_t host_mem_available_bytes()
{
  std::ifstream f("/proc/meminfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      std::istringstream iss(line);
      std::string key, unit;
      size_t kb = 0;
      iss >> key >> kb >> unit;
      return kb * 1024;
    }
  }
  return 0;
}

void benchmark_parallel_mode(const raft::handle_t& handle, const std::string& dir, int replicas,
                             double time_limit)
{
  namespace fs = std::filesystem;
  constexpr double GB          = 1.0 / (1024.0 * 1024.0 * 1024.0);
  // The kernel does no in-kernel malloc (every buffer is preallocated by the host and budgeted by
  // problem_static_bytes), so the VRAM pre-flight below is static bytes only.

  // One prepared, replicated problem instance.
  struct unique_node_t {
    std::string name;
    ds::lp_problem_t<int, double> lp;
    std::vector<ds::variable_status_t> vstatus;
    std::vector<double> edge_norms;
    ds::simplex_solver_settings_t<int, double> settings;
    ds::dual::status_t ref_status = ds::dual::status_t::UNSET;
    int ref_iters                 = 0;
    double ref_obj                = kNaN;
    int m                         = 0;
    int depth                     = -1;
    // Iteration count recorded in verify.log by the ORIGINAL B&B dump. Used ONLY as the heavy-first
    // ordering key. It is a file on disk, so unlike ref_iters (which is measured by this run's CPU
    // driver) no env var, driver config or reprice setting can change it -- the block layout is
    // therefore identical across every configuration being compared. See the sort below.
    int verify_iters              = 0;
    size_t static_bytes           = 0;
    explicit unique_node_t(ds::lp_problem_t<int, double>&& lp_in) : lp(std::move(lp_in)) {}
  };

  // ---- Collect (node dir, instance dir) pairs. Supports a flat dir of node_* (instance == dir)
  //      AND a collection of instance subdirs each holding node_* + its own edge_norms.txt. ----
  std::vector<std::pair<std::string, std::string>> node_dirs;  // (node path, instance dir)
  for (const auto& e : fs::directory_iterator(dir)) {
    if (!e.is_directory()) continue;
    if (e.path().filename().string().rfind("node_", 0) == 0) {
      node_dirs.emplace_back(e.path().string(), dir);
    } else {
      for (const auto& s : fs::directory_iterator(e.path())) {
        if (s.is_directory() && s.path().filename().string().rfind("node_", 0) == 0) {
          node_dirs.emplace_back(s.path().string(), e.path().string());
        }
      }
    }
  }
  std::sort(node_dirs.begin(), node_dirs.end());

  // Steepest-edge seed is one shared vector per instance dir; cache per instance.
  std::map<std::string, std::vector<double>> edge_cache;
  auto instance_edge_norms = [&](const std::string& inst) -> const std::vector<double>& {
    auto it = edge_cache.find(inst);
    if (it == edge_cache.end()) {
      std::vector<double> v;
      load_doubles_file(inst + "/edge_norms.txt", v);
      it = edge_cache.emplace(inst, std::move(v)).first;
    }
    return it->second;
  };

  std::vector<std::unique_ptr<unique_node_t>> nodes;
  int skipped = 0;
  for (const auto& [nd, inst] : node_dirs) {
    const std::string name  = fs::path(inst).filename().string() + "/" + fs::path(nd).filename().string();
    node_verify_info_t info = parse_verify_log(nd + "/verify.log");
    if (!info.ok) { skipped++; continue; }
    try {
      ds::lp_problem_t<int, double> child_lp = load_lp_problem_direct(handle, nd + "/child.mps");
      std::vector<ds::variable_status_t> vstatus_warm;
      if (!(load_vstatus_file(nd + "/child_vstatus.txt", vstatus_warm) &&
            static_cast<int>(vstatus_warm.size()) == child_lp.num_cols)) {
        std::printf("  skip %s (no saved warm-start basis)\n", name.c_str());
        skipped++;
        continue;
      }
      const std::vector<double>& inst_en = instance_edge_norms(inst);
      std::vector<double> edge_norms;
      if (static_cast<int>(inst_en.size()) == child_lp.num_cols) { edge_norms = inst_en; }
      auto settings = make_settings(time_limit);
      apply_dumped_settings(settings, info);

      auto un        = std::make_unique<unique_node_t>(std::move(child_lp));
      un->name       = name;
      un->vstatus    = std::move(vstatus_warm);
      un->edge_norms = std::move(edge_norms);
      un->settings   = settings;
      un->m          = un->lp.num_rows;
      un->depth      = info.depth;
      un->verify_iters = info.iterations;
      nodes.push_back(std::move(un));
    } catch (const std::exception& ex) {
      std::printf("  skip %s (exception: %s)\n", name.c_str(), ex.what());
      skipped++;
    }
  }

  const int uc = static_cast<int>(nodes.size());
  if (uc == 0) {
    std::printf("No usable nodes with a saved basis in %s — nothing to benchmark.\n", dir.c_str());
    return;
  }
  std::printf("Loaded %d unique nodes (%d skipped) from %s; replicating x%d => %d problems.\n\n",
              uc, skipped, dir.c_str(), replicas, uc * replicas);

  // ---- Sequential reference (single thread) + per-node size measurement ----
  for (auto& un : nodes) {
    // Fresh factorization purely to measure the exact device static-buffer footprint.
    auto p            = prepare_from_basis(un->lp, un->vstatus, un->settings, un->edge_norms);
    un->static_bytes  = estimate_static_bytes(un->lp, p->ft);
    // Reference result via the SAME function the CPU benchmark uses (bit-identical).
    auto s            = un->settings;
    double obj = kNaN, dinf = 0.0;
    int iters = 0, upd = 0;
    un->ref_status = run_phase2_from_basis(un->lp, un->vstatus, s, /*mode CPU simplified*/ 1, obj,
                                           iters, dinf, upd, /*init*/ false, un->edge_norms);
    un->ref_iters = iters;
    un->ref_obj   = obj;
  }

  // ---- Heavy-first order (anti-tailing): sort unique by verify.log iters desc; each unique's
  //      `replicas` copies are consecutive, so the heaviest problems get the low blockIdx /
  //      earliest OpenMP grab (LPT). Depth does NOT predict iters (r = -0.29 over this collection),
  //      so a measured iteration count is the right key.
  //
  //      The key is verify_iters (recorded in verify.log by the B&B dump), NOT ref_iters (measured
  //      by THIS run's CPU driver). That distinction is load-bearing: ref_iters changes with the CPU
  //      reprice setting, which reordered the sort, which moved nodes to different blockIdx. With
  //      ~100 co-resident single-warp blocks the kernel time is the makespan of one straggler, so
  //      relocating it against a different set of co-residents moved the reported kernel time by up
  //      to 8% with ZERO change in GPU work (per-node results byte-identical, same clocks). A
  //      GPU-only A/B was therefore silently comparing two different block layouts. verify_iters is
  //      a file on disk and cannot vary with any run-time setting.
  //
  //      Ties keep the original discovery order (std::stable_sort), so the layout is reproducible
  //      even when several nodes share an iteration count.
  std::vector<int> order(uc);
  for (int i = 0; i < uc; ++i) { order[i] = i; }
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return nodes[a]->verify_iters > nodes[b]->verify_iters; });
  std::vector<int> list;
  list.reserve(static_cast<size_t>(uc) * replicas);
  for (int rank = 0; rank < uc; ++rank) {
    for (int r = 0; r < replicas; ++r) { list.push_back(order[rank]); }
  }
  const size_t Ntot = list.size();

  // ---- Pre-flight memory estimate (fail loud BEFORE building anything big) ----
  size_t vram_static = 0, host_need = 0;
  for (int u : list) {
    vram_static += nodes[u]->static_bytes;
    host_need   += static_cast<size_t>(nodes[u]->static_bytes * 1.3);  // host copies + overhead
  }
  const size_t vram_need = vram_static;
  size_t free_vram = 0, total_vram = 0;
  cudaMemGetInfo(&free_vram, &total_vram);
  const size_t free_ram = host_mem_available_bytes();
  std::printf("Memory estimate for %zu problems:\n", Ntot);
  std::printf("  VRAM  needed ~%.2f GB (static) | free %.2f GB\n", vram_need * GB, free_vram * GB);
  std::printf("  Host  needed ~%.2f GB | available %.2f GB\n\n", host_need * GB, free_ram * GB);
  bool reject = false;
  std::string why;
  if (vram_need > static_cast<size_t>(0.92 * static_cast<double>(free_vram))) {
    reject = true;
    why    = "insufficient VRAM (need " + std::to_string(vram_need * GB) + " GB, free " +
          std::to_string(free_vram * GB) + " GB)";
  }
  if (free_ram > 0 && host_need > static_cast<size_t>(0.90 * static_cast<double>(free_ram))) {
    reject = true;
    why    = "insufficient host RAM (need " + std::to_string(host_need * GB) + " GB, available " +
          std::to_string(free_ram * GB) + " GB)";
  }
  if (reject) {
    std::printf("BENCHMARK REJECTED: %s — reduce --replicas or use a larger GPU/host.\n",
                why.c_str());
    return;
  }

  // ---- CPU benchmark: OpenMP across problems (each thread refactors + solves its own) ----
  std::vector<ds::dual::status_t> cpu_status(Ntot, ds::dual::status_t::UNSET);
  std::vector<int> cpu_iters(Ntot, 0);
  std::vector<double> cpu_obj(Ntot, kNaN);
  std::vector<double> cpu_driver_s(Ntot, 0.0);
  // Self-report the CPU leg, mirroring the GPU's [CONFIG] line. A run that states its own
  // configuration cannot be mislabeled afterwards, which is the whole point.
  std::printf("[CONFIG] leg=CPU(reference) problems=%zu\n", Ntot);
  const double cpu_t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
  for (size_t k = 0; k < Ntot; ++k) {
    const int u = list[k];
    auto s      = nodes[u]->settings;
    double obj = kNaN, dinf = 0.0, dsec = 0.0;
    int iters = 0, upd = 0;
    cpu_status[k] = run_phase2_from_basis(nodes[u]->lp, nodes[u]->vstatus, s, /*mode*/ 1, obj, iters,
                                          dinf, upd, /*init*/ false, nodes[u]->edge_norms, &dsec);
    cpu_iters[k]    = iters;
    cpu_obj[k]      = obj;
    cpu_driver_s[k] = dsec;
  }
  const double cpu_total_s = omp_get_wtime() - cpu_t0;
  double cpu_driver_sum    = 0.0;
  for (size_t k = 0; k < Ntot; ++k) { cpu_driver_sum += cpu_driver_s[k]; }

  int cpu_mismatch = 0;
  for (size_t k = 0; k < Ntot; ++k) {
    const int u = list[k];
    if (!reproduces_ref(cpu_status[k], cpu_iters[k], cpu_obj[k], nodes[u]->ref_status,
                        nodes[u]->ref_iters, nodes[u]->ref_obj)) {
      cpu_mismatch++;
    }
  }


  // ---- GPU benchmark: host-setup all N (OpenMP), then ONE multi-block launch ----
  std::vector<std::unique_ptr<prepared_problem_t>> prepared(Ntot);
  const double gpu_setup_t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
  for (size_t k = 0; k < Ntot; ++k) {
    const int u = list[k];
    prepared[k] =
      prepare_from_basis(nodes[u]->lp, nodes[u]->vstatus, nodes[u]->settings, nodes[u]->edge_norms);
  }
  const double gpu_setup_s = omp_get_wtime() - gpu_setup_t0;

  std::vector<gds::gpu_batch_item<int, double>> items(Ntot);
  for (size_t k = 0; k < Ntot; ++k) {
    const int u          = list[k];
    items[k].lp          = &nodes[u]->lp;
    items[k].settings    = &nodes[u]->settings;
    items[k].vstatus     = &prepared[k]->vstatus;
    items[k].ft          = &prepared[k]->ft;
    items[k].basic_list  = &prepared[k]->basic_list;
    items[k].nonbasic_list = &prepared[k]->nonbasic_list;
    items[k].sol         = &prepared[k]->sol;
    items[k].dy_se       = &prepared[k]->dy_se;
  }

  std::vector<ds::dual::status_t> gpu_status;
  gds::gpu_batch_timing timing;
  gds::gpu_simplified_dual_phase2_batch<int, double>(items, gpu_status, timing);
  if (timing.rejected) {
    std::printf("BENCHMARK REJECTED (GPU pre-flight): %s\n", timing.message.c_str());
    return;
  }

  // Timeout-aware verification: a completed GPU solve must be bit-exact to the reference; a
  // GPU TIME_LIMIT (single-thread too slow on a heavy node) is a GPU-speed limitation, NOT a
  // batching corruption, and is counted separately (its result is wall-clock-dependent, so it
  // cannot be bit-compared). Any other divergence is a true mismatch (a real bug).
  int gpu_exact = 0, gpu_timeout = 0, gpu_true_mismatch = 0;
  // Distinct [TRUE MISMATCH] lines, in first-seen order, with how many problems produced each.
  std::vector<std::pair<std::string, int>> mismatch_lines;
  std::unordered_map<std::string, size_t>  mismatch_index;
  for (size_t k = 0; k < Ntot; ++k) {
    const int u       = list[k];
    const double gobj = ds::compute_user_objective<int, double>(nodes[u]->lp, prepared[k]->sol.x);
    // Same full-precision dump as the sequential path so the two streams diff bitwise. NOTE: the
    // built-in "bit-exact vs ref" counter compares against the CPU SIMPLIFIED reference, not against
    // the sequential GPU -- so it cannot answer "does batching change the GPU result".
    if (env_on("TEST_NETLIB_NODE_DUMP")) {
      std::printf("[NODEDUMP] %-44s %-16s it=%-6d obj=%.17g\n",
                  nodes[u]->name.c_str(), ds::dual_status_to_string(gpu_status[k]).c_str(),
                  prepared[k]->sol.iterations, gobj);
    }
    const bool exact  = reproduces_ref(gpu_status[k], prepared[k]->sol.iterations, gobj,
                                      nodes[u]->ref_status, nodes[u]->ref_iters, nodes[u]->ref_obj);
    if (exact) {
      gpu_exact++;
    } else if (gpu_status[k] == ds::dual::status_t::TIME_LIMIT &&
               nodes[u]->ref_status != ds::dual::status_t::TIME_LIMIT) {
      gpu_timeout++;
    } else {
      gpu_true_mismatch++;
      // COLLAPSE IDENTICAL LINES instead of printing one per problem. With --replicas 100 every
      // mismatching node would otherwise emit 100 byte-identical lines, burying the ~50 distinct
      // findings in thousands of rows.
      //
      // The key is the WHOLE line, not the node name: replicas of one node are not guaranteed to
      // agree with each other (the GPU result for one node can depend on the batch it was launched
      // in), so two different outcomes for the same node must both survive and be
      // counted separately. Only truly identical results merge.
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "  [TRUE MISMATCH] %-30s ref{%s it=%d obj=%.10g} gpu{%s it=%d obj=%.10g}",
                    nodes[u]->name.c_str(), ds::dual_status_to_string(nodes[u]->ref_status).c_str(),
                    nodes[u]->ref_iters, nodes[u]->ref_obj,
                    ds::dual_status_to_string(gpu_status[k]).c_str(), prepared[k]->sol.iterations,
                    gobj);
      const std::string line(buf);
      auto it = mismatch_index.find(line);
      if (it == mismatch_index.end()) {
        mismatch_index.emplace(line, mismatch_lines.size());
        mismatch_lines.emplace_back(line, 1);
      } else {
        mismatch_lines[it->second].second++;
      }
    }
  }
  // Emitted in first-seen order (which is block order, i.e. heavy-first), each with the number of
  // problems that produced it. The "(xN)" suffix is appended AFTER the closing gpu{...} brace so a
  // log parser keyed on that brace still matches unchanged.
  for (const auto& ml : mismatch_lines) {
    if (ml.second > 1) {
      std::printf("%s  (x%d identical problems)\n", ml.first.c_str(), ml.second);
    } else {
      std::printf("%s\n", ml.first.c_str());
    }
  }
  if (!mismatch_lines.empty()) {
    std::printf("  [TRUE MISMATCH] %zu distinct result(s) over %d mismatching problem(s).\n",
                mismatch_lines.size(), gpu_true_mismatch);
  }
  const double gpu_e2e_s = gpu_setup_s + timing.upload_s + timing.kernel_s + timing.copyback_s;

  // ---- Report ----
  size_t total_iters = 0;
  int max_iters      = 0;
  for (int u : list) {
    total_iters += nodes[u]->ref_iters;
    max_iters = std::max(max_iters, nodes[u]->ref_iters);
  }
  // The GPU's OWN pivot count. This must be reported separately from the CPU reference total above:
  // the reference is produced by the CPU driver, so it does NOT move when a GPU-side switch changes
  // (reprice, driver, launch-bound regime). Reading the CPU line during a GPU A/B makes two genuinely
  // different configurations look identical, and reading it as "pivots" makes any ms/pivot figure
  // wrong on both sides. That mistake produced a confidently inverted performance conclusion on
  // 2026-08-12; the two lines are now labelled so it cannot recur.
  size_t gpu_total_iters = 0;
  int gpu_max_iters      = 0;
  for (size_t k = 0; k < Ntot; ++k) {
    const int it = prepared[k]->sol.iterations;
    gpu_total_iters += static_cast<size_t>(it);
    gpu_max_iters = std::max(gpu_max_iters, it);
  }
  std::printf("\n==== parallel throughput: %zu problems (%d unique x%d) ====\n", Ntot, uc, replicas);
  std::printf("total simplex iterations (CPU reference) : %zu   (max single node %d, mean %.0f)\n",
              total_iters, max_iters, static_cast<double>(total_iters) / static_cast<double>(Ntot));
  std::printf("total simplex iterations (GPU actual)    : %zu   (max single node %d, mean %.0f)\n",
              gpu_total_iters, gpu_max_iters,
              static_cast<double>(gpu_total_iters) / static_cast<double>(Ntot));

  // Heaviest unique nodes (the tail that dominates the single-thread GPU kernel). Depth is shown
  // to confirm it does NOT predict iterations (heavy nodes are shallow) — hence the iters-based sort.
  std::printf("\n-- heaviest nodes (iters-sorted; the GPU kernel tail) --\n");
  std::printf("  %-32s %6s %8s %8s\n", "node", "depth", "m", "iters");
  for (int rank = 0; rank < std::min(8, uc); ++rank) {
    const auto& un = nodes[order[rank]];
    std::printf("  %-32s %6d %8d %8d\n", un->name.c_str(), un->depth, un->m, un->ref_iters);
  }

  std::printf("\n-- CPU (OpenMP, %d threads) --\n", omp_get_max_threads());
  std::printf("  wall total (refactor+solve) : %8.3f s\n", cpu_total_s);
  std::printf("  sum of driver-only seconds  : %8.3f s  (serial driver work)\n", cpu_driver_sum);
  std::printf("  result mismatches vs ref    : %d\n", cpu_mismatch);
  std::printf("\n-- GPU (one block per problem, single launch) --\n");
  std::printf("  host setup (refactor, OpenMP): %8.3f s\n", gpu_setup_s);
  std::printf("  H2D upload                   : %8.3f s\n", timing.upload_s);
  std::printf("  kernel (device compute)      : %8.3f s\n", timing.kernel_s);
  std::printf("  D2H copyback                 : %8.3f s\n", timing.copyback_s);
  std::printf("  end-to-end                   : %8.3f s\n", gpu_e2e_s);
  std::printf("  bit-exact vs ref             : %d / %zu\n", gpu_exact, Ntot);
  std::printf("  GPU TIME_LIMIT (too slow)    : %d\n", gpu_timeout);
  std::printf("  true mismatches (bug!)       : %d\n", gpu_true_mismatch);
  if (gpu_timeout > 0) {
    std::printf("  NOTE: %d node(s) hit the %.0fs limit — the GPU kernel wall is time-limit-bound, "
                "not the true solve time. Raise --time-limit for a completing (meaningful) number.\n",
                gpu_timeout, time_limit);
  }
  std::printf("\n-- speedups (CPU wall / GPU) --\n");
  if (gpu_e2e_s > 0) {
    std::printf("  CPU wall / GPU end-to-end : %.2fx\n", cpu_total_s / gpu_e2e_s);
  }
  if (timing.kernel_s > 0) {
    std::printf("  CPU wall / GPU kernel     : %.2fx\n", cpu_total_s / timing.kernel_s);
  }
  const bool passed = cpu_mismatch == 0 && gpu_true_mismatch == 0;
  std::printf("\n%s\n",
              passed ? "VERIFICATION PASSED: every CPU result and every COMPLETED GPU result is "
                       "bit-for-bit == the sequential reference (batching has zero effect)."
                     : "VERIFICATION FAILED: a completed result diverged from the sequential "
                       "reference (true mismatch — a real bug).");
}

// --solve-mps mode: solve one MPS FROM SCRATCH (no warm start, no cutoff) two
// independent ways, to check whether a node child LP is genuinely dual-unbounded
// / primal-infeasible or whether that only appears when warm-started from the
// parent basis. Both paths are the regular dual simplex started cold.
void solve_mps_from_scratch_mode(const raft::handle_t& handle,
                                 const std::string& path,
                                 double time_limit,
                                 double mps_cutoff)
{
  std::printf("==== solve-from-scratch (cold start): %s (cutoff=%.10g) ====\n", path.c_str(),
              mps_cutoff);
  std::fflush(stdout);

  // (C) Production full dual simplex WITH the given cutoff. Shows whether a
  // from-scratch solve of this exact LP + cutoff terminates via CUTOFF (obj
  // crosses the bound) rather than INFEASIBLE — i.e. the cutoff-vs-unbounded
  // outcome is path/cutoff dependent, not a fixed property of the LP.
  if (std::isfinite(mps_cutoff)) {
    try {
      ds::user_problem_t<int, double> up = load_user_problem(handle, path);
      auto settings   = make_settings(time_limit);
      settings.cut_off = mps_cutoff;
      ds::lp_solution_t<int, double> sol(up.num_rows, up.num_cols);
      const ds::lp_status_t st = ds::solve_linear_program<int, double>(up, settings, sol);
      std::printf("  [solve_linear_program +cutoff] status=%-15s raw_obj=%.10g iters=%d\n",
                  ds::lp_status_to_string(st).c_str(), sol.objective, sol.iterations);
    } catch (const std::exception& ex) {
      std::printf("  [solve_linear_program +cutoff] exception: %s\n", ex.what());
    }
    std::fflush(stdout);
  }

  // (A) Production full dual simplex on the user_problem (presolve+scale as usual).
  try {
    ds::user_problem_t<int, double> up = load_user_problem(handle, path);
    auto settings = make_settings(time_limit);
    ds::lp_solution_t<int, double> sol(up.num_rows, up.num_cols);
    const ds::lp_status_t st = ds::solve_linear_program<int, double>(up, settings, sol);
    std::printf("  [solve_linear_program] status=%-15s user_obj=%.10g raw_obj=%.10g iters=%d\n",
                ds::lp_status_to_string(st).c_str(),
                sol.user_objective,
                sol.objective,
                sol.iterations);
  } catch (const std::exception& ex) {
    std::printf("  [solve_linear_program] exception: %s\n", ex.what());
  }
  std::fflush(stdout);

  // (B) Regular dual simplex (dual_phase2 phase 1 -> phase 2) on the direct
  // lp_problem_t (same representation the verify-nodes warm start used), no cutoff.
  try {
    ds::lp_problem_t<int, double> lp = load_lp_problem_direct(handle, path);
    auto base = make_settings(time_limit);
    std::vector<ds::variable_status_t> vs;
    double obj = kNaN;
    std::string note;
    const bool ok = solve_lp_to_optimal_vstatus(lp, base, vs, obj, note);
    if (ok) {
      std::printf("  [dual_phase2 cold]     status=OPTIMAL         raw_obj=%.10g\n", obj);
    } else {
      std::printf("  [dual_phase2 cold]     status=%s\n", note.c_str());
    }
  } catch (const std::exception& ex) {
    std::printf("  [dual_phase2 cold]     exception: %s\n", ex.what());
  }
  std::fflush(stdout);
}

// ===========================================================================
// --replay-node mode: re-solve ONE bad B&B node, captured by gpu_node_pool's
// CUOPT_BNB_DUMP_BAD_NODES writer, through all three drivers.
//
// The dump is the state AS UPLOADED to the device -- bounds already strengthened,
// vstatus already repaired by refactor_basis -- so this replay starts each driver
// from exactly the state the GPU consumed when it made the wrong decision. The
// header also carries what was OBSERVED at capture time, so the artifact reports
// whether the defect reproduces without needing the original log.
void replay_node_mode(const raft::handle_t& handle, const std::string& file, double time_limit)
{
  namespace bn = cuopt::mathematical_optimization::mip::bad_node;

  bn::header_t h{};
  ds::lp_problem_t<int, double> lp(&handle, 1, 1, 1);
  std::vector<double> lower, upper, dy_se;
  std::vector<ds::variable_status_t> vstatus;
  // v3 dumps carry the SOLVE STATE the device was launched with. A seat captured mid-solve (after a
  // REFACTOR_NEEDED resume) cannot be reproduced without them -- recomputing x/y/z from the basis
  // starts the replay somewhere else, which is what made the one wrong-objective node we caught
  // replay cleanly while the live run had failed.
  std::vector<double> dump_x, dump_y, dump_z;
  std::string why;
  if (!bn::read<int, double>(file, h, lp, lower, upper, vstatus, dy_se, dump_x, dump_y, dump_z,
                             why)) {
    std::printf("replay-node: cannot load %s (%s)\n", file.c_str(), why.c_str());
    std::exit(1);
  }

  // A/B (TEST_NETLIB_REPLAY_COLD_SE=1): drop the carried steepest-edge norms and let every driver
  // re-initialise them. The GPU driver was proven bit-exact against the CPU simplified driver with an
  // EMPTY dy_se; branch-and-bound is the first caller that carries norms in, so the carried path is
  // the under-validated one. If a divergence disappears here, it is in that path.
  const bool cold_se = std::getenv("TEST_NETLIB_REPLAY_COLD_SE") != nullptr;
  if (cold_se) { dy_se.clear(); }

  // The carried vector is the ONE input that differs between the working and failing configurations,
  // so characterise it before blaming any code that consumes it. A non-positive entry matters: the
  // ratio test divides the squared primal infeasibility by dy_se[basic_list[i]], so a zero or
  // negative norm on a column that later enters the basis produces exactly the 1e+70 blow-up seen.
  if (!dy_se.empty()) {
    double lo = dy_se[0], hi = dy_se[0];
    size_t nonpos = 0, zero = 0;
    for (double v : dy_se) {
      lo = std::min(lo, v); hi = std::max(hi, v);
      if (v <= 0.0) { ++nonpos; }
      if (v == 0.0) { ++zero; }
    }
    std::printf("  dy_se stats: n=%zu min=%.17g max=%.17g nonpositive=%zu zero=%zu\n",
                dy_se.size(), lo, hi, nonpos, zero);
  }

  std::printf("replay-node %s\n", file.c_str());
  std::printf("  m=%d n=%d nnz=%lld  dy_se=%s\n",
              h.m, h.n, static_cast<long long>(h.nnz),
              cold_se ? "CLEARED (cold-SE A/B)" : (h.has_dy_se ? "carried" : "none"));
  std::printf("  cut_off=%.17g  obj_constant=%.17g obj_scale=%.17g obj_integral=%d\n",
              h.cut_off, h.obj_constant, h.obj_scale, h.objective_is_integral);
  std::printf("  captured: gpu %-16s obj=%.17g it=%d\n",
              ds::dual_status_to_string(static_cast<ds::dual::status_t>(h.gpu_status)).c_str(),
              h.gpu_objective, h.gpu_iters);
  std::printf("  captured: cpu %-16s obj=%.17g it=%d\n\n",
              ds::dual_status_to_string(static_cast<ds::dual::status_t>(h.cpu_status)).c_str(),
              h.cpu_objective, h.cpu_iters);

  // Every device-read setting comes from the dump; the rest stay at harness defaults.
  auto settings_for = [&]() {
    auto s = make_settings(time_limit);
    bn::apply_settings<int, double>(h, s);
    // TEST_NETLIB_REPLAY_PIVOT_TRACE=1: emit the simplified driver's per-pivot [CMP] bit trace, to
    // diff against the device's [GCMP]. Both print the same fields in the same order.
    if (std::getenv("TEST_NETLIB_REPLAY_PIVOT_TRACE") != nullptr) { s.log.log = true; }
    // TEST_NETLIB_REPLAY_REFACFREQ=<n>: override the refactorization frequency. Raising it past the
    // node's iteration count removes the PERIODIC refactor trigger entirely, so whatever still exits
    // must be the unsafe-rank-1-update trigger. Applies to both drivers, so they stay comparable.
    if (const char* e = std::getenv("TEST_NETLIB_REPLAY_REFACFREQ")) {
      s.refactor_frequency = std::atoi(e);
    }
    // TEST_NETLIB_REPLAY_CUTOFF=<v>: override the cutoff the dump recorded.
    //
    // WHY THIS EXISTS. The dump stores `cut_off` AS IT WAS AT FILL TIME, but the live pool
    // broadcasts the newest incumbent to every seat before each relaunch, so a node can be solved
    // against a materially tighter cutoff than the dump preserves. A tighter cutoff makes
    // `obj >= cut_off` fire earlier -- which is a candidate explanation for why a node that fathoms
    // WRONGLY in the live run replays perfectly here (measured on dano3_5: live CUTOFF/773.116 vs
    // replay OPTIMAL/578.362 on the same dump).
    if (const char* e = std::getenv("TEST_NETLIB_REPLAY_CUTOFF")) { s.cut_off = std::atof(e); }
    return s;
  };

  // Sanitizer harness: execute exactly one fresh V2 kernel launch for this captured node and
  // return. The normal replay intentionally runs several CPU/GPU drivers and many refactor legs;
  // under compute-sanitizer that obscures which launch produced a report and takes minutes before
  // reaching the persistent kernel used by B&B.
  if (env_on("TEST_NETLIB_REPLAY_ONE_V2")) {
    const int seats = [] {
      const char* e = std::getenv("TEST_NETLIB_REPLAY_SEATS");
      return e != nullptr ? std::max(1, std::atoi(e)) : 1;
    }();
    struct one_v2_state_t {
      std::vector<ds::variable_status_t> vs;
      std::vector<int> basic, nonbasic, superbasic;
      ds::basis_update_mpf_t<int, double> ft;
      ds::lp_solution_t<int, double> sol;
      std::vector<double> se;
      ds::simplex_solver_settings_t<int, double> settings;
      int refactor_reason{0};
      one_v2_state_t(int m_, int n_, int rf) : ft(m_, rf), sol(m_, n_) {}
    };
    std::vector<std::unique_ptr<one_v2_state_t>> state;
    std::vector<gds::gpu_batch_item<int, double>> items;
    state.reserve(static_cast<size_t>(seats));
    items.reserve(static_cast<size_t>(seats));
    bool prepared = true;
    for (int k = 0; k < seats; ++k) {
      auto q      = std::make_unique<one_v2_state_t>(h.m, h.n, h.refactor_frequency);
      q->vs       = vstatus;
      q->se       = dy_se;
      q->settings = settings_for();
      q->basic.assign(static_cast<size_t>(h.m), 0);
      ds::get_basis_from_vstatus<int>(
        h.m, q->vs, q->basic, q->nonbasic, q->superbasic);
      const double ft0 = mo::tic();
      if (q->ft.refactor_basis(lp.A,
                               q->settings,
                               lp.lower,
                               lp.upper,
                               ft0,
                               q->basic,
                               q->nonbasic,
                               q->vs) != 0) {
        prepared = false;
        break;
      }
      if (!dump_x.empty()) { q->sol.x = dump_x; }
      if (!dump_y.empty() && !dump_z.empty()) {
        q->sol.y = dump_y;
        q->sol.z = dump_z;
      } else {
        compute_duals(lp, q->ft, q->basic, q->nonbasic, q->sol.y, q->sol.z);
      }
      gds::gpu_batch_item<int, double> item{};
      item.lp              = &lp;
      item.lower           = &lp.lower;
      item.upper           = &lp.upper;
      item.settings        = &q->settings;
      item.vstatus         = &q->vs;
      item.ft              = &q->ft;
      item.basic_list      = &q->basic;
      item.nonbasic_list   = &q->nonbasic;
      item.sol             = &q->sol;
      item.dy_se           = &q->se;
      item.refactor_reason = &q->refactor_reason;
      items.push_back(item);
      state.push_back(std::move(q));
    }
    if (!prepared) {
      std::printf("one-v2: initial factorization failed\n");
      return;
    }
    gds::gpu_batch_timing timing;
    auto* run = gds::gpu_simplified_dual_phase2_batch_launch<int, double>(
      items,
      timing,
      rmm::cuda_stream_default,
      /*allow_refactor_exit=*/true,
      /*ro_cache=*/nullptr,
      /*persistent_seats=*/true);
    if (run == nullptr) {
      std::printf("one-v2: launch rejected: %s\n", timing.message.c_str());
      return;
    }
    std::vector<ds::dual::status_t> out;
    gds::gpu_simplified_dual_phase2_batch_finish<int, double>(run, out, timing);
    for (size_t k = 0; k < out.size(); ++k) {
      std::printf("one-v2[%zu] %s obj=%.17g it=%d\n",
                  k,
                  ds::dual_status_to_string(out[k]).c_str(),
                  state[k]->sol.objective,
                  state[k]->sol.iterations);
    }
    return;
  }

  struct row_t {
    const char* name;
    ds::dual::status_t st;
    double obj;
    int it;
    double sec;
    bool ran;
  };
  row_t rows[3] = {{"full-cpu", ds::dual::status_t::OPTIMAL, kNaN, 0, 0.0, false},
                   {"simplified", ds::dual::status_t::OPTIMAL, kNaN, 0, 0.0, false},
                   {"gpu", ds::dual::status_t::OPTIMAL, kNaN, 0, 0.0, false}};

  // Filled in by the refactor-and-resume leg below; that is the result branch-and-bound consumes.
  ds::dual::status_t resumed_st = ds::dual::status_t::UNSET;
  double resumed_obj            = kNaN;
  bool resumed_ran              = false;
  auto obj_close = [](double a, double b) {
    return std::abs(a - b) <= 1e-6 * (1.0 + std::abs(b));
  };

  std::printf("%-12s %-18s %22s %8s %10s\n", "driver", "status", "objective", "iters", "seconds");
  // TEST_NETLIB_VERBOSE=1 turns on the CPU drivers' per-pivot [CMP] bit trace. Paired with a
  // GPU_PIVOT_TRACE build (its [GCMP] line carries the same fields in the same order) the two
  // sequences diff directly to the first iteration on which they part.
  const bool verbose_trace = std::getenv("TEST_NETLIB_VERBOSE") != nullptr;
  for (int mode = 0; mode < 3; ++mode) {
    auto s     = settings_for();
    if (verbose_trace && mode == 1) { enable_verbose(s); }
    double obj = kNaN, dinf = 0.0, sec = 0.0;
    int it = 0, upd = 0;
    try {
      rows[mode].st  = run_phase2_from_basis(lp, vstatus, s, mode, obj, it, dinf, upd,
                                             /*initialize_basis*/ false, dy_se, &sec);
      rows[mode].ran = true;
    } catch (const std::exception& ex) {
      std::printf("%-12s (exception: %s)\n", rows[mode].name, ex.what());
      continue;
    }
    rows[mode].obj = obj;
    rows[mode].it  = it;
    rows[mode].sec = sec;
    std::printf("%-12s %-18s %22.17g %8d %10.4f\n",
                rows[mode].name, ds::dual_status_to_string(rows[mode].st).c_str(), obj, it, sec);
    std::fflush(stdout);
  }


  // Fourth leg: the GPU the way B&B ACTUALLY drives it -- REFACTOR-AND-RESUME. The single-shot
  // mode-2 call above stops at the first REFACTOR_NEEDED, but `gpu_solve_node_serial` rebuilds the
  // factorization on the host and calls the device straight back, carrying the primal point, the
  // objective and the steepest-edge norms across. A defect that only appears after a resume is
  // invisible to the single-shot leg, so the two must be reported side by side.
  {
    const int m = lp.num_rows;
    const int n = lp.num_cols;
    std::vector<ds::variable_status_t> vs = vstatus;
    std::vector<int> basic_list(m), nonbasic_list, superbasic_list;
    ds::basis_update_mpf_t<int, double> ft(m, h.refactor_frequency);
    ds::lp_solution_t<int, double> sol(m, n);
    std::vector<double> se = dy_se;
    auto s = settings_for();
    const double t0 = mo::tic();

    auto refactorize_and_price = [&]() {
      basic_list.assign(static_cast<size_t>(m), 0);
      nonbasic_list.clear();
      superbasic_list.clear();
      ds::get_basis_from_vstatus<int>(m, vs, basic_list, nonbasic_list, superbasic_list);
      return ft.refactor_basis(lp.A, s, lp.lower, lp.upper, t0, basic_list, nonbasic_list, vs) == 0;
    };

    // CO-SEAT REPLICATION (TEST_NETLIB_REPLAY_SEATS=<n>, default 1).
    //
    // WHY. The live pool runs up to 10 problems in ONE kernel launch; this replay ran exactly one,
    // so a fault that needs co-resident seats could never appear here. Measured on dano3_5:
    // seats=1 gives 0 mismatches over 90 consumed nodes, seats=10 gives 39 over 282. Everything
    // else -- child-continue, the cutoff value, basis repair -- was eliminated by A/B.
    //
    // Each shadow seat gets its OWN mutable state (vstatus / basis lists / ft / sol / dy_se) and
    // shares only `lp`, exactly as the pool's seats share one problem through the read-only cache.
    // If seat 0's answer changes when shadows are present, the fault is cross-seat.
    struct shadow_seat_t {
      std::vector<ds::variable_status_t> vs;
      std::vector<int> basic_list, nonbasic_list, superbasic_list;
      ds::basis_update_mpf_t<int, double> ft;
      ds::lp_solution_t<int, double> sol;
      std::vector<double> se;
      ds::simplex_solver_settings_t<int, double> s;
      int refactor_reason{0};
      bool refactor_resume{false};
      bool cert_retry{false};
      bool live{true};
      shadow_seat_t(int m_, int n_, int rf) : ft(m_, rf), sol(m_, n_) {}
    };
    const int replay_seats = [] {
      const char* e = std::getenv("TEST_NETLIB_REPLAY_SEATS");
      return e != nullptr ? std::max(1, std::atoi(e)) : 1;
    }();
    std::vector<std::unique_ptr<shadow_seat_t>> shadows;
    for (int k = 1; k < replay_seats; ++k) {
      auto sh = std::make_unique<shadow_seat_t>(m, n, h.refactor_frequency);
      sh->vs = vstatus;
      sh->se = dy_se;
      sh->s  = settings_for();
      sh->basic_list.assign(static_cast<size_t>(m), 0);
      ds::get_basis_from_vstatus<int>(m, sh->vs, sh->basic_list, sh->nonbasic_list,
                                      sh->superbasic_list);
      if (sh->ft.refactor_basis(lp.A, sh->s, lp.lower, lp.upper, t0, sh->basic_list,
                                sh->nonbasic_list, sh->vs) != 0) {
        sh->live = false;
      } else {
        compute_duals(lp, sh->ft, sh->basic_list, sh->nonbasic_list, sh->sol.y, sh->sol.z);
      }
      shadows.push_back(std::move(sh));
    }
    if (replay_seats > 1) {
      std::printf("  co-seat replication: %d seats in one launch (%zu shadows live)\n",
                  replay_seats,
                  std::count_if(shadows.begin(), shadows.end(),
                                [](const auto& p) { return p->live; }));
    }

    std::printf("\n---- gpu refactor-and-resume (as B&B drives it) ----\n");
    if (!refactorize_and_price()) {
      std::printf("  initial refactorization failed\n");
    } else {
      compute_duals(lp, ft, basic_list, nonbasic_list, sol.y, sol.z);
      int total_iters = 0, resumes = 0;
      int revalidations = 0;   // times a returned OPTIMAL failed fresh pricing
      bool item_refactor_resume = false;
      bool item_terminal_certificate_retry = false;
      ds::dual::status_t st = ds::dual::status_t::UNSET;
      while (true) {
        // A resume that carries nothing looks exactly like a resume that makes no progress, so
        // count what actually came back changed rather than inferring it from the objective.
        const std::vector<ds::variable_status_t> vs_in = vs;
        const std::vector<double> se_in = se;
        // BATCH entry, not the single-problem one: `gpu_simplified_dual_phase2` copies back only
        // x / y / z / objective / iterations, NOT vstatus or dy_se. A resume driven through it
        // restarts from the basis it was first given every time and can never make progress --
        // which is exactly what B&B drives, so the replay must use the same path B&B does.
        gds::gpu_batch_item<int, double> item{};
        item.lp            = &lp;
        item.lower         = &lp.lower;
        item.upper         = &lp.upper;
        item.settings      = &s;
        item.vstatus       = &vs;
        item.ft            = &ft;
        item.basic_list    = &basic_list;
        item.nonbasic_list = &nonbasic_list;
        item.sol           = &sol;
        item.dy_se         = &se;
        item.refactor_resume = item_refactor_resume;
        item.terminal_certificate_retry = item_terminal_certificate_retry;
        int refactor_reason  = 0;  // 0 periodic, 1 rejected update, 2 certificate retry
        item.refactor_reason = &refactor_reason;
        std::vector<gds::gpu_batch_item<int, double>> batch{item};
        // Co-resident shadow seats. They share `lp` and nothing else; seat 0 stays at index 0 so
        // `out[0]` is still the node under test.
        for (auto& sh : shadows) {
          if (!sh->live) { continue; }
          gds::gpu_batch_item<int, double> e{};
          e.lp            = &lp;
          e.lower         = &lp.lower;
          e.upper         = &lp.upper;
          e.settings      = &sh->s;
          e.vstatus       = &sh->vs;
          e.ft            = &sh->ft;
          e.basic_list    = &sh->basic_list;
          e.nonbasic_list = &sh->nonbasic_list;
          e.sol           = &sh->sol;
          e.dy_se         = &sh->se;
          e.refactor_resume            = sh->refactor_resume;
          e.terminal_certificate_retry = sh->cert_retry;
          e.refactor_reason            = &sh->refactor_reason;
          batch.push_back(e);
        }
        std::vector<ds::dual::status_t> out;
        gds::gpu_batch_timing timing;
        // allow_refactor_exit=TRUE. Without it this leg gets the kernel with the refactor exits
        // compiled OUT, so at the 256-update MPF cap it returns NUMERICAL instead of
        // REFACTOR_NEEDED and the loop stops after one leg -- which is NOT how B&B drives it.
        gds::gpu_simplified_dual_phase2_batch<int, double>(batch, out, timing,
                                                          /*allow_refactor_exit*/ true);
        if (out.empty() || timing.rejected) {
          std::printf("  batch rejected: %s\n", timing.message.c_str());
          break;
        }
        st = out[0];
        // Advance the shadow seats the same way B&B would, so they stay co-resident for the whole
        // solve instead of going terminal after one leg and leaving seat 0 alone again.
        {
          size_t oi = 1;
          for (auto& sh : shadows) {
            if (!sh->live) { continue; }
            if (oi >= out.size()) { break; }
            const ds::dual::status_t sst = out[oi++];
            if (sst == ds::dual::status_t::REFACTOR_NEEDED) {
              sh->basic_list.assign(static_cast<size_t>(m), 0);
              sh->nonbasic_list.clear();
              sh->superbasic_list.clear();
              ds::get_basis_from_vstatus<int>(m, sh->vs, sh->basic_list, sh->nonbasic_list,
                                              sh->superbasic_list);
              sh->live = sh->ft.refactor_basis(lp.A, sh->s, lp.lower, lp.upper, t0,
                                               sh->basic_list, sh->nonbasic_list, sh->vs) == 0;
              sh->refactor_resume = false;   // rebuilt from vstatus, so the device re-derives x
            } else {
              sh->live = false;              // terminal: this shadow stops occupying a seat
            }
          }
        }
        int vs_changed = 0, se_changed = 0;
        for (size_t k = 0; k < vs.size(); ++k) { vs_changed += (vs[k] != vs_in[k]); }
        for (size_t k = 0; k < se.size() && k < se_in.size(); ++k) {
          se_changed += (se[k] != se_in[k]);
        }
        // The device carries the cumulative iteration count in `sol` across refactor resumes.
        // Assign it; accumulating it would count all earlier legs again on every resume.
        const int leg_iters = sol.iterations - total_iters;
        total_iters         = sol.iterations;
        // Is the CARRIED z still DUAL FEASIBLE for the basis that just came back? That, not its
        // distance from freshly priced duals, is what the ratio test depends on: a dual simplex is
        // only valid while every nonbasic reduced cost carries the sign its status demands.
        const double dinf_carried = dual_infeasibility(lp, vs, sol.z, s.dual_tol);
        std::printf(
          "  leg %-2d %-18s it=%-6d cum_it=%-6d obj=%.17g  dinf=%.3e vs_chg=%d se_chg=%d\n",
          resumes, ds::dual_status_to_string(st).c_str(), leg_iters, total_iters,
          ds::compute_user_objective<int, double>(lp, sol.x), dinf_carried, vs_changed, se_changed);
        std::fflush(stdout);
        // VALIDATE AN "OPTIMAL" AGAINST FRESHLY PRICED REDUCED COSTS -- the same check the pool's
        // resume loop applies. The driver's optimality test reads the MAINTAINED z, which has no
        // error control across a long stacked-update sequence; if the basis is not actually dual
        // feasible then the solve has stopped at a primal-feasible but NON-optimal vertex.
        if (st == ds::dual::status_t::OPTIMAL &&
            std::getenv("TEST_NETLIB_REPLAY_VALIDATE_OPTIMAL") != nullptr && revalidations < 4) {
          basic_list.assign(static_cast<size_t>(m), 0);
          nonbasic_list.clear();
          superbasic_list.clear();
          ds::get_basis_from_vstatus<int>(m, vs, basic_list, nonbasic_list, superbasic_list);
          ds::basis_update_mpf_t<int, double> ft_v(m, s.refactor_frequency);
          const double tv = cuopt::mathematical_optimization::tic();
          if (ft_v.refactor_basis(lp.A, s, lp.lower, lp.upper, tv, basic_list, nonbasic_list, vs) ==
              0) {
            std::vector<double> y_v(static_cast<size_t>(m), 0.0),
              z_v(static_cast<size_t>(lp.num_cols), 0.0);
            double w_v = 0.0;
            ds::phase2::compute_dual_solution_from_basis(lp, ft_v, basic_list, nonbasic_list, y_v,
                                                         z_v, w_v);
            const double dinf_fresh = dual_infeasibility(lp, vs, z_v, s.dual_tol);
            if (dinf_fresh > s.dual_tol) {
              ++revalidations;
              std::printf("  OPTIMAL REJECTED: fresh dinf=%.6e (maintained said %.6e) -- resuming\n",
                          dinf_fresh, dual_infeasibility(lp, vs, sol.z, s.dual_tol));
              sol.y = y_v;
              sol.z = z_v;
              std::vector<double> xB_v(m, 0.0);
              double wx_v = 0.0;
              ds::phase2::compute_primal_solution_from_basis(lp, ft_v, basic_list, nonbasic_list,
                                                             vs, sol.x, xB_v, wx_v);
              item_refactor_resume = true;
              item_terminal_certificate_retry = false;
              continue;
            }
          }
        }
        if (st != ds::dual::status_t::REFACTOR_NEEDED) { break; }
        if (++resumes > 64) {
          std::printf("  resume budget exhausted\n");
          break;
        }
        // MIRROR THE LIVE PATH EXACTLY, or this leg proves nothing about it. gpu_solve_node_serial
        // refactorizes WITHOUT re-pricing, rebuilds x with the CPU's post-refactor formula, and sets
        // refactor_resume so the device does not redo it. Re-pricing here (which is what this loop
        // used to do) is a different algorithm that converges on nodes the live path stalls on --
        // which is precisely why the stall never reproduced offline.
        // FAITHFUL TO THE POOL: do NOT rebuild the lists from vstatus here. The device returns
        // basic_list / nonbasic_list in PIVOT-HISTORY order (it maintains them by in-place swap) and
        // gpu_node_pool's refactorize() keeps that order on a resume -- rebuilding gives a SORTED
        // permutation, which changes the LU that refactor_basis produces and the order the
        // order-dependent ratio test scans the nonbasics in. This loop used to rebuild, so it ran a
        // DIFFERENT resume algorithm from the pool: that is why a dumped node could replay cleanly
        // while the live solve had failed, and it made every dump-based comparison unsound.
        // TEST_NETLIB_REPLAY_RESORT_LISTS=1 restores the old (unfaithful) behaviour for A/B.
        // WHEN does the carried y stop belonging to its basis -- during the device leg, or during
        // the host refactorization? Measure |B'y-c_B| against the basis the DEVICE handed back,
        // BEFORE refactor_basis can repair it, and keep that basis to diff against the repaired one.
        // A residual that is already large here indicts the leg; one that only grows after
        // refactor_basis indicts the repair, which changes the basis without re-pricing y.
        std::vector<int> bl_before(basic_list.begin(), basic_list.end());
        std::vector<int> nbl_before(nonbasic_list.begin(), nonbasic_list.end());
        std::vector<ds::variable_status_t> vs_before(vs);
        double pre_resid = 0.0;
        for (int i = 0; i < m; ++i) {
          const int j  = bl_before[i];
          double dot   = 0.0;
          for (int p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
            dot += lp.A.x[p] * sol.y[lp.A.i[p]];
          }
          pre_resid = std::max(pre_resid, std::fabs(dot - lp.objective[j]));
        }
        static const bool resort_lists = std::getenv("TEST_NETLIB_REPLAY_RESORT_LISTS") != nullptr;
        if (resort_lists) {
          basic_list.assign(static_cast<size_t>(m), 0);
          nonbasic_list.clear();
          superbasic_list.clear();
          ds::get_basis_from_vstatus<int>(m, vs, basic_list, nonbasic_list, superbasic_list);
        }
        bool repaired = false;
        if (ft.refactor_basis(
              lp.A, s, lp.lower, lp.upper, t0, basic_list, nonbasic_list, vs, &repaired) != 0) {
          std::printf("  resume refactorization failed\n");
          st = ds::dual::status_t::NUMERICAL;
          break;
        }
        if (repaired) {
          std::printf("  resume refactorization repaired the basis -- CPU takeback\n");
          st = ds::dual::status_t::NUMERICAL;
          break;
        }
        {
          // The factorization is fresh here, so this is the one place the CARRIED duals can be
          // checked against the truth mid-solve. A dual simplex maintains dual feasibility as an
          // INVARIANT and never re-tests it, so a drift that accumulates here is invisible to the
          // driver and decides the final answer.
          std::vector<double> y_chk(static_cast<size_t>(m), 0.0),
            z_chk(static_cast<size_t>(lp.num_cols), 0.0);
          double w_chk = 0.0;
          ds::phase2::compute_dual_solution_from_basis(lp, ft, basic_list, nonbasic_list, y_chk,
                                                       z_chk, w_chk);
          double dz = 0.0, dy = 0.0;
          int argmax = -1;
          for (int j = 0; j < lp.num_cols; ++j) {
            const double d = std::fabs(z_chk[j] - sol.z[j]);
            if (d > dz) { dz = d; argmax = j; }
          }
          for (int i = 0; i < m; ++i) { dy = std::max(dy, std::fabs(y_chk[i] - sol.y[i])); }
          // WHICH of the two duals is the liar? Both are tested against the DEFINING equation of
          // the dual solution, B'y = c_B, evaluated directly from A -- no factorization involved.
          // A large residual for y_fresh indicts the LU that refactor_basis just built (so the
          // basis ORDER is the fault); a large residual for y_kept alone indicts the carried
          // duals (so the resume is the fault). They are different bugs with different fixes.
          auto basis_residual = [&](const std::vector<double>& yv) {
            double r = 0.0;
            for (int i = 0; i < m; ++i) {
              const int j  = basic_list[i];
              double dot   = 0.0;
              for (int p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
                dot += lp.A.x[p] * yv[lp.A.i[p]];
              }
              r = std::max(r, std::fabs(dot - lp.objective[j]));
            }
            return r;
          };
          // Did refactor_basis change WHICH columns are basic? Compare as sets: a repair
          // substitutes slacks for deficient columns, so membership changes even though the
          // caller never asked for it.
          std::vector<int> a(bl_before), b(basic_list.begin(), basic_list.end());
          std::sort(a.begin(), a.end());
          std::sort(b.begin(), b.end());
          int set_changed = 0;
          {
            std::vector<int> diff;
            std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
            set_changed = static_cast<int>(diff.size());
          }
          // Is the post-repair dual infeasibility CONCENTRATED on the columns the repair ejected?
          // basis_repair picks the ejected column's status from bound availability alone and never
          // looks at the sign of its reduced cost, so it can seat it at a dual-INFEASIBLE bound.
          // If the whole violation sits on those few columns, a sign-correct restatement fixes it;
          // if it is spread, the node has to go to the CPU.
          if (set_changed > 0) {
            // IS THE DEFICIENCY GENUINE, OR AN ARTIFACT OF THE COLUMN ORDER? Re-factorize the SAME
            // pre-repair basis from a scratch factorization, once in pivot-history order (control,
            // must repair) and once SORTED. factorize_basis uses threshold partial pivoting, so the
            // elimination order decides which columns still have an acceptable pivot -- a sorted
            // permutation of the same column SET can therefore factorize cleanly where pivot order
            // does not. If it does, the repair was provoked by the order, not by a singular basis.
            for (int variant = 0; variant < 2; ++variant) {
              std::vector<int> bl_try(bl_before), nbl_try(nbl_before);
              std::vector<ds::variable_status_t> vs_try(vs_before);
              if (variant == 1) { std::sort(bl_try.begin(), bl_try.end()); }
              ds::basis_update_mpf_t<int, double> ft_try(m, s.refactor_frequency);
              const int rc_try = ft_try.refactor_basis(lp.A, s, lp.lower, lp.upper, t0, bl_try,
                                                       nbl_try, vs_try);
              std::vector<int> t(bl_try.begin(), bl_try.end()), o(bl_before);
              std::sort(t.begin(), t.end());
              std::sort(o.begin(), o.end());
              std::vector<int> d;
              std::set_difference(o.begin(), o.end(), t.begin(), t.end(), std::back_inserter(d));
              std::printf("      ORDER TEST [%s]: rc=%d basis_changed=%zu\n",
                          variant == 0 ? "pivot order (control)" : "SORTED", rc_try, d.size());
            }
            std::vector<int> left;
            std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(left));
            double tot = 0.0, on_left = 0.0;
            int nviol = 0;
            for (int j = 0; j < lp.num_cols; ++j) {
              double v = 0.0;
              if (vs[j] == ds::variable_status_t::NONBASIC_LOWER && z_chk[j] < -s.dual_tol) {
                v = -z_chk[j];
              } else if (vs[j] == ds::variable_status_t::NONBASIC_UPPER && z_chk[j] > s.dual_tol) {
                v = z_chk[j];
              } else if (vs[j] == ds::variable_status_t::NONBASIC_FREE) {
                v = std::fabs(z_chk[j]);
              }
              if (v > 0.0) { tot += v; nviol++; }
              if (v > 0.0 && std::find(left.begin(), left.end(), j) != left.end()) { on_left += v; }
            }
            std::printf("    REPAIR: ejected %zu col(s); dinf_total=%.4e over %d col(s); "
                        "dinf_on_ejected=%.4e (%.1f%%)\n",
                        left.size(), tot, nviol, on_left,
                        tot > 0.0 ? 100.0 * on_left / tot : 0.0);
            for (int j : left) {
              const bool lo_fin = lp.lower[j] > -1e30, up_fin = lp.upper[j] < 1e30;
              std::printf("      ejected col %d: vstatus=%d z_fresh=%+.6e lower_finite=%d "
                          "upper_finite=%d  sign-correct status would be %s\n",
                          j, (int)vs[j], z_chk[j], (int)lo_fin, (int)up_fin,
                          z_chk[j] >= 0.0 ? (lo_fin ? "NONBASIC_LOWER" : "<none: lower is -inf>")
                                          : (up_fin ? "NONBASIC_UPPER" : "<none: upper is +inf>"));
            }
          }
          std::printf("    leg-end drift: max|z_fresh-z_kept|=%.6e (col %d, vstatus=%d) "
                      "max|y_fresh-y_kept|=%.6e  dinf(kept)=%.3e dinf(fresh)=%.3e "
                      "|B'y-c_B|: pre_refac=%.3e kept=%.3e fresh=%.3e basis_set_changed=%d\n",
                      dz, argmax, argmax >= 0 ? (int)vs[argmax] : -1, dy,
                      dual_infeasibility(lp, vs, sol.z, s.dual_tol),
                      dual_infeasibility(lp, vs, z_chk, s.dual_tol),
                      pre_resid, basis_residual(sol.y), basis_residual(y_chk), set_changed);
          std::fflush(stdout);
          const double fresh_dinf = dual_infeasibility(lp, vs, z_chk, s.dual_tol);
          if (refactor_reason != 1 && fresh_dinf > s.dual_tol) {
            std::printf("  fresh pricing is dual infeasible (%.6e) -- CPU takeback\n", fresh_dinf);
            st = ds::dual::status_t::NUMERICAL;
            break;
          }
        }
        // The device reports WHY it stopped (0 periodic / 1 rejected update / 2 certificate retry).
        if (refactor_reason == 1) {
          std::printf("  resume: rank-1 update was REJECTED (dinf carried=%.6e)\n",
                      dual_infeasibility(lp, vs, sol.z, s.dual_tol));
        } else if (refactor_reason == 2) {
          std::printf("  resume: retry terminal certificate from a fresh factorization\n");
        }
        {
          std::vector<double> xB_workspace(m, 0.0);
          double work_est = 0.0;
          ds::phase2::compute_primal_solution_from_basis(
            lp, ft, basic_list, nonbasic_list, vs, sol.x, xB_workspace, work_est);
        }
        item_refactor_resume = true;
        item_terminal_certificate_retry = (refactor_reason == 2);
      }
      const double resume_user_obj = ds::compute_user_objective<int, double>(lp, sol.x);
      {
        // A dual simplex stops when its PRIMAL infeasibility list empties. That list is maintained
        // incrementally, so a stale one lets the driver report OPTIMAL at a point that is not
        // actually feasible -- the first thing to test when an "OPTIMAL" objective is wrong.
        double bv = 0.0, rr = 0.0;
        primal_residuals(lp, sol.x, bv, rr);
        std::printf("  gpu-resume FINAL x: max bound violation=%.6e  max |Ax-b|=%.6e\n", bv, rr);

        // `dinf` above is measured on the MAINTAINED z, which is what the driver's own optimality
        // test sees. Price the final basis fresh and measure again: if the maintained z has drifted,
        // the driver can believe it is dual feasible at a basis that is not, and stop early.
        {
          ds::basis_update_mpf_t<int, double> ft_chk(m, s.refactor_frequency);
          std::vector<int> b_chk(static_cast<size_t>(m), 0), nb_chk, sb_chk;
          ds::get_basis_from_vstatus<int>(m, vs, b_chk, nb_chk, sb_chk);
          const double tchk = cuopt::mathematical_optimization::tic();
          if (ft_chk.refactor_basis(lp.A, s, lp.lower, lp.upper, tchk, b_chk, nb_chk, vs) == 0) {
            std::vector<double> y_chk(static_cast<size_t>(m), 0.0),
              z_chk(static_cast<size_t>(lp.num_cols), 0.0);
            double w_chk = 0.0;
            ds::phase2::compute_dual_solution_from_basis(lp, ft_chk, b_chk, nb_chk, y_chk, z_chk,
                                                         w_chk);
            double dz = 0.0;
            for (int j = 0; j < lp.num_cols; ++j) {
              dz = std::max(dz, std::fabs(z_chk[j] - sol.z[j]));
            }
            std::printf("  gpu-resume FINAL basis: dinf(maintained z)=%.6e  dinf(FRESH z)=%.6e  "
                        "max|z_fresh-z_kept|=%.6e\n",
                        dual_infeasibility(lp, vs, sol.z, s.dual_tol),
                        dual_infeasibility(lp, vs, z_chk, s.dual_tol), dz);
          }
        }
      }
      std::printf("%-12s %-18s %22.17g %8d\n", "gpu-resume",
                  ds::dual_status_to_string(st).c_str(), resume_user_obj, total_iters);
      resumed_st  = st;
      // USER objective, to match the three table rows above (they come from
      // compute_user_objective, which applies obj_scale and adds obj_constant). Comparing the
      // RAW sol.objective against them reported a spurious DISAGREES on every node whose
      // obj_constant is non-zero -- germanrr's is 286120.9568, exactly the gap that appeared.
      resumed_obj = resume_user_obj;
      resumed_ran = true;
      // The node was mis-decided because the objective the device reported rose ABOVE the true
      // optimum. A dual simplex objective ascends toward the optimum from below, so exceeding it is
      // the defect itself -- state it explicitly rather than leaving it to be read off the numbers.
      if (rows[0].ran && resume_user_obj > rows[0].obj + 1e-6 * (1.0 + std::abs(rows[0].obj))) {
        std::printf("DEFECT: resumed objective %.17g EXCEEDS the true optimum %.17g\n",
                    resume_user_obj, rows[0].obj);
        if (resume_user_obj >= h.cut_off) {
          std::printf("        and it is >= cut_off %.17g, which is what fathomed a live subtree\n",
                      h.cut_off);
        }
      }
    }
  }

  // Exercise two concurrent V2 pools, which is the smallest live configuration that reproduces
  // the wrong cutoff. A single pool and repeated fresh-node reuse of its seats are bit-exact.
  {
    constexpr int v2_pools = 2;
    constexpr int v2_seats = 10;
    struct replay_pool_t {
      std::vector<mip::gpu_batch_node_t<int, double>> nodes;
      std::vector<mip::batch_seat_state_t<int, double>> seat_state;
      std::unique_ptr<rmm::cuda_stream> stream;
      mip::gpu_batch_run<int, double>* run{nullptr};
    };
    std::vector<replay_pool_t> pools(v2_pools);
    for (auto& pool : pools) {
      pool.nodes.reserve(v2_seats);
      for (int seat = 0; seat < v2_seats; ++seat) {
        pool.nodes.emplace_back(lp.num_rows, lp.num_cols);
        auto& node    = pool.nodes.back();
        node.lower    = lp.lower;
        node.upper    = lp.upper;
        node.settings = settings_for();
        node.vstatus  = vstatus;
        node.dy_se    = dy_se;
      }
    }
    auto* cache = gds::gpu_ro_cache_create<int, double>();
    #pragma omp parallel for num_threads(v2_pools)
    for (int p = 0; p < v2_pools; ++p) {
      cudaSetDevice(0);
      pools[p].stream = std::make_unique<rmm::cuda_stream>();
      pools[p].run    = mip::gpu_persistent_batch_begin<int, double>(
        lp, pools[p].nodes, pools[p].stream->view(), cache, &pools[p].seat_state);
    }
    for (int p = 0; p < v2_pools; ++p) {
      bool harvested = pools[p].run != nullptr &&
                       mip::gpu_persistent_batch_harvest<int, double>(pools[p].run);
      if (pools[p].run != nullptr) {
        mip::gpu_batch_run_end<int, double>(pools[p].run);
        pools[p].run = nullptr;
      }
      if (!harvested) {
        std::printf("gpu-v2[%d] INCONCLUSIVE\n", p);
        continue;
      }
      for (int seat = 0; seat < v2_seats; ++seat) {
        const auto& node = pools[p].nodes[seat];
        const double obj = ds::compute_user_objective<int, double>(lp, node.sol.x);
        std::printf("gpu-v2[%d,%d] %-18s %.17g it=%d\n", p, seat,
                    ds::dual_status_to_string(node.status).c_str(), obj, node.sol.iterations);
      }
    }
    gds::gpu_ro_cache_destroy<int, double>(cache);
  }

  // The verdict the artifact exists to give: does the GPU still disagree with the CPU?
  // Judge the RESUMED result, not the single-shot one. REFACTOR_NEEDED is not an answer about the
  // LP -- branch-and-bound never sees it, because the host refactorizes and calls the device back.
  // Comparing the single-shot leg would report every periodic refactorization as a disagreement.
  const row_t& full = rows[0];
  std::printf("\n==== replay verdict ====\n");
  if (!full.ran || !resumed_ran) {
    std::printf("INCONCLUSIVE: a driver did not complete.\n");
  } else if (resumed_st == ds::dual::status_t::CUTOFF &&
             full.st == ds::dual::status_t::OPTIMAL && full.obj < h.cut_off) {
    std::printf("WRONG PRUNE: gpu fathomed at obj=%.17g, but the node's true optimum %.17g is "
                "below the cutoff %.17g -- a live subtree was lost.\n",
                resumed_obj, full.obj, h.cut_off);
  } else if (resumed_st == full.st && obj_close(resumed_obj, full.obj)) {
    std::printf("AGREES: gpu (resumed) %s obj=%.17g  ==  full-cpu %s obj=%.17g\n",
                ds::dual_status_to_string(resumed_st).c_str(), resumed_obj,
                ds::dual_status_to_string(full.st).c_str(), full.obj);
  } else {
    std::printf("DISAGREES: gpu (resumed) %s obj=%.17g  vs  full-cpu %s obj=%.17g\n",
                ds::dual_status_to_string(resumed_st).c_str(), resumed_obj,
                ds::dual_status_to_string(full.st).c_str(), full.obj);
  }
}


// --replay-pool mode: steady-state emulation of one GPU from a pool snapshot
// (CUOPT_BNB_GPU_SNAPSHOT). Every stored node is factorized once up front; `clones` copies of the
// pool then run on their own host threads and streams for `seconds`, each finished or
// rebuild-requesting seat restarting with the next stored node (each clone starting at its own
// offset). Rates are taken over the window after `warmup` seconds.
struct replay_gen_t {
  double t_launch, t_done;
  int64_t iters, finished, dive_nodes, abandoned, other;
};

void replay_pool_mode(const raft::handle_t& handle,
                      const std::string& file,
                      int device,
                      int clones,
                      double seconds,
                      double warmup,
                      int generations,
                      int threads,
                      bool no_dive)
{
  namespace ps = mip::pool_snapshot;
  // Default benchmark mode: the fissioned kernels (CUOPT_BNB_GPU_FISSION=0 selects the existing
  // kernel), set before anything reads it. Inverses are built on the device every generation; FTRAN
  // uses them unless CUOPT_EXP_FTRAN=direct; phase kernels are separate unless CUOPT_EXP_FUSED=1.
  setenv("CUOPT_BNB_GPU_FISSION", "1", 0);
  {
    auto env = [](const char* k, const char* d) { const char* e = std::getenv(k); return std::string(e != nullptr ? e : d); };
    const bool fis = env("CUOPT_BNB_GPU_FISSION", "1") != "0";
    std::printf("[MODE] kernel=%s%s\n", fis ? "fission" : "existing (one warp per seat)",
                fis ? (std::string(" inverses=device-built-per-generation ftran=") +
                       (env("CUOPT_EXP_FTRAN", "") == "direct" ? "direct" : "inverse") +
                       " phases=" + (env("CUOPT_EXP_FUSED", "0") != "0" ? "fused" : "separate") +
                       " scatter=" + (env("CUOPT_EXP_SCATTER", "") == "segment" ? "segment" : "balanced-batched") +
                       " fresh-duals=" + (env("CUOPT_EXP_PRICE_FRESH", "") == "0" ? "host-uploaded" : "device") +
                       " tloop=" + (env("CUOPT_EXP_TLOOP", "") == "old" ? "old" : "split-sums+balanced-axpy") +
                       " update=" + (env("CUOPT_EXP_UPDATE", "") == "old" ? "old" : "fused-cached") +
                       " ftran-dy-U=" + (env("CUOPT_EXP_FTRAN_DY", "") == "gather" ? "all-rows" : env("CUOPT_EXP_FTRAN_DY", "") == "scatter" ? "scatter" : "se-rows") +
                       " bounds-strengthen+setup=device")
                          .c_str()
                    : "");
  }
  {
    // The stored nodes stand in for nodes built by B&B workers spread over the machine: interleave
    // this process's host pages over all NUMA nodes (one node's bandwidth caps the staging copies).
    unsigned long mask[16] = {};
    for (int nd = 0; nd < 1024; ++nd) {
      if (access(("/sys/devices/system/node/node" + std::to_string(nd)).c_str(), F_OK) == 0) {
        mask[nd / 64] |= 1ul << (nd % 64);
      }
    }
    constexpr int kMpolInterleave = 3;
    if (syscall(SYS_set_mempolicy, kMpolInterleave, mask, 1024) != 0) {
      std::perror("replay-pool: set_mempolicy(MPOL_INTERLEAVE)");
    }
  }
  ps::pool_t<int, double> snap(&handle);
  std::string why;
  if (!ps::read<int, double>(file, snap, why)) {
    std::fprintf(stderr, "replay-pool: cannot load %s (%s)\n", file.c_str(), why.c_str());
    std::exit(1);
  }
  const auto& h  = snap.h;
  const auto& lp = snap.lp;
  const int m = h.m, n = h.n;
  if (clones <= 0) { clones = h.pools_on_device; }
  std::printf("replay-pool %s: m=%d n=%d seats=%d (new %d, rebuild %d, child %d, empty %d) captured "
              "t=%.1fs on gpu %d with %d pools; replay on gpu %d with %d clones\n",
              file.c_str(), m, n, h.seats, h.count[0], h.count[1], h.count[2], h.count[3],
              h.elapsed, h.device, h.pools_on_device, device, clones);

  // Stored nodes: every non-empty seat, as a new node from its basis.
  std::vector<mip::gpu_batch_node_t<int, double>> stored;
  for (const auto& s : snap.seats) {
    if (s.rec.command == ps::PARKED) { continue; }
    stored.emplace_back(m, n);
    auto& node = stored.back();
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
  const double f0 = mo::tic();
  const auto ok   = mip::gpu_batch_factor_fresh<int, double>(lp, stored, prepared, threads);
  std::vector<int> usable;
  for (size_t k = 0; k < ok.size(); ++k) {
    if (ok[k]) { usable.push_back(static_cast<int>(k)); }
  }
  std::printf("prepared %zu of %zu stored nodes (host LU + duals) in %.2fs\n", usable.size(),
              stored.size(), mo::toc(f0));
  if (usable.empty()) { std::exit(1); }
  const int U = static_cast<int>(usable.size());
  // Setup, untimed: every stored node starts from exact steepest-edge weights for its basis, so the
  // kernels are compared on fresh weights. TEST_NETLIB_STORED_SE=1 keeps the snapshot's weights.
  if (std::getenv("TEST_NETLIB_STORED_SE") != nullptr) {
    std::printf("[SE] weights=stored (snapshot)\n");
  } else {
    cudaSetDevice(device);
    const double t_se = se_exact_weights(m, prepared, usable, stored);
    std::printf("[SE] weights=exact (device inverses + weight kernel, untimed setup) for %d stored nodes in %.3fs\n", U, t_se);
  }

  std::vector<ds::variable_type_t> var_types(h.n_var_types);
  for (int j = 0; j < h.n_var_types; ++j) {
    var_types[j] = static_cast<ds::variable_type_t>(snap.var_types[j]);
  }
  std::vector<int> pc_num_down(snap.pc_num_down.begin(), snap.pc_num_down.end());
  std::vector<int> pc_num_up(snap.pc_num_up.begin(), snap.pc_num_up.end());

  auto* cache = gds::gpu_ro_cache_create<int, double>();
  gds::gpu_ro_cache_prebuild<int, double>(cache, lp);

  const int N = h.seats;
  std::vector<std::vector<replay_gen_t>> gens(clones);
  std::vector<std::string> errors(clones);
  std::atomic<int> started{0};
  const double t0 = mo::tic();
  auto clone_main = [&](int c) {
    cudaSetDevice(device);
    rmm::cuda_stream stream;
    std::vector<mip::gpu_batch_node_t<int, double>> nodes;
    std::vector<mip::batch_seat_state_t<int, double>> seat_state;
    int cursor = static_cast<int>((static_cast<int64_t>(c) * U) / clones);
    std::vector<int> seat_node(N);
    for (int s = 0; s < N; ++s) {
      seat_node[s] = usable[cursor];
      cursor       = (cursor + 1) % U;
      nodes.push_back(stored[seat_node[s]]);
    }
    auto* run = mip::gpu_persistent_batch_begin<int, double>(
      lp, nodes, stream.view(), cache, &seat_state, true, var_types.empty() ? nullptr : &var_types);
    if (run == nullptr) {
      errors[c] = "first launch failed";
      return;
    }
    ++started;
    double t_launch = mo::toc(t0);
    std::string err;
    const bool exp_ht = std::getenv("CUOPT_EXP_HOST_TIME") != nullptr;
    std::vector<double> ht_h, ht_s, ht_r;
    for (int g = 0;; ++g) {
      const double th0 = mo::toc(t0);
      if (!mip::gpu_persistent_batch_harvest<int, double>(run)) {
        errors[c] = "harvest failed";
        break;
      }
      if (exp_ht) { ht_h.push_back(mo::toc(t0) - th0); }
      replay_gen_t r{t_launch, mo::toc(t0), 0, 0, 0, 0, 0};
      for (int s = 0; s < N; ++s) {
        if (h.dive_rounds > 0 && g > 0 && !no_dive) {
          const auto dive = mip::gpu_persistent_batch_dive<int, double>(run, s);
          for (int q = 0; q < dive.count; ++q) { r.iters += dive.records[q].iterations; }
          r.dive_nodes += dive.count;
        }
        const auto v = mip::gpu_persistent_batch_result<int, double>(run, s);
        r.iters += v.iterations;
        if (v.status == ds::dual::status_t::REFACTOR_NEEDED) {
          ++r.abandoned;
        } else if (v.status == ds::dual::status_t::NUMERICAL ||
                   v.status == ds::dual::status_t::ITERATION_LIMIT) {
          ++r.other;
        } else {
          ++r.finished;
        }
      }
      if (generations > 0 && c == 0 && g + 1 == generations) {
        for (int s = 0; s < N; ++s) {
          const auto v = mip::gpu_persistent_batch_result<int, double>(run, s);
          std::printf("seat %4d node %4d %-18s obj=%.17g it=%d\n", s, seat_node[s],
                      ds::dual_status_to_string(v.status).c_str(), v.objective, v.iterations);
        }
      }
      gens[c].push_back(r);
      if (generations > 0 ? g + 1 >= generations : r.t_done >= seconds) { break; }
      const double ts0 = mo::toc(t0);
      for (int s = 0; s < N; ++s) {
        seat_node[s] = usable[cursor];
        cursor       = (cursor + 1) % U;
        if (!mip::gpu_persistent_batch_stage_fresh_from<int, double>(
              run, s, stored[seat_node[s]], prepared[seat_node[s]], err)) {
          errors[c] = "stage: " + err;
          break;
        }
      }
      if (!errors[c].empty()) { break; }
      if (h.n_pc > 0) {
        mip::gpu_batch_run_set_pseudocosts<int, double>(run,
                                                        snap.pc_sum_down.data(),
                                                        snap.pc_sum_up.data(),
                                                        pc_num_down.data(),
                                                        pc_num_up.data(),
                                                        h.pc_avg_down,
                                                        h.pc_avg_up,
                                                        h.reliable_threshold,
                                                        h.integer_tol);
        if (h.dive_rounds > 0 && !no_dive) {
          mip::gpu_batch_run_set_dive<int, double>(run, h.dive_rounds, snap.root_x.data());
        }
      }
      t_launch = mo::toc(t0);
      if (exp_ht) { ht_s.push_back(t_launch - ts0); }
      if (!mip::gpu_persistent_batch_relaunch<int, double>(run, err)) {
        errors[c] = "relaunch: " + err;
        break;
      }
      if (exp_ht) { ht_r.push_back(mo::toc(t0) - t_launch); }
    }
    if (exp_ht && c == 0) {
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      std::printf("[EXP-HOST] per generation, median ms: harvest call %.1f (incl. GPU wait), stage loop %.1f, relaunch call %.1f\n",
                  1e3 * med(ht_h), 1e3 * med(ht_s), 1e3 * med(ht_r));
      const double* ss = gds::gpu_batch_stage_split();
      const double g   = std::max<size_t>(1, ht_s.size() + 1);
      std::printf("[EXP-HOST] stage split, mean ms per generation (clone 0's thread, first launch included): checks %.1f, bounds %.1f, seat vectors %.1f, factors+rest %.1f\n",
                  1e3 * ss[0] / g, 1e3 * ss[1] / g, 1e3 * ss[2] / g, 1e3 * ss[3] / g);
    }
    mip::gpu_batch_run_end<int, double>(run);
  };
  std::vector<std::thread> workers;
  for (int c = 0; c < clones; ++c) { workers.emplace_back(clone_main, c); }
  for (auto& w : workers) { w.join(); }
  gds::gpu_ro_cache_destroy<int, double>(cache);
  for (int c = 0; c < clones; ++c) {
    if (!errors[c].empty()) { std::printf("clone %d: ERROR %s\n", c, errors[c].c_str()); }
  }

  // Window: generations completing after `warmup`, up to the earliest clone's last completion.
  double t_end = std::numeric_limits<double>::infinity();
  for (const auto& gc : gens) {
    if (!gc.empty()) { t_end = std::min(t_end, gc.back().t_done); }
  }
  const double t_beg = generations > 0 ? 0.0 : warmup;
  int64_t iters = 0, finished = 0, dive_nodes = 0, abandoned = 0, other = 0, ngen = 0;
  std::vector<double> gen_s, host_s;
  for (const auto& gc : gens) {
    for (size_t g = 0; g < gc.size(); ++g) {
      const auto& r = gc[g];
      if (r.t_done <= t_beg || r.t_done > t_end) { continue; }
      if (g + 1 < gc.size()) { host_s.push_back(gc[g + 1].t_launch - r.t_done); }
      iters += r.iters;
      finished += r.finished;
      dive_nodes += r.dive_nodes;
      abandoned += r.abandoned;
      other += r.other;
      ++ngen;
      gen_s.push_back(r.t_done - r.t_launch);
    }
  }
  const double win = t_end - t_beg;
  std::sort(gen_s.begin(), gen_s.end());
  const auto pct = [&](double p) {
    return gen_s.empty() ? 0.0 : gen_s[std::min(gen_s.size() - 1, size_t(p * gen_s.size()))];
  };
  std::printf("window %.2f-%.2fs (%.2fs): generations=%lld iterations=%lld -> %.3fM it/s | nodes "
              "finished=%lld (+%lld in dives) -> %.0f nodes/s | abandoned (LU rebuild)=%lld "
              "numerical/iter-limit=%lld\n",
              t_beg, t_end, win, (long long)ngen, (long long)iters, iters / win / 1e6,
              (long long)finished, (long long)dive_nodes, (finished + dive_nodes) / win,
              (long long)abandoned, (long long)other);
  std::printf("generation launch-to-results seconds: p10=%.3f p50=%.3f p90=%.3f max=%.3f\n",
              pct(0.1), pct(0.5), pct(0.9), gen_s.empty() ? 0.0 : gen_s.back());
  std::sort(host_s.begin(), host_s.end());
  if (!host_s.empty()) {
    std::printf("host results-to-relaunch seconds (restage all seats): p50=%.3f max=%.3f\n",
                host_s[host_s.size() / 2], host_s.back());
  }
  {
    if (std::getenv("CUOPT_EXP_HOST_TIME") != nullptr) {
      const std::string hv = gds::gpu_batch_harvest_report();
      if (!hv.empty()) { std::printf("%s", hv.c_str()); }
    }
    const std::string fs = gds::gpu_fission_step_report();
    std::istringstream in(fs);
    for (std::string line; std::getline(in, line);) { std::printf("%s\n", line.c_str()); }
  }
  const std::string clk = gds::gpu_batch_seat_clock_report();
  if (!clk.empty()) {
    std::istringstream in(clk);
    for (std::string line; std::getline(in, line);) { std::printf("%s\n", line.c_str()); }
  }
}

}  // namespace

int main(int argc, char* argv[])
{
  argparse::ArgumentParser program("test_netlib");
  program.add_argument("--dataset-dir")
    .help("directory containing <name>/<name>.mps NETLIB instances")
    .default_value(std::string(
      "/home/nblin/Desktop/another_claude_cuopt/datasets/linear_programming"));
  program.add_argument("--time-limit")
    .help("per-instance time limit in seconds")
    .default_value(600.0)
    .scan<'g', double>();
  program.add_argument("--dual-inf-gate")
    .help("max dual infeasibility of the basis handed to simplified_dual_phase2 "
          "(0.1 admits pilot87, whose reconstructed phase-1 basis is 0.057 dual-infeasible; "
          "the minimal driver's perturbation absorbs it and solves to the optimum)")
    .default_value(0.1)
    .scan<'g', double>();
  program.add_argument("--instance").help("run only this single instance (by name)");
  program.add_argument("--verify-nodes")
    .help("directory of B&B node samples (node_*/ each with parent.mps, child.mps, verify.log) "
          "dumped by cuopt_cli with CUOPT_DUMP_NODES=<dir>; solve each parent to its optimal "
          "basis, then replay the child from that basis through the advanced and simplified "
          "drivers (with the recorded objective cutoff) and check the recorded status/objective");
  program.add_argument("--verify-nodes-batch")
    .help("GPU-only resumable batch verification of a node collection. Nodes are grouped by "
          "instance, solved through the production per-seat host-refactor loop, and emitted as "
          "full-precision [NODEDUMP] rows for comparison with a frozen full-CPU golden log");
  program.add_argument("--batch-seats")
    .help("maximum nodes per resumable GPU verification batch")
    .default_value(8)
    .scan<'i', int>();
  program.add_argument("--replay-node")
    .help("binary bad-node dump written by the B&B GPU hook under "
          "CUOPT_BNB_DUMP_BAD_NODES=<dir>; re-solve that single node from its captured "
          "state with the full CPU driver, the simplified CPU driver and the GPU driver");
  program.add_argument("--benchmark-parallel")
    .help("directory of B&B node samples (as for --verify-nodes); replicate the nodes --replicas "
          "times and solve them ALL on the CPU (OpenMP across problems) and the GPU (one block per "
          "problem, single launch), reporting total runtime for each and verifying every result is "
          "bit-for-bit identical to a single-thread sequential reference");
  program.add_argument("--ftran-bench")
    .help("pool snapshot: batched base-factor FTRAN kernels on its nodes, --replicas copies each "
          "(needs --device)");
  program.add_argument("--ftran-repeats").default_value(5).scan<'i', int>();
  program.add_argument("--se-bench")
    .help("pool snapshot: exact steepest-edge weights of its nodes from device-built explicit inverses vs "
          "the host initialize_steepest_edge_norms (needs --device)");
  program.add_argument("--se-host-seats")
    .help("--se-bench: seats timed single-threaded on the host")
    .default_value(32)
    .scan<'i', int>();
  program.add_argument("--inv-bench")
    .help("pool snapshot: explicit inverses of its nodes' base factors built on the device vs the "
          "single-thread host build plus upload (needs --device)");
  program.add_argument("--replay-pool")
    .help("pool snapshot written under CUOPT_BNB_GPU_SNAPSHOT=<file>: emulate one GPU's steady state "
          "with --clones copies of the pool for --seconds (needs --device)");
  program.add_argument("--device").help("CUDA device for --replay-pool").default_value(-1).scan<'i', int>();
  program.add_argument("--clones")
    .help("pool copies for --replay-pool (0: the captured GPU's pool count)")
    .default_value(0)
    .scan<'i', int>();
  program.add_argument("--seconds").help("--replay-pool duration").default_value(30.0).scan<'g', double>();
  program.add_argument("--warmup")
    .help("--replay-pool: seconds excluded from the rates")
    .default_value(5.0)
    .scan<'g', double>();
  program.add_argument("--generations")
    .help("--replay-pool: run exactly this many launches per clone and print clone 0's seats "
          "(correctness check); 0 = timed")
    .default_value(0)
    .scan<'i', int>();
  program.add_argument("--no-dive").help("--replay-pool: no in-kernel dives").flag();
  program.add_argument("--prepare-threads")
    .help("--replay-pool: host threads for the up-front LUs")
    .default_value(32)
    .scan<'i', int>();
  program.add_argument("--replicas")
    .help("replication factor for --benchmark-parallel (unique nodes x this = total problems)")
    .default_value(10)
    .scan<'i', int>();
  program.add_argument("--solve-mps")
    .help("solve a single MPS file FROM SCRATCH (cold start, no cutoff) via both "
          "solve_linear_program and dual_phase2 phase1+phase2, and report the status/objective "
          "(diagnostic: is a node child LP genuinely dual-unbounded, or only when warm-started?)");
  program.add_argument("--mps-cutoff")
    .help("optional finite objective cutoff for --solve-mps: also run solve_linear_program with "
          "this cutoff, to check whether a from-scratch solve terminates via CUTOFF vs INFEASIBLE")
    .default_value(std::numeric_limits<double>::infinity())
    .scan<'g', double>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "%s\n", err.what());
    std::cerr << program;
    return 1;
  }

  const std::string dataset_dir = program.get<std::string>("--dataset-dir");
  const double time_limit       = program.get<double>("--time-limit");
  const double dual_inf_gate    = program.get<double>("--dual-inf-gate");

  // Route the solvers' settings.log to stdout (raw messages) so the verbose
  // per-iteration traces are visible.
  cuopt::init_logger_t logger("", true);
  cuopt::default_logger().set_pattern("%v");

  // Set up the RMM memory pool, exactly as run_pdlp.cu does. Installing the device resource is the
  // APPLICATION's job, not the solver's: gpu_phase2 allocates rmm::device_buffers from whatever
  // resource is current, so without a pool here every marshalled buffer is its own cudaMalloc
  // (measured: 41,300 cudaMallocs, 0.234 s, on a 1,200-problem batch).
  //
  // NOTE make_pool() reserves 40% of FREE VRAM up front; the batch pre-flight adds the current RMM
  // pool's size back to free VRAM, so a batch is not rejected merely because the pool exists.
  if (program.is_used("--replay-pool") || program.is_used("--ftran-bench") || program.is_used("--inv-bench") ||
      program.is_used("--se-bench")) {
    const int dev = program.get<int>("--device");
    if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) {
      std::fprintf(stderr, "--replay-pool needs a valid --device\n");
      return 1;
    }
    cudaDeviceProp p{};
    int clk_khz = 0, mclk_khz = 0;
    cudaGetDeviceProperties(&p, dev);
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, dev);
    std::printf("[DEVICE] index=%d name=\"%s\" cc=%d.%d sms=%d max_sm_clock_mhz=%d mem_clock_mhz=%d "
                "smem_per_sm_kb=%zu mem_gb=%.1f pci=%04x:%02x:%02x\n",
                dev, p.name, p.major, p.minor, p.multiProcessorCount, clk_khz / 1000, mclk_khz / 1000,
                p.sharedMemPerMultiprocessor / 1024, p.totalGlobalMem / 1e9, p.pciDomainID, p.pciBusID,
                p.pciDeviceID);
  }
  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource);

  const raft::handle_t handle{};

  // --verify-nodes mode: replay dumped B&B node samples and exit.
  if (program.is_used("--verify-nodes")) {
    verify_nodes_mode(handle, program.get<std::string>("--verify-nodes"), time_limit);
    return 0;
  }

  if (program.is_used("--verify-nodes-batch")) {
    verify_nodes_batch_mode(handle,
                            program.get<std::string>("--verify-nodes-batch"),
                            time_limit,
                            program.get<int>("--batch-seats"));
    return 0;
  }

  if (program.is_used("--se-bench")) {
    return se_bench_main(handle, program.get<std::string>("--se-bench"), program.get<int>("--ftran-repeats"),
                         program.get<int>("--prepare-threads"), program.get<int>("--se-host-seats"));
  }
  if (program.is_used("--inv-bench")) {
    return inv_bench_main(handle, program.get<std::string>("--inv-bench"), program.get<int>("--ftran-repeats"),
                          program.get<int>("--prepare-threads"));
  }
  if (program.is_used("--ftran-bench")) {
    return ftran_bench_main(handle,
                            program.get<std::string>("--ftran-bench"),
                            program.get<int>("--replicas"),
                            program.get<int>("--ftran-repeats"),
                            program.get<int>("--prepare-threads"));
  }
  if (program.is_used("--replay-pool")) {
    replay_pool_mode(handle,
                     program.get<std::string>("--replay-pool"),
                     program.get<int>("--device"),
                     program.get<int>("--clones"),
                     program.get<double>("--seconds"),
                     program.get<double>("--warmup"),
                     program.get<int>("--generations"),
                     program.get<int>("--prepare-threads"),
                     program.get<bool>("--no-dive"));
    return 0;
  }

  // --replay-node mode: re-solve one captured bad node with all three drivers and exit.
  if (program.is_used("--replay-node")) {
    replay_node_mode(handle, program.get<std::string>("--replay-node"), time_limit);
    return 0;
  }

  // --benchmark-parallel mode: replicate the node samples and race CPU-OpenMP vs GPU-multiblock.
  if (program.is_used("--benchmark-parallel")) {
    benchmark_parallel_mode(handle, program.get<std::string>("--benchmark-parallel"),
                            program.get<int>("--replicas"), time_limit);
    return 0;
  }

  // --solve-mps mode: cold-solve a single MPS and exit (diagnostic).
  if (program.is_used("--solve-mps")) {
    solve_mps_from_scratch_mode(handle,
                                program.get<std::string>("--solve-mps"),
                                time_limit,
                                program.get<double>("--mps-cutoff"));
    return 0;
  }

  // Collect <name>/<name>.mps instances.
  std::vector<std::pair<std::string, std::string>> instances;  // (name, path)
  namespace fs = std::filesystem;
  if (program.is_used("--instance")) {
    const std::string name = program.get<std::string>("--instance");
    instances.emplace_back(name, dataset_dir + "/" + name + "/" + name + ".mps");
  } else {
    for (const auto& entry : fs::directory_iterator(dataset_dir)) {
      if (!entry.is_directory()) continue;
      const std::string name = entry.path().filename().string();
      const std::string path = entry.path().string() + "/" + name + ".mps";
      if (fs::exists(path)) instances.emplace_back(name, path);
    }
    std::sort(instances.begin(), instances.end());
  }

  std::printf("Found %zu instances in %s\n\n", instances.size(), dataset_dir.c_str());
  std::printf(
    "%-12s | %-10s %14s %6s | %-16s %6s %8s | %-16s %14s %6s %8s | %-16s %14s %6s | %-9s | %s\n",
    "instance", "base", "base_obj", "it", "adv(full)", "adv_it", "adv_s", "simpl(CPU)", "simpl_obj",
    "it", "sim_s", "gpu", "gpu_obj", "it", "dual_inf", "gpu==cpu");
  std::fflush(stdout);

  std::vector<instance_result_t> results;
  results.reserve(instances.size());

  // Verbose per-iteration trace only makes sense for a single instance.
  const bool verbose = program.is_used("--instance");

  for (const auto& [name, path] : instances) {
    instance_result_t r;
    try {
      process_instance(handle, name, path, time_limit, dual_inf_gate, verbose, r);
    } catch (const std::exception& e) {
      r.name = name;
      r.note = std::string("exception: ") + e.what();
      std::fprintf(stderr, "[%s] %s\n", name.c_str(), r.note.c_str());
    }

    const std::string base_s = ds::lp_status_to_string(r.base_status);
    const std::string adv_s  = ds::dual_status_to_string(r.adv_status);
    const std::string sim_s  = r.simpl_ran ? ds::dual_status_to_string(r.simpl_status)
                                           : (r.note.empty() ? "-" : r.note);
    const std::string gpu_s  = r.gpu_ran ? ds::dual_status_to_string(r.gpu_status) : "-";
    // Results match: same status, and for OPTIMAL the objectives agree. (The CPU
    // does not set sol.iterations on its mid-loop NUMERICAL return, so iteration
    // equality is only meaningful for OPTIMAL.)
    const bool gpu_results_match =
      r.gpu_ran && r.simpl_ran && r.gpu_status == r.simpl_status &&
      (r.simpl_status != ds::dual::status_t::OPTIMAL || obj_matches(r.gpu_obj, r.simpl_obj));
    const bool gpu_bit_exact = gpu_results_match &&
                               r.simpl_status == ds::dual::status_t::OPTIMAL &&
                               r.gpu_iters == r.simpl_iters;
    const char* match_str =
      !r.gpu_ran ? "-" : (gpu_bit_exact ? "yes" : (gpu_results_match ? "obj" : "no"));
    std::printf(
      "%-12s | %-10s %14.6g %6d | %-16s %6d %8.4f | %-16s %14.6g %6d %8.4f | %-16s %14.6g %6d | %9.1e | %s\n",
      r.name.c_str(), base_s.c_str(), r.base_obj, r.base_iters, adv_s.c_str(), r.adv_iters,
      r.adv_time, sim_s.c_str(), r.simpl_obj, r.simpl_iters, r.simpl_time, gpu_s.c_str(), r.gpu_obj,
      r.gpu_iters, r.dual_inf, match_str);
    std::fflush(stdout);
    results.push_back(std::move(r));
  }

  // ---- Summary ----
  int base_optimal = 0, adv_optimal_match = 0, simpl_optimal_match = 0, gate_skipped = 0,
      setup_failed = 0;
  for (const auto& r : results) {
    if (r.base_status == ds::lp_status_t::OPTIMAL) base_optimal++;
    if (r.adv_ran && r.adv_status == ds::dual::status_t::OPTIMAL && obj_matches(r.adv_obj, r.base_obj))
      adv_optimal_match++;
    if (!r.setup_ok) setup_failed++;
    else if (!r.simpl_ran) gate_skipped++;
    else if (r.simpl_status == ds::dual::status_t::OPTIMAL && obj_matches(r.simpl_obj, r.base_obj))
      simpl_optimal_match++;
  }

  // Categorise the remaining failures now that the driver refactors.
  int fail_iter_limit = 0, fail_numerical = 0, fail_unbounded = 0, fail_other = 0;
  for (const auto& r : results) {
    if (!r.simpl_ran) continue;
    if (r.simpl_status == ds::dual::status_t::OPTIMAL && obj_matches(r.simpl_obj, r.base_obj))
      continue;
    switch (r.simpl_status) {
      case ds::dual::status_t::ITERATION_LIMIT: fail_iter_limit++; break;
      case ds::dual::status_t::NUMERICAL: fail_numerical++; break;
      case ds::dual::status_t::DUAL_UNBOUNDED: fail_unbounded++; break;
      default: fail_other++; break;
    }
  }

  // GPU vs CPU comparison (the metric of success).
  int gpu_optimal_match = 0;  // GPU OPTIMAL & obj matches baseline
  int gpu_results_match = 0;  // GPU reproduces CPU result (status, + obj if OPTIMAL)
  int gpu_bit_exact     = 0;  // also identical iteration count (OPTIMAL)
  int gpu_time_limited  = 0;  // GPU hit its wall-clock limit (single-thread too slow)
  int gpu_ran_count     = 0;
  for (const auto& r : results) {
    if (!r.gpu_ran) continue;
    gpu_ran_count++;
    if (r.gpu_status == ds::dual::status_t::OPTIMAL && obj_matches(r.gpu_obj, r.base_obj))
      gpu_optimal_match++;
    if (r.gpu_status == ds::dual::status_t::TIME_LIMIT) gpu_time_limited++;
    if (!r.simpl_ran) continue;
    const bool rmatch =
      r.gpu_status == r.simpl_status &&
      (r.simpl_status != ds::dual::status_t::OPTIMAL || obj_matches(r.gpu_obj, r.simpl_obj));
    if (rmatch) {
      gpu_results_match++;
      if (r.simpl_status == ds::dual::status_t::OPTIMAL && r.gpu_iters == r.simpl_iters)
        gpu_bit_exact++;
    }
  }

  std::printf("\n==== Summary ====\n");
  std::printf("instances                              : %zu\n", results.size());
  std::printf("baseline OPTIMAL (full dual simplex)   : %d / %zu\n", base_optimal, results.size());
  std::printf("full phase2 (advanced) OPTIMAL & match : %d / %zu\n", adv_optimal_match,
              results.size());
  std::printf("minimal (simplified) OPTIMAL & match   : %d / %zu\n", simpl_optimal_match,
              results.size());
  std::printf("minimal solved vs full phase2          : %d / %d  (gap %d)\n", simpl_optimal_match,
              adv_optimal_match, adv_optimal_match - simpl_optimal_match);
  if (gpu_ran_count > 0) {
    std::printf("GPU OPTIMAL & obj matches base         : %d / %zu\n", gpu_optimal_match,
                results.size());
    std::printf("GPU reproduces CPU result (status/obj) : %d / %d ran\n", gpu_results_match,
                gpu_ran_count);
    std::printf("  of which bit-exact (same iters)      : %d\n", gpu_bit_exact);
    std::printf("GPU TIME_LIMIT (single-thread too slow): %d\n", gpu_time_limited);
  }
  std::printf("setup failed (no dual-feasible basis)  : %d\n", setup_failed);
  std::printf("gate-skipped (basis not dual feasible) : %d\n", gate_skipped);
  std::printf("simplified ITERATION_LIMIT             : %d\n", fail_iter_limit);
  std::printf("simplified NUMERICAL                   : %d\n", fail_numerical);
  std::printf("simplified DUAL_UNBOUNDED              : %d\n", fail_unbounded);
  std::printf("simplified other failure               : %d\n", fail_other);

  // Failure breakdown for instances that had a valid basis but simplified did not match.
  std::printf("\n---- simplified failures (had valid basis) ----\n");
  std::printf("%-12s %-16s %8s %14s %14s\n", "instance", "simpl_status", "iters", "simpl_obj",
              "base_obj");
  for (const auto& r : results) {
    if (!r.simpl_ran) continue;
    const bool ok =
      r.simpl_status == ds::dual::status_t::OPTIMAL && obj_matches(r.simpl_obj, r.base_obj);
    if (ok) continue;
    std::printf("  %-12s %-16s %8d %14.6g %14.6g\n", r.name.c_str(),
                ds::dual_status_to_string(r.simpl_status).c_str(), r.simpl_iters, r.simpl_obj,
                r.base_obj);
  }

  // GPU-vs-CPU mismatches (where the GPU port did NOT exactly reproduce the CPU
  // simplified driver). The success metric is that this list is empty. Only shown
  // when the GPU driver actually ran (TEST_NETLIB_GPU=1).
  if (gpu_ran_count > 0) {
    std::printf("\n---- GPU vs CPU simplified mismatches ----\n");
    std::printf("%-12s %-16s %-16s %8s %8s %14s %14s\n", "instance", "cpu_status", "gpu_status",
                "cpu_it", "gpu_it", "cpu_obj", "gpu_obj");
    int n_mismatch = 0;
    for (const auto& r : results) {
      if (!r.gpu_ran || !r.simpl_ran) continue;
      const bool ok =
        r.gpu_status == r.simpl_status &&
        (r.simpl_status != ds::dual::status_t::OPTIMAL || obj_matches(r.gpu_obj, r.simpl_obj));
      if (ok) continue;
      n_mismatch++;
      std::printf("  %-12s %-16s %-16s %8d %8d %14.6g %14.6g\n", r.name.c_str(),
                  ds::dual_status_to_string(r.simpl_status).c_str(),
                  ds::dual_status_to_string(r.gpu_status).c_str(), r.simpl_iters, r.gpu_iters,
                  r.simpl_obj, r.gpu_obj);
    }
    if (n_mismatch == 0) { std::printf("  (none) - GPU exactly reproduced the CPU driver\n"); }
  }

  if (base_optimal != static_cast<int>(results.size())) {
    std::printf(
      "\nWARNING: not all instances solved to OPTIMAL by the baseline dual simplex solver.\n");
  }

  return 0;
}
