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

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>

#include <pdlp/translate.hpp>

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
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>

#include <argparse/argparse.hpp>

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark_helper.hpp"

namespace ds = cuopt::linear_programming::dual_simplex;

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
  auto model = cuopt::linear_programming::io::parse_mps<int, double>(path);
  auto opt =
    cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(&handle, model);
  return cuopt::linear_programming::cuopt_optimization_problem_to_user_problem<int, double>(&handle,
                                                                                           opt);
}

// y solves B' y = c_B ; z_N = c_N - N' y ; z_B = 0.  Mirrors
// compute_dual_solution_from_basis (phase2.cpp), which is not header-declared.
void compute_duals(const ds::lp_problem_t<int, double>& lp,
                   ds::basis_update_mpf_t<int, double>& ft,
                   const std::vector<int>& basic_list,
                   const std::vector<int>& nonbasic_list,
                   std::vector<double>& y,
                   std::vector<double>& z)
{
  const int m = lp.num_rows;
  const int n = lp.num_cols;
  std::vector<double> cB(m);
  for (int k = 0; k < m; ++k) {
    cB[k] = lp.objective[basic_list[k]];
  }
  y.assign(m, 0.0);
  ft.b_transpose_solve(cB, y);

  z.assign(n, 0.0);
  for (std::size_t k = 0; k < nonbasic_list.size(); ++k) {
    const int j = nonbasic_list[k];
    double dot  = 0.0;
    for (int p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
      dot += lp.A.x[p] * y[lp.A.i[p]];
    }
    z[j] = lp.objective[j] - dot;
  }
  for (int k = 0; k < m; ++k) {
    z[basic_list[k]] = 0.0;
  }
}

// Sum of dual-feasibility sign violations of the reduced costs against the
// nonbasic statuses.  ~0 means the basis is dual feasible (minimization form).
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
  ds::column_scaling(presolved_lp, settings, lp, column_scales);

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
  const double start_time = ds::tic();
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
  const double start_time = ds::tic();
  ft.refactor_basis(
    lp.A, settings, lp.lower, lp.upper, start_time, basic_list, nonbasic_list, vstatus);

  ds::lp_solution_t<int, double> sol(m, n);
  compute_duals(lp, ft, basic_list, nonbasic_list, sol.y, sol.z);
  out_dual_inf = dual_infeasibility(lp, vstatus, sol.z, settings.dual_tol);

  std::vector<double> dy_se;  // empty -> cold-start steepest-edge weights
  // Diagnostic (TEST_NETLIB_EXACT_SE=1, CPU minimal only): seed dy_se with the EXACT
  // initial steepest-edge reference norms gamma_k = ||B^{-T} e_k||^2 for basic vars,
  // instead of the minimal driver's crude slack-basis init (basic=1.0). Tests whether
  // the SE-norm initialization is what makes pilot/woodw land on a suboptimal vertex.
  if (mode == 1 && std::getenv("TEST_NETLIB_EXACT_SE") != nullptr) {
    dy_se.assign(n, 1e-4);  // nonbasic default (unused until a var enters)
    std::vector<double> ek(m, 0.0), y(m, 0.0);
    for (int k = 0; k < m; ++k) {
      ek[k] = 1.0;
      ft.b_transpose_solve(ek, y);
      double g = 0.0;
      for (int i = 0; i < m; ++i) g += y[i] * y[i];
      dy_se[basic_list[k]] = g;
      ek[k] = 0.0;
    }
  }
  // Exact replay: feed the B&B's carried steepest-edge norms verbatim (the driver
  // uses a non-empty dy_se as-is instead of reinitializing).
  if (!dy_se_override.empty()) { dy_se = dy_se_override; }
  ds::dual::status_t status;
  ft.dbg_se_reset();  // [SESTAT] temp: count only THIS driver's SE-context b_solves
  // Time JUST the phase-2 driver call (the refactor_basis setup above is identical for both
  // the advanced and the minimal driver, so isolating the driver is the fair time comparison).
  const double driver_t0 = ds::tic();
  if (mode == 1) {
    status =
      ds::simplified_dual_phase2(lp, settings, vstatus, ft, basic_list, nonbasic_list, sol, dy_se);
    out_iters = sol.iterations;
  } else if (mode == 2) {
    status = ds::gpu_simplified_dual_phase2(
      lp, settings, vstatus, ft, basic_list, nonbasic_list, sol, dy_se);
    out_iters = sol.iterations;
  } else {
    int iter = 0;
    status   = ds::dual_phase2_with_advanced_basis(2,
                                                 0,
                                                 initialize_basis,
                                                 start_time,
                                                 lp,
                                                 settings,
                                                 vstatus,
                                                 ft,
                                                 basic_list,
                                                 nonbasic_list,
                                                 sol,
                                                 iter,
                                                 dy_se,
                                                 nullptr);
    out_iters = iter;
  }
  if (out_seconds != nullptr) { *out_seconds = ds::toc(driver_t0); }
  // [SESTAT] temp: for CPU drivers (mode 0/1) the SE counters live on this ft; for the GPU (mode 2)
  // they live in the device g_se global and are emitted separately as [SEPROF].
  if (out_se != nullptr) {
    out_se->calls   = ft.dbg_se_calls_;
    out_se->hyper_l = ft.dbg_se_hyper_l_;
    out_se->hyper_u = ft.dbg_se_hyper_u_;
    out_se->nz_in   = ft.dbg_se_nz_in_;
    out_se->nz_l    = ft.dbg_se_nz_l_;
    out_se->nz_u    = ft.dbg_se_nz_u_;
    out_se->numupd  = ft.dbg_se_numupd_sum_;
  }
  out_obj         = ds::compute_user_objective<int, double>(lp, sol.x);
  out_num_updates = ft.num_updates();
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
    const double t0 = ds::tic();
    r.base_status   = ds::solve_linear_program<int, double>(up, settings, sol);
    r.base_time     = ds::toc(t0);
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
    ds::dual_phase2(1, 1, ds::tic(), phase1_problem, s, vstatus, p1sol, iter1, en1, nullptr);
  if (st1 != ds::dual::status_t::OPTIMAL) {
    note = "parent phase1: " + ds::dual::status_to_string(st1);
    return false;
  }

  // Phase 2: solve `lp` to optimality from the dual-feasible basis (no cutoff).
  int iter2 = 0;
  ds::lp_solution_t<int, double> sol(m, n);
  std::vector<double> en2;
  const ds::dual::status_t st2 =
    ds::dual_phase2(2, 0, ds::tic(), lp, s, vstatus, sol, iter2, en2, nullptr);
  if (st2 != ds::dual::status_t::OPTIMAL) {
    note = "parent phase2: " + ds::dual::status_to_string(st2);
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
  s.cut_off = info.cut_off;
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
    const std::string name = fs::path(nd).filename().string();
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
        // The steepest-edge seed is a single shared vector per instance (dir/edge_norms.txt);
        // fall back to the legacy per-node file for older dumps.
        if ((!load_doubles_file(dir + "/edge_norms.txt", edge_norms) &&
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
      const bool se_stats = std::getenv("TEST_NETLIB_SE_STATS") != nullptr;
      if (se_stats) { std::printf("[SENODE] %s\n", name.c_str()); std::fflush(stdout); }
      // TEST_NETLIB_VERBOSE=1 turns on both drivers' per-iteration bit traces
      // ([ADV] from advanced, [CMP] from simplified) so the exact iteration where the
      // two pivot sequences diverge can be located. Use on a single-node dir.
      const bool verbose_trace = std::getenv("TEST_NETLIB_VERBOSE") != nullptr;
      auto s_adv = make_settings(time_limit);
      apply_dumped_settings(s_adv, info);
      if (verbose_trace) { enable_verbose(s_adv); std::printf(">>> ADV %s\n", name.c_str()); }
      const ds::dual::status_t adv_st = run_phase2_from_basis(
        child_lp, vstatus_warm, s_adv, /*advanced*/ 0, adv_obj, adv_it, dinf, upd, init_basis,
        edge_norms, &adv_sec, &se_adv);
      auto s_sim = make_settings(time_limit);
      apply_dumped_settings(s_sim, info);
      if (verbose_trace) { enable_verbose(s_sim); std::printf(">>> SIM %s\n", name.c_str()); }
      const ds::dual::status_t sim_st = run_phase2_from_basis(
        child_lp, vstatus_warm, s_sim, /*simplified*/ 1, sim_obj, sim_it, dinf, upd, false,
        edge_norms, &sim_sec, &se_sim);
      // [SESTAT] TEMP debug: print the SE-context b_solve profile for full (mode 0) and minimal
      // (mode 1). hyperL/hyperU = #calls that took the hypersparse L/U path (dense = calls - that).
      if (se_stats) {
        auto pr = [&](const char* tag, const se_dbg_t& s) {
          const double c = s.calls ? static_cast<double>(s.calls) : 1.0;
          std::printf("[SESTAT] %-20s %-4s se_calls=%ld hyperL=%ld/%ld hyperU=%ld/%ld "
                      "avg_nupd=%.1f avg_nz_in=%.1f avg_nz_l=%.1f avg_nz_u=%.1f\n",
                      name.c_str(), tag, s.calls, s.hyper_l, s.calls, s.hyper_u, s.calls,
                      s.numupd / c, s.nz_in / c, s.nz_l / c, s.nz_u / c);
        };
        pr("full", se_adv);
        pr("min", se_sim);
        std::fflush(stdout);
      }

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
                  ds::dual::status_to_string(adv_st).c_str(),
                  adv_obj,
                  adv_it,
                  adv_sec,
                  ds::dual::status_to_string(sim_st).c_str(),
                  sim_obj,
                  sim_it,
                  sim_sec,
                  info.iterations,
                  pass ? "PASS" : "FAIL",
                  basis_src);
      if (run_gpu) {
        std::printf(" || GPU %-10s %10.6g %8d %8.4f [%s]",
                    ds::dual::status_to_string(gpu_st).c_str(),
                    gpu_obj,
                    gpu_it,
                    gpu_sec,
                    gpu_match ? "cpu==gpu" : "CPU!=GPU");
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
  const double start_time = ds::tic();
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
  b += bi(L0.col_start.size())  + bi(L0.nnz())  + bd(L0.nnz());
  b += bi(U0.col_start.size())  + bi(U0.nnz())  + bd(U0.nnz());
  b += bi(L0t.col_start.size()) + bi(L0t.nnz()) + bd(L0t.nnz());
  b += bi(U0t.col_start.size()) + bi(U0t.nnz()) + bd(U0t.nnz());
  b += bi(m) + bi(m) + bi(2 * m) + bd(m);
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
  constexpr double kSparsity   = 0.30;  // per-update S_ fill (fraction of m) from the sparsity study
  constexpr double kHeapSafety = 2.0;   // gpu_vector doubles capacity on growth + fragmentation

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
    size_t static_bytes           = 0;
    size_t heap_bytes             = 0;
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
    un->heap_bytes =
      static_cast<size_t>(static_cast<double>(iters) * kSparsity * static_cast<double>(un->m) *
                          12.0 * kHeapSafety);
  }

  // ---- Heavy-first order (anti-tailing): sort unique by measured iters desc; each unique's
  //      `replicas` copies are consecutive, so the heaviest problems get the low blockIdx /
  //      earliest OpenMP grab (LPT). Depth does NOT predict iters, so sort by measured iters. ----
  std::vector<int> order(uc);
  for (int i = 0; i < uc; ++i) { order[i] = i; }
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return nodes[a]->ref_iters > nodes[b]->ref_iters; });
  std::vector<int> list;
  list.reserve(static_cast<size_t>(uc) * replicas);
  for (int rank = 0; rank < uc; ++rank) {
    for (int r = 0; r < replicas; ++r) { list.push_back(order[rank]); }
  }
  const size_t Ntot = list.size();

  // ---- Pre-flight memory estimate (fail loud BEFORE building anything big) ----
  size_t vram_static = 0, vram_heap = 0, host_need = 0;
  for (int u : list) {
    vram_static += nodes[u]->static_bytes;
    vram_heap   += nodes[u]->heap_bytes;
    host_need   += static_cast<size_t>(nodes[u]->static_bytes * 1.3);  // host copies + overhead
  }
  const size_t vram_need = vram_static + vram_heap;
  size_t free_vram = 0, total_vram = 0;
  cudaMemGetInfo(&free_vram, &total_vram);
  const size_t free_ram = host_mem_available_bytes();
  std::printf("Memory estimate for %zu problems:\n", Ntot);
  std::printf("  VRAM  needed ~%.2f GB (static %.2f + heap %.2f) | free %.2f GB\n",
              vram_need * GB, vram_static * GB, vram_heap * GB, free_vram * GB);
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

  std::vector<ds::gpu_batch_item<int, double>> items(Ntot);
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
  ds::gpu_batch_timing timing;
  ds::gpu_simplified_dual_phase2_batch<int, double>(items, vram_heap, gpu_status, timing);
  if (timing.rejected) {
    std::printf("BENCHMARK REJECTED (GPU pre-flight): %s\n", timing.message.c_str());
    return;
  }

  // Timeout-aware verification: a completed GPU solve must be bit-exact to the reference; a
  // GPU TIME_LIMIT (single-thread too slow on a heavy node) is a GPU-speed limitation, NOT a
  // batching corruption, and is counted separately (its result is wall-clock-dependent, so it
  // cannot be bit-compared). Any other divergence is a true mismatch (a real bug).
  int gpu_exact = 0, gpu_timeout = 0, gpu_true_mismatch = 0;
  for (size_t k = 0; k < Ntot; ++k) {
    const int u       = list[k];
    const double gobj = ds::compute_user_objective<int, double>(nodes[u]->lp, prepared[k]->sol.x);
    const bool exact  = reproduces_ref(gpu_status[k], prepared[k]->sol.iterations, gobj,
                                      nodes[u]->ref_status, nodes[u]->ref_iters, nodes[u]->ref_obj);
    if (exact) {
      gpu_exact++;
    } else if (gpu_status[k] == ds::dual::status_t::TIME_LIMIT &&
               nodes[u]->ref_status != ds::dual::status_t::TIME_LIMIT) {
      gpu_timeout++;
    } else {
      gpu_true_mismatch++;
      std::printf("  [TRUE MISMATCH] %-30s ref{%s it=%d obj=%.10g} gpu{%s it=%d obj=%.10g}\n",
                  nodes[u]->name.c_str(), ds::dual::status_to_string(nodes[u]->ref_status).c_str(),
                  nodes[u]->ref_iters, nodes[u]->ref_obj,
                  ds::dual::status_to_string(gpu_status[k]).c_str(), prepared[k]->sol.iterations,
                  gobj);
    }
  }
  const double gpu_e2e_s = gpu_setup_s + timing.upload_s + timing.kernel_s + timing.copyback_s;

  // ---- Report ----
  size_t total_iters = 0;
  int max_iters      = 0;
  for (int u : list) {
    total_iters += nodes[u]->ref_iters;
    max_iters = std::max(max_iters, nodes[u]->ref_iters);
  }
  std::printf("\n==== parallel throughput: %zu problems (%d unique x%d) ====\n", Ntot, uc, replicas);
  std::printf("total simplex iterations : %zu   (max single node %d, mean %.0f)\n", total_iters,
              max_iters, static_cast<double>(total_iters) / static_cast<double>(Ntot));

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
  program.add_argument("--benchmark-parallel")
    .help("directory of B&B node samples (as for --verify-nodes); replicate the nodes --replicas "
          "times and solve them ALL on the CPU (OpenMP across problems) and the GPU (one block per "
          "problem, single launch), reporting total runtime for each and verifying every result is "
          "bit-for-bit identical to a single-thread sequential reference");
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

  const raft::handle_t handle{};

  // --verify-nodes mode: replay dumped B&B node samples and exit.
  if (program.is_used("--verify-nodes")) {
    verify_nodes_mode(handle, program.get<std::string>("--verify-nodes"), time_limit);
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
    const std::string adv_s  = ds::dual::status_to_string(r.adv_status);
    const std::string sim_s  = r.simpl_ran ? ds::dual::status_to_string(r.simpl_status)
                                           : (r.note.empty() ? "-" : r.note);
    const std::string gpu_s  = r.gpu_ran ? ds::dual::status_to_string(r.gpu_status) : "-";
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
                ds::dual::status_to_string(r.simpl_status).c_str(), r.simpl_iters, r.simpl_obj,
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
                  ds::dual::status_to_string(r.simpl_status).c_str(),
                  ds::dual::status_to_string(r.gpu_status).c_str(), r.simpl_iters, r.gpu_iters,
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
