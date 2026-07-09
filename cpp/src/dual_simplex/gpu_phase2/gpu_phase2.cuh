/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// gpu_simplified_dual_phase2: a faithful, single-thread GPU port of the CPU
// dual-simplex driver `simplified_dual_phase2` (dual_simplex/phase2.cpp).
//
// It has the EXACT same signature and produces the EXACT same results as the
// CPU function. The wrapper is just a prologue (marshal inputs CPU->GPU) and
// epilogue (marshal outputs GPU->CPU) around a single <<<1,1>>> kernel that
// replays the CPU algorithm line-for-line on the device.
//
// Scope: PURE MPF STACKING only (no on-device LU refactorization). It matches
// the CPU default behavior (CUOPT_SIMPLIFIED_REFACTOR unset / 0).

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// GPU port of simplified_dual_phase2. Same parameters, same returned status and
// solution. Only instantiated for <int, double>.
template <typename i_t, typename f_t>
dual::status_t gpu_simplified_dual_phase2(const lp_problem_t<i_t, f_t>& lp,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          std::vector<variable_status_t>& vstatus,
                                          basis_update_mpf_t<i_t, f_t>& ft,
                                          std::vector<i_t>& basic_list,
                                          std::vector<i_t>& nonbasic_list,
                                          lp_solution_t<i_t, f_t>& sol,
                                          std::vector<f_t>& delta_y_steepest_edge);

// ---------------------------------------------------------------------------
// Multi-problem batch: solve N independent problems in ONE launch (one block
// per problem). Used by the throughput benchmark. Each item must already be set
// up exactly as a single solve is — a FRESH factorized `ft` (num_updates == 0),
// `basic_list`/`nonbasic_list` from `vstatus`, and duals in `sol.y`/`sol.z` —
// i.e. the same steps run_phase2_from_basis performs before the driver call.
// The pointers are borrowed (not owned) and must outlive the call.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
struct gpu_batch_item {
  const lp_problem_t<i_t, f_t>* lp;
  const simplex_solver_settings_t<i_t, f_t>* settings;
  std::vector<variable_status_t>* vstatus;
  basis_update_mpf_t<i_t, f_t>* ft;
  std::vector<i_t>* basic_list;
  std::vector<i_t>* nonbasic_list;
  lp_solution_t<i_t, f_t>* sol;  // in: y/z (duals); out: x, objective, user_objective, iterations
  std::vector<f_t>* dy_se;       // carried steepest-edge norms (may be empty)
};

// Timing + pre-flight result for a batch launch. If `rejected` is true the batch
// did NOT run (VRAM pre-flight failed); `message` explains needed-vs-free.
struct gpu_batch_timing {
  double upload_s          = 0.0;  // host: marshal + H2D of all N problems
  double kernel_s          = 0.0;  // device compute (CUDA events, kernel only)
  double copyback_s        = 0.0;  // host: D2H of all N results
  std::size_t vram_needed_bytes = 0;
  std::size_t vram_free_bytes   = 0;
  bool rejected            = false;
  std::string message;
};

// Solve every item in one <<<N,1>>> launch. `heap_estimate_bytes` sizes the device
// malloc arena for all N concurrent S_ stacks (and is included in the VRAM check).
// Fails loud: if the batch will not fit it sets timing.rejected + timing.message and
// returns without launching (never a silent OOM). On success fills out_status[i] and
// each item->sol. Only instantiated for <int, double>.
template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch(const std::vector<gpu_batch_item<i_t, f_t>>& items,
                                      std::size_t heap_estimate_bytes,
                                      std::vector<dual::status_t>& out_status,
                                      gpu_batch_timing& timing);

}  // namespace cuopt::linear_programming::dual_simplex
