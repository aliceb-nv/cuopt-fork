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

}  // namespace cuopt::linear_programming::dual_simplex
