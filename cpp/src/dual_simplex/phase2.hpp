/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <math_optimization/types.hpp>
#include <utilities/memory_instrumentation.hpp>

#include <vector>

namespace cuopt {
struct work_limit_context_t;
}

namespace cuopt::mathematical_optimization::simplex {

enum class dual_status_t {
  OPTIMAL          = 0,
  DUAL_UNBOUNDED   = 1,
  NUMERICAL        = 2,
  CUTOFF           = 3,
  TIME_LIMIT       = 4,
  ITERATION_LIMIT  = 5,
  CONCURRENT_LIMIT = 6,
  WORK_LIMIT       = 7,
  UNSET            = 8,
  // NON-TERMINAL. The solve stopped because its basis representation must be rebuilt, not because
  // it reached an answer: the periodic update limit was reached, the rank-1 update was unsafe
  // (|mu| outside [1e-8,1e8]), or the preallocated update store is full. The caller refactorizes
  // the basis the solve handed back and
  // calls again to CONTINUE the same solve.
  //
  // Only the GPU driver produces it. The CPU drivers refactorize in place and keep iterating, which
  // the device cannot do -- it has no LU factorization of its own. Nothing outside the solver may
  // see this value: branch-and-bound must never treat it as a node result.
  REFACTOR_NEEDED = 9
};

static std::string dual_status_to_string(dual_status_t status)
{
  switch (status) {
    case dual_status_t::OPTIMAL: return "OPTIMAL";
    case dual_status_t::DUAL_UNBOUNDED: return "DUAL_UNBOUNDED";
    case dual_status_t::NUMERICAL: return "NUMERICAL";
    case dual_status_t::CUTOFF: return "CUTOFF";
    case dual_status_t::TIME_LIMIT: return "TIME_LIMIT";
    case dual_status_t::ITERATION_LIMIT: return "ITERATION_LIMIT";
    case dual_status_t::CONCURRENT_LIMIT: return "CONCURRENT_LIMIT";
    case dual_status_t::WORK_LIMIT: return "WORK_LIMIT";
    case dual_status_t::UNSET: return "UNSET";
    case dual_status_t::REFACTOR_NEEDED: return "REFACTOR_NEEDED";
  }
  return "UNKNOWN";
}

// The simplified dual-simplex drivers (CPU `simplified_dual_phase2` and its GPU
// port) spell the phase-2 status enum as `dual::status_t`. Provide that name as
// an alias of the canonical `dual_status_t` so it resolves in every TU that
// includes this header.
namespace dual {
using status_t = dual_status_t;
}

template <typename i_t, typename f_t>
dual_status_t dual_phase2(i_t phase,
                          i_t slack_basis,
                          f_t start_time,
                          const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          std::vector<variable_status_t>& vstatus,
                          lp_solution_t<i_t, f_t>& sol,
                          i_t& iter,
                          std::vector<f_t>& steepest_edge_norms,
                          work_limit_context_t* work_unit_context = nullptr);

template <typename i_t, typename f_t>
dual_status_t dual_phase2_with_advanced_basis(i_t phase,
                                              i_t slack_basis,
                                              bool initialize_basis,
                                              f_t start_time,
                                              const lp_problem_t<i_t, f_t>& lp,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              std::vector<variable_status_t>& vstatus,
                                              basis_update_mpf_t<i_t, f_t>& ft,
                                              std::vector<i_t>& basic_list,
                                              std::vector<i_t>& nonbasic_list,
                                              lp_solution_t<i_t, f_t>& sol,
                                              i_t& iter,
                                              std::vector<f_t>& delta_y_steepest_edge,
                                              work_limit_context_t* work_unit_context = nullptr);

// Minimal dual simplex driver. Assumes the caller provides an LU factorization
// `ft` matching the basis described by `vstatus`, `basic_list`, `nonbasic_list`.
// Implements only the 5 essential pieces: sparse triangular solves (BTran/FTran),
// simple ratio test, middle-product-form rank-1 basis updates, steepest-edge
// pricing with weight updates, and SpMv-based reduced-cost computation.
template <typename i_t, typename f_t>
dual::status_t simplified_dual_phase2(const lp_problem_t<i_t, f_t>& lp,
                                      const simplex_solver_settings_t<i_t, f_t>& settings,
                                      std::vector<variable_status_t>& vstatus,
                                      basis_update_mpf_t<i_t, f_t>& ft,
                                      std::vector<i_t>& basic_list,
                                      std::vector<i_t>& nonbasic_list,
                                      lp_solution_t<i_t, f_t>& sol,
                                      std::vector<f_t>& delta_y_steepest_edge);

template <typename i_t, typename f_t>
void compute_reduced_cost_update(const lp_problem_t<i_t, f_t>& lp,
                                 const std::vector<i_t>& basic_list,
                                 const std::vector<i_t>& nonbasic_list,
                                 const std::vector<f_t>& delta_y,
                                 i_t leaving_index,
                                 i_t direction,
                                 std::vector<i_t>& delta_z_mark,
                                 std::vector<i_t>& delta_z_indices,
                                 std::vector<f_t>& delta_z,
                                 f_t& work_estimate);

template <typename i_t, typename f_t>
void compute_delta_z(const csr_matrix_t<i_t, f_t>& Arow,
                     const sparse_vector_t<i_t, f_t>& delta_y,
                     i_t leaving_index,
                     i_t direction,
                     const std::vector<i_t>& nonbasic_end,
                     std::vector<i_t>& delta_z_mark,
                     std::vector<i_t>& delta_z_indices,
                     std::vector<f_t>& delta_z,
                     f_t& work_estimate);

template <typename i_t, typename f_t>
void compute_initial_nonbasic_end(const std::vector<i_t>& basic_mark,
                                  csr_matrix_t<i_t, f_t>& Arow,
                                  std::vector<i_t>& nonbasic_end);

namespace phase2 {
template <typename i_t, typename f_t>
i_t compute_primal_solution_from_basis(const lp_problem_t<i_t, f_t>& lp,
                                       basis_update_mpf_t<i_t, f_t>& ft,
                                       const std::vector<i_t>& basic_list,
                                       const std::vector<i_t>& nonbasic_list,
                                       const std::vector<variable_status_t>& vstatus,
                                       std::vector<f_t>& x,
                                       std::vector<f_t>& xB_workspace,
                                       f_t& work_estimate);

// Bounds-explicit form, for a caller whose bounds are NOT `lp.lower` / `lp.upper`. The GPU node pool
// shares ONE lp_problem_t across every seat -- the read-only upload dedups on `lp.A.col_start.data()`,
// so a per-seat copy would re-upload the matrix per node -- and carries each node's bounds
// separately. Its REFACTOR_NEEDED resume has to rebuild the primal point with THIS node's bounds,
// using the same formula the CPU driver's in-place refactor uses. The form above delegates here, so
// there is only one implementation.
template <typename i_t, typename f_t>
i_t compute_primal_solution_from_basis(const lp_problem_t<i_t, f_t>& lp,
                                       const std::vector<f_t>& lower,
                                       const std::vector<f_t>& upper,
                                       basis_update_mpf_t<i_t, f_t>& ft,
                                       const std::vector<i_t>& basic_list,
                                       const std::vector<i_t>& nonbasic_list,
                                       const std::vector<variable_status_t>& vstatus,
                                       std::vector<f_t>& x,
                                       std::vector<f_t>& xB_workspace,
                                       f_t& work_estimate);

// y and z from a basis: B^T y = c_B, then z_N = c_N - N^T y. Declared here because callers outside
// phase2.cpp need it to set up a solve -- the GPU batch entry point requires the duals as input, and
// its harness used to carry a private copy of this function.
template <typename i_t, typename f_t>
void compute_dual_solution_from_basis(const lp_problem_t<i_t, f_t>& lp,
                                      basis_update_mpf_t<i_t, f_t>& ft,
                                      const std::vector<i_t>& basic_list,
                                      const std::vector<i_t>& nonbasic_list,
                                      std::vector<f_t>& y,
                                      std::vector<f_t>& z,
                                      f_t& work_estimate);
}  // namespace phase2

}  // namespace cuopt::mathematical_optimization::simplex
