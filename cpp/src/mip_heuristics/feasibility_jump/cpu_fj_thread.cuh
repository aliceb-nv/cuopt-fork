/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <mip_heuristics/utilities/cpu_worker_thread.cuh>

#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

template <typename i_t, typename f_t>
class fj_t;

template <typename i_t, typename f_t>
struct cpu_fj_thread_t : public cpu_worker_thread_base_t<cpu_fj_thread_t<i_t, f_t>> {
  ~cpu_fj_thread_t();

  void run_worker();
  void on_terminate();
  void on_start();
  bool get_result() { return cpu_fj_solution_found; }

  void stop_cpu_solver();

  std::atomic<bool> cpu_fj_solution_found{false};
  f_t time_limit{+std::numeric_limits<f_t>::infinity()};
  double work_unit_limit{std::numeric_limits<double>::infinity()};
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> fj_cpu;
  fj_t<i_t, f_t>* fj_ptr{nullptr};
};

template <typename i_t, typename f_t>
bool run_fj_cpu_from_host_lp(
  const dual_simplex::lp_problem_t<i_t, f_t>& problem,
  const std::vector<dual_simplex::variable_type_t>& variable_types,
  const std::vector<f_t>& seed_assignment,
  const dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  std::atomic<bool>& preemption_flag,
  f_t time_limit,
  double work_unit_limit,
  std::function<void(f_t, const std::vector<f_t>&, double)> improvement_callback,
  std::string log_prefix);

}  // namespace cuopt::linear_programming::detail
