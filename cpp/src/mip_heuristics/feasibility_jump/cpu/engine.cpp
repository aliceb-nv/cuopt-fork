/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPUFJ is intentionally instantiated in one host translation unit. The search headers contain the
// latency-sensitive template chain; setup, seed, audit, and orchestration implementations retain
// their final source-file boundaries while sharing this instantiation boundary.
#include "problem.hpp"
#include "state.hpp"
#include "search/score.hpp"
#include "search/batching.hpp"
#include "audit.cpp"
#include "search/moves.hpp"
#include "search/update.hpp"
#include "search/escape.cpp"
#include "setup/structure.cpp"
#include "setup/bounds.cpp"
#include "setup/lp.cpp"
#include "seeds/seeds.hpp"
#include "seeds/covering.cpp"
#include "seeds/cardinality.cpp"
#include "seeds/matching.cpp"
#include "seeds/chain.cpp"
#include "seeds/affine.cpp"
#include "climber.cpp"
#include "loop.cpp"
#include "portfolio.cpp"

namespace cuopt::mathematical_optimization::mip {

#if MIP_INSTANTIATE_FLOAT
template struct fj_cpu_worker_t<int, float>;
template std::shared_ptr<fj_cpu_shared_incumbent_t<int, float>>
make_fj_cpu_shared_incumbent<int, float>();
template void cpufj_solve(fj_cpu_climber_t<int, float>*, float, double);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_clone(
  const fj_cpu_climber_t<int, float>&, std::atomic<bool>&, fj_settings_t);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_from_host_model(
  int,
  int,
  int,
  bool,
  float,
  float,
  std::vector<float>,
  std::vector<int>,
  std::vector<int>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<char>,
  std::vector<var_t>,
  const typename mip_solver_settings_t<int, float>::tolerances_t&,
  std::atomic<bool>&,
  fj_settings_t);
template void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<int, float>&,
  fj_cpu_problem_t<int, float>&,
  int,
  int,
  int,
  int,
  const typename mip_solver_settings_t<int, float>::tolerances_t&);
template void apply_lane_diversification<int, float>(fj_cpu_climber_t<int, float>&, int, int64_t);
template void complete_climber_portfolio<int, float>(
  std::unique_ptr<fj_cpu_climber_t<int, float>>,
  const std::vector<int64_t>&,
  std::vector<std::atomic<bool>>&,
  std::vector<std::unique_ptr<fj_cpu_climber_t<int, float>>>&,
  int64_t,
  bool);
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct fj_cpu_worker_t<int, double>;
template std::shared_ptr<fj_cpu_shared_incumbent_t<int, double>>
make_fj_cpu_shared_incumbent<int, double>();
template void cpufj_solve(fj_cpu_climber_t<int, double>*, double, double);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_clone(
  const fj_cpu_climber_t<int, double>&, std::atomic<bool>&, fj_settings_t);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_from_host_model(
  int,
  int,
  int,
  bool,
  double,
  double,
  std::vector<double>,
  std::vector<int>,
  std::vector<int>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<char>,
  std::vector<var_t>,
  const typename mip_solver_settings_t<int, double>::tolerances_t&,
  std::atomic<bool>&,
  fj_settings_t);
template void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<int, double>&,
  fj_cpu_problem_t<int, double>&,
  int,
  int,
  int,
  int,
  const typename mip_solver_settings_t<int, double>::tolerances_t&);
template void apply_lane_diversification<int, double>(fj_cpu_climber_t<int, double>&, int, int64_t);
template void complete_climber_portfolio<int, double>(
  std::unique_ptr<fj_cpu_climber_t<int, double>>,
  const std::vector<int64_t>&,
  std::vector<std::atomic<bool>>&,
  std::vector<std::unique_ptr<fj_cpu_climber_t<int, double>>>&,
  int64_t,
  bool);
#endif

}  // namespace cuopt::mathematical_optimization::mip
