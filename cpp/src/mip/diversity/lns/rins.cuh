/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mip/diversity/population.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <utilities/timer.hpp>

#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

// forward declare
template <typename i_t, typename f_t>
class diversity_manager_t;

template <typename i_t, typename f_t>
struct rins_settings_t {
  i_t node_freq          = 10;
  f_t min_frac           = 0.3;
  f_t max_frac           = 0.8;
  f_t default_frac       = 0.5;
  f_t min_time_limit     = 1.;
  f_t max_time_limit     = 20.;
  f_t default_time_limit = 4.;
};

template <typename i_t, typename f_t>
class rins_t {
 public:
  rins_t(mip_solver_context_t<i_t, f_t>& context,
         diversity_manager_t<i_t, f_t>& dm,
         rins_settings_t<i_t, f_t> settings = rins_settings_t<i_t, f_t>());

  void node_callback(const std::vector<f_t>& solution, f_t objective);

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  diversity_manager_t<i_t, f_t>& dm;
  rins_settings_t<i_t, f_t> settings;

  f_t frac{0.5};

  i_t total_calls{0};
  i_t total_success{0};
};

}  // namespace cuopt::linear_programming::detail
