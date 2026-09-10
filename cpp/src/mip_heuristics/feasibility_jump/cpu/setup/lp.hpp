/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"
namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
void apply_lp_rounded_seed(fj_cpu_climber_t<i_t, f_t>&, f_t);
template <typename i_t, typename f_t>
bool apply_lp_polish(fj_cpu_climber_t<i_t, f_t>&, double);
template <typename i_t, typename f_t>
void apply_lp_feasibility_dive(fj_cpu_climber_t<i_t, f_t>&, f_t);
}  // namespace cuopt::mathematical_optimization::mip
