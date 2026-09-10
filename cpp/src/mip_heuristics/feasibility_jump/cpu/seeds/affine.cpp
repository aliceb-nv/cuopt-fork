/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void apply_affine_equality_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  phase_timer_t timer(fj_cpu.t_seed);
  std::vector<i_t> rows;
  for (i_t row = 0; row < fj_cpu.problem->n_constraints; ++row) {
    const f_t lb = fj_cpu.problem->cstr_lb[row], ub = fj_cpu.problem->cstr_ub[row];
    if (!std::isfinite(lb) || !std::isfinite(ub) ||
        std::fabs(lb - ub) > fj_cpu.problem->tolerances.absolute_tolerance) continue;
    const auto [begin, end] = model_range_for_row<i_t, f_t>(fj_cpu, row);
    for (i_t p = begin; p < end; ++p) {
      const i_t var = fj_cpu.problem->variables[p];
      const auto bounds = fj_cpu.h_var_bounds[var].get();
      if (!is_integer_var<i_t, f_t>(fj_cpu, var) && get_lower(bounds) < get_upper(bounds) &&
          fj_cpu.problem->coefficients[p] != 0) { rows.push_back(row); break; }
    }
  }
  if (rows.size() < 8 || 2 * rows.size() < (size_t)fj_cpu.problem->n_constraints) return;
  recompute_lhs(fj_cpu);
  auto best = std::vector<f_t>(fj_cpu.h_assignment.begin(), fj_cpu.h_assignment.end());
  i_t best_count = fj_cpu.violated_constraints.size();
  f_t best_severity = -fj_cpu.total_violations;
  const auto started = std::chrono::steady_clock::now();
  for (i_t pass = 0; pass < 128; ++pass) {
    bool changed = false;
    for (i_t row : rows) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() > 0.45)
        break;
      const f_t residual = fj_cpu.problem->cstr_lb[row] - fj_cpu.h_lhs[row];
      if (std::fabs(residual) <= fj_cpu.row_tolerance) continue;
      i_t chosen = -1, chosen_degree = std::numeric_limits<i_t>::max();
      f_t chosen_value = 0, remainder = std::fabs(residual);
      const auto [begin, end] = model_range_for_row<i_t, f_t>(fj_cpu, row);
      for (i_t p = begin; p < end; ++p) {
        const i_t var = fj_cpu.problem->variables[p];
        if (is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
        const f_t coeff = fj_cpu.problem->coefficients[p];
        if (coeff == 0) continue;
        const auto bounds = fj_cpu.h_var_bounds[var].get();
        const f_t old = fj_cpu.h_assignment[var];
        const f_t value = std::clamp(old + residual / coeff,
                                     get_lower(bounds), get_upper(bounds));
        const f_t error = std::fabs(residual - coeff * (value - old));
        const i_t degree = fj_cpu.problem->reverse_offsets[var + 1] -
                           fj_cpu.problem->reverse_offsets[var];
        if (std::isfinite(value) && value != old &&
            (error < remainder || (error == remainder && degree < chosen_degree))) {
          chosen = var; chosen_value = value; remainder = error; chosen_degree = degree;
        }
      }
      if (chosen < 0) continue;
      fj_cpu.h_assignment[chosen] = chosen_value;
      changed = true;
      recompute_lhs(fj_cpu);
    }
    const i_t count = fj_cpu.violated_constraints.size();
    const f_t severity = -fj_cpu.total_violations;
    if (count < best_count || (count == best_count && severity < best_severity)) {
      best_count = count; best_severity = severity;
      best.assign(fj_cpu.h_assignment.begin(), fj_cpu.h_assignment.end());
      if (!count) break;
    }
    if (!changed) break;
  }
  std::copy(best.begin(), best.end(), fj_cpu.h_assignment.begin());
  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

template <typename i_t, typename f_t>
void apply_structural_completion_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  apply_lock_weighted_seed<i_t, f_t>(fj_cpu);
  apply_exact_k_seed<i_t, f_t>(fj_cpu);
  apply_greedy_covering_seed<i_t, f_t>(fj_cpu);
  repair_difficult_anchor<i_t, f_t>(fj_cpu);
}


}  // namespace cuopt::mathematical_optimization::mip
