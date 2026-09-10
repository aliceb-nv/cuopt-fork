/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
bool apply_cumulative_chain_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu, uint32_t seed)
{
  if (fj_cpu.problem->equality_fraction <= 0.5 || fj_cpu.problem->nnz > fj_seed_nnz_limit) return false;
  const i_t n = fj_cpu.problem->n_variables;
  std::vector<std::vector<i_t>> groups;
  std::vector<uint8_t> grouped(n, 0);
  for (i_t row = 0; row < fj_cpu.problem->n_constraints; ++row) {
    const f_t lb = fj_cpu.problem->cstr_lb[row], ub = fj_cpu.problem->cstr_ub[row];
    const i_t begin = fj_cpu.problem->offsets[row], end = fj_cpu.problem->offsets[row + 1];
    if (!std::isfinite(lb) || !std::isfinite(ub) || std::fabs(lb - ub) > fj_exact_k_tol ||
        end - begin < 4 || end - begin > fj_exact_k_max_width) continue;
    const f_t scale = fj_cpu.problem->coefficients[begin];
    if (!(scale > 0) || std::fabs(lb / scale - 1) > fj_exact_k_tol) continue;
    bool valid = true;
    for (i_t p = begin; p < end; ++p) {
      const i_t var = fj_cpu.problem->variables[p];
      valid &= fj_cpu.h_is_binary_variable[var] && !grouped[var] &&
               std::fabs(fj_cpu.problem->coefficients[p] - scale) <=
                 fj_exact_k_tol * std::max((f_t)1, std::fabs(scale));
    }
    if (!valid) continue;
    groups.emplace_back();
    for (i_t p = begin; p < end; ++p) {
      const i_t var = fj_cpu.problem->variables[p];
      groups.back().push_back(var); grouped[var] = 1;
    }
  }
  if (groups.size() < 8) return false;

  struct edge_t { i_t row, a, b; };
  std::vector<edge_t> edges;
  std::vector<std::pair<i_t, i_t>> anchors;
  std::vector<std::vector<i_t>> incident(n);
  for (i_t row = 0; row < fj_cpu.problem->n_constraints; ++row) {
    const f_t lb = fj_cpu.problem->cstr_lb[row], ub = fj_cpu.problem->cstr_ub[row];
    if (!std::isfinite(lb) || !std::isfinite(ub) || std::fabs(lb - ub) > fj_exact_k_tol) continue;
    i_t cont[2] = {-1, -1}, count = 0;
    f_t coeff[2] = {0, 0};
    bool valid = true;
    for (i_t p = fj_cpu.problem->offsets[row]; p < fj_cpu.problem->offsets[row + 1]; ++p) {
      const i_t var = fj_cpu.problem->variables[p];
      if (!is_integer_var<i_t, f_t>(fj_cpu, var)) {
        if (count == 2) { valid = false; break; }
        cont[count] = var; coeff[count++] = fj_cpu.problem->coefficients[p];
      } else if (!fj_cpu.h_is_binary_variable[var]) { valid = false; break; }
    }
    if (!valid || !count) continue;
    if (count == 1) anchors.emplace_back(row, cont[0]);
    else if (std::fabs(coeff[0] + coeff[1]) <= fj_exact_k_tol * std::max((f_t)1, std::fabs(coeff[0]))) {
      const i_t e = edges.size(); edges.push_back({row, cont[0], cont[1]});
      incident[cont[0]].push_back(e); incident[cont[1]].push_back(e);
    }
  }
  std::vector<i_t> chain, chain_rows;
  for (auto [anchor_row, anchor_var] : anchors) {
    std::vector<uint8_t> used(edges.size(), 0);
    std::vector<i_t> states{anchor_var}, rows{anchor_row};
    i_t current = anchor_var;
    while (true) {
      i_t e = -1;
      for (i_t candidate : incident[current]) if (!used[candidate]) { e = candidate; break; }
      if (e < 0) break;
      used[e] = 1; current = edges[e].a == current ? edges[e].b : edges[e].a;
      states.push_back(current); rows.push_back(edges[e].row);
    }
    if (states.size() > chain.size()) { chain = std::move(states); chain_rows = std::move(rows); }
  }
  if (chain.size() < 16) return false;

  auto propagate = [&] {
    f_t excess = 0;
    for (size_t k = 0; k < chain.size(); ++k) {
      const i_t row = chain_rows[k], target = chain[k];
      f_t rhs = fj_cpu.problem->cstr_lb[row], target_coeff = 0;
      for (i_t p = fj_cpu.problem->offsets[row]; p < fj_cpu.problem->offsets[row + 1]; ++p) {
        const i_t var = fj_cpu.problem->variables[p]; const f_t c = fj_cpu.problem->coefficients[p];
        if (var == target) target_coeff = c; else rhs -= c * fj_cpu.h_assignment[var];
      }
      if (target_coeff == 0) return std::numeric_limits<f_t>::infinity();
      const f_t value = rhs / target_coeff;
      const auto bounds = fj_cpu.h_var_bounds[target].get();
      excess += std::max((f_t)0, get_lower(bounds) - value) +
                std::max((f_t)0, value - get_upper(bounds));
      fj_cpu.h_assignment[target] = value;
    }
    return excess;
  };
  std::mt19937 rng(seed);
  std::vector<i_t> choice(groups.size());
  auto install = [&] {
    for (size_t g = 0; g < groups.size(); ++g) {
      for (i_t var : groups[g]) fj_cpu.h_assignment[var] = 0;
      fj_cpu.h_assignment[groups[g][choice[g]]] = 1;
    }
  };
  for (size_t g = 0; g < groups.size(); ++g)
    choice[g] = std::uniform_int_distribution<i_t>(0, groups[g].size() - 1)(rng);
  install(); f_t current = propagate(), best = current; auto best_choice = choice;
  const auto started = std::chrono::steady_clock::now();
  for (i_t iteration = 0; iteration < 30000 && best > fj_exact_k_tol; ++iteration) {
    if (!(iteration & 255) && std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count() > 0.15) break;
    const i_t g = std::uniform_int_distribution<i_t>(0, groups.size() - 1)(rng);
    const i_t old = choice[g];
    choice[g] = std::uniform_int_distribution<i_t>(0, groups[g].size() - 1)(rng);
    install(); const f_t candidate = propagate();
    if (candidate <= current) { current = candidate; if (candidate < best) { best = candidate; best_choice = choice; } }
    else choice[g] = old;
  }
  if (best > fj_exact_k_tol) return false;
  choice = best_choice; install(); propagate(); recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
  return true;
}

template <typename i_t, typename f_t>
void apply_precedence_completion_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  phase_timer_t timer(fj_cpu.t_seed);
  const i_t n_constraints = fj_cpu.problem->n_constraints;

  i_t lower_only = 0;
  for (i_t row = 0; row < n_constraints; ++row)
    lower_only +=
      std::isfinite(fj_cpu.problem->cstr_lb[row]) && !std::isfinite(fj_cpu.problem->cstr_ub[row]);
  if (lower_only * fj_precedence_lower_den < n_constraints * fj_precedence_lower_num) return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_precedence_budget_s;
  };

  recompute_lhs(fj_cpu);
  const auto anchor         = fj_cpu.h_assignment;
  auto best                 = anchor;
  const i_t anchor_count    = fj_cpu.violated_constraints.size();
  const f_t anchor_severity = -fj_cpu.total_violations;
  i_t best_count            = anchor_count;
  f_t best_severity         = anchor_severity;

  for (i_t pass = 0; pass < fj_precedence_passes; ++pass) {
    bool changed = false;
    for (i_t row = 0; row < n_constraints; ++row) {
      if ((row & 0x1FF) == 0 && timed_out()) break;
      const f_t lb = fj_cpu.problem->cstr_lb[row];
      if (!std::isfinite(lb) || std::isfinite(fj_cpu.problem->cstr_ub[row])) continue;
      const f_t lhs     = fj_cpu.h_lhs[row];
      const f_t deficit = lb - lhs;
      if (deficit <= fj_cpu.row_tolerance) continue;

      // The head is the row's only positive continuous coefficient. A row with none, or with
      // several, is not a precedence row and is left to the search.
      i_t head                = -1;
      f_t head_coeff          = 0;
      const auto [begin, end] = model_range_for_row<i_t, f_t>(fj_cpu, row);
      for (i_t p = begin; p < end; ++p) {
        const i_t var   = fj_cpu.problem->variables[p];
        const f_t coeff = fj_cpu.problem->coefficients[p];
        if (coeff <= 0 || is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
        if (head >= 0 && head != var) {
          head = -2;
          break;
        }
        head       = var;
        head_coeff = coeff;
      }
      if (head < 0 || head_coeff == 0) continue;

      const auto bounds = fj_cpu.h_var_bounds[head].get();
      const f_t old_val = fj_cpu.h_assignment[head];
      const f_t value   = std::min(get_upper(bounds), old_val + deficit / head_coeff);
      const f_t delta   = value - old_val;
      if (!(delta > 0) || !std::isfinite(value)) continue;

      fj_cpu.h_assignment[head]       = value;
      const auto [col_begin, col_end] = model_range_for_var<i_t, f_t>(fj_cpu, head);
      for (i_t q = col_begin; q < col_end; ++q) {
        const i_t touched   = fj_cpu.problem->reverse_constraints[q];
        const f_t patched   = fj_cpu.h_lhs[touched];
        fj_cpu.h_lhs[touched] = patched + fj_cpu.problem->reverse_coefficients[q] * delta;
      }
      changed = true;

      cuopt_assert((f_t)fj_cpu.h_lhs[row] >= lb - fj_cpu.row_tolerance ||
                     value >= get_upper(bounds),
                   "precedence step neither repaired the row nor saturated its head");
    }

    recompute_lhs(fj_cpu);
    const i_t count    = fj_cpu.violated_constraints.size();
    const f_t severity = -fj_cpu.total_violations;
    if (count < best_count || (count == best_count && severity < best_severity)) {
      best_count    = count;
      best_severity = severity;
      best          = fj_cpu.h_assignment;
      if (count == 0) break;
    }
    if (!changed || timed_out()) break;
  }

  cuopt_assert(fj_cpu.h_assignment.size() == anchor.size(),
               "incumbent_assignment span would be invalidated");
  const bool keep = best_count < anchor_count ||
                    (best_count == anchor_count && best_severity < anchor_severity);
  if (keep) {
    fj_cpu.h_assignment = best;
  } else {
    fj_cpu.h_assignment = anchor;
  }
  recompute_lhs(fj_cpu);
  cuopt_func_call(audit_assignment_bounds(fj_cpu, "precedence seed"));
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}


}  // namespace cuopt::mathematical_optimization::mip
