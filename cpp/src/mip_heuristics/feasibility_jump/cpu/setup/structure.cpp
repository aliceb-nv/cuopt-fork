/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure.hpp"
#include "../internal.hpp"
#include <utilities/integer_scaling.hpp>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
double implied_integer_row_scale(const fj_cpu_problem_t<i_t, f_t>& problem,
                                 i_t row_idx,
                                 f_t pivot_coeff)
{
  const i_t begin = problem.offsets[row_idx], end = problem.offsets[row_idx + 1];
  const f_t rhs = problem.cstr_lb[row_idx];
  if (!scaling_bound_finite(rhs)) return 0;
  const double scale = row_int_scale(problem.coefficients.data() + begin,
                                     end - begin,
                                     rhs,
                                     rhs,
                                     end - begin,
                                     std::numeric_limits<int16_t>::max());
  if (scale == 0) return 0;
  const int64_t divisor = std::llround(scale * pivot_coeff);
  if (divisor == 0 || std::llround(scale * rhs) % divisor != 0) return 0;
  for (i_t entry_idx = begin; entry_idx < end; ++entry_idx)
    if (std::llround(scale * problem.coefficients[entry_idx]) % divisor != 0) return 0;
  return scale;
}

template <typename i_t, typename f_t>
void detect_implied_integers(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                             fj_cpu_problem_t<i_t, f_t>& problem)
{
  const i_t n_variables = problem.n_variables, n_constraints = problem.n_constraints;
  // Snapshot the original types for the initial equality counts.
  std::vector<uint8_t> was_continuous(n_variables);
  bool has_continuous = false;
  for (i_t var_idx = 0; var_idx < n_variables; ++var_idx)
    has_continuous |= was_continuous[var_idx] = problem.h_var_types[var_idx] == var_t::CONTINUOUS;
  if (!has_continuous) return;
  // Equalities with one continuous variable can force it integer; newly integral
  // variables may expose another such equality through the existing transpose.
  std::vector<i_t> n_continuous_in_row(n_constraints, 0), pending_rows;
  for (i_t row_idx = 0; row_idx < n_constraints; ++row_idx) {
    if (!std::isfinite(problem.cstr_lb[row_idx]) ||
        problem.cstr_lb[row_idx] != problem.cstr_ub[row_idx])
      continue;
    for (i_t entry_idx = problem.offsets[row_idx]; entry_idx < problem.offsets[row_idx + 1];
         ++entry_idx)
      n_continuous_in_row[row_idx] += was_continuous[problem.variables[entry_idx]];
    if (n_continuous_in_row[row_idx] == 1) pending_rows.push_back(row_idx);
  }
  i_t n_forced = 0, n_completable = 0;
  std::vector<i_t> changed_vars;
  // A row reaches one unknown at most once. Each newly integral column updates
  // its incident counts once, so even a long chain costs O(nnz + rows + columns).
  for (size_t queue_idx = 0; queue_idx < pending_rows.size(); ++queue_idx) {
    const i_t row_idx = pending_rows[queue_idx];
    if (n_continuous_in_row[row_idx] != 1) continue;
    i_t var_idx     = -1;
    f_t pivot_coeff = 0;
    for (i_t entry_idx = problem.offsets[row_idx]; entry_idx < problem.offsets[row_idx + 1];
         ++entry_idx) {
      if (problem.h_var_types[problem.variables[entry_idx]] == var_t::CONTINUOUS) {
        var_idx     = problem.variables[entry_idx];
        pivot_coeff = problem.coefficients[entry_idx];
        break;
      }
    }
    if (var_idx < 0 || !implied_integer_row_scale(problem, row_idx, pivot_coeff)) continue;
    const auto bounds = fj_cpu.h_var_bounds[var_idx].get();
    bool valid        = scaling_bound_finite(get_lower(bounds)) &&
                 scaling_bound_finite(get_upper(bounds)) &&
                 get_lower(bounds) == std::round(get_lower(bounds)) &&
                 get_upper(bounds) == std::round(get_upper(bounds));
    if (!valid) continue;
    problem.h_var_types[var_idx] = var_t::INTEGER;
    changed_vars.push_back(var_idx);
    ++n_forced;
    for (i_t entry_idx = problem.reverse_offsets[var_idx];
         entry_idx < problem.reverse_offsets[var_idx + 1];
         ++entry_idx) {
      const i_t incident_row = problem.reverse_constraints[entry_idx];
      if (n_continuous_in_row[incident_row] > 0 && --n_continuous_in_row[incident_row] == 1)
        pending_rows.push_back(incident_row);
    }
  }
  // Each pair variable occurs only in this equality, so changing the pair cannot
  // affect other rows. After scaling, the equality fixes their difference to an
  // integer. With zero lower bounds, integral upper bounds and equal positive
  // costs, setting the smaller variable to zero gives an integral optimal completion.
  // Fractional feasible pairs can still exist; integrality here preserves an optimum.
  for (i_t row_idx = 0; row_idx < n_constraints; ++row_idx) {
    if (n_continuous_in_row[row_idx] != 2) continue;
    i_t first_var = -1, second_var = -1;
    f_t pivot_coeff = 0, second_coeff = 0;
    for (i_t entry_idx = problem.offsets[row_idx]; entry_idx < problem.offsets[row_idx + 1];
         ++entry_idx) {
      const i_t var_idx = problem.variables[entry_idx];
      if (problem.h_var_types[var_idx] != var_t::CONTINUOUS) continue;
      if (first_var < 0) {
        first_var   = var_idx;
        pivot_coeff = problem.coefficients[entry_idx];
      } else {
        second_var   = var_idx;
        second_coeff = problem.coefficients[entry_idx];
      }
    }
    if (first_var < 0 || second_var < 0 || first_var == second_var) continue;
    const double scale = implied_integer_row_scale(problem, row_idx, pivot_coeff);
    if (scale == 0 || std::llround(scale * second_coeff) != -std::llround(scale * pivot_coeff))
      continue;
    bool valid = problem.h_obj_coeffs[first_var] > 0 &&
                 std::isfinite(problem.h_obj_coeffs[first_var]) &&
                 problem.h_obj_coeffs[first_var] == problem.h_obj_coeffs[second_var];
    for (i_t var_idx : {first_var, second_var}) {
      const auto bounds = fj_cpu.h_var_bounds[var_idx].get();
      valid &= problem.reverse_offsets[var_idx + 1] - problem.reverse_offsets[var_idx] == 1 &&
               get_lower(bounds) == f_t{0} && get_upper(bounds) >= f_t{0} &&
               scaling_bound_finite(get_upper(bounds)) &&
               get_upper(bounds) == std::round(get_upper(bounds));
    }
    if (!valid) continue;
    problem.h_var_types[first_var] = problem.h_var_types[second_var] = var_t::INTEGER;
    changed_vars.push_back(first_var);
    changed_vars.push_back(second_var);
    n_completable += 2;
  }

  // update classification for the newly-implied-integer vars
  for (i_t var_idx : changed_vars) {
    fj_cpu.h_assignment[var_idx]      = std::round((f_t)fj_cpu.h_assignment[var_idx]);
    fj_cpu.h_best_assignment[var_idx] = std::round((f_t)fj_cpu.h_best_assignment[var_idx]);
    const auto bounds                 = fj_cpu.h_var_bounds[var_idx].get();
    if (get_lower(bounds) == f_t{0} && get_upper(bounds) == f_t{1}) {
      fj_cpu.h_is_binary_variable[var_idx] = 1;
      fj_cpu.h_binary_indices.push_back(var_idx);
    }
  }
  CUOPT_LOG_DEBUG(
    "CPUFJ implied integrality: %d forced, %d integral-completable", n_forced, n_completable);
}

template <typename i_t, typename f_t>
void precompute_problem_features(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                 fj_cpu_problem_t<i_t, f_t>& problem)
{
  phase_timer_t timer(fj_cpu.t_features);
  fj_cpu.n_binary_vars  = 0;
  fj_cpu.n_integer_vars = 0;
  for (i_t i = 0; i < (i_t)fj_cpu.h_is_binary_variable.size(); i++) {
    if (fj_cpu.h_is_binary_variable[i]) {
      fj_cpu.n_binary_vars++;
    } else if (problem.h_var_types[i] == var_t::INTEGER) {
      fj_cpu.n_integer_vars++;
    }
  }

  i_t total_nnz = problem.reverse_offsets.back();
  i_t n_vars    = problem.reverse_offsets.size() - 1;
  i_t n_cstrs   = problem.offsets.size() - 1;

  problem.avg_var_degree = (double)total_nnz / n_vars;

  problem.max_var_degree = 0;
  std::vector<i_t> var_degrees(n_vars);
  for (i_t i = 0; i < n_vars; i++) {
    i_t degree             = problem.reverse_offsets[i + 1] - problem.reverse_offsets[i];
    var_degrees[i]         = degree;
    problem.max_var_degree = std::max(problem.max_var_degree, degree);
  }

  i_t equalities = 0;
  for (i_t row = 0; row < n_cstrs; ++row)
    equalities += problem.cstr_lb[row] == problem.cstr_ub[row];
  problem.equality_fraction = n_cstrs ? (double)equalities / n_cstrs : 0.0;
}

template <typename i_t, typename f_t>
void detect_free_equality_singletons(fj_cpu_climber_t<i_t, f_t>& c)
{
  const auto& model = *c.problem;
  const i_t n = model.reverse_offsets.size() - 1, m = model.offsets.size() - 1;
  c.bin_ignore_row.assign(m, 0);
  c.bin_ignore_var.assign(n, 0);
  for (i_t row = 0; row < m; ++row) {
    const f_t rhs = model.cstr_lb[row];
    if (!std::isfinite(rhs) || rhs != model.cstr_ub[row]) continue;
    typename fj_cpu_climber_t<i_t, f_t>::bin_eliminated_row_t rec{row, rhs};
    bool valid = true, raises = false, lowers = false;
    for (i_t p = model.offsets[row]; p < model.offsets[row + 1]; ++p) {
      const i_t var = model.variables[p]; const f_t a = model.coefficients[p];
      if (!a || c.h_is_binary_variable[var]) continue;
      if (c.problem->h_var_types[var] != var_t::CONTINUOUS ||
          model.reverse_offsets[var + 1] - model.reverse_offsets[var] != 1) { valid = false; break; }
      const auto bounds = c.h_var_bounds[var].get();
      const bool up = (a > 0 && !std::isfinite(get_upper(bounds))) ||
                      (a < 0 && !std::isfinite(get_lower(bounds)));
      const bool down = (a > 0 && !std::isfinite(get_lower(bounds))) ||
                        (a < 0 && !std::isfinite(get_upper(bounds)));
      if (!up && !down) { valid = false; break; }
      rec.all.push_back(var);
      if (up) { raises = true; rec.positive.push_back(var); rec.positive_coeff.push_back(a); }
      if (down) { lowers = true; rec.negative.push_back(var); rec.negative_coeff.push_back(a); }
    }
    if (!valid || rec.all.empty() || !raises || !lowers) continue;
    c.bin_ignore_row[row] = 1;
    for (i_t var : rec.all) c.bin_ignore_var[var] = 1;
    c.bin_eliminated_rows.push_back(std::move(rec));
  }
  c.has_bin_elimination = !c.bin_eliminated_rows.empty();
}

template <typename i_t, typename f_t>
void build_cardinality_index(fj_cpu_climber_t<i_t, f_t>& c,
                             fj_cpu_problem_t<i_t, f_t>& problem)
{
  auto& offsets       = problem.card_row_offsets;
  auto& members       = problem.card_variables;
  auto& cardinalities = problem.card_cardinalities;
  offsets.assign(1, 0);
  members.clear();
  cardinalities.clear();
  problem.card_group_of_variable.assign(problem.n_variables, -1);

  for (i_t row = 0; row < problem.n_constraints; ++row) {
    const f_t lb = problem.cstr_lb[row], ub = problem.cstr_ub[row];
    const i_t begin = problem.offsets[row], end = problem.offsets[row + 1];
    if (!std::isfinite(lb) || !std::isfinite(ub) || std::abs(lb - ub) > 1e-6 ||
        end - begin < 2 || end - begin > 20000)
      continue;

    const f_t common = problem.coefficients[begin];
    if (std::abs(common) <= 1e-6) continue;
    const f_t cardinality = lb / common;
    if (std::abs(cardinality - std::round(cardinality)) > 1e-6 ||
        cardinality < 0 || cardinality > end - begin)
      continue;

    bool valid = true;
    for (i_t p = begin; p < end && valid; ++p) {
      const i_t var = problem.variables[p];
      valid = c.h_is_binary_variable[var] &&
              std::abs(problem.coefficients[p] - common) <=
                1e-6 * std::max<f_t>(1, std::abs(common));
    }
    if (!valid) continue;
    const i_t group = (i_t)offsets.size() - 1;
    for (i_t p = begin; p < end; ++p) {
      const i_t var = problem.variables[p];
      members.push_back(var);
      i_t& owner = problem.card_group_of_variable[var];
      owner = owner == -1 ? group : (owner == group ? group : -2);
    }
    offsets.push_back((i_t)members.size());
    cardinalities.push_back((i_t)std::llround(cardinality));
  }
}

template <typename i_t, typename f_t>
void certify_epigraph_variables(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t n_variables)
{
  fj_cpu.epigraph_push.assign(n_variables, 0);
  fj_cpu.epigraph_vars.clear();

  for (i_t var = 0; var < n_variables; ++var) {
    if (is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
    const f_t obj_coeff = fj_cpu.problem->h_obj_coeffs[var];
    if (obj_coeff == f_t{0}) continue;

    const auto [begin, end] = model_range_for_var<i_t, f_t>(fj_cpu, var);
    if (begin == end) continue;

    // A positive coefficient is minimised by pushing the variable down, so its rows must be the
    // only thing holding it up, and it must be free to rise as far as they demand.
    const bool push_up = obj_coeff > f_t{0};
    const auto bounds  = fj_cpu.h_var_bounds[var].get();
    if (std::isfinite(push_up ? get_upper(bounds) : get_lower(bounds))) continue;

    bool certified = true;
    for (i_t p = begin; p < end && certified; ++p) {
      const i_t row     = fj_cpu.problem->reverse_constraints[p];
      const f_t coeff   = fj_cpu.problem->reverse_coefficients[p];
      const bool has_lb = std::isfinite((f_t)fj_cpu.problem->cstr_lb[row]);
      const bool has_ub = std::isfinite((f_t)fj_cpu.problem->cstr_ub[row]);
      if (coeff == f_t{0}) continue;
      certified = push_up ? ((coeff > 0 && has_lb && !has_ub) || (coeff < 0 && has_ub && !has_lb))
                          : ((coeff > 0 && has_ub && !has_lb) || (coeff < 0 && has_lb && !has_ub));
    }
    if (!certified) continue;

    fj_cpu.epigraph_push[var] = push_up ? 1 : -1;
    fj_cpu.epigraph_vars.push_back(var);
  }
}

template <typename i_t, typename f_t>
void build_one_sided_rows(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::build_one_sided_rows");
  const i_t n_model_rows = fj_cpu.problem->n_constraints;

  i_t n_rows = 0;
  i_t nnz    = 0;
  for (i_t row = 0; row < n_model_rows; ++row) {
    const i_t sides = (i_t)std::isfinite((f_t)fj_cpu.problem->cstr_lb[row]) +
                      (i_t)std::isfinite((f_t)fj_cpu.problem->cstr_ub[row]);
    n_rows += sides;
    nnz += sides * (fj_cpu.problem->offsets[row + 1] - fj_cpu.problem->offsets[row]);
  }
  cuopt_assert(n_rows > 0, "model has no bounded rows");

  auto& fwd_offsets      = fj_cpu.h_offsets;
  auto& fwd_variables    = fj_cpu.h_variables;
  auto& fwd_coefficients = fj_cpu.h_coefficients;
  fwd_offsets.clear();
  fwd_variables.clear();
  fwd_coefficients.clear();
  fwd_offsets.reserve((size_t)n_rows + 1);
  fwd_variables.reserve((size_t)nnz);
  fwd_coefficients.reserve((size_t)nnz);
  fwd_offsets.push_back(0);

  fj_cpu.h_bound.clear();
  fj_cpu.h_bound.reserve((size_t)n_rows);
  fj_cpu.h_row_state.underlying().assign((size_t)n_rows, {});
  fj_cpu.h_row_is_integral.clear();
  fj_cpu.h_row_is_integral.reserve((size_t)n_rows);

  for (i_t row = 0; row < n_model_rows; ++row) {
    const f_t lb    = fj_cpu.problem->cstr_lb[row];
    const f_t ub    = fj_cpu.problem->cstr_ub[row];
    const i_t begin = fj_cpu.problem->offsets[row];
    const i_t end   = fj_cpu.problem->offsets[row + 1];
    bool integral_activity = true;
    for (i_t k = begin; k < end; ++k) {
      const i_t var   = fj_cpu.problem->variables[k];
      const f_t coeff = fj_cpu.problem->coefficients[k];
      if (coeff != f_t{0} &&
          (fj_cpu.problem->h_var_types[var] != var_t::INTEGER || coeff != std::round(coeff))) {
        integral_activity = false;
        break;
      }
    }

    for (i_t side = 0; side < 2; ++side) {
      const f_t bound = side == 0 ? lb : ub;
      if (!std::isfinite(bound)) continue;
      const f_t sign = side == 0 ? (f_t)-1 : (f_t)1;

      for (i_t k = begin; k < end; ++k) {
        fwd_variables.push_back(fj_cpu.problem->variables[k]);
        fwd_coefficients.push_back(sign * (f_t)fj_cpu.problem->coefficients[k]);
      }
      fwd_offsets.push_back((i_t)fwd_variables.size());

      const i_t r = (i_t)fj_cpu.h_bound.size();
      fj_cpu.h_bound.push_back(sign * bound);
      fj_cpu.h_row_is_integral.push_back(integral_activity && bound == std::round(bound));
      fj_cpu.row_state()[r].weight =
        side == 0 ? fj_cpu.h_cstr_left_weights[row] : fj_cpu.h_cstr_right_weights[row];
    }
  }
  cuopt_assert((i_t)fj_cpu.h_bound.size() == n_rows, "row count mismatch");
  cuopt_assert((i_t)fj_cpu.h_row_is_integral.size() == n_rows, "row count mismatch");
  cuopt_assert((i_t)fwd_variables.size() == nnz, "nonzero count mismatch");

  fj_cpu.n_rows = n_rows;
  fj_cpu.h_slack_sumcomp.underlying().assign((size_t)n_rows, f_t{0});
  fj_cpu.h_cstr_version.assign((size_t)n_rows, 0);
  fj_cpu.violated_constraints.resize(n_rows);
  fj_cpu.satisfied_constraints.resize(n_rows);
  // Indexed by forward nonzero, so they follow the search rows rather than the model.
  fj_cpu.cached_mtm_moves.assign((size_t)nnz,
                                 std::make_pair(f_t{0}, fj_staged_score_t::zero()));
  fj_cpu.cached_mtm_moves_version.assign((size_t)nnz, -1);
  // One binary-variable table per row, so 2-opt can key it by the row it is iterating.
  fj_cpu.h_binrow_offsets.resize((size_t)n_rows + 1);
  fj_cpu.h_binrow_vars.clear();
  for (i_t r = 0; r < n_rows; ++r) {
    fj_cpu.h_binrow_offsets[r] = (i_t)fj_cpu.h_binrow_vars.size();
    const i_t row_begin = fwd_offsets[r];
    const i_t row_end   = fwd_offsets[r + 1];
    for (i_t k = row_begin; k < row_end; ++k) {
      const i_t var_idx = fwd_variables[k];
      if (fj_cpu.h_is_binary_variable[var_idx])
        fj_cpu.h_binrow_vars.push_back(var_idx);
    }
  }
  fj_cpu.h_binrow_offsets[n_rows] = (i_t)fj_cpu.h_binrow_vars.size();

  // Counting-sort transpose. Rows are emitted in increasing order, so each variable's slice comes
  // out row-ascending, which the 2-opt merges require.
  const i_t n_variables = fj_cpu.problem->n_variables;
  auto& rev_offsets     = fj_cpu.h_reverse_offsets.underlying();
  rev_offsets.assign((size_t)n_variables + 1, 0);
  for (i_t k = 0; k < nnz; ++k) {
    const i_t var_idx = fwd_variables[k];
    rev_offsets[var_idx + 1]++;
  }
  for (i_t v = 0; v < n_variables; ++v) rev_offsets[v + 1] += rev_offsets[v];
  fj_cpu.h_reverse_constraints.resize((size_t)nnz);
  fj_cpu.h_reverse_coefficients.resize((size_t)nnz);
  {
    std::vector<i_t> cursor(rev_offsets.begin(), rev_offsets.begin() + n_variables);
    for (i_t r = 0; r < n_rows; ++r) {
      const i_t row_begin = fwd_offsets[r];
      const i_t row_end   = fwd_offsets[r + 1];
      for (i_t k = row_begin; k < row_end; ++k) {
        const i_t var_idx                   = fwd_variables[k];
        const i_t slot                      = cursor[var_idx]++;
        fj_cpu.h_reverse_constraints[slot]  = r;
        fj_cpu.h_reverse_coefficients[slot] = fwd_coefficients[k];
      }
    }
  }

  recompute_slack(fj_cpu);
}


}  // namespace cuopt::mathematical_optimization::mip
