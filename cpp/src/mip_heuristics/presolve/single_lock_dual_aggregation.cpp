/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "single_lock_dual_aggregation.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <vector>

namespace cuopt::linear_programming::detail {

// Single-Lock Dual Aggregation
//
// For a binary variable x with exactly one "up-lock" (one constraint preventing
// it from increasing), we try to prove an implication y=0 => x=0 via activity
// bounds on the locking row. If additionally the row is non-binding when y=1
// (no capacity competition), we can substitute x = y, eliminating a variable.
//
// Symmetric logic applies for "down-lock" candidates (one constraint preventing
// decrease), proving y=1 => x=1.

template <typename f_t>
papilo::PresolveStatus SingleLockDualAggregation<f_t>::execute(
  const papilo::Problem<f_t>& problem,
  const papilo::ProblemUpdate<f_t>& problemUpdate,
  const papilo::Num<f_t>& num,
  papilo::Reductions<f_t>& reductions,
  const papilo::Timer& timer,
  int& reason_of_infeasibility)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;
  const auto& objective         = problem.getObjective().coefficients;

  const int nrows   = constraint_matrix.getNRows();
  const int ncols   = problem.getNCols();
  const double tlim = problemUpdate.getPresolveOptions().tlim;
  const f_t tol     = num.getFeasTol();

  // =========================================================================
  // Step 1: Lock Counting — O(nnz)
  //
  // An "up-lock" on column j means a constraint prevents j from increasing:
  //   - a_j > 0 in a <= row, or a_j < 0 in a >= row.
  // "Down-lock" is the reverse. Equality rows lock both directions.
  // We record the row index of the first lock; a second lock invalidates it.
  // =========================================================================

  enum lock_dir { UP = 0, DOWN = 1 };
  enum bound_side { LOWER = 0, UPPER = 1 };
  std::vector<int> locks[2]    = {std::vector<int>(ncols, 0), std::vector<int>(ncols, 0)};
  std::vector<int> lock_row[2] = {std::vector<int>(ncols, -1), std::vector<int>(ncols, -1)};

  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;

    // Row direction: 'E' = equality, 'L' = <=, 'G' = >=, 'R' = ranged/free (skip)
    bool lhs_inf = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf = row_flags[row].test(papilo::RowFlag::kRhsInf);
    char row_dir = (!lhs_inf && !rhs_inf) ? 'E' : (lhs_inf ? 'L' : (rhs_inf ? 'G' : 'R'));
    if (row_dir == 'R') continue;

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* coefs = row_coeff.getValues();
    const int length = row_coeff.getLength();

    // Record the index of the locking row.
    // If more than one lock exists, mark the col as excluded from the search.
    auto record_lock = [&](lock_dir dir, int col) {
      if (locks[dir][col]++ == 0)
        lock_row[dir][col] = row;
      else
        lock_row[dir][col] = -1;
    };

    if (row_dir == 'E') {
      // Equality: locks both directions
      for (int j = 0; j < length; ++j) {
        record_lock(UP, cols[j]);
        record_lock(DOWN, cols[j]);
      }
    } else {
      // One-sided: directions swap between L (<=) and G (>=)
      lock_dir pos_dir = (row_dir == 'L') ? UP : DOWN;
      lock_dir neg_dir = (row_dir == 'L') ? DOWN : UP;
      for (int j = 0; j < length; ++j) {
        if (coefs[j] > 0)
          record_lock(pos_dir, cols[j]);
        else if (coefs[j] < 0)
          record_lock(neg_dir, cols[j]);
      }
    }
  }

  // =========================================================================
  // Step 2: Candidate Identification — O(ncols)
  //
  // Upward candidates: binary, single up-lock, c <= 0 (objective doesn't
  // penalize increase — needed so x pushes against the lock or is indifferent).
  // Downward: symmetric with single down-lock, c >= 0.
  // =========================================================================

  struct candidate_t {
    int col;
    int locking_row;
    lock_dir dir;
  };
  std::vector<candidate_t> candidates;
  candidates.reserve(std::min(ncols, nrows));

  for (int col = 0; col < ncols; ++col) {
    if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;
    if (!is_binary_or_implied(col, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
      continue;
    // Skip singletons: PaPILO's stuffing presolver handles these.
    if (constraint_matrix.getColumnCoefficients(col).getLength() <= 1) continue;

    // can be turned into strict checks if we need to guarantee
    // that we never cut off any optimal solution
    if (locks[UP][col] == 1 && objective[col] <= 0)
      candidates.push_back({col, lock_row[UP][col], UP});
    else if (locks[DOWN][col] == 1 && objective[col] >= 0)
      candidates.push_back({col, lock_row[DOWN][col], DOWN});
  }

  if (this->is_time_exceeded(timer, tlim) || candidates.empty())
    return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 3: Mini-Probing — O(L + K) per row
  //
  // For each locking row (L nonzeros, K candidates), we prove implications by
  // fixing two variables and checking if the row's activity bounds are violated:
  //   - Fix candidate x to its "bad" bound (ub for upward, lb for downward)
  //   - Fix master y to its "unfavorable" bound (0 for upward, 1 for downward)
  //   - If the resulting minimum (LEQ) or maximum (GEQ) activity exceeds the
  //     row's bound, the combination is infeasible, proving y_unfav => x_safe.
  //
  // The master y is the binary variable in the row whose coefficient best
  // amplifies the violation. We track the top-2 most extreme coefficients
  // (neg_y for most negative, pos_y for most positive) so that if the
  // candidate itself is the top-1 extremum, we can fall back to top-2.
  // This keeps master selection O(1) per candidate instead of O(L).
  //
  // Candidates are sorted by lock_row so all K candidates sharing a row are
  // processed together in a single O(L) scan, yielding O(L+K) per row group.
  //
  // dense_row_coefs[] is an ncols-sized scratch array giving O(1) coefficient
  // lookup by column index; populated and cleaned per row in O(L).
  // =========================================================================

  std::sort(candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
    return a.locking_row < b.locking_row;
  });

  struct top2_t {
    std::pair<int, f_t> top1{-1, 0}, top2{-1, 0};

    void update(int idx, f_t val, bound_side side)
    {
      auto better = [side](f_t a, f_t b) { return side == LOWER ? a < b : a > b; };
      if (top1.first == -1 || better(val, top1.second)) {
        top2 = top1;
        top1 = {idx, val};
      } else if (top2.first == -1 || better(val, top2.second)) {
        top2 = {idx, val};
      }
    }
  };

  int n_substitutions = 0;
  std::vector<f_t> dense_row_coefs(ncols, f_t{0});
  std::vector<bool> substituted(ncols, false);

  auto cand_it = candidates.begin();
  while (cand_it != candidates.end()) {
    if (this->is_time_exceeded(timer, tlim)) break;

    int r = cand_it->locking_row;
    if (r < 0) {
      ++cand_it;
      continue;
    }

    // advance row_end to the first candidate with a different locking_row
    auto row_end = std::find_if(
      cand_it, candidates.end(), [r](const candidate_t& c) { return c.locking_row != r; });

    auto row_coeff   = constraint_matrix.getRowCoefficients(r);
    const int* cols  = row_coeff.getIndices();
    const f_t* coefs = row_coeff.getValues();
    const int length = row_coeff.getLength();

    bool has_lhs = !row_flags[r].test(papilo::RowFlag::kLhsInf);
    bool has_rhs = !row_flags[r].test(papilo::RowFlag::kRhsInf);

    // A_min / A_max: tightest possible activity of the row over all variable bounds
    f_t A_min = 0, A_max = 0;
    bool can_reach_neg_inf = false, can_reach_pos_inf = false;
    top2_t neg_y, pos_y;

    for (int j = 0; j < length; ++j) {
      int col     = cols[j];
      f_t coef    = coefs[j];
      bool lb_inf = col_flags[col].test(papilo::ColFlag::kLbInf);
      bool ub_inf = col_flags[col].test(papilo::ColFlag::kUbInf);

      dense_row_coefs[col] = coef;

      // coef > 0: min activity uses lb, max uses ub; coef < 0: swapped
      bool min_inf  = (coef > 0) ? lb_inf : ub_inf;
      bool max_inf  = (coef > 0) ? ub_inf : lb_inf;
      f_t min_bound = (coef > 0) ? lower_bounds[col] : upper_bounds[col];
      f_t max_bound = (coef > 0) ? upper_bounds[col] : lower_bounds[col];

      if (min_inf)
        can_reach_neg_inf = true;
      else
        A_min += coef * min_bound;
      if (max_inf)
        can_reach_pos_inf = true;
      else
        A_max += coef * max_bound;

      if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;
      if (!is_binary_or_implied(col, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
        continue;
      if (lower_bounds[col] == upper_bounds[col]) continue;

      neg_y.update(col, coef, LOWER);
      pos_y.update(col, coef, UPPER);
    }

    // LEQ probe needs finite A_min; GEQ probe needs finite A_max
    bool use_leq_check = has_rhs && !can_reach_neg_inf;
    bool use_geq_check = has_lhs && !can_reach_pos_inf;

    // Probe: replace cand and y's min/max contributions with their fixed test
    // values, then check if the resulting activity violates the row bound.
    // Both candidate and master are binary [0,1], so min/max contributions simplify.
    // anti=false: direct (y_unfav=0 for upward, y_unfav=1 for downward)
    // anti=true:  complement (y_unfav=1 for upward, y_unfav=0 for downward)
    auto evaluate = [&](f_t cand_coeff, bool is_upward, int y_col, f_t y_coef, bool anti) -> bool {
      if (y_col < 0) return false;
      f_t cand_test = is_upward ? cand_coeff : f_t{0};
      f_t y_test    = (is_upward == anti) ? y_coef : f_t{0};
      f_t test      = cand_test + y_test;

      if (use_leq_check) {
        f_t probed_min = A_min - std::min(f_t{0}, cand_coeff) - std::min(f_t{0}, y_coef) + test;
        if (probed_min > rhs_values[r] + tol) return true;
      }
      if (use_geq_check) {
        f_t probed_max = A_max - std::max(f_t{0}, cand_coeff) - std::max(f_t{0}, y_coef) + test;
        if (probed_max < lhs_values[r] - tol) return true;
      }
      return false;
    };

    // Return the best master from the top-2 tracker, skipping excluded columns.
    auto pick_master = [&substituted](const top2_t& t, int exclude) -> std::pair<int, f_t> {
      if (t.top1.first >= 0 && t.top1.first != exclude && !substituted[t.top1.first]) return t.top1;
      if (t.top2.first >= 0 && t.top2.first != exclude && !substituted[t.top2.first]) return t.top2;
      return {-1, f_t{0}};
    };

    for (auto ci = cand_it; ci != row_end; ++ci) {
      auto [cand, locking_row, dir] = *ci;
      if (substituted[cand]) continue;

      bool is_upward = (dir == UP);
      f_t cand_coeff = dense_row_coefs[cand];

      bool proven    = false;
      int master_col = -1;
      bool is_anti   = false;

      // For LEQ upward direct: y=0 zeroes out y's contribution, so the best
      // master is the one with the most negative coefficient (maximizes
      // probed_min). For anti (complement): y=1, so pick most positive instead.
      auto try_prove = [&](bool check, const top2_t& direct_trk, const top2_t& anti_trk) {
        if (!check || proven) return;
        auto [yd, ycd] = pick_master(direct_trk, cand);
        if (evaluate(cand_coeff, is_upward, yd, ycd, false)) {
          proven     = true;
          master_col = yd;
          is_anti    = false;
          return;
        }
        auto [ya, yca] = pick_master(anti_trk, cand);
        if (evaluate(cand_coeff, is_upward, ya, yca, true)) {
          proven     = true;
          master_col = ya;
          is_anti    = true;
          return;
        }
      };
      try_prove(use_leq_check, is_upward ? neg_y : pos_y, is_upward ? pos_y : neg_y);
      try_prove(use_geq_check, is_upward ? pos_y : neg_y, is_upward ? neg_y : pos_y);
      if (!proven) continue;

      // The probe proves a one-directional implication (e.g. y=0 => x=0).
      // The substitution x=y also asserts the reverse (y=1 => x=1), which is
      // only safe if forcing x to its bound doesn't starve other variables of
      // capacity in the locking row. Verify the row becomes globally non-binding
      // when y is in its favorable state.
      // For direct: favorable y=1 (upward) or y=0 (downward).
      // For anti (complement): favorable state flips.
      f_t y_coef_val    = dense_row_coefs[master_col];
      f_t fav_y_contrib = (is_upward != is_anti) ? y_coef_val : f_t{0};

      auto check_side =
        [&](bool active, bool unbounded, f_t activity, f_t orig_y, f_t bound, bound_side side) {
          if (!active || !proven) return;
          if (unbounded) {
            proven = false;
            return;
          }
          f_t fav = activity - orig_y + fav_y_contrib;
          if (side == UPPER ? fav > bound + tol : fav < bound - tol) proven = false;
        };
      check_side(
        has_rhs, can_reach_pos_inf, A_max, std::max(0.0, y_coef_val), rhs_values[r], UPPER);
      check_side(
        has_lhs, can_reach_neg_inf, A_min, std::min(0.0, y_coef_val), lhs_values[r], LOWER);
      if (!proven) continue;

      substituted[cand] = true;
      if (is_anti)
        reductions.replaceCol(cand, master_col, f_t{-1}, f_t{1});  // x = 1 - y
      else
        reductions.replaceCol(cand, master_col, f_t{1}, f_t{0});  // x = y
      ++n_substitutions;
    }

    for (int j = 0; j < length; ++j)
      dense_row_coefs[cols[j]] = 0;
    cand_it = row_end;
  }

  if (n_substitutions == 0) return papilo::PresolveStatus::kUnchanged;

  CUOPT_LOG_DEBUG("Single-lock dual aggregation: %d candidates, %d substitutions",
                  (int)candidates.size(),
                  n_substitutions);

  return papilo::PresolveStatus::kReduced;
}

#define INSTANTIATE(F_TYPE) template class SingleLockDualAggregation<F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
