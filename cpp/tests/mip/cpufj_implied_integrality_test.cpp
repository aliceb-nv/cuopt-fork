/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::mip {
namespace {

auto make_climber(std::vector<std::vector<double>> rows,
                  std::vector<double> rhs,
                  std::vector<var_t> types,
                  std::vector<double> cost  = {},
                  std::vector<double> upper = {},
                  bool maximize             = false,
                  std::vector<char> senses  = {})
{
  const int n = types.size(), m = rows.size();
  std::vector<double> coefficients;
  std::vector<int> variables, offsets{0};
  for (const auto& row : rows) {
    for (int v = 0; v < n; ++v) {
      if (row[v] == 0) continue;
      coefficients.push_back(row[v]);
      variables.push_back(v);
    }
    offsets.push_back(variables.size());
  }
  if (cost.empty()) cost.assign(n, 0);
  if (upper.empty()) upper.assign(n, 3);
  if (senses.empty()) senses.assign(m, 'E');
  static std::atomic<bool> stop{false};
  return init_fj_cpu_from_host_model<int, double>(n,
                                                  m,
                                                  coefficients.size(),
                                                  maximize,
                                                  1,
                                                  0,
                                                  coefficients,
                                                  variables,
                                                  offsets,
                                                  cost,
                                                  std::vector<double>(n, 0),
                                                  upper,
                                                  {},
                                                  {},
                                                  rhs,
                                                  senses,
                                                  types,
                                                  {},
                                                  stop,
                                                  {});
}

constexpr auto C = var_t::CONTINUOUS;
constexpr auto I = var_t::INTEGER;

}  // namespace

TEST(CpufjImpliedIntegrality, PropagatesEqualityChainAndUpdatesBinaryMetadata)
{
  auto c = make_climber({{0, -1, 0.5}, {-1, 1, 0}}, {0, 0}, {I, C, C}, {}, {1, 1, 3});
  EXPECT_EQ(c->problem->h_var_types, (std::vector<var_t>{I, I, I}));
  EXPECT_EQ(c->h_is_binary_variable[1], 1);
  EXPECT_EQ(c->n_binary_vars, 2);
  EXPECT_EQ(c->n_integer_vars, 1);
  EXPECT_EQ(c->h_binary_indices.size(), 2);
}

TEST(CpufjImpliedIntegrality, CertifiesIndependentPairsAfterPropagation)
{
  auto c = make_climber({{-1, 1, 0, 0}, {0, 1, 1, -1}}, {0, 0}, {I, C, C, C}, {0, 0, 40, 40});
  EXPECT_EQ(c->problem->h_var_types, (std::vector<var_t>{I, I, I, I}));
  EXPECT_EQ(c->problem->h_obj_coeffs, (std::vector<double>{0, 0, 40, 40}));
}

TEST(CpufjImpliedIntegrality, RejectsFractionalRhsCoefficientAndBounds)
{
  EXPECT_EQ(make_climber({{-1, 1}}, {0.5}, {I, C})->problem->h_var_types[1], C);
  EXPECT_EQ(make_climber({{-0.5, 1}}, {0}, {I, C})->problem->h_var_types[1], C);
  EXPECT_EQ(make_climber({{-1, 1}}, {0}, {I, C}, {}, {3, 2.5})->problem->h_var_types[1], C);
  EXPECT_EQ(make_climber({{-1, 1}}, {1e-5}, {I, C})->problem->h_var_types[1], C);
  EXPECT_EQ(make_climber({{-2, 3}}, {0}, {I, C})->problem->h_var_types[1], C);
}

TEST(CpufjImpliedIntegrality, AcceptsNonDyadicPivotsButRejectsInequalities)
{
  EXPECT_EQ(make_climber({{-3, 3}}, {0}, {I, C})->problem->h_var_types[1], I);
  EXPECT_EQ(make_climber({{-1, 1}}, {0}, {I, C}, {}, {}, false, {'L'})->problem->h_var_types[1], C);
}

TEST(CpufjImpliedIntegrality, UsesSharedScalingTolerance)
{
  EXPECT_EQ(make_climber({{-1, 1}}, {1e-12}, {I, C})->problem->h_var_types[1], I);
  EXPECT_EQ(make_climber({{-1 + 1e-12, 1}}, {0}, {I, C})->problem->h_var_types[1], I);
  EXPECT_EQ(make_climber({{-0.3, 0.3}}, {0}, {I, C})->problem->h_var_types[1], I);
  EXPECT_EQ(make_climber({{-0.3, 0.3, -0.3}}, {0}, {I, C, C}, {0, 1, 1})->problem->h_var_types,
            (std::vector<var_t>{I, I, I}));
}

TEST(CpufjImpliedIntegrality, RejectsCoupledOrUnsafePairs)
{
  EXPECT_EQ(
    make_climber({{-1, 1, -1}, {0, 1, 1}}, {0, 1}, {I, C, C}, {0, 1, 1})->problem->h_var_types,
    (std::vector<var_t>{I, C, C}));
  EXPECT_EQ(make_climber({{-1, 1, -1}}, {0.5}, {I, C, C}, {0, 1, 1})->problem->h_var_types,
            (std::vector<var_t>{I, C, C}));
  EXPECT_EQ(make_climber({{-1, 1, -1}}, {0}, {I, C, C}, {0, 1, 2})->problem->h_var_types,
            (std::vector<var_t>{I, C, C}));
  EXPECT_EQ(
    make_climber({{-1, 1, -1}}, {0}, {I, C, C}, {0, 1, 1}, {3, 2.5, 3})->problem->h_var_types,
    (std::vector<var_t>{I, C, C}));
  EXPECT_EQ(make_climber({{-1, 1, -1}}, {0}, {I, C, C}, {0, 1, 1}, {}, true)->problem->h_var_types,
            (std::vector<var_t>{I, C, C}));
}

TEST(CpufjImpliedIntegrality, RejectsZeroScaledPivotUnsafeMagnitudeAndNonfiniteBounds)
{
  EXPECT_EQ(make_climber({{-1, 1e-15}}, {0}, {I, C})->problem->h_var_types[1], C);
  EXPECT_EQ(make_climber({{-1e16, 1}}, {0}, {I, C})->problem->h_var_types[1], C);
  EXPECT_EQ(make_climber({{-1, 1}}, {0}, {I, C}, {}, {3, INFINITY})->problem->h_var_types[1], C);
}

TEST(CpufjImpliedIntegrality, LongReverseOrderedChain)
{
  constexpr int n = 256;
  std::vector<std::vector<double>> rows(n - 1, std::vector<double>(n, 0));
  for (int r = 0; r < n - 1; ++r) {
    rows[r][n - 2 - r] = -1;
    rows[r][n - 1 - r] = 1;
  }
  std::vector<var_t> types(n, C);
  types[0] = I;
  auto c   = make_climber(rows, std::vector<double>(n - 1, 0), types);
  EXPECT_EQ(c->problem->h_var_types, std::vector<var_t>(n, I));
}

}  // namespace cuopt::mathematical_optimization::mip
