/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/heuristic_hyper_params.hpp>
#include <mip_heuristics/heuristic_hyper_params_loader.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <string>

namespace cuopt::linear_programming::test {

class HeuristicHyperParamsTest : public ::testing::Test {
 protected:
  std::string tmp_path;

  void SetUp() override
  {
    tmp_path = std::filesystem::temp_directory_path() / "cuopt_heuristic_params_test.config";
  }

  void TearDown() override { std::remove(tmp_path.c_str()); }
};

TEST_F(HeuristicHyperParamsTest, DefaultRoundTrip)
{
  mip_heuristic_hyper_params_t original;
  dump_mip_heuristic_hyper_params(tmp_path, original);

  mip_heuristic_hyper_params_t loaded;
  // Perturb all fields to ensure the loader actually writes them back
  loaded.population_size                    = -999;
  loaded.num_cpufj_threads                  = -999;
  loaded.presolve_time_ratio                = -999;
  loaded.presolve_max_time                  = -999;
  loaded.root_lp_time_ratio                 = -999;
  loaded.root_lp_max_time                   = -999;
  loaded.rins_time_limit                    = -999;
  loaded.rins_max_time_limit                = -999;
  loaded.rins_fix_rate                      = -999;
  loaded.stagnation_trigger                 = -999;
  loaded.max_iterations_without_improvement = -999;
  loaded.initial_infeasibility_weight       = -999;
  loaded.n_of_minimums_for_exit             = -999;
  loaded.enabled_recombiners                = -999;
  loaded.cycle_detection_length             = -999;
  loaded.relaxed_lp_time_limit              = -999;
  loaded.related_vars_time_limit            = -999;

  fill_mip_heuristic_hyper_params(tmp_path, loaded);

  EXPECT_EQ(loaded.population_size, original.population_size);
  EXPECT_EQ(loaded.num_cpufj_threads, original.num_cpufj_threads);
  EXPECT_DOUBLE_EQ(loaded.presolve_time_ratio, original.presolve_time_ratio);
  EXPECT_DOUBLE_EQ(loaded.presolve_max_time, original.presolve_max_time);
  EXPECT_DOUBLE_EQ(loaded.root_lp_time_ratio, original.root_lp_time_ratio);
  EXPECT_DOUBLE_EQ(loaded.root_lp_max_time, original.root_lp_max_time);
  EXPECT_DOUBLE_EQ(loaded.rins_time_limit, original.rins_time_limit);
  EXPECT_DOUBLE_EQ(loaded.rins_max_time_limit, original.rins_max_time_limit);
  EXPECT_DOUBLE_EQ(loaded.rins_fix_rate, original.rins_fix_rate);
  EXPECT_EQ(loaded.stagnation_trigger, original.stagnation_trigger);
  EXPECT_EQ(loaded.max_iterations_without_improvement, original.max_iterations_without_improvement);
  EXPECT_DOUBLE_EQ(loaded.initial_infeasibility_weight, original.initial_infeasibility_weight);
  EXPECT_EQ(loaded.n_of_minimums_for_exit, original.n_of_minimums_for_exit);
  EXPECT_EQ(loaded.enabled_recombiners, original.enabled_recombiners);
  EXPECT_EQ(loaded.cycle_detection_length, original.cycle_detection_length);
  EXPECT_DOUBLE_EQ(loaded.relaxed_lp_time_limit, original.relaxed_lp_time_limit);
  EXPECT_DOUBLE_EQ(loaded.related_vars_time_limit, original.related_vars_time_limit);
}

TEST_F(HeuristicHyperParamsTest, CustomValuesRoundTrip)
{
  mip_heuristic_hyper_params_t original;
  original.population_size                    = 64;
  original.num_cpufj_threads                  = 4;
  original.presolve_time_ratio                = 0.2;
  original.presolve_max_time                  = 120.0;
  original.root_lp_time_ratio                 = 0.05;
  original.root_lp_max_time                   = 30.0;
  original.rins_time_limit                    = 5.0;
  original.rins_max_time_limit                = 40.0;
  original.rins_fix_rate                      = 0.7;
  original.stagnation_trigger                 = 5;
  original.max_iterations_without_improvement = 12;
  original.initial_infeasibility_weight       = 500.0;
  original.n_of_minimums_for_exit             = 10000;
  original.enabled_recombiners                = 5;
  original.cycle_detection_length             = 50;
  original.relaxed_lp_time_limit              = 2.0;
  original.related_vars_time_limit            = 60.0;

  dump_mip_heuristic_hyper_params(tmp_path, original);

  mip_heuristic_hyper_params_t loaded;
  fill_mip_heuristic_hyper_params(tmp_path, loaded);

  EXPECT_EQ(loaded.population_size, 64);
  EXPECT_EQ(loaded.num_cpufj_threads, 4);
  EXPECT_DOUBLE_EQ(loaded.presolve_time_ratio, 0.2);
  EXPECT_DOUBLE_EQ(loaded.presolve_max_time, 120.0);
  EXPECT_DOUBLE_EQ(loaded.root_lp_time_ratio, 0.05);
  EXPECT_DOUBLE_EQ(loaded.root_lp_max_time, 30.0);
  EXPECT_DOUBLE_EQ(loaded.rins_time_limit, 5.0);
  EXPECT_DOUBLE_EQ(loaded.rins_max_time_limit, 40.0);
  EXPECT_DOUBLE_EQ(loaded.rins_fix_rate, 0.7);
  EXPECT_EQ(loaded.stagnation_trigger, 5);
  EXPECT_EQ(loaded.max_iterations_without_improvement, 12);
  EXPECT_DOUBLE_EQ(loaded.initial_infeasibility_weight, 500.0);
  EXPECT_EQ(loaded.n_of_minimums_for_exit, 10000);
  EXPECT_EQ(loaded.enabled_recombiners, 5);
  EXPECT_EQ(loaded.cycle_detection_length, 50);
  EXPECT_DOUBLE_EQ(loaded.relaxed_lp_time_limit, 2.0);
  EXPECT_DOUBLE_EQ(loaded.related_vars_time_limit, 60.0);
}

TEST_F(HeuristicHyperParamsTest, PartialConfigKeepsDefaults)
{
  // Write a config with only two keys
  {
    std::ofstream f(tmp_path);
    f << "population_size = 128\n";
    f << "rins_fix_rate = 0.3\n";
  }

  mip_heuristic_hyper_params_t loaded;
  fill_mip_heuristic_hyper_params(tmp_path, loaded);

  EXPECT_EQ(loaded.population_size, 128);
  EXPECT_DOUBLE_EQ(loaded.rins_fix_rate, 0.3);
  // Everything else should be unchanged from struct defaults
  mip_heuristic_hyper_params_t defaults;
  EXPECT_EQ(loaded.num_cpufj_threads, defaults.num_cpufj_threads);
  EXPECT_DOUBLE_EQ(loaded.presolve_time_ratio, defaults.presolve_time_ratio);
  EXPECT_EQ(loaded.n_of_minimums_for_exit, defaults.n_of_minimums_for_exit);
  EXPECT_EQ(loaded.enabled_recombiners, defaults.enabled_recombiners);
}

TEST_F(HeuristicHyperParamsTest, CommentsAndBlankLinesIgnored)
{
  {
    std::ofstream f(tmp_path);
    f << "# This is a comment\n";
    f << "\n";
    f << "# Another comment\n";
    f << "population_size = 42\n";
    f << "\n";
  }

  mip_heuristic_hyper_params_t loaded;
  fill_mip_heuristic_hyper_params(tmp_path, loaded);
  EXPECT_EQ(loaded.population_size, 42);
}

}  // namespace cuopt::linear_programming::test
