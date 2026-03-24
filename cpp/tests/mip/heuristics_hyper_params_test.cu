/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/heuristics_hyper_params.hpp>
#include <mip_heuristics/heuristics_hyper_params_loader.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <string>

namespace cuopt::linear_programming::test {

class HeuristicsHyperParamsTest : public ::testing::Test {
 protected:
  std::string tmp_path;

  void SetUp() override
  {
    tmp_path = std::filesystem::temp_directory_path() / "cuopt_heuristic_params_test.config";
  }

  void TearDown() override { std::remove(tmp_path.c_str()); }
};

TEST_F(HeuristicsHyperParamsTest, DumpedFileIsAllCommentedOut)
{
  mip_heuristics_hyper_params_t original;
  dump_mip_heuristics_hyper_params(tmp_path, original);

  // Loading the commented-out dump should leave struct defaults unchanged
  mip_heuristics_hyper_params_t loaded;
  loaded.population_size = 9999;
  fill_mip_heuristics_hyper_params(tmp_path, loaded);
  EXPECT_EQ(loaded.population_size, 9999);
}

TEST_F(HeuristicsHyperParamsTest, DumpedFileIsParseable)
{
  mip_heuristics_hyper_params_t original;
  dump_mip_heuristics_hyper_params(tmp_path, original);

  // The dumped file should parse without errors (all lines are comments)
  mip_heuristics_hyper_params_t loaded;
  EXPECT_NO_THROW(fill_mip_heuristics_hyper_params(tmp_path, loaded));
}

TEST_F(HeuristicsHyperParamsTest, CustomValuesRoundTrip)
{
  {
    std::ofstream f(tmp_path);
    f << "population_size = 64\n";
    f << "num_cpufj_threads = 4\n";
    f << "presolve_time_ratio = 0.2\n";
    f << "presolve_max_time = 120\n";
    f << "root_lp_time_ratio = 0.05\n";
    f << "root_lp_max_time = 30\n";
    f << "rins_time_limit = 5\n";
    f << "rins_max_time_limit = 40\n";
    f << "rins_fix_rate = 0.7\n";
    f << "stagnation_trigger = 5\n";
    f << "max_iterations_without_improvement = 12\n";
    f << "initial_infeasibility_weight = 500\n";
    f << "n_of_minimums_for_exit = 10000\n";
    f << "enabled_recombiners = 5\n";
    f << "cycle_detection_length = 50\n";
    f << "relaxed_lp_time_limit = 2\n";
    f << "related_vars_time_limit = 60\n";
  }

  mip_heuristics_hyper_params_t loaded;
  fill_mip_heuristics_hyper_params(tmp_path, loaded);

  EXPECT_EQ(loaded.population_size, 64);
  codex EXPECT_EQ(loaded.num_cpufj_threads, 4);
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

TEST_F(HeuristicsHyperParamsTest, PartialConfigKeepsDefaults)
{
  // Write a config with only two keys
  {
    std::ofstream f(tmp_path);
    f << "population_size = 128\n";
    f << "rins_fix_rate = 0.3\n";
  }

  mip_heuristics_hyper_params_t loaded;
  fill_mip_heuristics_hyper_params(tmp_path, loaded);

  EXPECT_EQ(loaded.population_size, 128);
  EXPECT_DOUBLE_EQ(loaded.rins_fix_rate, 0.3);
  // Everything else should be unchanged from struct defaults
  mip_heuristics_hyper_params_t defaults;
  EXPECT_EQ(loaded.num_cpufj_threads, defaults.num_cpufj_threads);
  EXPECT_DOUBLE_EQ(loaded.presolve_time_ratio, defaults.presolve_time_ratio);
  EXPECT_EQ(loaded.n_of_minimums_for_exit, defaults.n_of_minimums_for_exit);
  EXPECT_EQ(loaded.enabled_recombiners, defaults.enabled_recombiners);
}

TEST_F(HeuristicsHyperParamsTest, CommentsAndBlankLinesIgnored)
{
  {
    std::ofstream f(tmp_path);
    f << "# This is a comment\n";
    f << "\n";
    f << "# Another comment\n";
    f << "population_size = 42\n";
    f << "\n";
  }

  mip_heuristics_hyper_params_t loaded;
  fill_mip_heuristics_hyper_params(tmp_path, loaded);
  EXPECT_EQ(loaded.population_size, 42);
}

TEST_F(HeuristicsHyperParamsTest, UnknownKeyThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "bogus_key = 42\n";
  }
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params(tmp_path, loaded), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, BadNumericValueThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "population_size = not_a_number\n";
  }
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params(tmp_path, loaded), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, TrailingJunkThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "population_size = 64foo\n";
  }
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params(tmp_path, loaded), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, RangeViolationCycleDetectionThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "cycle_detection_length = 0\n";
  }
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params(tmp_path, loaded), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, RangeViolationFixRateThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "rins_fix_rate = 2.0\n";
  }
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params(tmp_path, loaded), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, NonexistentFileThrows)
{
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params("/tmp/does_not_exist_cuopt_test.config", loaded),
               cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, DirectoryPathThrows)
{
  mip_heuristics_hyper_params_t loaded;
  EXPECT_THROW(fill_mip_heuristics_hyper_params("/tmp", loaded), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, IndentedCommentAndWhitespaceLinesIgnored)
{
  {
    std::ofstream f(tmp_path);
    f << "   # indented comment\n";
    f << "  \t  \n";
    f << "population_size = 99\n";
  }
  mip_heuristics_hyper_params_t loaded;
  fill_mip_heuristics_hyper_params(tmp_path, loaded);
  EXPECT_EQ(loaded.population_size, 99);
}

}  // namespace cuopt::linear_programming::test
