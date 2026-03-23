/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/heuristic_hyper_params.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

namespace {

template <typename T>
void parse_hyper_param_value(std::istringstream& iss, T& value)
{
  iss >> value;
}

struct double_entry_t {
  const char* name;
  double mip_heuristic_hyper_params_t::* field;
};

struct int_entry_t {
  const char* name;
  int mip_heuristic_hyper_params_t::* field;
};

// clang-format off
constexpr double_entry_t double_entries[] = {
  {"presolve_time_ratio",             &mip_heuristic_hyper_params_t::presolve_time_ratio},
  {"presolve_max_time",               &mip_heuristic_hyper_params_t::presolve_max_time},
  {"root_lp_time_ratio",              &mip_heuristic_hyper_params_t::root_lp_time_ratio},
  {"root_lp_max_time",                &mip_heuristic_hyper_params_t::root_lp_max_time},
  {"rins_time_limit",                 &mip_heuristic_hyper_params_t::rins_time_limit},
  {"rins_max_time_limit",             &mip_heuristic_hyper_params_t::rins_max_time_limit},
  {"rins_fix_rate",                   &mip_heuristic_hyper_params_t::rins_fix_rate},
  {"initial_infeasibility_weight",    &mip_heuristic_hyper_params_t::initial_infeasibility_weight},
  {"relaxed_lp_time_limit",           &mip_heuristic_hyper_params_t::relaxed_lp_time_limit},
  {"related_vars_time_limit",         &mip_heuristic_hyper_params_t::related_vars_time_limit},
};

constexpr int_entry_t int_entries[] = {
  {"population_size",                    &mip_heuristic_hyper_params_t::population_size},
  {"num_cpufj_threads",                  &mip_heuristic_hyper_params_t::num_cpufj_threads},
  {"stagnation_trigger",                 &mip_heuristic_hyper_params_t::stagnation_trigger},
  {"max_iterations_without_improvement", &mip_heuristic_hyper_params_t::max_iterations_without_improvement},
  {"n_of_minimums_for_exit",             &mip_heuristic_hyper_params_t::n_of_minimums_for_exit},
  {"enabled_recombiners",                &mip_heuristic_hyper_params_t::enabled_recombiners},
  {"cycle_detection_length",             &mip_heuristic_hyper_params_t::cycle_detection_length},
};
// clang-format on

}  // namespace

/**
 * @brief Load MIP heuristic hyper-parameters from a key=value text file.
 *
 * Format: one assignment per line, e.g.  population_size = 64
 * Unknown keys cause a hard error. Partial files are fine — unmentioned
 * fields keep their struct defaults.
 */
inline void fill_mip_heuristic_hyper_params(const std::string& path,
                                            mip_heuristic_hyper_params_t& params)
{
  if (!std::filesystem::exists(path)) {
    std::cerr << "MIP heuristic config file path is not valid: " << path << std::endl;
    exit(-1);
  }
  std::ifstream file(path);
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string var;

    if (!(iss >> var >> std::ws && iss.get() == '=')) {
      std::cerr << "MIP heuristic config: bad line: " << line << std::endl;
      exit(-1);
    }

    bool found = false;
    for (const auto& e : double_entries) {
      if (var == e.name) {
        parse_hyper_param_value(iss, params.*e.field);
        found = true;
        break;
      }
    }
    if (!found) {
      for (const auto& e : int_entries) {
        if (var == e.name) {
          parse_hyper_param_value(iss, params.*e.field);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      std::cerr << "MIP heuristic config: unknown parameter: " << var << std::endl;
      exit(-1);
    }
  }
}

/**
 * @brief Dump current hyper-parameters to a key=value text file.
 *
 * The output is a valid config file that can be loaded back with
 * fill_mip_heuristic_hyper_params().
 */
inline void dump_mip_heuristic_hyper_params(const std::string& path,
                                            const mip_heuristic_hyper_params_t& params)
{
  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Cannot open file for writing: " << path << std::endl;
    return;
  }
  file << "# MIP heuristic hyper-parameters (auto-generated)\n\n";
  for (const auto& e : int_entries) {
    file << e.name << " = " << params.*e.field << "\n";
  }
  for (const auto& e : double_entries) {
    file << e.name << " = " << params.*e.field << "\n";
  }
}

}  // namespace cuopt::linear_programming
