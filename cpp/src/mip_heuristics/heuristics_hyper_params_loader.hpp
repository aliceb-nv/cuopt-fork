/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/heuristics_hyper_params.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace cuopt::linear_programming {

namespace {

using hp_t              = mip_heuristics_hyper_params_t;
using double_member_ptr = double hp_t::*;
using int_member_ptr    = int hp_t::*;

struct double_param_t {
  const char* name;
  double_member_ptr field;
  double min_val;
  double max_val;
  const char* description;
};

struct int_param_t {
  const char* name;
  int_member_ptr field;
  int min_val;
  int max_val;
  const char* description;
};

constexpr double inf = std::numeric_limits<double>::infinity();

// clang-format off
constexpr double_param_t double_params[] = {
  {"presolve_time_ratio",          &hp_t::presolve_time_ratio,          0.0,  1.0, "fraction of total time for presolve"},
  {"presolve_max_time",            &hp_t::presolve_max_time,            0.0,  inf, "hard cap on presolve seconds"},
  {"root_lp_time_ratio",           &hp_t::root_lp_time_ratio,           0.0,  1.0, "fraction of total time for root LP"},
  {"root_lp_max_time",             &hp_t::root_lp_max_time,             0.0,  inf, "hard cap on root LP seconds"},
  {"rins_time_limit",              &hp_t::rins_time_limit,              0.0,  inf, "per-call RINS sub-MIP time"},
  {"rins_max_time_limit",          &hp_t::rins_max_time_limit,          0.0,  inf, "ceiling for RINS adaptive time budget"},
  {"rins_fix_rate",                &hp_t::rins_fix_rate,                0.0,  1.0, "RINS variable fix rate"},
  {"initial_infeasibility_weight", &hp_t::initial_infeasibility_weight, 1e-9, inf, "constraint violation penalty seed"},
  {"relaxed_lp_time_limit",        &hp_t::relaxed_lp_time_limit,       1e-9, inf, "base relaxed LP time cap in heuristics"},
  {"related_vars_time_limit",      &hp_t::related_vars_time_limit,     1e-9, inf, "time for related-variable structure build"},
};

constexpr int_param_t int_params[] = {
  {"population_size",                    &hp_t::population_size,                    1, std::numeric_limits<int>::max(), "max solutions in pool"},
  {"num_cpufj_threads",                  &hp_t::num_cpufj_threads,                  0, std::numeric_limits<int>::max(), "parallel CPU FJ climbers"},
  {"stagnation_trigger",                 &hp_t::stagnation_trigger,                 1, std::numeric_limits<int>::max(), "FP loops w/o improvement before recombination"},
  {"max_iterations_without_improvement", &hp_t::max_iterations_without_improvement, 1, std::numeric_limits<int>::max(), "diversity step depth after stagnation"},
  {"n_of_minimums_for_exit",             &hp_t::n_of_minimums_for_exit,             1, std::numeric_limits<int>::max(), "FJ baseline local-minima exit threshold"},
  {"enabled_recombiners",                &hp_t::enabled_recombiners,                0, 15,                              "bitmask: 1=BP 2=FP 4=LS 8=SubMIP"},
  {"cycle_detection_length",             &hp_t::cycle_detection_length,             1, std::numeric_limits<int>::max(), "FP assignment cycle ring buffer length"},
};
// clang-format on

template <typename ParamDesc>
bool try_parse_param(const ParamDesc* params,
                     size_t n_params,
                     const std::string& var,
                     std::istringstream& iss,
                     hp_t& hp,
                     const std::string& line)
{
  for (size_t i = 0; i < n_params; ++i) {
    const auto& p = params[i];
    if (var != p.name) continue;
    cuopt_expects(bool(iss >> hp.*p.field),
                  error_type_t::ValidationError,
                  "MIP heuristic config: bad value for %s: %s",
                  p.name,
                  line.c_str());
    std::string trailing;
    cuopt_expects(!bool(iss >> trailing),
                  error_type_t::ValidationError,
                  "MIP heuristic config: trailing junk for %s: %s",
                  p.name,
                  line.c_str());
    cuopt_expects(hp.*p.field >= p.min_val && hp.*p.field <= p.max_val,
                  error_type_t::ValidationError,
                  "MIP heuristic config: %s = %s out of range [%s, %s]",
                  p.name,
                  std::to_string(hp.*p.field).c_str(),
                  std::to_string(p.min_val).c_str(),
                  std::to_string(p.max_val).c_str());
    CUOPT_LOG_INFO("MIP heuristic config: %s = %s", p.name, std::to_string(hp.*p.field).c_str());
    return true;
  }
  return false;
}

}  // namespace

/**
 * @brief Load MIP heuristic hyper-parameters from a key=value text file.
 *
 * Format: one assignment per line, e.g.  population_size = 64
 * Lines starting with # (optionally indented) are comments.
 * Unknown keys, malformed values, and out-of-range values all throw.
 * Partial files are fine — omitted keys keep their struct defaults.
 */
inline void fill_mip_heuristics_hyper_params(const std::string& path,
                                             mip_heuristics_hyper_params_t& params)
{
  cuopt_expects(!std::filesystem::is_directory(path) && std::filesystem::exists(path),
                error_type_t::ValidationError,
                "MIP heuristic config: not a valid file: %s",
                path.c_str());
  std::ifstream file(path);
  cuopt_expects(file.is_open(),
                error_type_t::ValidationError,
                "MIP heuristic config: cannot open: %s",
                path.c_str());
  std::string line;

  while (std::getline(file, line)) {
    // Trim leading whitespace, then skip blank lines and comments
    auto first_non_ws = std::find_if_not(line.begin(), line.end(), ::isspace);
    if (first_non_ws == line.end() || *first_non_ws == '#') continue;
    line.erase(line.begin(), first_non_ws);

    std::istringstream iss(line);
    std::string var;

    cuopt_expects(iss >> var >> std::ws && iss.get() == '=',
                  error_type_t::ValidationError,
                  "MIP heuristic config: bad line: %s",
                  line.c_str());

    bool found = try_parse_param(double_params, std::size(double_params), var, iss, params, line) ||
                 try_parse_param(int_params, std::size(int_params), var, iss, params, line);

    cuopt_expects(found,
                  error_type_t::ValidationError,
                  "MIP heuristic config: unknown parameter: %s",
                  var.c_str());
  }

  CUOPT_LOG_INFO("MIP heuristic config loaded from: %s", path.c_str());
}

/**
 * @brief Dump current hyper-parameters to a key=value text file.
 *
 * Each entry is commented out and annotated with its type, allowed range,
 * and description. The output is a valid config file that can be loaded
 * back with fill_mip_heuristics_hyper_params() after uncommenting the
 * desired overrides.
 */
inline bool dump_mip_heuristics_hyper_params(const std::string& path,
                                             const mip_heuristics_hyper_params_t& params)
{
  std::ofstream file(path);
  if (!file.is_open()) {
    CUOPT_LOG_ERROR("Cannot open file for writing: %s", path.c_str());
    return false;
  }
  file << "# MIP heuristic hyper-parameters (auto-generated)\n";
  file << "# Uncomment and change only the values you want to override.\n\n";
  for (const auto& p : int_params) {
    file << "# " << p.description << " (int, range: [" << p.min_val << ", " << p.max_val << "])\n";
    file << "# " << p.name << " = " << params.*p.field << "\n\n";
  }
  for (const auto& p : double_params) {
    file << "# " << p.description << " (double, range: [" << p.min_val << ", " << p.max_val
         << "])\n";
    file << "# " << p.name << " = " << params.*p.field << "\n\n";
  }
  return true;
}

}  // namespace cuopt::linear_programming
