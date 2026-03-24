/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/backend_selection.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/logger.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <unistd.h>
#include <argparse/argparse.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <math_optimization/solution_reader.hpp>

#include <cuopt/version_config.hpp>

static char cuda_module_loading_env[] = "CUDA_MODULE_LOADING=EAGER";

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline cuopt::init_logger_t dummy_logger(
  const cuopt::linear_programming::solver_settings_t<int, double>& settings)
{
  return cuopt::init_logger_t(settings.template get_parameter<std::string>(CUOPT_LOG_FILE),
                              settings.template get_parameter<bool>(CUOPT_LOG_TO_CONSOLE));
}

int run_single_file(const std::string& file_path,
                    const std::string& initial_solution_file,
                    bool solve_relaxation,
                    const std::map<std::string, std::string>& settings_strings,
                    const std::string& params_file = "")
{
  cuopt::linear_programming::solver_settings_t<int, double> settings;

  try {
    if (!params_file.empty()) { settings.load_parameters_from_file(params_file); }
    for (auto& [key, val] : settings_strings) {
      settings.set_parameter_from_string(key, val);
    }
  } catch (const std::exception& e) {
    auto log = dummy_logger(settings);
    CUOPT_LOG_ERROR("Error: %s", e.what());
    return -1;
  }

  std::string base_filename = file_path.substr(file_path.find_last_of("/\\") + 1);

  constexpr bool input_mps_strict = false;
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model;
  bool parsing_failed = false;
  {
    CUOPT_LOG_INFO("Reading file %s", base_filename.c_str());
    try {
      mps_data_model = cuopt::mps_parser::parse_mps<int, double>(file_path, input_mps_strict);
    } catch (const std::logic_error& e) {
      CUOPT_LOG_ERROR("MPS parser execption: %s", e.what());
      parsing_failed = true;
    }
  }
  if (parsing_failed) {
    auto log = dummy_logger(settings);
    CUOPT_LOG_ERROR("Parsing MPS failed. Exiting!");
    return -1;
  }

  auto memory_backend = cuopt::linear_programming::get_memory_backend_type();
  std::unique_ptr<raft::handle_t> handle_ptr;
  std::unique_ptr<cuopt::linear_programming::optimization_problem_interface_t<int, double>>
    problem_interface;

  if (memory_backend == cuopt::linear_programming::memory_backend_t::GPU) {
    handle_ptr = std::make_unique<raft::handle_t>();
    problem_interface =
      std::make_unique<cuopt::linear_programming::optimization_problem_t<int, double>>(
        handle_ptr.get());
  } else {
    problem_interface =
      std::make_unique<cuopt::linear_programming::cpu_optimization_problem_t<int, double>>();
  }

  cuopt::linear_programming::populate_from_mps_data_model(problem_interface.get(), mps_data_model);

  const bool is_mip = (problem_interface->get_problem_category() ==
                         cuopt::linear_programming::problem_category_t::MIP ||
                       problem_interface->get_problem_category() ==
                         cuopt::linear_programming::problem_category_t::IP) &&
                      !solve_relaxation;

  try {
    auto initial_solution =
      initial_solution_file.empty()
        ? std::vector<double>()
        : cuopt::linear_programming::solution_reader_t::get_variable_values_from_sol_file(
            initial_solution_file, mps_data_model.get_variable_names());

    if (is_mip) {
      auto& mip_settings = settings.get_mip_settings();
      if (initial_solution.size() > 0) {
        mip_settings.add_initial_solution(initial_solution.data(), initial_solution.size());
      }
    } else {
      auto& lp_settings = settings.get_pdlp_settings();
      if (initial_solution.size() > 0) {
        lp_settings.set_initial_primal_solution(initial_solution.data(), initial_solution.size());
      }
    }
  } catch (const std::exception& e) {
    auto log = dummy_logger(settings);
    CUOPT_LOG_ERROR("Error: %s", e.what());
    return -1;
  }

  try {
    if (is_mip) {
      auto& mip_settings = settings.get_mip_settings();
      auto solution = cuopt::linear_programming::solve_mip(problem_interface.get(), mip_settings);
    } else {
      auto& lp_settings = settings.get_pdlp_settings();
      auto solution     = cuopt::linear_programming::solve_lp(problem_interface.get(), lp_settings);
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "cuopt_cli error: %s\n", e.what());
    CUOPT_LOG_ERROR("Error: %s", e.what());
    return -1;
  }

  return 0;
}

std::string param_name_to_arg_name(const std::string& input)
{
  std::string result = "--";
  result += input;
  std::replace(result.begin(), result.end(), '_', '-');
  return result;
}

int set_cuda_module_loading(int argc, char* argv[])
{
  int method_int = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--method" || arg == "-m") && i + 1 < argc) {
      try {
        method_int = std::stoi(argv[i + 1]);
      } catch (...) {
        std::cerr << "Invalid value for --method: " << argv[i + 1] << std::endl;
        return 1;
      }
      break;
    }
    if (arg.rfind("--method=", 0) == 0) {
      try {
        method_int = std::stoi(arg.substr(9));
      } catch (...) {
        std::cerr << "Invalid value for --method: " << arg << std::endl;
        return 1;
      }
      break;
    }
  }

  char* env_val = getenv("CUDA_MODULE_LOADING");
  if (method_int == 0 && (!env_val || env_val[0] == '\0')) {
    CUOPT_LOG_INFO("Setting CUDA_MODULE_LOADING to EAGER");
    putenv(cuda_module_loading_env);
  }
  return 0;
}

int main(int argc, char* argv[])
{
  // Handle --dump-hyper-params before argparse so no MPS file is required
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dump-hyper-params" && i + 1 < argc) {
      cuopt::linear_programming::solver_settings_t<int, double> settings;
      bool ok = settings.dump_parameters_to_file(argv[i + 1], true);
      return ok ? 0 : 1;
    }
    if (arg == "--show-hyper-params") {
      cuopt::linear_programming::solver_settings_t<int, double> settings;
      settings.dump_parameters_to_file("/dev/stdout", true);
      return 0;
    }
  }

  if (set_cuda_module_loading(argc, argv) != 0) { return 1; }

  const std::string version_string = std::string("cuOpt ") + std::to_string(CUOPT_VERSION_MAJOR) +
                                     "." + std::to_string(CUOPT_VERSION_MINOR) + "." +
                                     std::to_string(CUOPT_VERSION_PATCH);

  argparse::ArgumentParser program("cuopt_cli", version_string);

  program.add_argument("filename").help("input mps file").nargs(1).required();

  program.add_argument("--initial-solution")
    .help("path to the initial solution .sol file")
    .default_value("");

  program.add_argument("--relaxation")
    .help("solve the LP relaxation of the MIP")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--presolve")
    .help("enable/disable presolve (default: true for MIP problems, false for LP problems)")
    .default_value(true)
    .implicit_value(true);

  program.add_argument("--params-file")
    .help("path to parameter config file (key = value format, supports all parameters)")
    .default_value(std::string(""));

  program.add_argument("--dump-hyper-params")
    .help("write default hyper-parameters to the given file and exit")
    .default_value(std::string(""));

  program.add_argument("--show-hyper-params")
    .help("print hyper-parameters in config-file format and exit")
    .default_value(false)
    .implicit_value(true);

  std::map<std::string, std::string> arg_name_to_param_name;

  program.add_argument("--pdlp-precision")
    .help(
      "PDLP precision mode. default: native type, single: FP32 internally, "
      "double: FP64 explicitly, mixed: mixed-precision SpMV (FP32 matrix, FP64 vectors).")
    .default_value(std::string("-1"))
    .choices("default", "single", "double", "mixed", "-1", "0", "1", "2");
  arg_name_to_param_name["--pdlp-precision"] = CUOPT_PDLP_PRECISION;

  {
    cuopt::linear_programming::solver_settings_t<int, double> dummy_settings;

    auto int_params    = dummy_settings.get_int_parameters();
    auto double_params = dummy_settings.get_float_parameters();
    auto bool_params   = dummy_settings.get_bool_parameters();
    auto string_params = dummy_settings.get_string_parameters();

    for (auto& param : int_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      if (arg_name_to_param_name.count(arg_name) == 0) {
        auto& arg = program.add_argument(arg_name.c_str()).default_value(param.default_value);
        if (param.is_hyperparameter) { arg.hidden(); }
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }

    for (auto& param : double_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      if (arg_name_to_param_name.count(arg_name) == 0) {
        auto& arg = program.add_argument(arg_name.c_str()).default_value(param.default_value);
        if (param.is_hyperparameter) { arg.hidden(); }
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }

    for (auto& param : bool_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      if (arg_name_to_param_name.count(arg_name) == 0) {
        auto& arg = program.add_argument(arg_name.c_str()).default_value(param.default_value);
        if (param.is_hyperparameter) { arg.hidden(); }
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }

    for (auto& param : string_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      if (arg_name_to_param_name.count(arg_name) == 0) {
        auto& arg = program.add_argument(arg_name.c_str()).default_value(param.default_value);
        if (param.is_hyperparameter) { arg.hidden(); }
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }
  }

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  static const std::map<std::string, std::string> precision_name_to_value = {
    {"default", "-1"}, {"single", "0"}, {"double", "1"}, {"mixed", "2"}};

  std::map<std::string, std::string> settings_strings;
  for (auto& [arg_name, param_name] : arg_name_to_param_name) {
    if (program.is_used(arg_name.c_str())) {
      auto val = program.get<std::string>(arg_name.c_str());
      if (param_name == CUOPT_PDLP_PRECISION) {
        auto it = precision_name_to_value.find(val);
        if (it != precision_name_to_value.end()) { val = it->second; }
      }
      settings_strings[param_name] = val;
    }
  }
  std::string file_name = program.get<std::string>("filename");

  const auto initial_solution_file = program.get<std::string>("--initial-solution");
  const auto solve_relaxation      = program.get<bool>("--relaxation");

  auto memory_backend = cuopt::linear_programming::get_memory_backend_type();
  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> memory_resources;

  if (memory_backend == cuopt::linear_programming::memory_backend_t::GPU) {
    const auto num_gpus = program.is_used("--num-gpus")
                            ? std::stoi(program.get<std::string>("--num-gpus"))
                            : program.get<int>("--num-gpus");

    for (int i = 0; i < std::min(raft::device_setter::get_device_count(), num_gpus); ++i) {
      RAFT_CUDA_TRY(cudaSetDevice(i));
      memory_resources.push_back(make_async());
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, memory_resources.back().get());
    }
    RAFT_CUDA_TRY(cudaSetDevice(0));
  }

  const auto params_file = program.get<std::string>("--params-file");

  return run_single_file(
    file_name, initial_solution_file, solve_relaxation, settings_strings, params_file);
}
