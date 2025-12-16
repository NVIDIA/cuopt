/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
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

/**
 * @file cuopt_cli.cpp
 * @brief Command line interface for solving Linear Programming (LP) and Mixed Integer Programming
 * (MIP) problems using cuOpt
 *
 * This CLI provides a simple interface to solve LP/MIP problems using cuOpt. It accepts MPS format
 * input files and various solver parameters.
 *
 * Usage:
 * ```
 * cuopt_cli <mps_file_path> [OPTIONS]
 * cuopt_cli [OPTIONS] <mps_file_path>
 * ```
 *
 * Required arguments:
 * - <mps_file_path>: Path to the MPS format input file containing the optimization problem
 *
 * Optional arguments:
 * - --initial-solution: Path to initial solution file in SOL format
 * - Various solver parameters that can be passed as command line arguments
 *   (e.g. --max-iterations, --tolerance, etc.)
 *
 * Example:
 * ```
 * cuopt_cli problem.mps --max-iterations 1000
 * ```
 *
 * The solver will read the MPS file, solve the optimization problem according to the specified
 * parameters, and write the solution to a .sol file in the output directory.
 */

/**
 * @brief Make an async memory resource for RMM
 * @return std::shared_ptr<rmm::mr::cuda_async_memory_resource>
 */
inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

/**
 * @brief Handle logger when error happens before logger is initialized
 * @param settings Solver settings
 * @return cuopt::init_logger_t
 */
inline cuopt::init_logger_t dummy_logger(
  const cuopt::linear_programming::solver_settings_t<int, double>& settings)
{
  return cuopt::init_logger_t(settings.get_parameter<std::string>(CUOPT_LOG_FILE),
                              settings.get_parameter<bool>(CUOPT_LOG_TO_CONSOLE));
}

/**
 * @brief Run a single file
 * @param file_path Path to the MPS format input file containing the optimization problem
 * @param initial_solution_file Path to initial solution file in SOL format
 * @param settings_strings Map of solver parameters
 */
int run_single_file(const std::string& file_path,
                    const std::string& initial_solution_file,
                    bool solve_relaxation,
                    const std::map<std::string, std::string>& settings_strings)
{
  const raft::handle_t handle_{};
  cuopt::linear_programming::solver_settings_t<int, double> settings;

  try {
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

  auto op_problem =
    cuopt::linear_programming::mps_data_model_to_optimization_problem(&handle_, mps_data_model);

  const bool is_mip =
    (op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::MIP ||
     op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::IP) &&
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
      auto solution      = cuopt::linear_programming::solve_mip(op_problem, mip_settings);
    } else {
      auto& lp_settings = settings.get_pdlp_settings();
      auto solution     = cuopt::linear_programming::solve_lp(op_problem, lp_settings);
    }
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("Error: %s", e.what());
    return -1;
  }

  return 0;
}

/**
 * @brief Convert a parameter name to an argument name
 * @param input Parameter name
 * @return Argument name
 */
std::string param_name_to_arg_name(const std::string& input)
{
  std::string result = "--";
  result += input;

  // Replace underscores with hyphens
  std::replace(result.begin(), result.end(), '_', '-');

  return result;
}

/**
 * @brief Set the CUDA module loading environment variable
 * If the method is 0, set the CUDA module loading environment variable to EAGER
 * This needs to be done before the first call to the CUDA API. In this file before dummy settings
 * default constructor is called.
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return 0 on success, 1 on failure
 */
int set_cuda_module_loading(int argc, char* argv[])
{
  // Parse method_int from argv
  int method_int = 0;  // Default value
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
    // Also support --method=1 style
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

/**
 * @brief Main function for the cuOpt CLI
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return 0 on success, 1 on failure
 */
int main(int argc, char* argv[])
{
  if (set_cuda_module_loading(argc, argv) != 0) { return 1; }

  // Get the version string from the version_config.hpp file
  const std::string version_string = std::string("cuOpt ") + std::to_string(CUOPT_VERSION_MAJOR) +
                                     "." + std::to_string(CUOPT_VERSION_MINOR) + "." +
                                     std::to_string(CUOPT_VERSION_PATCH);

  // Create the argument parser
  argparse::ArgumentParser program("cuopt_cli", version_string);

  // Define all arguments with appropriate defaults and help messages
  program.add_argument("filename").help("input mps file").nargs(1).required();

  // FIXME: use a standard format for initial solution file
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

  std::map<std::string, std::string> arg_name_to_param_name;
  {
    // Add all solver settings as arguments
    cuopt::linear_programming::solver_settings_t<int, double> dummy_settings;

    auto int_params    = dummy_settings.get_int_parameters();
    auto double_params = dummy_settings.get_float_parameters();
    auto bool_params   = dummy_settings.get_bool_parameters();
    auto string_params = dummy_settings.get_string_parameters();

    for (auto& param : int_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      // handle duplicate parameters appearing in MIP and LP settings
      if (arg_name_to_param_name.count(arg_name) == 0) {
        program.add_argument(arg_name.c_str()).default_value(param.default_value);
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }

    for (auto& param : double_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      // handle duplicate parameters appearing in MIP and LP settings
      if (arg_name_to_param_name.count(arg_name) == 0) {
        program.add_argument(arg_name.c_str()).default_value(param.default_value);
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }

    for (auto& param : bool_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      if (arg_name_to_param_name.count(arg_name) == 0) {
        program.add_argument(arg_name.c_str()).default_value(param.default_value);
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }

    for (auto& param : string_params) {
      std::string arg_name = param_name_to_arg_name(param.param_name);
      // handle duplicate parameters appearing in MIP and LP settings
      if (arg_name_to_param_name.count(arg_name) == 0) {
        program.add_argument(arg_name.c_str()).default_value(param.default_value);
        arg_name_to_param_name[arg_name] = param.param_name;
      }
    }  // done with solver settings
  }

  // Parse arguments
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  // Read everything as a string
  std::map<std::string, std::string> settings_strings;
  for (auto& [arg_name, param_name] : arg_name_to_param_name) {
    if (program.is_used(arg_name.c_str())) {
      settings_strings[param_name] = program.get<std::string>(arg_name.c_str());
    }
  }
  // Get the values
  std::string file_name = program.get<std::string>("filename");

  const auto initial_solution_file = program.get<std::string>("--initial-solution");
  const auto solve_relaxation      = program.get<bool>("--relaxation");

  // All arguments are parsed as string, default values are parsed as int if unused.
  const auto num_gpus = program.is_used("--num-gpus")
                          ? std::stoi(program.get<std::string>("--num-gpus"))
                          : program.get<int>("--num-gpus");

  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> memory_resources;

  for (int i = 0; i < std::min(raft::device_setter::get_device_count(), num_gpus); ++i) {
    cudaSetDevice(i);
    memory_resources.push_back(make_async());
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, memory_resources.back().get());
  }
  cudaSetDevice(0);

  return run_single_file(file_name, initial_solution_file, solve_relaxation, settings_strings);
}
