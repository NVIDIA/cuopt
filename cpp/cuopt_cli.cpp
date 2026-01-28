/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/data_model_view.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/logger.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <unistd.h>
#include <algorithm>
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
 * @brief Create a data_model_view_t from mps_data_model_t
 *
 * This creates a non-owning view with spans pointing to the CPU data in the mps_data_model.
 * Used for remote solve where data stays in CPU memory.
 *
 * @param mps_data_model The owning mps_data_model_t
 * @return data_model_view_t with spans pointing to the mps_data_model's vectors
 */
template <typename i_t, typename f_t>
cuopt::linear_programming::data_model_view_t<i_t, f_t> create_view_from_mps_data_model(
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model)
{
  cuopt::linear_programming::data_model_view_t<i_t, f_t> view;

  view.set_maximize(mps_data_model.get_sense());

  if (!mps_data_model.get_constraint_matrix_values().empty()) {
    view.set_csr_constraint_matrix(mps_data_model.get_constraint_matrix_values().data(),
                                   mps_data_model.get_constraint_matrix_values().size(),
                                   mps_data_model.get_constraint_matrix_indices().data(),
                                   mps_data_model.get_constraint_matrix_indices().size(),
                                   mps_data_model.get_constraint_matrix_offsets().data(),
                                   mps_data_model.get_constraint_matrix_offsets().size());
  }

  if (!mps_data_model.get_constraint_bounds().empty()) {
    view.set_constraint_bounds(mps_data_model.get_constraint_bounds().data(),
                               mps_data_model.get_constraint_bounds().size());
  }

  if (!mps_data_model.get_objective_coefficients().empty()) {
    view.set_objective_coefficients(mps_data_model.get_objective_coefficients().data(),
                                    mps_data_model.get_objective_coefficients().size());
  }

  view.set_objective_scaling_factor(mps_data_model.get_objective_scaling_factor());
  view.set_objective_offset(mps_data_model.get_objective_offset());

  if (!mps_data_model.get_variable_lower_bounds().empty()) {
    view.set_variable_lower_bounds(mps_data_model.get_variable_lower_bounds().data(),
                                   mps_data_model.get_variable_lower_bounds().size());
  }

  if (!mps_data_model.get_variable_upper_bounds().empty()) {
    view.set_variable_upper_bounds(mps_data_model.get_variable_upper_bounds().data(),
                                   mps_data_model.get_variable_upper_bounds().size());
  }

  if (!mps_data_model.get_variable_types().empty()) {
    view.set_variable_types(mps_data_model.get_variable_types().data(),
                            mps_data_model.get_variable_types().size());
  }

  if (!mps_data_model.get_row_types().empty()) {
    view.set_row_types(mps_data_model.get_row_types().data(),
                       mps_data_model.get_row_types().size());
  }

  if (!mps_data_model.get_constraint_lower_bounds().empty()) {
    view.set_constraint_lower_bounds(mps_data_model.get_constraint_lower_bounds().data(),
                                     mps_data_model.get_constraint_lower_bounds().size());
  }

  if (!mps_data_model.get_constraint_upper_bounds().empty()) {
    view.set_constraint_upper_bounds(mps_data_model.get_constraint_upper_bounds().data(),
                                     mps_data_model.get_constraint_upper_bounds().size());
  }

  view.set_objective_name(mps_data_model.get_objective_name());
  view.set_problem_name(mps_data_model.get_problem_name());

  if (!mps_data_model.get_variable_names().empty()) {
    view.set_variable_names(mps_data_model.get_variable_names());
  }

  if (!mps_data_model.get_row_names().empty()) {
    view.set_row_names(mps_data_model.get_row_names());
  }

  if (!mps_data_model.get_initial_primal_solution().empty()) {
    view.set_initial_primal_solution(mps_data_model.get_initial_primal_solution().data(),
                                     mps_data_model.get_initial_primal_solution().size());
  }

  if (!mps_data_model.get_initial_dual_solution().empty()) {
    view.set_initial_dual_solution(mps_data_model.get_initial_dual_solution().data(),
                                   mps_data_model.get_initial_dual_solution().size());
  }

  if (mps_data_model.has_quadratic_objective()) {
    view.set_quadratic_objective_matrix(mps_data_model.get_quadratic_objective_values().data(),
                                        mps_data_model.get_quadratic_objective_values().size(),
                                        mps_data_model.get_quadratic_objective_indices().data(),
                                        mps_data_model.get_quadratic_objective_indices().size(),
                                        mps_data_model.get_quadratic_objective_offsets().data(),
                                        mps_data_model.get_quadratic_objective_offsets().size());
  }

  return view;
}

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
                    const std::map<std::string, std::string>& settings_strings,
                    bool is_remote_solve)
{
  // Only create raft handle for local solve - it triggers CUDA initialization
  std::unique_ptr<raft::handle_t> handle_ptr;
  if (!is_remote_solve) { handle_ptr = std::make_unique<raft::handle_t>(); }
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

  // Determine if this is a MIP problem by checking variable types
  bool has_integers = false;
  for (const auto& vt : mps_data_model.get_variable_types()) {
    if (vt == 'I' || vt == 'B') {
      has_integers = true;
      break;
    }
  }
  const bool is_mip = has_integers && !solve_relaxation;

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

  // Create a non-owning view from the mps_data_model
  // solve_lp/solve_mip will handle remote vs local solve based on env vars
  auto view = create_view_from_mps_data_model(mps_data_model);

  try {
    // Pass handle_ptr.get() - can be nullptr for remote solve
    if (is_mip) {
      auto& mip_settings = settings.get_mip_settings();
      auto solution = cuopt::linear_programming::solve_mip(handle_ptr.get(), view, mip_settings);
      if (solution.get_error_status().get_error_type() != cuopt::error_type_t::Success) {
        auto log = dummy_logger(settings);
        CUOPT_LOG_ERROR("MIP solve failed: %s", solution.get_error_status().what());
        return -1;
      }
    } else {
      auto& lp_settings = settings.get_pdlp_settings();
      auto solution     = cuopt::linear_programming::solve_lp(handle_ptr.get(), view, lp_settings);
      if (solution.get_error_status().get_error_type() != cuopt::error_type_t::Success) {
        auto log = dummy_logger(settings);
        CUOPT_LOG_ERROR("LP solve failed: %s", solution.get_error_status().what());
        return -1;
      }
      // Note: Solution output is now handled by solve_lp/solve_lp_remote via CUOPT_LOG_INFO
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

  // Check for remote solve BEFORE any CUDA initialization
  const bool is_remote_solve = cuopt::linear_programming::is_remote_solve_enabled();

  std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> memory_resources;

  if (!is_remote_solve) {
    // Only initialize CUDA resources for local solve
    // All arguments are parsed as string, default values are parsed as int if unused.
    const auto num_gpus = program.is_used("--num-gpus")
                            ? std::stoi(program.get<std::string>("--num-gpus"))
                            : program.get<int>("--num-gpus");

    for (int i = 0; i < std::min(raft::device_setter::get_device_count(), num_gpus); ++i) {
      cudaSetDevice(i);
      memory_resources.push_back(make_async());
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, memory_resources.back().get());
    }
    cudaSetDevice(0);
  }

  const auto initial_solution_file = program.get<std::string>("--initial-solution");
  const auto solve_relaxation      = program.get<bool>("--relaxation");
  return run_single_file(
    file_name, initial_solution_file, solve_relaxation, settings_strings, is_remote_solve);
}
