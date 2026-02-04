/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <mps_parser/parser.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>

#include <argparse/argparse.hpp>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <rmm/mr/pool_memory_resource.hpp>

#include "benchmark_helper.hpp"

static void parse_arguments(argparse::ArgumentParser& program)
{
  program.add_argument("--path").help("path to mps file").required();

  program.add_argument("--time-limit")
    .help("Time limit in seconds")
    .default_value(3600.0)
    .scan<'g', double>();

  program.add_argument("--iteration-limit")
    .help("Iteration limit")
    .default_value(std::numeric_limits<int>::max())
    .scan<'i', int>();

  program.add_argument("--optimality-tolerance")
    .help("Optimality tolerance")
    .default_value(1e-4)
    .scan<'g', double>();

  // TODO replace all comments with Stable2 with Stable3
  program.add_argument("--pdlp-solver-mode")
    .help("Solver mode for PDLP. Possible values: Stable3 (default), Methodical1, Fast1")
    .default_value("Stable3")
    .choices("Stable3", "Stable2", "Stable1", "Methodical1", "Fast1");

  program.add_argument("--method")
    .help(
      "Method to solve the linear programming problem. 0: Concurrent (default), 1: PDLP, 2: "
      "DualSimplex, 3: Barrier")
    .default_value(0)
    .scan<'i', int>()
    .choices(0, 1, 2, 3);

  program.add_argument("--crossover")
    .help("Enable crossover. 0: disabled (default), 1: enabled")
    .default_value(0)
    .scan<'i', int>()
    .choices(0, 1);

  program.add_argument("--pdlp-hyper-params-path")
    .help(
      "Path to PDLP hyper-params file to configure PDLP solver. Has priority over PDLP solver "
      "modes.");

  program.add_argument("--presolve")
    .help("enable/disable presolve (default: true for MIP problems, false for LP problems)")
    .default_value(0)
    .scan<'i', int>()
    .choices(0, 1);

  program.add_argument("--solution-path").help("Path where solution file will be generated");

  program.add_argument("--pdlp-fp32")
    .help("Use FP32 (float) precision instead of FP64 (double). Only PDLP method without presolve and crossover is supported.")
    .default_value(false)
    .implicit_value(true);
}

static cuopt::linear_programming::pdlp_solver_mode_t string_to_pdlp_solver_mode(
  const std::string& mode)
{
  if (mode == "Stable1") return cuopt::linear_programming::pdlp_solver_mode_t::Stable1;
  if (mode == "Stable2")
    return cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
  else if (mode == "Methodical1")
    return cuopt::linear_programming::pdlp_solver_mode_t::Methodical1;
  else if (mode == "Fast1")
    return cuopt::linear_programming::pdlp_solver_mode_t::Fast1;
  else if (mode == "Stable3")
    return cuopt::linear_programming::pdlp_solver_mode_t::Stable3;
  return cuopt::linear_programming::pdlp_solver_mode_t::Stable3;
}

template <typename f_t>
static cuopt::linear_programming::pdlp_solver_settings_t<int, f_t> create_solver_settings(
  const argparse::ArgumentParser& program)
{
  cuopt::linear_programming::pdlp_solver_settings_t<int, f_t> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, f_t>{};

  settings.time_limit      = static_cast<f_t>(program.get<double>("--time-limit"));
  settings.iteration_limit = program.get<int>("--iteration-limit");
  settings.set_optimality_tolerance(static_cast<f_t>(program.get<double>("--optimality-tolerance")));
  settings.pdlp_solver_mode =
    string_to_pdlp_solver_mode(program.get<std::string>("--pdlp-solver-mode"));
  settings.method = static_cast<cuopt::linear_programming::method_t>(program.get<int>("--method"));
  settings.crossover = program.get<int>("--crossover");
  settings.presolve  = program.get<int>("--presolve");

  return settings;
}

template <typename f_t>
static int run_solver(const argparse::ArgumentParser& program, const raft::handle_t& handle_)
{
  // Initialize solver settings from binary arguments
  cuopt::linear_programming::pdlp_solver_settings_t<int, f_t> settings =
    create_solver_settings<f_t>(program);

  bool use_pdlp_solver_mode = true;
  if (program.is_used("--pdlp-hyper-params-path")) {
    std::string pdlp_hyper_params_path = program.get<std::string>("--pdlp-hyper-params-path");
    fill_pdlp_hyper_params(pdlp_hyper_params_path, settings.hyper_params);
    use_pdlp_solver_mode = false;
  }

  // Parse MPS file
  cuopt::mps_parser::mps_data_model_t<int, f_t> op_problem =
    cuopt::mps_parser::parse_mps<int, f_t>(program.get<std::string>("--path"));

  // Solve LP problem
  bool problem_checking = true;
  cuopt::linear_programming::optimization_problem_solution_t<int, f_t> solution =
    cuopt::linear_programming::solve_lp(
      &handle_, op_problem, settings, problem_checking, use_pdlp_solver_mode);

  // Write solution to file if requested
  if (program.is_used("--solution-path"))
    solution.write_to_file(program.get<std::string>("--solution-path"), handle_.get_stream());

  return 0;
}

int main(int argc, char* argv[])
{
  // Parse binary arguments
  argparse::ArgumentParser program("solve_LP");
  parse_arguments(program);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  // Setup up RMM memory pool
  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource.get());

  // Initialize raft handle and running stream
  const raft::handle_t handle_{};

  // Run solver with appropriate precision
  bool use_fp32 = program.get<bool>("--pdlp-fp32");
  if (use_fp32) {
    return run_solver<float>(program, handle_);
  } else {
    return run_solver<double>(program, handle_);
  }
}
