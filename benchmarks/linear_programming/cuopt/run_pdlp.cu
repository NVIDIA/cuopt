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

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
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

  program.add_argument("--presolver")
    .help("Presolver to use. Possible values: None, Papilo, PSLP, Default")
    .default_value("Default")
    .choices("None", "Papilo", "PSLP", "Default");

  program.add_argument("--solution-path").help("Path where solution file will be generated");
}

static cuopt::linear_programming::presolver_t string_to_presolver(const std::string& presolver)
{
  if (presolver == "None") return cuopt::linear_programming::presolver_t::None;
  if (presolver == "Papilo") return cuopt::linear_programming::presolver_t::Papilo;
  if (presolver == "PSLP") return cuopt::linear_programming::presolver_t::PSLP;
  if (presolver == "Default") return cuopt::linear_programming::presolver_t::Default;
  return cuopt::linear_programming::presolver_t::Default;
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

static cuopt::linear_programming::pdlp_solver_settings_t<int, double> create_solver_settings(
  const argparse::ArgumentParser& program)
{
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  settings.time_limit      = program.get<double>("--time-limit");
  settings.iteration_limit = program.get<int>("--iteration-limit");
  settings.set_optimality_tolerance(program.get<double>("--optimality-tolerance"));
  settings.pdlp_solver_mode =
    string_to_pdlp_solver_mode(program.get<std::string>("--pdlp-solver-mode"));
  settings.method = static_cast<cuopt::linear_programming::method_t>(program.get<int>("--method"));
  settings.crossover = program.get<int>("--crossover");
  settings.presolver = string_to_presolver(program.get<std::string>("--presolver"));

  return settings;
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

  // Initialize solver settings from binary arguments
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    create_solver_settings(program);

  bool use_pdlp_solver_mode = true;
  if (program.is_used("--pdlp-hyper-params-path")) {
    std::string pdlp_hyper_params_path = program.get<std::string>("--pdlp-hyper-params-path");
    fill_pdlp_hyper_params(pdlp_hyper_params_path, settings.hyper_params);
    use_pdlp_solver_mode = false;
  }

  // Setup up RMM memory pool
  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource.get());

  // Initialize raft handle and running stream
  const raft::handle_t handle_{};

  // Parse MPS file
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(program.get<std::string>("--path"));

  // Solve LP problem
  bool problem_checking = true;
  cuopt::linear_programming::optimization_problem_solution_t<int, double> solution =
    cuopt::linear_programming::solve_lp(
      &handle_, op_problem, settings, problem_checking, use_pdlp_solver_mode);

  // Write solution to file if requested
  if (program.is_used("--solution-path"))
    solution.write_to_file(program.get<std::string>("--solution-path"), handle_.get_stream());

  return 0;
}
