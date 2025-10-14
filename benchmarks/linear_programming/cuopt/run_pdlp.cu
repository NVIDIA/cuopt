/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <mps_parser/parser.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>

#include <argparse/argparse.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

#include <rmm/mr/device/pool_memory_resource.hpp>

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
    .choices("Stable3", "Methodical1", "Fast1");

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
}

static cuopt::linear_programming::pdlp_solver_mode_t string_to_pdlp_solver_mode(
  const std::string& mode)
{
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
  settings.presolve  = program.get<int>("--presolve");

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
    fill_pdlp_hyper_params(pdlp_hyper_params_path);
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
