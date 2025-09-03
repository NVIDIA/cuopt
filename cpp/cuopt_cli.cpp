/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/logger.hpp>
#include <mps_parser/parser.hpp>

#include <raft/core/handle.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <unistd.h>
#include <argparse/argparse.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <math_optimization/solution_reader.hpp>

#include <cuopt/version_config.hpp>

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

static int get_physical_cores()
{
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo.is_open()) return 0;

  std::string line;
  int physical_id = -1, core_id = -1;
  std::set<std::pair<int, int>> cores;

  while (std::getline(cpuinfo, line)) {
    if (line.find("physical id") != std::string::npos) {
      physical_id = std::stoi(line.substr(line.find(":") + 1));
    } else if (line.find("core id") != std::string::npos) {
      core_id = std::stoi(line.substr(line.find(":") + 1));
    }

    if (physical_id != -1 && core_id != -1) {
      cores.insert({physical_id, core_id});
      physical_id = -1;
      core_id     = -1;
    }
  }

  if (cores.empty()) {
    cpuinfo.clear();
    cpuinfo.seekg(0);
    while (std::getline(cpuinfo, line)) {
      if (line.find("cpu cores") != std::string::npos) {
        return std::stoi(line.substr(line.find(":") + 1));
      }
    }
    return 1;
  }
  return cores.size();
}

static std::string get_cpu_model_from_proc()
{
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo.is_open()) return "";

  std::string line;
  while (std::getline(cpuinfo, line)) {
    std::size_t pos = line.find("model name");
    if (pos == std::string::npos) pos = line.find("Processor");
    if (pos != std::string::npos) {
      std::size_t colon = line.find(':', pos);
      if (colon != std::string::npos) return line.substr(colon + 2);  // Skip ": "
    }
  }
  return "";
}

// From https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html
// Also supported by clang
static std::string get_cpu_model_builtin()
{
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_is("amd")               ? "AMD CPU"
         : __builtin_cpu_is("intel")           ? "Intel CPU"
         : __builtin_cpu_is("atom")            ? "Intel Atom CPU"
         : __builtin_cpu_is("slm")             ? "Intel Silvermont CPU"
         : __builtin_cpu_is("core2")           ? "Intel Core 2 CPU"
         : __builtin_cpu_is("corei7")          ? "Intel Core i7 CPU"
         : __builtin_cpu_is("nehalem")         ? "Intel Core i7 Nehalem CPU"
         : __builtin_cpu_is("westmere")        ? "Intel Core i7 Westmere CPU"
         : __builtin_cpu_is("sandybridge")     ? "Intel Core i7 Sandy Bridge CPU"
         : __builtin_cpu_is("ivybridge")       ? "Intel Core i7 Ivy Bridge CPU"
         : __builtin_cpu_is("haswell")         ? "Intel Core i7 Haswell CPU"
         : __builtin_cpu_is("broadwell")       ? "Intel Core i7 Broadwell CPU"
         : __builtin_cpu_is("skylake")         ? "Intel Core i7 Skylake CPU"
         : __builtin_cpu_is("skylake-avx512")  ? "Intel Core i7 Skylake AVX512 CPU"
         : __builtin_cpu_is("cannonlake")      ? "Intel Core i7 Cannon Lake CPU"
         : __builtin_cpu_is("icelake-client")  ? "Intel Core i7 Ice Lake Client CPU"
         : __builtin_cpu_is("icelake-server")  ? "Intel Core i7 Ice Lake Server CPU"
         : __builtin_cpu_is("cascadelake")     ? "Intel Core i7 Cascadelake CPU"
         : __builtin_cpu_is("tigerlake")       ? "Intel Core i7 Tigerlake CPU"
         : __builtin_cpu_is("cooperlake")      ? "Intel Core i7 Cooperlake CPU"
         : __builtin_cpu_is("sapphirerapids")  ? "Intel Core i7 sapphirerapids CPU"
         : __builtin_cpu_is("alderlake")       ? "Intel Core i7 Alderlake CPU"
         : __builtin_cpu_is("rocketlake")      ? "Intel Core i7 Rocketlake CPU"
         : __builtin_cpu_is("graniterapids")   ? "Intel Core i7 graniterapids CPU"
         : __builtin_cpu_is("graniterapids-d") ? "Intel Core i7 graniterapids D CPU"
         : __builtin_cpu_is("bonnell")         ? "Intel Atom Bonnell CPU"
         : __builtin_cpu_is("silvermont")      ? "Intel Atom Silvermont CPU"
         : __builtin_cpu_is("goldmont")        ? "Intel Atom Goldmont CPU"
         : __builtin_cpu_is("goldmont-plus")   ? "Intel Atom Goldmont Plus CPU"
         : __builtin_cpu_is("tremont")         ? "Intel Atom Tremont CPU"
         : __builtin_cpu_is("sierraforest")    ? "Intel Atom Sierra Forest CPU"
         : __builtin_cpu_is("grandridge")      ? "Intel Atom Grand Ridge CPU"
         : __builtin_cpu_is("amdfam10h")       ? "AMD Family 10h CPU"
         : __builtin_cpu_is("barcelona")       ? "AMD Family 10h Barcelona CPU"
         : __builtin_cpu_is("shanghai")        ? "AMD Family 10h Shanghai CPU"
         : __builtin_cpu_is("istanbul")        ? "AMD Family 10h Istanbul CPU"
         : __builtin_cpu_is("btver1")          ? "AMD Family 14h CPU"
         : __builtin_cpu_is("amdfam15h")       ? "AMD Family 15h CPU"
         : __builtin_cpu_is("bdver1")          ? "AMD Family 15h Bulldozer version 1"
         : __builtin_cpu_is("bdver2")          ? "AMD Family 15h Bulldozer version 2"
         : __builtin_cpu_is("bdver3")          ? "AMD Family 15h Bulldozer version 3"
         : __builtin_cpu_is("bdver4")          ? "AMD Family 15h Bulldozer version 4"
         : __builtin_cpu_is("btver2")          ? "AMD Family 16h CPU"
         : __builtin_cpu_is("amdfam17h")       ? "AMD Family 17h CPU"
         : __builtin_cpu_is("znver1")          ? "AMD Family 17h Zen version 1"
         : __builtin_cpu_is("znver2")          ? "AMD Family 17h Zen version 2"
         : __builtin_cpu_is("amdfam19h")       ? "AMD Family 19h CPU"
                                               : "Unknown";
#else
  return "Unknown";
#endif
}

static std::string get_cpu_model()
{
  if (auto model_from_proc = get_cpu_model_from_proc(); !model_from_proc.empty()) {
    return model_from_proc;
  } else if (auto model_from_builtin = get_cpu_model_builtin(); !model_from_builtin.empty()) {
    return model_from_builtin;
  }
  return "Unknown";
}

static double get_available_memory_gb()
{
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) return 0.0;

  std::string line;
  long kb = 0;
  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") == 0 || line.find("MemFree:") == 0) {
      std::size_t pos = line.find_first_of("0123456789");
      if (pos != std::string::npos) {
        kb = std::stol(line.substr(pos));
        break;
      }
    }
  }

  return kb / (1024.0 * 1024.0);  // Convert KB to GB
}

/**
 * @brief Make an async memory resource for RMM
 * @return std::shared_ptr<rmm::mr::cuda_async_memory_resource>
 */
inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

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
    CUOPT_LOG_ERROR("Error: %s", e.what());
    return -1;
  }

  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  cudaUUID_t uuid   = device_prop.uuid;
  char uuid_str[37] = {0};
  snprintf(uuid_str,
           sizeof(uuid_str),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           uuid.bytes[0],
           uuid.bytes[1],
           uuid.bytes[2],
           uuid.bytes[3],
           uuid.bytes[4],
           uuid.bytes[5],
           uuid.bytes[6],
           uuid.bytes[7],
           uuid.bytes[8],
           uuid.bytes[9],
           uuid.bytes[10],
           uuid.bytes[11],
           uuid.bytes[12],
           uuid.bytes[13],
           uuid.bytes[14],
           uuid.bytes[15]);
  int version = 0;
  cudaRuntimeGetVersion(&version);
  int major = version / 1000;
  int minor = (version % 1000) / 10;
  CUOPT_LOG_INFO("cuOpt version: %d.%d.%d, git hash: %s, host arch: %s, device archs: %s",
                 CUOPT_VERSION_MAJOR,
                 CUOPT_VERSION_MINOR,
                 CUOPT_VERSION_PATCH,
                 CUOPT_GIT_COMMIT_HASH,
                 CUOPT_CPU_ARCHITECTURE,
                 CUOPT_CUDA_ARCHITECTURES);
  CUOPT_LOG_INFO("CPU: %s, threads (physical/logical): %d/%d, RAM: %.2f GiB",
                 get_cpu_model().c_str(),
                 get_physical_cores(),
                 std::thread::hardware_concurrency(),
                 get_available_memory_gb());
  CUOPT_LOG_INFO("CUDA %d.%d, device: %s (ID %d), VRAM: %.2f GiB",
                 major,
                 minor,
                 device_prop.name,
                 device_id,
                 (double)device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  CUOPT_LOG_INFO("CUDA device UUID: %s\n", uuid_str);

  std::string base_filename = file_path.substr(file_path.find_last_of("/\\") + 1);

  constexpr bool input_mps_strict = false;
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model;
  bool parsing_failed = false;
  {
    CUOPT_LOG_INFO("Running file %s", base_filename.c_str());
    try {
      mps_data_model = cuopt::mps_parser::parse_mps<int, double>(file_path, input_mps_strict);
    } catch (const std::logic_error& e) {
      CUOPT_LOG_ERROR("MPS parser execption: %s", e.what());
      parsing_failed = true;
    }
  }
  if (parsing_failed) {
    CUOPT_LOG_ERROR("Parsing MPS failed. Exiting!");
    return -1;
  }

  auto op_problem =
    cuopt::linear_programming::mps_data_model_to_optimization_problem(&handle_, mps_data_model);

  const bool is_mip =
    (op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::MIP ||
     op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::IP);

  auto initial_solution =
    initial_solution_file.empty()
      ? std::vector<double>()
      : cuopt::linear_programming::solution_reader_t::get_variable_values_from_sol_file(
          initial_solution_file, mps_data_model.get_variable_names());

  try {
    if (is_mip && !solve_relaxation) {
      auto& mip_settings = settings.get_mip_settings();
      if (initial_solution.size() > 0) {
        mip_settings.add_initial_solution(initial_solution.data(), initial_solution.size());
      }
      auto solution = cuopt::linear_programming::solve_mip(op_problem, mip_settings);
    } else {
      auto& lp_settings = settings.get_pdlp_settings();
      if (initial_solution.size() > 0) {
        lp_settings.set_initial_primal_solution(initial_solution.data(), initial_solution.size());
      }
      auto solution = cuopt::linear_programming::solve_lp(op_problem, lp_settings);
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
 * @brief Main function for the cuOpt CLI
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return 0 on success, 1 on failure
 */
int main(int argc, char* argv[])
{
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

  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  return run_single_file(file_name, initial_solution_file, solve_relaxation, settings_strings);
}
