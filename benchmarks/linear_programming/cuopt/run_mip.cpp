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
#include "initial_solution_reader.hpp"
#include "mip_test_instances.hpp"

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/logger.hpp>
#include <cuopt/version_config.hpp>
#include <mps_parser/parser.hpp>

#include <omp.h>
#include <raft/core/handle.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <rmm/mr/device/owning_wrapper.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "initial_problem_check.hpp"

void merge_result_files(const std::string& out_dir,
                        const std::string& final_result_file,
                        int n_gpus,
                        int batch_id)
{
  std::ofstream final_file(final_result_file, std::ios_base::app);
  if (!final_file.is_open()) {
    std::cerr << "Error opening final result file" << std::endl;
    return;
  }
  int batch_offset = n_gpus * batch_id;
  for (int i = 0; i < n_gpus; ++i) {
    int res_id            = i + batch_offset;
    std::string temp_file = out_dir + "/result_" + std::to_string(res_id) + ".txt";
    std::ifstream infile(temp_file);
    if (infile.is_open()) {
      final_file << infile.rdbuf();
      infile.close();
      std::remove(temp_file.c_str());  // Delete the temporary file
    } else {
      printf("could not open result file! %s\n", temp_file.c_str());
    }
  }

  final_file.close();
}

void write_to_output_file(const std::string& out_dir,
                          const std::string& base_filename,
                          int gpu_id,
                          int batch_id,
                          const std::string& data)
{
  int output_id        = batch_id * 8 + gpu_id;
  std::string filename = out_dir + "/result_" + std::to_string(output_id) + ".txt";
  std::ofstream outfile(filename, std::ios_base::app);
  if (outfile.is_open()) {
    outfile << data;
    outfile.close();
  } else {
    std::cerr << "Error opening file " << filename << std::endl;
  }
}

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

void read_single_solution_from_path(const std::string& path,
                                    const std::vector<std::string>& var_names,
                                    std::vector<std::vector<double>>& solutions)
{
  solution_reader_t reader;
  bool success = reader.read_from_sol(path);
  if (!success) {
    CUOPT_LOG_ERROR("Initial solution reading error!");
  } else {
    CUOPT_LOG_INFO(
      "Success reading file %s Number of var vals %lu", path.c_str(), reader.data_map.size());
  }
  std::vector<double> assignment;
  for (auto name : var_names) {
    auto it = reader.data_map.find(name);
    double val;
    if ((it != reader.data_map.end())) {
      val = it->second;
    } else {
      CUOPT_LOG_TRACE("Variable %s has no input value ", name.c_str());
      val = 0.;
    }
    assignment.push_back(val);
  }
  if (assignment.size() > 0) {
    CUOPT_LOG_INFO("Adding a solution with size %lu ", assignment.size());
    solutions.push_back(assignment);
  }
}

// reads a solution from an input file. The input file needs to be csv formatted
// var_name,val
std::vector<std::vector<double>> read_solution_from_dir(const std::string file_path,
                                                        const std::string& mps_file_name,
                                                        const std::vector<std::string>& var_names)
{
  std::vector<std::vector<double>> initial_solutions;
  std::string mps_file_name_no_ext = mps_file_name.substr(0, mps_file_name.find_last_of("."));
  // check if a directory with the given mps file exists
  std::string initial_solution_dir = file_path + "/" + mps_file_name_no_ext;
  if (std::filesystem::exists(initial_solution_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(initial_solution_dir)) {
      read_single_solution_from_path(entry.path(), var_names, initial_solutions);
    }
  } else {
    read_single_solution_from_path(file_path, var_names, initial_solutions);
  }
  return initial_solutions;
}

int run_single_file(std::string file_path,
                    int device,
                    int batch_id,
                    std::string out_dir,
                    std::optional<std::string> initial_solution_dir,
                    bool heuristics_only,
                    int num_cpu_threads,
                    bool write_log_file,
                    bool log_to_console,
                    double time_limit)
{
  const raft::handle_t handle_{};
  cuopt::linear_programming::mip_solver_settings_t<int, double> settings;
  std::string base_filename = file_path.substr(file_path.find_last_of("/\\") + 1);
  // if output directory is given, set the log file
  if (write_log_file) {
    if (out_dir != "") {
      std::string log_file =
        out_dir + "/" + base_filename.substr(0, base_filename.find(".mps")) + ".log";
      settings.log_file = log_file;
    } else {
      std::string log_file = base_filename.substr(0, base_filename.find(".mps")) + ".log";
      settings.log_file    = log_file;
    }
  }

  constexpr bool input_mps_strict = false;
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model;
  bool parsing_failed = false;
  {
    CUOPT_LOG_INFO("running file %s on gpu : %d", base_filename.c_str(), device);
    try {
      mps_data_model = cuopt::mps_parser::parse_mps<int, double>(file_path, input_mps_strict);
    } catch (const std::logic_error& e) {
      CUOPT_LOG_ERROR("MPS parser execption: %s", e.what());
      parsing_failed = true;
    }
  }
  if (parsing_failed) {
    CUOPT_LOG_ERROR("Parsing MPS failed exiting!");
    return -1;
  }
  if (initial_solution_dir.has_value()) {
    auto initial_solutions = read_solution_from_dir(
      initial_solution_dir.value(), base_filename, mps_data_model.get_variable_names());
    for (auto& initial_solution : initial_solutions) {
      bool feasible_variables =
        test_constraint_and_variable_sanity(mps_data_model,
                                            initial_solution,
                                            settings.tolerances.absolute_tolerance,
                                            settings.tolerances.relative_tolerance,
                                            settings.tolerances.integrality_tolerance);
      if (feasible_variables) {
        settings.add_initial_solution(
          initial_solution.data(), initial_solution.size(), handle_.get_stream());
      }
    }
  }

  settings.time_limit                    = time_limit;
  settings.heuristics_only               = heuristics_only;
  settings.num_cpu_threads               = num_cpu_threads;
  settings.log_to_console                = log_to_console;
  settings.tolerances.relative_tolerance = 1e-12;
  settings.tolerances.absolute_tolerance = 1e-6;
  cuopt::linear_programming::benchmark_info_t benchmark_info;
  settings.benchmark_info_ptr = &benchmark_info;
  auto start_run_solver       = std::chrono::high_resolution_clock::now();
  auto solution = cuopt::linear_programming::solve_mip(&handle_, mps_data_model, settings);
  CUOPT_LOG_INFO(
    "first obj: %f last improvement of best feasible: %f last improvement after recombination: %f",
    benchmark_info.objective_of_initial_population,
    benchmark_info.last_improvement_of_best_feasible,
    benchmark_info.last_improvement_after_recombination);
  // solution.write_to_sol_file(base_filename + ".sol", handle_.get_stream());
  std::chrono::milliseconds duration;
  auto end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_run_solver);
  CUOPT_LOG_INFO("run_solver %d", duration.count());
  handle_.sync_stream();
  int sol_found  = int(solution.get_termination_status() ==
                        cuopt::linear_programming::mip_termination_status_t::FeasibleFound ||
                      solution.get_termination_status() ==
                        cuopt::linear_programming::mip_termination_status_t::Optimal);
  double obj_val = sol_found ? solution.get_objective_value() : std::numeric_limits<double>::max();
  if (sol_found) {
    CUOPT_LOG_INFO("%s: solution found, obj: %f", base_filename.c_str(), obj_val);
  } else {
    CUOPT_LOG_INFO("%s: no solution found", base_filename.c_str());
  }
  std::stringstream ss;
  int decimal_places = 2;
  ss << std::fixed << std::setprecision(decimal_places) << base_filename << "," << sol_found << ","
     << obj_val << "," << benchmark_info.objective_of_initial_population << ","
     << benchmark_info.last_improvement_of_best_feasible << ","
     << benchmark_info.last_improvement_after_recombination << "\n";
  write_to_output_file(out_dir, base_filename, device, batch_id, ss.str());
  CUOPT_LOG_INFO("Results written to the file %s", base_filename.c_str());
  return sol_found;
}

void run_single_file_mp(std::string file_path,
                        int device,
                        int batch_id,
                        std::string out_dir,
                        std::optional<std::string> input_file_dir,
                        bool heuristics_only,
                        int num_cpu_threads,
                        bool write_log_file,
                        bool log_to_console,
                        double time_limit)
{
  std::cout << "running file " << file_path << " on gpu : " << device << std::endl;
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  int sol_found = run_single_file(file_path,
                                  device,
                                  batch_id,
                                  out_dir,
                                  input_file_dir,
                                  heuristics_only,
                                  num_cpu_threads,
                                  write_log_file,
                                  log_to_console,
                                  time_limit);
  // this is a bad design to communicate the result but better than adding complexity of IPC or
  // pipes
  exit(sol_found);
}

void return_gpu_to_the_queue(std::unordered_map<pid_t, int>& pid_gpu_map,
                             std::unordered_map<pid_t, std::string>& pid_file_map,
                             std::queue<int>& gpu_queue)
{
  int status;
  pid_t pid = wait(&status);
  if (!WIFEXITED(status)) {
    auto file_name    = pid_file_map[pid];
    int signal_number = WTERMSIG(status);
    printf("error occured on %s with signal %d\n", file_name.c_str(), signal_number);
  }
  int gpu        = pid_gpu_map[pid];
  auto file_name = pid_file_map[pid];
  gpu_queue.push(gpu);
  pid_gpu_map.erase(pid);
  pid_file_map.erase(pid);
}

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

int main(int argc, char* argv[])
{
  argparse::ArgumentParser program("solve_MIP");

  // Define all arguments with appropriate defaults and help messages
  program.add_argument("--path").help("input path").required();

  program.add_argument("--run-dir")
    .help("run directory flag with optional time limit (t[time] format)")
    .default_value(std::string("f"));

  program.add_argument("--run-selected")
    .help("run selected flag (t/f)")
    .default_value(std::string("f"));

  program.add_argument("--n-gpus").help("number of GPUs").scan<'i', int>().default_value(1);

  program.add_argument("--out-dir").help("output directory for results");

  program.add_argument("--batch-num").help("batch number").scan<'i', int>().default_value(-1);

  program.add_argument("--n-batches")
    .help("total number of batches")
    .scan<'i', int>()
    .default_value(-1);

  program.add_argument("--initial-solution-path").help("path to the initial solution csv file");

  program.add_argument("--heuristics-only")
    .help("run heuristics only (t/f)")
    .default_value(std::string("f"));

  program.add_argument("--num-cpu-threads")
    .help("number of CPU threads")
    .scan<'i', int>()
    .default_value(-1);

  program.add_argument("--write-log-file")
    .help("write log file (t/f)")
    .default_value(std::string("f"));

  program.add_argument("--log-to-console")
    .help("log to console (t/f)")
    .default_value(std::string("t"));

  program.add_argument("--time-limit")
    .help("time limit")
    .scan<'g', double>()
    .default_value(std::numeric_limits<double>::max());

  // Parse arguments
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  // Get the values
  std::string path        = program.get<std::string>("--path");
  std::string run_dir_arg = program.get<std::string>("--run-dir");
  bool run_dir            = run_dir_arg[0] == 't';
  double time_limit       = program.get<double>("--time-limit");

  bool run_selected = program.get<std::string>("--run-selected")[0] == 't';
  int n_gpus        = program.get<int>("--n-gpus");

  std::string out_dir;
  std::string result_file;
  int batch_num = -1;

  bool heuristics_only = program.get<std::string>("--heuristics-only")[0] == 't';
  int num_cpu_threads  = program.get<int>("--num-cpu-threads");
  bool write_log_file  = program.get<std::string>("--write-log-file")[0] == 't';
  bool log_to_console  = program.get<std::string>("--log-to-console")[0] == 't';

  if (program.is_used("--out-dir")) {
    out_dir     = program.get<std::string>("--out-dir");
    result_file = out_dir + "/final_result.csv";

    batch_num = program.get<int>("--batch-num");
    if (batch_num != -1) {
      result_file = out_dir + "/final_result_" + std::to_string(batch_num) + ".csv";
    }
  }

  int n_batches = program.get<int>("--n-batches");
  std::optional<std::string> initial_solution_file;
  if (program.is_used("--initial-solution-path")) {
    initial_solution_file = program.get<std::string>("--initial-solution-path");
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
  printf("cuOpt version: %d.%d.%d, git hash: %s, host arch: %s, device archs: %s\n",
         CUOPT_VERSION_MAJOR,
         CUOPT_VERSION_MINOR,
         CUOPT_VERSION_PATCH,
         CUOPT_GIT_COMMIT_HASH,
         CUOPT_CPU_ARCHITECTURE,
         CUOPT_CUDA_ARCHITECTURES);
  printf("CPU: %s, threads (physical/logical): %d/%d, RAM: %.2f GiB\n",
         get_cpu_model().c_str(),
         get_physical_cores(),
         std::thread::hardware_concurrency(),
         get_available_memory_gb());
  printf("CUDA %d.%d, device: %s (ID %d), VRAM: %.2f GiB\n",
         major,
         minor,
         device_prop.name,
         device_id,
         (double)device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("CUDA device UUID: %s\n\n", uuid_str);

  if (run_dir) {
    std::queue<std::string> task_queue;
    std::queue<int> gpu_queue;
    std::unordered_map<pid_t, int> pid_gpu_map;
    std::unordered_map<pid_t, std::string> pid_file_map;
    // Populate the task queue
    for (int i = 0; i < n_gpus; ++i) {
      gpu_queue.push(i);
    }
    int tests_ran = 0;
    std::vector<std::string> paths;
    if (run_selected) {
      for (const auto& instance : instances) {
        paths.push_back(path + "/" + instance);
      }
    } else {
      for (const auto& entry : std::filesystem::directory_iterator(path)) {
        paths.push_back(entry.path());
      }
    }
    // if batch_num is given, trim the paths to only concerned batch
    if (batch_num != -1) {
      if (n_batches <= 0) {
        std::cout << "Error on number of batches!\n";
        exit(1);
      }
      int batch_size  = std::ceil(static_cast<double>(paths.size()) / n_batches);
      int start_index = batch_num * batch_size;
      int end_index   = std::min((batch_num + 1) * batch_size, int(paths.size()));
      paths = std::vector<std::string>(paths.begin() + start_index, paths.begin() + end_index);
    } else {
      batch_num = 0;
    }
    std::cout << "Running from directory n_files: " << paths.size() << std::endl;

    bool static_dispatch = false;
    if (static_dispatch) {
      for (size_t i = 0; i < paths.size(); ++i) {
        // TODO implement
      }
    } else {
      for (size_t i = 0; i < paths.size(); ++i) {
        task_queue.push(paths[i]);
      }
      while (!task_queue.empty()) {
        if (!gpu_queue.empty()) {
          int gpu_id     = gpu_queue.front();
          auto file_name = task_queue.front();
          gpu_queue.pop();
          task_queue.pop();
          auto sys_pid = fork();
          // if parent
          if (sys_pid > 0) {
            pid_gpu_map.insert({sys_pid, gpu_id});
            pid_file_map.insert({sys_pid, file_name});
          }
          if (sys_pid == 0) {
            RAFT_CUDA_TRY(cudaSetDevice(gpu_id));
            run_single_file_mp(file_name,
                               gpu_id,
                               batch_num,
                               out_dir,
                               initial_solution_file,
                               heuristics_only,
                               num_cpu_threads,
                               write_log_file,
                               log_to_console,
                               time_limit);
          } else if (sys_pid < 0) {
            std::cerr << "Fork failed!" << std::endl;
            exit(1);
          }
        } else {
          return_gpu_to_the_queue(pid_gpu_map, pid_file_map, gpu_queue);
        }
        sleep(1);
      }
      int remaining = paths.size() - tests_ran;
      // wait for all processes to finish
      for (int i = 0; i < remaining; ++i) {
        return_gpu_to_the_queue(pid_gpu_map, pid_file_map, gpu_queue);
      }
    }
    merge_result_files(out_dir, result_file, n_gpus, batch_num);
  } else {
    auto memory_resource = make_async();
    rmm::mr::set_current_device_resource(memory_resource.get());
    run_single_file(path,
                    0,
                    0,
                    out_dir,
                    initial_solution_file,
                    heuristics_only,
                    num_cpu_threads,
                    write_log_file,
                    log_to_console,
                    time_limit);
  }

  return 0;
}
