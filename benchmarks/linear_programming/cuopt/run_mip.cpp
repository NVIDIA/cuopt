/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include "initial_solution_reader.hpp"
#include "mip_test_instances.hpp"
#include "miplib2017_bks.hpp"
#include "row_audit.hpp"

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cstdio>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>
#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <fcntl.h>
#include <omp.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#include <argparse/argparse.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <syncstream>
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
                          int n_gpus,
                          int batch_id,
                          const std::string& data)
{
  int output_id        = batch_id * n_gpus + gpu_id;
  std::string filename = out_dir + "/result_" + std::to_string(output_id) + ".txt";
  std::ofstream outfile(filename, std::ios_base::app);
  if (outfile.is_open()) {
    outfile << data;
    outfile.close();
  } else {
    std::cerr << "Error opening file " << filename << std::endl;
  }
}

inline auto make_async() { return rmm::mr::cuda_async_memory_resource(); }

constexpr bool use_c_api = true;

struct c_api_handles_t {
  c_api_handles_t()                                  = default;
  c_api_handles_t(const c_api_handles_t&)            = delete;
  c_api_handles_t& operator=(const c_api_handles_t&) = delete;

  ~c_api_handles_t()
  {
    cuOptDestroySolution(&solution);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroyProblem(&problem);
  }

  cuOptOptimizationProblem problem{};
  cuOptSolverSettings settings{};
  cuOptSolution solution{};
};

struct solve_result_t {
  double objective_value{};
  double solution_bound{};
  double mip_gap{};
  cuopt_int_t termination_status{};

  double get_objective_value() const { return objective_value; }
  double get_solution_bound() const { return solution_bound; }
  double get_mip_gap() const { return mip_gap; }
};

static void configure_c_api_settings(
  cuOptSolverSettings c_settings,
  const cuopt::mathematical_optimization::mip_solver_settings_t<int, double>& settings)
{
  cuOptSetFloatParameter(c_settings, CUOPT_TIME_LIMIT, settings.time_limit);
  cuOptSetFloatParameter(c_settings, CUOPT_WORK_LIMIT, settings.work_limit);
  cuOptSetIntegerParameter(c_settings, CUOPT_NUM_CPU_THREADS, settings.num_cpu_threads);
  cuOptSetIntegerParameter(c_settings, CUOPT_MIP_DETERMINISM_MODE, settings.determinism_mode);
  cuOptSetFloatParameter(
    c_settings, CUOPT_MIP_RELATIVE_TOLERANCE, settings.tolerances.relative_tolerance);
  cuOptSetFloatParameter(
    c_settings, CUOPT_MIP_ABSOLUTE_TOLERANCE, settings.tolerances.absolute_tolerance);
  cuOptSetFloatParameter(
    c_settings, CUOPT_MIP_INTEGRALITY_TOLERANCE, settings.tolerances.integrality_tolerance);
  cuOptSetIntegerParameter(c_settings, CUOPT_PRESOLVE, (cuopt_int_t)settings.presolver);
  cuOptSetIntegerParameter(
    c_settings, CUOPT_MIP_RELIABILITY_BRANCHING, settings.reliability_branching);
  cuOptSetIntegerParameter(c_settings, CUOPT_MIP_CLIQUE_CUTS, settings.clique_cuts);
  cuOptSetIntegerParameter(c_settings, CUOPT_RANDOM_SEED, settings.seed);
  cuOptSetParameter(
    c_settings, CUOPT_MIP_HEURISTICS_ONLY, settings.heuristics_only ? "true" : "false");
  cuOptSetParameter(c_settings, CUOPT_LOG_TO_CONSOLE, settings.log_to_console ? "true" : "false");
  cuOptSetParameter(c_settings, CUOPT_LOG_FILE, settings.log_file.c_str());
}

static bool verify_solution(
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& problem,
  const std::vector<double>& solution,
  double reported_objective,
  const cuopt::mathematical_optimization::mip_solver_settings_t<int, double>::tolerances_t&
    tolerances,
  size_t incumbent)
{
  const auto& values                 = problem.get_constraint_matrix_values();
  const auto& indices                = problem.get_constraint_matrix_indices();
  const auto& offsets                = problem.get_constraint_matrix_offsets();
  const auto& row_lower_bounds       = problem.get_constraint_lower_bounds();
  const auto& row_upper_bounds       = problem.get_constraint_upper_bounds();
  const auto& variable_lower_bounds  = problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds  = problem.get_variable_upper_bounds();
  const auto& variable_types         = problem.get_variable_types();
  const auto& objective_coefficients = problem.get_objective_coefficients();

  if (variable_lower_bounds.size() != variable_types.size() ||
      variable_upper_bounds.size() != variable_types.size() ||
      objective_coefficients.size() != variable_types.size()) {
    std::osyncstream(std::cerr) << "Incumbent " << incumbent
                                << " cannot be checked against malformed variable data\n";
    return false;
  }
  if (solution.size() != variable_types.size()) {
    std::osyncstream(std::cerr) << "Incumbent " << incumbent << " has " << solution.size()
                                << " variables; expected " << variable_types.size() << "\n";
    return false;
  }

  _Float128 objective = 0;
  for (size_t variable = 0; variable < solution.size(); ++variable) {
    const double value = solution[variable];
    if (!std::isfinite(value)) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                  << " variable " << variable << " is not finite: " << value
                                  << "\n";
      return false;
    }
    const auto bound_limits = cuopt_bench::scaled_bound_limits(tolerances.absolute_tolerance,
                                                               value,
                                                               variable_lower_bounds[variable],
                                                               variable_upper_bounds[variable]);
    const double lower_bound_tolerance = variable_lower_bounds[variable] - bound_limits.first;
    const double upper_bound_tolerance = bound_limits.second - variable_upper_bounds[variable];
    if (value < bound_limits.first || value > bound_limits.second) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                  << " variable " << variable << " violates bounds: value=" << value
                                  << " lb=" << variable_lower_bounds[variable]
                                  << " ub=" << variable_upper_bounds[variable]
                                  << " lower_tolerance=" << lower_bound_tolerance
                                  << " upper_tolerance=" << upper_bound_tolerance << "\n";
      return false;
    }
    if (variable_types[variable] == 'I' && value != std::round(value)) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                  << " variable " << variable
                                  << " is not exactly integral: value=" << value
                                  << " residual=" << std::abs(value - std::round(value)) << "\n";
      return false;
    }
    objective += (_Float128)objective_coefficients[variable] * (_Float128)solution[variable];
  }

  if (row_upper_bounds.size() != row_lower_bounds.size() ||
      offsets.size() != row_lower_bounds.size() + 1) {
    std::osyncstream(std::cerr) << "Incumbent " << incumbent
                                << " cannot be checked against malformed row data\n";
    return false;
  }
  for (size_t row = 0; row < row_lower_bounds.size(); ++row) {
    const double lower_bound = row_lower_bounds[row];
    const double upper_bound = row_upper_bounds[row];

    // fp64 first. _Float128 is soft-float on x86-64; only rows whose rounding-error
    // interval meets a bound pay for it. Bound is (nnz+1)*eps*abs_sum.
    const auto verdict = cuopt_bench::check_row(
      values.data(),
      indices.data(),
      (int64_t)offsets[row],
      (int64_t)offsets[row + 1],
      solution.data(),
      lower_bound,
      upper_bound,
      [&](double positive_activity) {
        return cuopt_bench::scaled_row_limits(
          tolerances.absolute_tolerance, positive_activity, lower_bound, upper_bound);
      });
    if (verdict.excess > 0.0) {
      std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent << " row "
                                  << row << " violates bounds: activity=" << verdict.activity
                                  << " lb=" << lower_bound << " ub=" << upper_bound
                                  << " lower_tolerance=" << (lower_bound - verdict.lower_limit)
                                  << " upper_tolerance=" << (verdict.upper_limit - upper_bound)
                                  << "\n";
      return false;
    }
  }

  const double recomputed_objective =
    problem.get_objective_scaling_factor() * ((double)objective + problem.get_objective_offset());
  const double objective_difference = std::abs(recomputed_objective - reported_objective);
  const double objective_scale =
    std::max(std::abs(recomputed_objective), std::abs(reported_objective));
  const double relative_objective_tolerance = objective_scale * tolerances.relative_tolerance;
  if (!std::isfinite(recomputed_objective) || !std::isfinite(reported_objective) ||
      (objective_difference > tolerances.absolute_tolerance &&
       objective_difference > relative_objective_tolerance)) {
    std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << incumbent
                                << " objective mismatch: reported=" << reported_objective
                                << " recomputed=" << recomputed_objective
                                << " difference=" << objective_difference
                                << " absolute_tolerance=" << tolerances.absolute_tolerance
                                << " relative_threshold=" << relative_objective_tolerance << "\n";
    return false;
  }
  return true;
}

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

struct incumbent_record_t {
  std::vector<double> solution;
  double reported_objective;
  double work_timestamp;
  double wall_time;
};

class incumbent_tracker_t : public cuopt::internals::get_solution_callback_t {
 public:
  incumbent_tracker_t(std::chrono::high_resolution_clock::time_point start_time,
                      size_t num_variables)
    : start_time_(start_time), num_variables_(num_variables)
  {
  }

  void get_solution(void* data, void* cost, void* /*solution_bound*/, void* /*user_data*/) override
  {
    record_solution(static_cast<double*>(data), *static_cast<double*>(cost));
  }

  void record_solution(const double* solution, double objective)
  {
    const auto now = std::chrono::high_resolution_clock::now();
    records_.push_back({std::vector<double>(solution, solution + num_variables_),
                        objective,
                        0.0,
                        std::chrono::duration<double>(now - start_time_).count()});
  }

  void write_csv(
    const std::string& path,
    const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& problem,
    const cuopt::mathematical_optimization::mip_solver_settings_t<int, double>::tolerances_t&
      tolerances,
    int num_cpu_threads) const
  {
    std::ofstream file(path);
    if (!file.is_open()) {
      std::cerr << "Error opening incumbent file " << path << std::endl;
      return;
    }
    constexpr double bks_rounding_threshold = 0.5e-6;
    const auto bks = cuopt_bench::lookup_miplib_bks(problem.get_problem_name());
    file << "index,objective,work_timestamp,wall_time_s,valid\n";
    std::vector<char> valid(records_.size());
#pragma omp parallel for schedule(static) num_threads(num_cpu_threads)
    for (size_t i = 0; i < records_.size(); ++i) {
      valid[i] = verify_solution(
        problem, records_[i].solution, records_[i].reported_objective, tolerances, i);
      if (bks.has_value()) {
        const double objective      = records_[i].reported_objective;
        const double scale          = std::max({std::abs(objective), std::abs(*bks), 1.0});
        const double normalized_gap = (objective - *bks) / scale;
        if (normalized_gap < -bks_rounding_threshold) {
          std::osyncstream(std::cerr) << std::setprecision(17) << "Incumbent " << i << " objective "
                                      << objective << " is below MIPLIB BKS " << *bks << "\n";
          valid[i] = 0;
        }
      }
    }
    for (size_t i = 0; i < records_.size(); ++i) {
      file << i << "," << std::setprecision(15) << records_[i].reported_objective << ","
           << records_[i].work_timestamp << "," << std::setprecision(6) << records_[i].wall_time
           << "," << int(valid[i]) << "\n";
    }
  }

  size_t size() const { return records_.size(); }

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
  size_t num_variables_;
  std::vector<incumbent_record_t> records_;
};

static void c_api_incumbent_callback(const cuopt_float_t* solution,
                                     const cuopt_float_t* objective_value,
                                     const cuopt_float_t* /*solution_bound*/,
                                     void* user_data)
{
  static_cast<incumbent_tracker_t*>(user_data)->record_solution(solution, *objective_value);
}

int run_single_file(std::string file_path,
                    int device,
                    int batch_id,
                    int n_gpus,
                    std::string out_dir,
                    std::optional<std::string> initial_solution_dir,
                    bool heuristics_only,
                    int num_cpu_threads,
                    bool write_log_file,
                    bool log_to_console,
                    int reliability_branching,
                    double time_limit,
                    double work_limit,
                    bool deterministic)
{
  (void)cudaFree(0);

  std::unique_ptr<raft::handle_t> handle;
  if constexpr (!use_c_api) { handle = std::make_unique<raft::handle_t>(); }
  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings;
  c_api_handles_t c_api;
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
  settings.time_limit       = time_limit;
  settings.work_limit       = work_limit;
  settings.heuristics_only  = heuristics_only;
  settings.num_cpu_threads  = num_cpu_threads;
  settings.log_to_console   = log_to_console;
  settings.determinism_mode = deterministic ? CUOPT_MODE_DETERMINISTIC : CUOPT_MODE_OPPORTUNISTIC;
  settings.tolerances.relative_tolerance = 1e-12;
  settings.tolerances.absolute_tolerance = 1e-6;
  settings.presolver                     = cuopt::mathematical_optimization::presolver_t::Default;
  settings.reliability_branching         = reliability_branching;
  settings.clique_cuts                   = -1;
  settings.seed                          = 42;

  // This benchmark and the solver library have separate loggers, both writing settings.log_file.
  // Configure the solver's first so its own initializer reuses that configuration rather than
  // truncating the file mid-solve; this image's logger then appends to it. Without one, this
  // image's own messages would sit unflushed in the buffer sink.
  auto solver_log = cuopt::mathematical_optimization::configure_logging(
    settings.log_file, log_to_console, /*truncate=*/true);
  cuopt::init_logger_t bench_log(settings.log_file, log_to_console, /*truncate=*/false);
  if constexpr (use_c_api) {
    cuOptCreateSolverSettings(&c_api.settings);
    configure_c_api_settings(c_api.settings, settings);
  }

  constexpr bool input_mps_strict = false;
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> mps_data_model;
  bool parsing_failed = false;
  {
    CUOPT_LOG_INFO("running file %s on gpu : %d", base_filename.c_str(), device);
    try {
      mps_data_model =
        cuopt::mathematical_optimization::io::read_mps<int, double>(file_path, input_mps_strict);
    } catch (const std::logic_error& e) {
      CUOPT_LOG_ERROR("MPS parser execption: %s", e.what());
      parsing_failed = true;
    }
  }
  if (parsing_failed) {
    CUOPT_LOG_ERROR("Parsing MPS failed exiting!");
    return -1;
  }
  // Use the benchmark filename for downstream instance-level reporting.
  // This keeps per-instance metrics aligned with the run list even if the MPS NAME card differs.
  mps_data_model.set_problem_name(base_filename);

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
        if constexpr (use_c_api) {
          cuOptAddMIPStart(c_api.settings, initial_solution.data(), initial_solution.size());
        } else {
          settings.add_initial_solution(
            initial_solution.data(), initial_solution.size(), handle->get_stream());
        }
      }
    }
  }
  cuopt::mathematical_optimization::benchmark_info_t benchmark_info;
  if constexpr (!use_c_api) { settings.benchmark_info_ptr = &benchmark_info; }
  std::chrono::high_resolution_clock::time_point start_run_solver;
  std::unique_ptr<incumbent_tracker_t> incumbent_tracker;
  solve_result_t solution;
  if constexpr (use_c_api) {
    if (mps_data_model.get_objective_scaling_factor() != 1.0) {
      throw std::runtime_error("cuOptCreateRangedProblem does not support objective scaling");
    }
    cuOptCreateRangedProblem(mps_data_model.get_n_constraints(),
                             mps_data_model.get_n_variables(),
                             mps_data_model.get_sense() ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE,
                             mps_data_model.get_objective_offset(),
                             mps_data_model.get_objective_coefficients().data(),
                             mps_data_model.get_constraint_matrix_offsets().data(),
                             mps_data_model.get_constraint_matrix_indices().data(),
                             mps_data_model.get_constraint_matrix_values().data(),
                             mps_data_model.get_constraint_lower_bounds().data(),
                             mps_data_model.get_constraint_upper_bounds().data(),
                             mps_data_model.get_variable_lower_bounds().data(),
                             mps_data_model.get_variable_upper_bounds().data(),
                             mps_data_model.get_variable_types().data(),
                             &c_api.problem);

    start_run_solver = std::chrono::high_resolution_clock::now();
    incumbent_tracker =
      std::make_unique<incumbent_tracker_t>(start_run_solver, mps_data_model.get_n_variables());
    cuOptSetMIPGetSolutionCallback(
      c_api.settings, c_api_incumbent_callback, incumbent_tracker.get());
    cuOptSolve(c_api.problem, c_api.settings, &c_api.solution);
    cuOptGetTerminationStatus(c_api.solution, &solution.termination_status);
    cuOptGetObjectiveValue(c_api.solution, &solution.objective_value);
    cuOptGetSolutionBound(c_api.solution, &solution.solution_bound);
    cuOptGetMIPGap(c_api.solution, &solution.mip_gap);
  } else {
    start_run_solver = std::chrono::high_resolution_clock::now();
    incumbent_tracker =
      std::make_unique<incumbent_tracker_t>(start_run_solver, mps_data_model.get_n_variables());
    settings.set_mip_callback(incumbent_tracker.get());
    auto cpp_solution =
      cuopt::mathematical_optimization::solve_mip(handle.get(), mps_data_model, settings);
    solution.objective_value    = cpp_solution.get_objective_value();
    solution.solution_bound     = cpp_solution.get_solution_bound();
    solution.mip_gap            = cpp_solution.get_mip_gap();
    solution.termination_status = (cuopt_int_t)cpp_solution.get_termination_status();
    // solution.write_to_sol_file(base_filename + ".sol", handle_.get_stream());
  }
  CUOPT_LOG_INFO(
    "first obj: %f last improvement of best feasible: %f last improvement after recombination: %f",
    benchmark_info.objective_of_initial_population,
    benchmark_info.last_improvement_of_best_feasible,
    benchmark_info.last_improvement_after_recombination);
  std::chrono::milliseconds duration;
  auto end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_run_solver);
  CUOPT_LOG_INFO("run_solver %d", duration.count());
  if constexpr (!use_c_api) { handle->sync_stream(); }
  int sol_found  = int(solution.termination_status == CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND ||
                      solution.termination_status == CUOPT_TERMINATION_STATUS_OPTIMAL);
  double obj_val = sol_found ? solution.get_objective_value() : std::numeric_limits<double>::max();
  if (sol_found) {
    CUOPT_LOG_INFO("%s: solution found, obj: %f", base_filename.c_str(), obj_val);
  } else {
    CUOPT_LOG_INFO("%s: no solution found", base_filename.c_str());
  }

  // Per-instance "gap closed to BKS" stat. Emits a single
  // grep-friendly "MIPLIBGapStat ..." line via printf so cross-branch
  // comparison is just `grep '^MIPLIBGapStat' branchA.log` then diff.
  // BKS values are looked up from the in-source MIPLIB2017 benchmark-set
  // table (miplib2017_bks.hpp); unknown instances emit "opt=TBD"
  // and infeasibility-flagged instances emit "opt=Infeasible".
  {
    const double _gap_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_run_solver)
                                  .count() /
                                1000.0;
    std::string _status_str;
    switch (solution.termination_status) {
      case CUOPT_TERMINATION_STATUS_OPTIMAL: _status_str = "Optimal"; break;
      case CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND: _status_str = "FeasibleFound"; break;
      case CUOPT_TERMINATION_STATUS_TIME_LIMIT: _status_str = "TimeLimit"; break;
      case CUOPT_TERMINATION_STATUS_INFEASIBLE: _status_str = "Infeasible"; break;
      default: _status_str = "Other"; break;
    }
    cuopt_bench::print_miplib_gap_stat(base_filename,
                                       solution,
                                       _gap_seconds,
                                       _status_str,
                                       benchmark_info.root_lp_no_cuts,
                                       benchmark_info.root_lp_with_cuts,
                                       benchmark_info.cut_generation_time_sec);
  }

  std::stringstream ss;
  int decimal_places = 2;
  double mip_gap     = solution.get_mip_gap();
  int is_optimal     = solution.termination_status == CUOPT_TERMINATION_STATUS_OPTIMAL ? 1 : 0;
  ss << std::fixed << std::setprecision(decimal_places) << base_filename << "," << sol_found << ","
     << obj_val << "," << benchmark_info.objective_of_initial_population << ","
     << benchmark_info.last_improvement_of_best_feasible << ","
     << benchmark_info.last_improvement_after_recombination << "," << mip_gap << "," << is_optimal
     << "\n";
  write_to_output_file(out_dir, base_filename, device, n_gpus, batch_id, ss.str());
  CUOPT_LOG_INFO("Results written to the file %s", base_filename.c_str());
  if (out_dir != "") {
    std::string csv_path =
      out_dir + "/" + base_filename.substr(0, base_filename.find(".mps")) + "_incumbents.csv";
    const auto csv_start = std::chrono::high_resolution_clock::now();
    incumbent_tracker->write_csv(
      csv_path, mps_data_model, settings.get_tolerances(), num_cpu_threads);
    const auto csv_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Incumbent csv generation took "
              << std::chrono::duration<double>(csv_end - csv_start).count() << " s ("
              << incumbent_tracker->size() << " entries) -> " << csv_path << std::endl;
    CUOPT_LOG_INFO(
      "Incumbent trace (%zu entries) written to %s", incumbent_tracker->size(), csv_path.c_str());
  }
  return sol_found;
}

void run_single_file_mp(std::string file_path,
                        int device,
                        int batch_id,
                        int n_gpus,
                        std::string out_dir,
                        std::optional<std::string> input_file_dir,
                        bool heuristics_only,
                        int num_cpu_threads,
                        bool write_log_file,
                        bool log_to_console,
                        int reliability_branching,
                        double time_limit,
                        double work_limit,
                        bool deterministic)
{
  std::cout << "running file " << file_path << " on gpu : " << device << std::endl;
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource);
  int sol_found = run_single_file(file_path,
                                  device,
                                  batch_id,
                                  n_gpus,
                                  out_dir,
                                  input_file_dir,
                                  heuristics_only,
                                  num_cpu_threads,
                                  write_log_file,
                                  log_to_console,
                                  reliability_branching,
                                  time_limit,
                                  work_limit,
                                  deterministic);
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
    .help("time limit in seconds")
    .scan<'g', double>()
    .default_value(std::numeric_limits<double>::infinity());

  program.add_argument("--work-limit")
    .help("work unit limit (for deterministic mode)")
    .scan<'g', double>()
    .default_value(std::numeric_limits<double>::infinity());

  program.add_argument("--memory-limit")
    .help("memory limit in MB")
    .scan<'g', double>()
    .default_value(0.0);

  program.add_argument("--track-allocations")
    .help("track allocations (t/f)")
    .default_value(std::string("f"));

  program.add_argument("--reliability-branching")
    .help("reliability branching: -1 (automatic), 0 (disable) or k > 0 (use k)")
    .scan<'i', int>()
    .default_value(-1);

  program.add_argument("-d", "--determinism")
    .help("enable deterministic mode")
    .default_value(false)
    .implicit_value(true);

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
  double work_limit       = program.get<double>("--work-limit");

  bool run_selected = program.get<std::string>("--run-selected")[0] == 't';
  int n_gpus        = program.get<int>("--n-gpus");

  std::string out_dir;
  std::string result_file;
  int batch_num = -1;

  bool heuristics_only      = program.get<std::string>("--heuristics-only")[0] == 't';
  int num_cpu_threads       = program.get<int>("--num-cpu-threads");
  bool write_log_file       = program.get<std::string>("--write-log-file")[0] == 't';
  bool log_to_console       = program.get<std::string>("--log-to-console")[0] == 't';
  double memory_limit       = program.get<double>("--memory-limit");
  bool track_allocations    = program.get<std::string>("--track-allocations")[0] == 't';
  int reliability_branching = program.get<int>("--reliability-branching");
  bool deterministic        = program.get<bool>("--determinism");

  if (num_cpu_threads < 0) {
    num_cpu_threads = omp_get_max_threads() / n_gpus;
    // std::ifstream smt_file("/sys/devices/system/cpu/smt/active");
    // if (smt_file.is_open()) {
    //   int smt_active = 0;
    //   smt_file >> smt_active;
    //   if (smt_active) { num_cpu_threads /= 2; }
    // }
    num_cpu_threads = std::max(num_cpu_threads, 2);
  }

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
                               n_gpus,
                               out_dir,
                               initial_solution_file,
                               heuristics_only,
                               num_cpu_threads,
                               write_log_file,
                               log_to_console,
                               reliability_branching,
                               time_limit,
                               work_limit,
                               deterministic);
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
    auto run_single      = [&]() {
      run_single_file(path,
                      0,
                      0,
                      n_gpus,
                      out_dir,
                      initial_solution_file,
                      heuristics_only,
                      num_cpu_threads,
                      write_log_file,
                      log_to_console,
                      reliability_branching,
                      time_limit,
                      work_limit,
                      deterministic);
    };
    if (memory_limit > 0) {
      auto limiting_adaptor =
        rmm::mr::limiting_resource_adaptor(memory_resource, memory_limit * 1024ULL * 1024ULL);
      rmm::mr::set_current_device_resource(limiting_adaptor);
      run_single();
    } else if (track_allocations) {
      rmm::mr::tracking_resource_adaptor tracking_adaptor(memory_resource,
                                                          /*capture_stacks=*/true);
      rmm::mr::set_current_device_resource(tracking_adaptor);
      run_single();
    } else {
      rmm::mr::set_current_device_resource(memory_resource);
      run_single();
    }
  }

  return 0;
}
