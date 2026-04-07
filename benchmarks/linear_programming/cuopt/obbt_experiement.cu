/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>

#include <raft/core/handle.hpp>

#include <argparse/argparse.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include <rmm/mr/pool_memory_resource.hpp>

#include "benchmark_helper.hpp"

using i_t = int;
using f_t = double;

using termination_t = cuopt::linear_programming::pdlp_termination_status_t;
using method_t      = cuopt::linear_programming::method_t;

static constexpr i_t obbt_iteration_limit = 200000;
static constexpr f_t obbt_time_limit      = 60.0;
static constexpr f_t improvement_eps          = 1e-4;
static constexpr f_t obbt_overall_time_limit  = 1800.0;  // 30 minutes

struct obbt_result_t {
  std::vector<f_t> lb;
  std::vector<f_t> ub;
  i_t lb_improved_count    = 0;
  i_t ub_improved_count    = 0;
  i_t lb_optimal_count     = 0;
  i_t ub_optimal_count     = 0;
  i_t lb_non_optimal_count = 0;
  i_t ub_non_optimal_count = 0;
  std::vector<f_t> solve_times;
  f_t total_elapsed = 0.0;
  f_t ws_elapsed    = 0.0;
};

struct obbt_summary_t {
  std::string instance;
  i_t n_vars        = 0;
  i_t n_constraints = 0;
  i_t n_nnz         = 0;
  i_t pdlp_lb_improved = 0;
  i_t pdlp_ub_improved = 0;
  f_t pdlp_time        = 0.0;
  i_t ds_lb_improved   = 0;
  i_t ds_ub_improved   = 0;
  f_t ds_time          = 0.0;
};

static obbt_result_t run_obbt_solver(
  cuopt::mps_parser::mps_data_model_t<i_t, f_t> mps_model,
  const std::vector<f_t>& original_lb,
  const std::vector<f_t>& original_ub,
  f_t tolerance,
  bool warm_start,
  method_t method,
  const raft::handle_t& handle)
{
  const i_t n       = static_cast<i_t>(original_lb.size());
  const bool is_pdlp = (method == method_t::PDLP);
  const char* method_name = is_pdlp ? "PDLP" : "DS";

  obbt_result_t result;
  result.lb = original_lb;
  result.ub = original_ub;

  bool has_warm_start = false;
  rmm::device_uvector<f_t> ws_primal(0, handle.get_stream());
  rmm::device_uvector<f_t> ws_dual(0, handle.get_stream());
  f_t ws_step_size      = 0.0;
  f_t ws_primal_weight  = 0.0;
  i_t ws_pdlp_iteration = 0;

  if (warm_start && is_pdlp) {
    printf("[%s] Solving root LP relaxation for warm start...\n", method_name);

    cuopt::linear_programming::pdlp_solver_settings_t<i_t, f_t> ws_settings{};
    ws_settings.method          = method_t::PDLP;
    ws_settings.presolver       = cuopt::linear_programming::presolver_t::None;
    ws_settings.iteration_limit = obbt_iteration_limit;
    ws_settings.time_limit      = obbt_time_limit;
    ws_settings.set_optimality_tolerance(tolerance);
    ws_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable3;

    auto ws_solution = cuopt::linear_programming::solve_lp(
      &handle, mps_model, ws_settings, false, true);
    result.ws_elapsed = ws_solution.get_solve_time();

    if (ws_solution.get_termination_status() == termination_t::Optimal) {
      const auto& warm_data = ws_solution.get_pdlp_warm_start_data();
      ws_primal = rmm::device_uvector<f_t>(
        ws_solution.get_primal_solution(), handle.get_stream());
      ws_dual = rmm::device_uvector<f_t>(
        ws_solution.get_dual_solution(), handle.get_stream());
      ws_step_size      = warm_data.initial_step_size_;
      ws_primal_weight  = warm_data.initial_primal_weight_;
      ws_pdlp_iteration = warm_data.total_pdlp_iterations_;
      has_warm_start    = true;
      printf("[%s] Root LP solved in %.3fs (%d iterations). Warm start cached.\n",
             method_name, result.ws_elapsed, ws_pdlp_iteration);
    } else {
      printf("[%s] Root LP did not reach optimality (status=%d) in %.3fs.\n",
             method_name,
             static_cast<int>(ws_solution.get_termination_status()),
             result.ws_elapsed);
    }
  }

  mps_model.set_objective_scaling_factor(1.0);
  mps_model.set_objective_offset(0.0);
  mps_model.set_maximize(false);

  // Per-variable arrays: thread i writes only to slot i, zero contention.
  std::vector<f_t> lb_times(n, 0.0);
  std::vector<f_t> ub_times(n, 0.0);
  std::vector<int8_t> lb_status(n, 0);  // 0=skipped, 1=optimal, -1=non-optimal
  std::vector<int8_t> ub_status(n, 0);

  auto make_settings = [&](const raft::handle_t& h) {
    cuopt::linear_programming::pdlp_solver_settings_t<i_t, f_t> s{};
    s.method          = method;
    s.presolver       = cuopt::linear_programming::presolver_t::None;
    s.iteration_limit = obbt_iteration_limit;
    s.time_limit      = obbt_time_limit;
    if (is_pdlp) {
      s.set_optimality_tolerance(tolerance);
      s.tolerances.absolute_dual_tolerance = 1e-8;
      s.tolerances.relative_dual_tolerance = 1e-8;
      s.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable3;
    }
    if (has_warm_start) {
      s.set_initial_primal_solution(
        ws_primal.data(), ws_primal.size(), h.get_stream());
      s.set_initial_dual_solution(
        ws_dual.data(), ws_dual.size(), h.get_stream());
      s.set_initial_step_size(ws_step_size);
      s.set_initial_primal_weight(ws_primal_weight);
      s.set_initial_pdlp_iteration(ws_pdlp_iteration);
    }
    return s;
  };

  auto process_var = [&](i_t i,
                         cuopt::mps_parser::mps_data_model_t<i_t, f_t>& model,
                         const raft::handle_t& h) {
    auto& obj = model.get_objective_coefficients();
    std::fill(obj.begin(), obj.end(), 0.0);

    obj[i] = 1.0;
    auto s_lb  = make_settings(h);
    auto sol_lb = cuopt::linear_programming::solve_lp(&h, model, s_lb, false, true);
    lb_times[i] = sol_lb.get_solve_time();

    if (sol_lb.get_termination_status() == termination_t::Optimal) {
      lb_status[i] = 1;
      f_t delta = 0.0;
      if (is_pdlp) {
        auto info = sol_lb.get_additional_termination_information();
        delta = tolerance * (1.0 + std::fabs(info.primal_objective)
                                 + std::fabs(info.dual_objective));
      }
      f_t val = sol_lb.get_objective_value() - delta;
      if (val > original_lb[i] + improvement_eps) { result.lb[i] = val; }
    } else {
      lb_status[i] = -1;
    }

    obj[i] = -1.0;
    auto s_ub  = make_settings(h);
    auto sol_ub = cuopt::linear_programming::solve_lp(&h, model, s_ub, false, true);
    ub_times[i] = sol_ub.get_solve_time();

    if (sol_ub.get_termination_status() == termination_t::Optimal) {
      ub_status[i] = 1;
      f_t delta = 0.0;
      if (is_pdlp) {
        auto info = sol_ub.get_additional_termination_information();
        delta = tolerance * (1.0 + std::fabs(info.primal_objective)
                                 + std::fabs(info.dual_objective));
      }
      f_t val = -sol_ub.get_objective_value() + delta;
      if (val < original_ub[i] - improvement_eps) { result.ub[i] = val; }
    } else {
      ub_status[i] = -1;
    }

    obj[i] = 0.0;
  };

  auto total_start = std::chrono::steady_clock::now();

  auto is_timed_out = [&]() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<f_t>(now - total_start).count() >= obbt_overall_time_limit;
  };

  if (method == method_t::DualSimplex) {
    #pragma omp parallel
    {
      auto par_model = mps_model;
      const raft::handle_t par_handle{};

      #pragma omp for schedule(dynamic)
      for (i_t i = 0; i < n; ++i) {
        if (is_timed_out()) continue;
        process_var(i, par_model, par_handle);
      }
    }
  } else {
    for (i_t i = 0; i < n; ++i) {
      if (is_timed_out()) {
        printf("[%s] Overall time limit (%.0fs) reached after %d/%d variables\n",
               method_name, obbt_overall_time_limit, i, n);
        break;
      }
      process_var(i, mps_model, handle);
      if ((i + 1) % 100 == 0 || i == n - 1) {
        printf("[%s] OBBT progress: %d/%d variables done\n", method_name, i + 1, n);
      }
    }
  }

  auto total_end = std::chrono::steady_clock::now();
  result.total_elapsed = std::chrono::duration<f_t>(total_end - total_start).count();

  for (i_t i = 0; i < n; ++i) {
    if (lb_status[i] == 1) {
      result.lb_optimal_count++;
      if (result.lb[i] != original_lb[i]) result.lb_improved_count++;
    } else if (lb_status[i] == -1) {
      result.lb_non_optimal_count++;
    }
    if (ub_status[i] == 1) {
      result.ub_optimal_count++;
      if (result.ub[i] != original_ub[i]) result.ub_improved_count++;
    } else if (ub_status[i] == -1) {
      result.ub_non_optimal_count++;
    }
    if (lb_status[i] != 0) result.solve_times.push_back(lb_times[i]);
    if (ub_status[i] != 0) result.solve_times.push_back(ub_times[i]);
  }

  return result;
}

static void print_solver_results(const char* name,
                                 const obbt_result_t& r,
                                 const std::vector<f_t>& original_lb,
                                 const std::vector<f_t>& original_ub)
{
  const i_t n = static_cast<i_t>(original_lb.size());

  f_t time_min = r.solve_times.empty() ? 0.0
    : *std::min_element(r.solve_times.begin(), r.solve_times.end());
  f_t time_max = r.solve_times.empty() ? 0.0
    : *std::max_element(r.solve_times.begin(), r.solve_times.end());
  f_t time_avg = r.solve_times.empty() ? 0.0
    : std::accumulate(r.solve_times.begin(), r.solve_times.end(), 0.0)
      / r.solve_times.size();

  printf("\n=== %s OBBT Results ===\n", name);
  printf("Variables: %d\n", n);
  printf("LB: %d/%d optimal, %d non-optimal, %d improved\n",
         r.lb_optimal_count, n, r.lb_non_optimal_count, r.lb_improved_count);
  printf("UB: %d/%d optimal, %d non-optimal, %d improved\n",
         r.ub_optimal_count, n, r.ub_non_optimal_count, r.ub_improved_count);
  printf("--- Timing ---\n");
  printf("Root LP warm start: %.3fs\n", r.ws_elapsed);
  printf("OBBT sub-LPs wall time: %.3fs\n", r.total_elapsed);
  printf("Total wall time: %.3fs\n", r.ws_elapsed + r.total_elapsed);
  printf("Per sub-LP solve: min=%.4fs  max=%.4fs  avg=%.4fs  (%d solves)\n",
         time_min, time_max, time_avg, (int)r.solve_times.size());

  if (r.lb_improved_count > 0 || r.ub_improved_count > 0) {
    printf("--- Per-variable improvements ---\n");
    printf("%6s  %18s %18s %14s  %18s %18s %14s\n",
           "var", "old_lb", "new_lb", "lb_delta",
           "old_ub", "new_ub", "ub_delta");

    f_t total_domain_reduction = 0.0;
    i_t domain_reduced_count   = 0;

    for (i_t i = 0; i < n; ++i) {
      f_t lb_delta = r.lb[i] - original_lb[i];
      f_t ub_delta = original_ub[i] - r.ub[i];
      if (lb_delta > improvement_eps || ub_delta > improvement_eps) {
        printf("%6d  %18.10e %18.10e %+14.6e  %18.10e %18.10e %+14.6e\n",
               i, original_lb[i], r.lb[i], lb_delta,
               original_ub[i], r.ub[i], -ub_delta);

        f_t orig_domain = original_ub[i] - original_lb[i];
        f_t new_domain  = r.ub[i] - r.lb[i];
        if (std::isfinite(orig_domain) && orig_domain > 1e-12) {
          total_domain_reduction += (orig_domain - new_domain) / orig_domain * 100.0;
          domain_reduced_count++;
        }
      }
    }

    if (domain_reduced_count > 0) {
      printf("Average domain reduction: %.4f%% (over %d variables)\n",
             total_domain_reduction / domain_reduced_count, domain_reduced_count);
    }
  } else {
    printf("No bound improvements found.\n");
  }
}

static void print_comparison(const obbt_result_t& pdlp,
                             const obbt_result_t& ds,
                             const std::vector<f_t>& original_lb,
                             const std::vector<f_t>& original_ub)
{
  const i_t n = static_cast<i_t>(original_lb.size());

  printf("\n=== PDLP vs Dual Simplex Comparison ===\n");
  printf("%6s  %18s %18s %14s  %18s %18s %14s\n",
         "var", "pdlp_lb", "ds_lb", "lb_diff",
         "pdlp_ub", "ds_ub", "ub_diff");

  i_t pdlp_tighter_lb = 0, ds_tighter_lb = 0, equal_lb = 0;
  i_t pdlp_tighter_ub = 0, ds_tighter_ub = 0, equal_ub = 0;
  i_t vars_with_diff = 0;

  for (i_t i = 0; i < n; ++i) {
    f_t lb_diff = pdlp.lb[i] - ds.lb[i];
    f_t ub_diff = pdlp.ub[i] - ds.ub[i];

    bool lb_changed = (pdlp.lb[i] != original_lb[i]) || (ds.lb[i] != original_lb[i]);
    bool ub_changed = (pdlp.ub[i] != original_ub[i]) || (ds.ub[i] != original_ub[i]);

    if (!lb_changed && !ub_changed) continue;

    bool has_meaningful_diff = std::fabs(lb_diff) > improvement_eps ||
                               std::fabs(ub_diff) > improvement_eps;

    if (has_meaningful_diff) {
      printf("%6d  %18.10e %18.10e %+14.6e  %18.10e %18.10e %+14.6e\n",
             i, pdlp.lb[i], ds.lb[i], lb_diff,
             pdlp.ub[i], ds.ub[i], ub_diff);
      vars_with_diff++;
    }

    if (lb_diff > improvement_eps) pdlp_tighter_lb++;
    else if (lb_diff < -improvement_eps) ds_tighter_lb++;
    else equal_lb++;

    if (ub_diff < -improvement_eps) pdlp_tighter_ub++;
    else if (ub_diff > improvement_eps) ds_tighter_ub++;
    else equal_ub++;
  }

  printf("\n--- Summary ---\n");
  printf("LB: PDLP tighter=%d  DS tighter=%d  equal=%d\n",
         pdlp_tighter_lb, ds_tighter_lb, equal_lb);
  printf("UB: PDLP tighter=%d  DS tighter=%d  equal=%d\n",
         pdlp_tighter_ub, ds_tighter_ub, equal_ub);
  printf("Variables with meaningful differences: %d/%d\n", vars_with_diff, n);
  printf("Speedup: DS is %.2fx faster (DS=%.3fs, PDLP=%.3fs)\n",
         (pdlp.ws_elapsed + pdlp.total_elapsed) /
           std::max(ds.ws_elapsed + ds.total_elapsed, 1e-9),
         ds.ws_elapsed + ds.total_elapsed,
         pdlp.ws_elapsed + pdlp.total_elapsed);
}

static obbt_summary_t run_obbt(const std::string& mps_path,
                                f_t tolerance,
                                bool warm_start,
                                const raft::handle_t& handle)
{
  obbt_summary_t summary;
  summary.instance = std::filesystem::path(mps_path).filename().string();

  auto mps_model = cuopt::mps_parser::parse_mps<i_t, f_t>(mps_path);

  summary.n_vars        = mps_model.get_n_variables();
  summary.n_constraints = mps_model.get_n_constraints();
  summary.n_nnz         = mps_model.get_nnz();

  const i_t n = static_cast<i_t>(mps_model.get_objective_coefficients().size());

  const std::vector<f_t> original_lb = mps_model.get_variable_lower_bounds();
  const std::vector<f_t> original_ub = mps_model.get_variable_upper_bounds();

  printf("OBBT: %d variables, solving %d sub-LPs per solver\n", n, 2 * n);
  printf("OBBT: tolerance=%.2e, iter_limit=%d, time_limit=%.0fs\n",
         tolerance, obbt_iteration_limit, obbt_time_limit);

  auto pdlp_result = run_obbt_solver(
    mps_model, original_lb, original_ub, tolerance, warm_start, method_t::PDLP, handle);
  print_solver_results("PDLP", pdlp_result, original_lb, original_ub);

  auto ds_result = run_obbt_solver(
    mps_model, original_lb, original_ub, tolerance, false, method_t::DualSimplex, handle);
  print_solver_results("DualSimplex", ds_result, original_lb, original_ub);

  print_comparison(pdlp_result, ds_result, original_lb, original_ub);

  summary.pdlp_lb_improved = pdlp_result.lb_improved_count;
  summary.pdlp_ub_improved = pdlp_result.ub_improved_count;
  summary.pdlp_time        = pdlp_result.ws_elapsed + pdlp_result.total_elapsed;
  summary.ds_lb_improved   = ds_result.lb_improved_count;
  summary.ds_ub_improved   = ds_result.ub_improved_count;
  summary.ds_time          = ds_result.ws_elapsed + ds_result.total_elapsed;

  return summary;
}

static void write_to_output_file(const std::string& out_dir,
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
    fprintf(stderr, "Error opening file %s\n", filename.c_str());
  }
}

static void merge_result_files(const std::string& out_dir,
                                const std::string& final_result_file,
                                int n_gpus,
                                int batch_id)
{
  std::ofstream final_file(final_result_file, std::ios_base::app);
  if (!final_file.is_open()) {
    fprintf(stderr, "Error opening final result file %s\n", final_result_file.c_str());
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
      std::remove(temp_file.c_str());
    }
  }
  final_file.close();
}

static void return_gpu_to_the_queue(std::unordered_map<pid_t, int>& pid_gpu_map,
                                     std::unordered_map<pid_t, std::string>& pid_file_map,
                                     std::queue<int>& gpu_queue)
{
  int status;
  pid_t pid = wait(&status);
  auto file_name = pid_file_map[pid];
  int gpu        = pid_gpu_map[pid];
  std::string base = std::filesystem::path(file_name).filename().string();
  if (WIFEXITED(status)) {
    printf("[GPU %d] %s finished (exit %d)\n", gpu, base.c_str(), WEXITSTATUS(status));
  } else {
    int signal_number = WTERMSIG(status);
    printf("[GPU %d] %s killed by signal %d\n", gpu, base.c_str(), signal_number);
  }
  fflush(stdout);
  gpu_queue.push(gpu);
  pid_gpu_map.erase(pid);
  pid_file_map.erase(pid);
}

static void run_single_instance_mp(const std::string& mps_path,
                                   f_t tolerance,
                                   bool warm_start,
                                   const std::string& out_dir,
                                   int gpu_id,
                                   int batch_id,
                                   int n_gpus)
{
  std::string base_filename = std::filesystem::path(mps_path).filename().string();
  std::string log_file = out_dir + "/" +
    base_filename.substr(0, base_filename.find(".mps")) + ".log";

  FILE* log_fp = freopen(log_file.c_str(), "w", stdout);
  if (!log_fp) {
    fprintf(stderr, "Failed to redirect stdout to %s\n", log_file.c_str());
    exit(1);
  }
  dup2(fileno(stdout), fileno(stderr));

  printf("Running OBBT on %s (gpu=%d)\n", base_filename.c_str(), gpu_id);

  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource.get());

  const raft::handle_t handle{};

  obbt_summary_t summary;
  try {
    summary = run_obbt(mps_path, tolerance, warm_start, handle);
  } catch (const std::exception& e) {
    printf("Exception: %s\n", e.what());
    summary.instance = base_filename;
  }

  std::stringstream ss;
  ss << std::fixed << std::setprecision(2)
     << summary.instance << ","
     << summary.n_vars << ","
     << summary.n_constraints << ","
     << summary.n_nnz << ","
     << summary.pdlp_lb_improved << ","
     << summary.pdlp_ub_improved << ","
     << summary.pdlp_time << ","
     << summary.ds_lb_improved << ","
     << summary.ds_ub_improved << ","
     << summary.ds_time << "\n";
  write_to_output_file(out_dir, gpu_id, n_gpus, batch_id, ss.str());

  fclose(log_fp);
  exit(0);
}

int main(int argc, char* argv[])
{
  argparse::ArgumentParser program("obbt_experiment");

  program.add_argument("--path").help("Path to MPS file or directory of MPS files").required();
  program.add_argument("--tolerance")
    .help("Optimality tolerance for PDLP")
    .default_value(1e-6)
    .scan<'g', double>();
  program.add_argument("--warm-start")
    .help("Enable warm start from root LP solve")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("--n-gpus")
    .help("Number of GPUs on this node")
    .default_value(1)
    .scan<'i', int>();
  program.add_argument("--out-dir")
    .help("Output directory for result files (required for directory mode)");
  program.add_argument("--batch-num")
    .help("Batch number (0-based)")
    .default_value(-1)
    .scan<'i', int>();
  program.add_argument("--n-batches")
    .help("Total number of batches")
    .default_value(-1)
    .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  std::string path  = program.get<std::string>("--path");
  f_t tolerance     = program.get<double>("--tolerance");
  bool warm_start   = program.get<bool>("--warm-start");
  bool run_dir      = std::filesystem::is_directory(path);
  int n_gpus        = program.get<int>("--n-gpus");
  int batch_num     = program.get<int>("--batch-num");
  int n_batches     = program.get<int>("--n-batches");

  if (!run_dir) {
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());
    const raft::handle_t handle{};
    run_obbt(path, tolerance, warm_start, handle);
    return 0;
  }

  std::string out_dir;
  if (program.is_used("--out-dir")) {
    out_dir = program.get<std::string>("--out-dir");
    std::filesystem::create_directories(out_dir);
  } else {
    std::cerr << "--out-dir is required when using --run-dir" << std::endl;
    return 1;
  }

  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ".mps") {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());

  if (batch_num != -1) {
    if (n_batches <= 0) {
      std::cerr << "Error: --n-batches must be > 0 when using --batch-num" << std::endl;
      return 1;
    }
    int batch_size  = static_cast<int>(std::ceil(
      static_cast<double>(paths.size()) / n_batches));
    int start_index = batch_num * batch_size;
    int end_index   = std::min((batch_num + 1) * batch_size, static_cast<int>(paths.size()));
    if (start_index >= static_cast<int>(paths.size())) {
      printf("Batch %d has no files (total %zu files, %d batches)\n",
             batch_num, paths.size(), n_batches);
      return 0;
    }
    paths = std::vector<std::string>(paths.begin() + start_index, paths.begin() + end_index);
  } else {
    batch_num = 0;
  }

  printf("Running OBBT on %zu files (batch %d, %d GPUs)\n", paths.size(), batch_num, n_gpus);

  std::string result_file = out_dir + "/final_result_" + std::to_string(batch_num) + ".csv";

  std::queue<std::string> task_queue;
  std::queue<int> gpu_queue;
  std::unordered_map<pid_t, int> pid_gpu_map;
  std::unordered_map<pid_t, std::string> pid_file_map;

  for (int i = 0; i < n_gpus; ++i) {
    gpu_queue.push(i);
  }
  for (const auto& p : paths) {
    task_queue.push(p);
  }

  while (!task_queue.empty()) {
    if (!gpu_queue.empty()) {
      int gpu_id     = gpu_queue.front();
      auto file_name = task_queue.front();
      gpu_queue.pop();
      task_queue.pop();

      std::string base = std::filesystem::path(file_name).filename().string();
      printf("[batch %d] GPU %d -> %s (starting)\n", batch_num, gpu_id, base.c_str());
      fflush(stdout);

      auto sys_pid = fork();
      if (sys_pid > 0) {
        pid_gpu_map.insert({sys_pid, gpu_id});
        pid_file_map.insert({sys_pid, file_name});
      } else if (sys_pid == 0) {
        RAFT_CUDA_TRY(cudaSetDevice(gpu_id));
        run_single_instance_mp(file_name, tolerance, warm_start,
                               out_dir, gpu_id, batch_num, n_gpus);
      } else {
        std::cerr << "Fork failed!" << std::endl;
        exit(1);
      }
    } else {
      return_gpu_to_the_queue(pid_gpu_map, pid_file_map, gpu_queue);
    }
    sleep(1);
  }

  while (!pid_gpu_map.empty()) {
    return_gpu_to_the_queue(pid_gpu_map, pid_file_map, gpu_queue);
  }

  merge_result_files(out_dir, result_file, n_gpus, batch_num);
  printf("Results merged to %s\n", result_file.c_str());

  return 0;
}
