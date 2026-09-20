/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "miplib2017_bks.hpp"

#include <mip_heuristics/feasibility_jump/cpu/tuning.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/presolve/presolve_budget_policy.hpp>
#include <mip_heuristics/presolve/probing_cache.cuh>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>

#include <math_optimization/solution_writer.hpp>

#include <argparse/argparse.hpp>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utilities/seed_generator.cuh>
#include <vector>

#include "initial_problem_check.hpp"

namespace {

using i_t     = int;
using f_t     = double;
namespace mip = cuopt::mathematical_optimization::mip;

using clk = std::chrono::high_resolution_clock;
double since(clk::time_point t0)
{
  return std::chrono::duration_cast<std::chrono::duration<double>>(clk::now() - t0).count();
}

struct climber_result_t {
  bool crossed{false};
  double t_first{-1.0};
  f_t best_objective{std::numeric_limits<f_t>::infinity()};
  i_t iterations{0};
  double seconds{0.0};
};

void pin_to_core(int core)
{
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// The CPUs this process is actually permitted to run on. A cgroup mask can be non-contiguous, so
// indexing hardware_concurrency() directly would collide several climbers onto one core.
std::vector<int> allowed_cpus()
{
  std::vector<int> allowed;
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &set)) allowed.push_back(cpu);
    }
  }
  if (allowed.empty()) allowed.push_back(0);
  return allowed;
}

void run_climber(mip::fj_cpu_climber_t<i_t, f_t>* climber,
                 f_t time_limit,
                 int core,
                 climber_result_t& result)
{
  pin_to_core(core);
  const auto t0 = clk::now();

  climber->improvement_callback = [&result, t0](f_t objective, const std::vector<f_t>&, double) {
    if (!result.crossed) {
      result.crossed = true;
      result.t_first = since(t0);
    }
    result.best_objective = objective;
  };

  mip::cpufj_solve(climber, time_limit);

  result.seconds    = since(t0);
  result.iterations = climber->iterations;
}

std::vector<f_t> uncrush_assignment(mip::problem_t<i_t, f_t>& problem,
                                    const std::vector<f_t>& assignment,
                                    rmm::cuda_stream_view stream)
{
  rmm::device_uvector<f_t> d_assignment(assignment.size(), stream);
  raft::copy(d_assignment.data(), assignment.data(), assignment.size(), stream);
  problem.post_process_assignment(d_assignment, true, stream);
  if (problem.has_papilo_presolve_data()) {
    problem.papilo_uncrush_assignment(d_assignment, stream);
  }
  return cuopt::host_copy(d_assignment, stream);
}

bool write_lane_solutions(
  const std::string& dir,
  const std::string& instance,
  const std::vector<std::string>& var_names,
  const std::vector<std::unique_ptr<mip::fj_cpu_climber_t<i_t, f_t>>>& climbers,
  const std::vector<climber_result_t>& results,
  mip::problem_t<i_t, f_t>& problem,
  rmm::cuda_stream_view stream,
  int& written)
{
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) { return false; }

  const std::string stem = std::filesystem::path(instance).stem().string();
  for (size_t k = 0; k < climbers.size(); ++k) {
    const auto& c = *climbers[k];
    if (!c.feasible_found) { continue; }

    const std::vector<f_t> solver(c.h_best_assignment.data(),
                                  c.h_best_assignment.data() + c.h_best_assignment.size());
    const std::vector<f_t> user = uncrush_assignment(problem, solver, stream);
    if (user.size() != var_names.size()) { return false; }

    const std::string file =
      (std::filesystem::path(dir) / (stem + ".lane" + std::to_string(k) + ".sol")).string();
    cuopt::mathematical_optimization::solution_writer_t::write_solution_to_sol_file<f_t>(
      file,
      "Feasible",
      problem.get_user_obj_from_solver_obj(results[k].best_objective),
      var_names,
      user);
    ++written;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  argparse::ArgumentParser program("solve_CPUFJ");

  program.add_argument("instance").help("input .mps path");

  program.add_argument("time_limit")
    .help("time limit in seconds")
    .scan<'g', double>()
    .default_value(60.0)
    .nargs(argparse::nargs_pattern::optional);

  program.add_argument("climbers")
    .help("number of climbers")
    .scan<'i', int>()
    .default_value(16)
    .nargs(argparse::nargs_pattern::optional);

  program.add_argument("seed")
    .help("base seed")
    .scan<'i', unsigned>()
    .default_value(12345u)
    .nargs(argparse::nargs_pattern::optional);

  program.add_argument("--sol-dir")
    .help("directory to write one .sol per feasible lane into")
    .default_value(std::string(""));

  program.add_argument("--no-related-vars")
    .help("skip the related-variable structure build in trivial presolve")
    .flag();

  program.add_argument("--low-latency")
    .help("skip bound propagation and the binary fast path scan")
    .flag();

  program.add_argument("--probing")
    .help("compute the probing cache before the solve window and hand it to the portfolio")
    .flag();

  program.add_argument("--probing-time-limit")
    .help("wall clock cap for the probing cache, in seconds; not charged to the solve window")
    .scan<'g', double>()
    .default_value(60.0);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 2;
  }

  const std::string path        = program.get<std::string>("instance");
  const f_t time_limit          = program.get<double>("time_limit");
  const int n_climbers          = program.get<int>("climbers");
  const unsigned base_seed      = program.get<unsigned>("seed");
  const bool no_related_vars    = program.get<bool>("--no-related-vars");
  const std::string sol_dir     = program.get<std::string>("--sol-dir");
  const bool low_latency        = program.get<bool>("--low-latency");
  const bool run_probing        = program.get<bool>("--probing");
  const double probing_time     = program.get<double>("--probing-time-limit");

  // Console sink so the engine's end-of-solve incumbent audit is visible, as solve_MIP does it.
  cuopt::init_logger_t log_guard("", true);

  raft::handle_t handle;

  const auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<i_t, f_t>(path, false);
  const auto op_problem =
    cuopt::mathematical_optimization::mps_data_model_to_optimization_problem<i_t, f_t>(
      &handle, mps_data_model);
  mip::problem_t<i_t, f_t> problem(op_problem);

  // Anonymise the instance before anything under evolution can see it.
  //
  // problem_t exposes var_names, row_names and objective_name as public members, and
  // the FJ code receives problem_t&. For a fixed benchmark set those strings are an
  // exact fingerprint -- row_names[0] alone identifies most MIPLIB instances -- so a
  // candidate could branch on identity and return a memorised objective. Reading the
  // MODEL is intended and useful: coefficients, bounds, variable types, sparsity and
  // row structure are all untouched here, so recognising set-packing rows, knapsack
  // substructure or GUB constraints still works exactly as before. Only the labels go.
  //
  // Each string is cleared in place rather than the vectors being emptied, so size()
  // and indexing stay valid and any code that walks names by variable index still
  // works -- it just gets empty strings.
  //
  // This file is outside target_code and is sha256-gated by evaluate.py's FROZEN_FILES,
  // so a candidate cannot restore the names. Do not move this below the solve.
  for (auto& name : problem.var_names)
    name.clear();
  for (auto& name : problem.row_names)
    name.clear();
  problem.objective_name.clear();

  // The same pair solve.cu runs before handing the model to the heuristics. Without it the harness
  // solves a model the production path never produces: trivial presolve is what drops explicit zero
  // coefficients, and every consumer downstream is entitled to assume a nonzero is nonzero.
  // trivial_presolve requires preprocess_problem first and says so.
  problem.preprocess_problem();
  mip::trivial_presolve(
    problem, /*remap_cache_ids=*/true, /*compute_related_vars=*/!no_related_vars);

  std::unique_ptr<mip::mip_solver_context_t<i_t, f_t>> probing_context;
  std::unique_ptr<mip::bound_presolve_t<i_t, f_t>> probing_presolve;
  if (run_probing) {
    cuopt::mathematical_optimization::mip_solver_settings_t<i_t, f_t> probing_settings;
    probing_settings.seed = (i_t)base_seed;
    probing_context =
      std::make_unique<mip::mip_solver_context_t<i_t, f_t>>(&handle, &problem, probing_settings);
    probing_presolve = std::make_unique<mip::bound_presolve_t<i_t, f_t>>(*probing_context);

    const auto probing_features = mip::probing_presolve_features(problem);
    const auto probing_budget =
      mip::evaluate_presolve_budget(probing_settings.heuristic_params, probing_features);
    const int probing_threads            = std::max(1, (int)allowed_cpus().size());
    probing_presolve->settings.num_tasks = std::max(1, probing_threads - 1);

    const auto probing_t0             = clk::now();
    bool infeasible                   = false;
    const int saved_max_active_levels = omp_get_max_active_levels();
    if (saved_max_active_levels < 2) { omp_set_max_active_levels(2); }
#pragma omp parallel num_threads(probing_threads)
    {
#pragma omp masked
      {
        infeasible = mip::compute_probing_cache(*probing_presolve,
                                                problem,
                                                cuopt::timer_t{probing_time},
                                                probing_budget.probing_work_limit,
                                                (size_t)probing_budget.probing_step_size);
      }
    }
    if (saved_max_active_levels < 2) { omp_set_max_active_levels(saved_max_active_levels); }
    handle.sync_stream();

    if (infeasible) {
      std::printf("probing: problem proved infeasible\n");
      return 1;
    }
    mip::trivial_presolve(
      problem, /*remap_cache_ids=*/true, /*compute_related_vars=*/!no_related_vars);
    std::printf("probing: %zu cached vars  %.3fs  threads=%d  work_limit=%.3f  step=%d\n",
                probing_presolve->probing_cache.probing_cache.size(),
                since(probing_t0),
                probing_threads,
                probing_budget.probing_work_limit,
                probing_budget.probing_step_size);
  }

  std::printf("instance: %s  n_vars=%d n_cstrs=%d nnz=%d\n",
              path.c_str(),
              problem.n_variables,
              problem.n_constraints,
              problem.nnz);

  // Taken from the host-side parse, so it is independent of everything under target_code.
  {
    const auto& col_indices = mps_data_model.get_constraint_matrix_indices();
    const auto& row_lb      = mps_data_model.get_constraint_lower_bounds();
    const auto& row_ub      = mps_data_model.get_constraint_upper_bounds();
    const int64_t nnz       = (int64_t)col_indices.size();

    const i_t n_cols = mps_data_model.get_n_variables();
    std::vector<i_t> degree(n_cols, 0);
    for (i_t index : col_indices) {
      if (index >= 0 && index < n_cols) ++degree[index];
    }
    std::sort(degree.begin(), degree.end());

    const i_t max_degree = degree.empty() ? 0 : degree.back();
    auto quantile        = [&](double q) {
      return degree.empty()
                      ? 0
                      : degree[std::min<size_t>(degree.size() - 1, (size_t)(q * degree.size()))];
    };
    int64_t top10 = 0;
    for (size_t k = 0; k < 10 && k < degree.size(); ++k)
      top10 += degree[degree.size() - 1 - k];
    const double mean_degree = n_cols > 0 ? (double)nnz / n_cols : 0.0;
    std::printf(
      "census cols: n=%d degree max=%d p99=%d p90=%d median=%d mean=%.1f"
      "  widest=%.1f%% top10=%.1f%% of nnz  hub=%.0fx mean\n",
      n_cols,
      max_degree,
      quantile(0.99),
      quantile(0.90),
      quantile(0.50),
      mean_degree,
      nnz > 0 ? 100.0 * max_degree / nnz : 0.0,
      nnz > 0 ? 100.0 * top10 / nnz : 0.0,
      mean_degree > 0 ? max_degree / mean_degree : 0.0);

    const i_t n_rows = (i_t)std::min(row_lb.size(), row_ub.size());
    i_t lb_only = 0, ub_only = 0, equality = 0, ranged = 0, free_rows = 0;
    for (i_t r = 0; r < n_rows; ++r) {
      const bool has_lb = std::isfinite((double)row_lb[r]);
      const bool has_ub = std::isfinite((double)row_ub[r]);
      if (has_lb && has_ub) {
        ++(row_lb[r] == row_ub[r] ? equality : ranged);
      } else if (has_lb) {
        ++lb_only;
      } else if (has_ub) {
        ++ub_only;
      } else {
        ++free_rows;
      }
    }
    std::printf(
      "census rows: n=%d lb_only=%d ub_only=%d equality=%d ranged=%d free=%d"
      "  one_sided=%.1f%%\n",
      n_rows,
      lb_only,
      ub_only,
      equality,
      ranged,
      free_rows,
      n_rows > 0 ? 100.0 * (lb_only + ub_only) / n_rows : 0.0);
  }

  // FROZEN -- defines t=0 for the benchmark. Everything above it (the MPS parse,
  // problem construction under problem/, preprocess and trivial presolve, and the
  // name anonymisation) is outside target_code; everything below it is editable. A
  // marker any later would leave editable code ahead of the clock, which is somewhere
  // to do unmeasured work; any earlier would charge the budget for a parse and a CUDA
  // context no candidate can influence.
  CUOPT_LOG_INFO("CPUFJ solve window start");

  std::vector<std::atomic<bool>> preemption_flags(n_climbers);
  std::vector<std::unique_ptr<mip::fj_cpu_climber_t<i_t, f_t>>> climbers(n_climbers);
  mip::build_climber_portfolio<i_t, f_t>(
    problem, preemption_flags, climbers, base_seed, low_latency);
  if (probing_presolve != nullptr) {
    for (int k = 0; k < n_climbers; ++k)
      const_cast<mip::fj_cpu_problem_t<i_t, f_t>*>(climbers[k]->problem.get())->probing_cache =
        &probing_presolve->probing_cache;
  }
  for (int k = 0; k < n_climbers; ++k) {
    climbers[k]->log_prefix = "[climber " + std::to_string(k) + "] ";
  }

  const std::vector<int> cpus = allowed_cpus();
  std::printf("running %d climbers x %.0fs, base seed %u, %zu allowed CPUs (%d..%d)\n",
              n_climbers,
              (double)time_limit,
              base_seed,
              cpus.size(),
              cpus.front(),
              cpus.back());

  std::vector<climber_result_t> results(n_climbers);
  std::vector<std::thread> threads;
  threads.reserve(n_climbers);
  const auto wall0 = clk::now();
  for (int k = 0; k < n_climbers; ++k) {
    threads.emplace_back(
      run_climber, climbers[k].get(), time_limit, cpus[k % cpus.size()], std::ref(results[k]));
  }
  for (auto& t : threads) {
    t.join();
  }
  const double wall = since(wall0);

  int crossed      = 0;
  double sum_iters = 0;
  f_t best_overall = std::numeric_limits<f_t>::infinity();
  std::printf("\n climber | crossed | t_first(s) |          obj |    iters |  iters/s\n");
  std::printf("---------+---------+------------+--------------+----------+---------\n");
  for (int k = 0; k < n_climbers; ++k) {
    const auto& r = results[k];
    sum_iters += r.iterations;
    if (r.crossed) {
      ++crossed;
      best_overall = std::min(best_overall, r.best_objective);
    }
    std::printf(" %7d | %7s | %10s | %12.6g | %8d | %8.0f\n",
                k,
                r.crossed ? "YES" : "no",
                r.crossed ? std::to_string(r.t_first).c_str() : "-",
                r.crossed ? (double)problem.get_user_obj_from_solver_obj(r.best_objective) : 0.0,
                r.iterations,
                r.seconds > 0 ? r.iterations / r.seconds : 0.0);
  }
  // Runs after the measured window closes, so its cost is off the clock.
  // Solver space is always a minimisation, so beating the best known is always a smaller value.
  const auto bks_user = cuopt_bench::lookup_miplib_bks(path);
  const double bks = bks_user ? (double)problem.get_solver_obj_from_user_obj((f_t)*bks_user) : 0.0;
  const double bks_slack = std::max(1e-6, std::fabs(bks) * 1e-9);

  int audited = 0, invalid = 0;
  int64_t escalated_rows = 0;
  std::printf(
    "\n climber | viol rows  worst/tol | bnd viol  worst/tol | int viol  worst/tol |"
    "     obj drift    rel |    vs bks\n");
  std::printf(
    "---------+----------------------+---------------------+---------------------+"
    "----------------------+----------\n");
  for (int k = 0; k < n_climbers; ++k) {
    auto& c = *climbers[k];
    if (c.feasible_found != results[k].crossed) {
      std::printf(" %7d | feasible_found=%d disagrees with a reported incumbent=%d\n",
                  k,
                  (int)c.feasible_found,
                  (int)results[k].crossed);
      ++invalid;
      continue;
    }
    if (!c.feasible_found) continue;
    ++audited;

    const auto& cpu_problem = *c.problem;
    const double int_tol    = cpu_problem.tolerances.integrality_tolerance;

    i_t rows_over          = 0;
    i_t rows_exact         = 0;
    double worst_row_ratio = 0.0;
    // Both counters and the max are order-independent, so any schedule leaves the verdict
    // identical. Per-row cost tracks the row's nnz, which is skewed, hence guided.
#pragma omp parallel for num_threads(n_climbers) schedule(guided) \
  reduction(+ : rows_over, rows_exact) reduction(max : worst_row_ratio)
    for (i_t r = 0; r < cpu_problem.n_constraints; ++r) {
      const i_t begin = cpu_problem.offsets[r];
      const i_t end   = cpu_problem.offsets[r + 1];
      const f_t lb    = cpu_problem.cstr_lb[r];
      const f_t ub    = cpu_problem.cstr_ub[r];

      const double row_tol =
        mip::get_cstr_tolerance<i_t, f_t>(lb,
                                          ub,
                                          cpu_problem.tolerances.absolute_tolerance,
                                          cpu_problem.tolerances.relative_tolerance);
      const double tol = std::max(row_tol, 1e-12);

      // Naive summation over w products: each product carries eps/2 and each of the w-1 additions
      // carries eps, both against the running magnitude, so the row's error is within
      // (w + 1) * eps * abs_sum. A verdict farther from the tolerance than that cannot flip.
      // Only rows the double pass cannot place on one side of the tolerance pay for _Float128,
      // which is soft-float on x86-64. Includes rows whose double excess is zero but whose error
      // bound alone exceeds the tolerance: a real violation can hide there.
      const auto verdict = check_row(
        cpu_problem.coefficients.data(),
        cpu_problem.variables.data(),
        (int64_t)begin,
        (int64_t)end,
        c.h_best_assignment.data(),
        (double)lb,
        (double)ub,
        [&](double) { return std::pair<double, double>{(double)lb - tol, (double)ub + tol}; });
      if (verdict.escalated) ++rows_exact;
      if (verdict.raw_excess <= 0.0) continue;

      const double ratio =
        tol > 0 ? verdict.raw_excess / tol : std::numeric_limits<double>::infinity();
      if (ratio > 1.0) ++rows_over;
      worst_row_ratio = std::max(worst_row_ratio, ratio);
    }
    escalated_rows += rows_exact;

    i_t bounds_over            = 0;
    i_t integers_over          = 0;
    double worst_bound_ratio   = 0.0;
    double worst_integer_ratio = 0.0;
    _Float128 objective        = 0;
    for (i_t v = 0; v < cpu_problem.n_variables; ++v) {
      const auto bounds = c.h_var_bounds[v].get();
      const double x    = (double)c.h_best_assignment[v];
      const double out  = std::max(
        std::max((double)cuopt::get_lower(bounds) - x, x - (double)cuopt::get_upper(bounds)), 0.0);
      if (out > int_tol) ++bounds_over;
      worst_bound_ratio = std::max(worst_bound_ratio, int_tol > 0 ? out / int_tol : 0.0);

      if (cpu_problem.h_var_types[v] == cuopt::mathematical_optimization::var_t::INTEGER) {
        const double residual = std::fabs(x - std::round(x));
        if (residual > int_tol) ++integers_over;
        worst_integer_ratio = std::max(worst_integer_ratio, int_tol > 0 ? residual / int_tol : 0.0);
      }
      const double coefficient = cpu_problem.h_obj_coeffs[v];
      objective += (_Float128)coefficient * (_Float128)x;
    }

    // Differenced before narrowing; the drift is smaller than a double ulp of the sum.
    const _Float128 difference = objective - (_Float128)results[k].best_objective;
    const double drift         = (double)(difference < 0 ? -difference : difference);
    const double exact         = (double)objective;
    const double scale         = std::max(std::fabs(exact), 1.0);
    const bool below_bks       = bks_user && exact < bks - bks_slack;
    const bool bad             = rows_over > 0 || bounds_over > 0 || integers_over > 0 || below_bks;
    if (bad) ++invalid;
    std::printf(" %7d | %9d %10.3g | %8d %10.3g | %8d %10.3g | %12.3g %6.1e | %9.3g%s%s\n",
                k,
                rows_over,
                worst_row_ratio,
                bounds_over,
                worst_bound_ratio,
                integers_over,
                worst_integer_ratio,
                drift,
                drift / scale,
                bks_user ? exact - bks : 0.0,
                below_bks ? "  BELOW BKS" : "",
                bad ? "  INVALID" : "");
  }
  std::printf(
    "AUDIT: %d/%d reporting climbers checked, %d invalid, %lld rows re-summed exactly,"
    " bks %s\n",
    audited,
    crossed,
    invalid,
    (long long)escalated_rows,
    bks_user ? std::to_string(*bks_user).c_str()
             : (cuopt_bench::is_known_infeasible(path) ? "known infeasible" : "unknown"));

  // checked the uncrushed solution against the original model
  {
    const auto& A_val     = mps_data_model.get_constraint_matrix_values();
    const auto& A_idx     = mps_data_model.get_constraint_matrix_indices();
    const auto& A_off     = mps_data_model.get_constraint_matrix_offsets();
    const auto& row_lb    = mps_data_model.get_constraint_lower_bounds();
    const auto& row_ub    = mps_data_model.get_constraint_upper_bounds();
    const auto& col_lb    = mps_data_model.get_variable_lower_bounds();
    const auto& col_ub    = mps_data_model.get_variable_upper_bounds();
    const auto& v_type    = mps_data_model.get_variable_types();
    const i_t n_orig_rows = (i_t)A_off.size() - 1;
    const double abs_tol  = problem.tolerances.absolute_tolerance;
    const double int_tol  = problem.tolerances.integrality_tolerance;

    int lifted_checked = 0, lifted_bad = 0;
    for (int k = 0; k < n_climbers; ++k) {
      const auto& c = *climbers[k];
      if (!c.feasible_found) continue;
      const std::vector<f_t> solver(c.h_best_assignment.data(),
                                    c.h_best_assignment.data() + c.h_best_assignment.size());
      const std::vector<f_t> user = uncrush_assignment(problem, solver, handle.get_stream());
      if ((i_t)user.size() != (i_t)col_lb.size()) {
        std::printf("LIFTED AUDIT: climber %d produced %d values for %d original columns\n",
                    k,
                    (int)user.size(),
                    (int)col_lb.size());
        ++lifted_bad;
        continue;
      }
      ++lifted_checked;

      i_t bad_rows = 0, bad_bnd = 0, bad_int = 0;
      double worst_row = 0.0;
      i_t worst_row_id = -1;
      for (i_t r = 0; r < n_orig_rows; ++r) {
        const double lb = (double)row_lb[r];
        const double ub = (double)row_ub[r];
        const auto verdict =
          check_row(A_val.data(),
                    A_idx.data(),
                    (int64_t)A_off[r],
                    (int64_t)A_off[r + 1],
                    user.data(),
                    lb,
                    ub,
                    [&](double positive) { return scaled_row_limits(abs_tol, positive, lb, ub); });
        if (verdict.excess > 0.0) {
          ++bad_rows;
          if (verdict.excess > worst_row) {
            worst_row    = verdict.excess;
            worst_row_id = r;
          }
        }
      }
      for (i_t v = 0; v < (i_t)user.size(); ++v) {
        const double x = (double)user[v];
        if (x < (double)col_lb[v] - int_tol || x > (double)col_ub[v] + int_tol) ++bad_bnd;
        if ((v_type[v] == 'I' || v_type[v] == 'B') && std::fabs(x - std::round(x)) > int_tol)
          ++bad_int;
      }
      if (bad_rows || bad_bnd || bad_int) {
        ++lifted_bad;
        std::printf(
          "LIFTED AUDIT: climber %d INVALID on the original model -- %d rows, %d bounds,"
          " %d integrality; worst row %d by %.6g\n",
          k,
          (int)bad_rows,
          (int)bad_bnd,
          (int)bad_int,
          (int)worst_row_id,
          worst_row);
      }
    }
    std::printf(
      "LIFTED AUDIT: %d/%d crossing climbers verified against the original model,"
      " %d INVALID, %d original rows\n",
      lifted_checked,
      crossed,
      lifted_bad,
      (int)n_orig_rows);
  }

  std::printf(
    "\n climber |     moves |  apply nnz | nnz/move | bitmap elems | ratio |"
    " bump/apply | bump/weight | mtm inval | cache hit%%\n");
  std::printf(
    "---------+-----------+------------+----------+--------------+-------+"
    "------------+-------------+-----------+-----------\n");
  for (int k = 0; k < n_climbers; ++k) {
    const auto& c        = *climbers[k];
    const int64_t bitmap = 2 * c.n_moves_applied * (int64_t)c.problem->n_variables;
    const int64_t probes = c.hit_count + c.miss_count;
    std::printf(
      " %7d | %9lld | %10lld | %8.1f | %12lld | %5.0f | %10lld | %11lld | %9lld |"
      " %9.2f\n",
      k,
      (long long)c.n_moves_applied,
      (long long)c.apply_move_nnz,
      c.n_moves_applied > 0 ? (double)c.apply_move_nnz / c.n_moves_applied : 0.0,
      (long long)bitmap,
      c.apply_move_nnz > 0 ? (double)bitmap / c.apply_move_nnz : 0.0,
      (long long)c.n_version_bumps_apply,
      (long long)c.n_version_bumps_weights,
      (long long)c.n_mtm_cache_invalidations,
      probes > 0 ? 100.0 * c.hit_count / probes : 0.0);
  }

  std::printf(
    "\n climber | mtm calls | row entries | ent/call |  capped ent | capped/call |"
    " score calls | score nnz | nnz/score | nnz budget\n");
  std::printf(
    "---------+-----------+-------------+----------+-------------+-------------+"
    "-------------+-----------+-----------+-----------\n");
  for (int k = 0; k < n_climbers; ++k) {
    const auto& c = *climbers[k];
    std::printf(
      " %7d | %9lld | %11lld | %8.0f | %11lld | %11.0f | %11lld | %9lld | %9.1f |"
      " %10d\n",
      k,
      (long long)c.n_mtm_calls,
      (long long)c.mtm_row_entries,
      c.n_mtm_calls > 0 ? (double)c.mtm_row_entries / c.n_mtm_calls : 0.0,
      (long long)c.mtm_entries_capped,
      c.n_mtm_calls > 0 ? (double)c.mtm_entries_capped / c.n_mtm_calls : 0.0,
      (long long)c.n_compute_score_calls,
      (long long)c.compute_score_nnz,
      c.n_compute_score_calls > 0 ? (double)c.compute_score_nnz / c.n_compute_score_calls : 0.0,
      c.nnz_samples);
  }

  std::printf(
    "\n climber | refresh period | lhs total | periodic | bigval | perturb | restart |"
    " epi vars | epi projections\n");
  std::printf(
    "---------+----------------+-----------+----------+--------+---------+---------+"
    "----------+----------------\n");
  for (int k = 0; k < n_climbers; ++k) {
    const auto& c = *climbers[k];
    std::printf(" %7d | %14d | %9lld | %8lld | %6lld | %7lld | %7lld | %8zu | %15lld\n",
                k,
                c.lhs_refresh_period_used,
                (long long)c.n_lhs_recompute_total,
                (long long)c.n_lhs_recompute_periodic,
                (long long)c.n_lhs_recompute_bigval,
                (long long)c.n_lhs_recompute_perturb,
                (long long)c.n_lhs_recompute_restart,
                c.epigraph_vars.size(),
                (long long)c.n_epigraph_projections);
  }

  // Everything a climber spends outside the search loop. lp solve is the simplex share of the LP
  // start, so it is shown for attribution and left out of the total. A phase a lane does not run
  // reads 0.
  std::printf(
    "\n climber |    start | bnd prop | lp start | (lp solve) | colouring | features |"
    " init lhs | bin setup |    total\n");
  std::printf(
    "---------+----------+----------+----------+------------+-----------+----------+"
    "----------+-----------+---------\n");
  for (int k = 0; k < n_climbers; ++k) {
    const auto& c      = *climbers[k];
    const double total = c.t_start + c.t_bound_prop + c.t_lp_start + c.t_coloring + c.t_features +
                         c.t_init_lhs + c.bin_setup.total();
    std::printf(" %7d | %8.4f | %8.4f | %8.4f | %10.4f | %9.4f | %8.4f | %8.4f | %9.4f | %8.4f\n",
                k,
                c.t_start,
                c.t_bound_prop,
                c.t_lp_start,
                c.t_lp_relaxation,
                c.t_coloring,
                c.t_features,
                c.t_init_lhs,
                c.bin_setup.total(),
                total);
  }

  // The bin setup column above, by phase. Charged even when the fast path declines, so a scan that
  // only produces a rejection still shows. narrow and transpose are the all-binary path; encode is
  // the general-integer one and runs twice when int8 is enough.
  std::printf(
    "\n climber | bin scan | bin narrow | transpose | bin encode | engine init | bin total\n");
  std::printf(
    "---------+----------+------------+-----------+------------+-------------+----------\n");
  for (int k = 0; k < n_climbers; ++k) {
    const auto& b = climbers[k]->bin_setup;
    std::printf(" %7d | %8.4f | %10.4f | %9.4f | %10.4f | %11.4f | %9.4f\n",
                k,
                b.scan,
                b.narrow,
                b.transpose,
                b.encode,
                b.engine_init,
                b.total());
  }

  std::printf("\nSUMMARY: %d/%d crossed (%.0f%%)  wall=%.1fs  total_iters=%.0f  agg_iters/s=%.0f\n",
              crossed,
              n_climbers,
              100.0 * crossed / n_climbers,
              wall,
              sum_iters,
              wall > 0 ? sum_iters / wall : 0.0);
  if (crossed > 0) {
    std::printf("BEST OBJECTIVE: %.10g\n",
                (double)problem.get_user_obj_from_solver_obj(best_overall));
  }

  if (!sol_dir.empty()) {
    int written   = 0;
    const bool ok = write_lane_solutions(sol_dir,
                                         path,
                                         mps_data_model.get_variable_names(),
                                         climbers,
                                         results,
                                         problem,
                                         handle.get_stream(),
                                         written);
    std::printf("SOLUTIONS: %d/%d lanes -> %s (%s)\n",
                written,
                n_climbers,
                sol_dir.c_str(),
                ok ? "written" : "WRITE FAILED");
  }

  return 0;
}
