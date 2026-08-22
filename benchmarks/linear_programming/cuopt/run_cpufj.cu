/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "miplib2017_bks.hpp"

#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/utils.cuh>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

using i_t = int;
using f_t = double;
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

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <instance.mps> [time_limit_s=60] [climbers=16] [seed=12345]\n",
                 argv[0]);
    return 2;
  }
  const std::string path   = argv[1];
  const f_t time_limit     = argc > 2 ? std::atof(argv[2]) : 60.0;
  const int n_climbers     = argc > 3 ? std::atoi(argv[3]) : 16;
  const unsigned base_seed = argc > 4 ? (unsigned)std::atoll(argv[4]) : 12345u;

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
  for (auto& name : problem.var_names) name.clear();
  for (auto& name : problem.row_names) name.clear();
  problem.objective_name.clear();

  std::printf("instance: %s  n_vars=%d n_cstrs=%d nnz=%d\n",
              path.c_str(),
              problem.n_variables,
              problem.n_constraints,
              problem.nnz);

  // FROZEN -- defines t=0 for the benchmark. Everything above it (the MPS parse,
  // problem construction under problem/, and the name anonymisation) is outside
  // target_code; everything below it is editable. A marker any later would leave
  // editable code ahead of the clock, which is somewhere to do unmeasured work; any
  // earlier would charge the budget for a parse and a CUDA context no candidate can
  // influence.
  CUOPT_LOG_INFO("CPUFJ solve window start");

  // Shared by every climber. Built by build_start_assignment, which is editable --
  // this driver is not.
  mip::solution_t<i_t, f_t> solution(problem);
  mip::build_start_assignment<i_t, f_t>(problem, solution, &handle);

  std::vector<std::atomic<bool>> preemption_flags(n_climbers);
  std::vector<std::unique_ptr<mip::fj_cpu_climber_t<i_t, f_t>>> climbers(n_climbers);
  // Composition and per-climber parameters come from build_climber_portfolio, which
  // is editable. The log prefix is assigned here and not there, so every climber
  // stays identifiable in the log whatever the portfolio does.
  mip::build_climber_portfolio<i_t, f_t>(problem, solution, preemption_flags, climbers, base_seed);
  for (int k = 0; k < n_climbers; ++k) {
    climbers[k]->log_prefix = "[climber " + std::to_string(k) + "] ";
  }

  const std::vector<int> cpus = allowed_cpus();
  std::printf("running %d climbers x %.0fs, base seed %u, %zu allowed CPUs (%d..%d)\n",
              n_climbers, (double)time_limit, base_seed, cpus.size(), cpus.front(), cpus.back());

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

  int crossed       = 0;
  double sum_iters  = 0;
  f_t best_overall  = std::numeric_limits<f_t>::infinity();
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
                r.crossed ? (double)r.best_objective : 0.0,
                r.iterations,
                r.seconds > 0 ? r.iterations / r.seconds : 0.0);
  }
  // Runs after the measured window closes, so its cost is off the clock.
  // Solver space is always a minimisation, so beating the best known is always a smaller value.
  const auto bks_user = cuopt_bench::lookup_miplib_bks(path);
  const double bks = bks_user ? (double)problem.get_solver_obj_from_user_obj((f_t)*bks_user) : 0.0;
  const double bks_slack = std::max(1e-6, std::fabs(bks) * 1e-9);

  int audited = 0, invalid = 0;
  std::printf("\n climber | viol rows  worst/tol | bnd viol  worst/tol | int viol  worst/tol |"
              "     obj drift    rel |    vs bks\n");
  std::printf("---------+----------------------+---------------------+---------------------+"
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

    const double int_tol = c.view.pb.tolerances.integrality_tolerance;

    i_t rows_over          = 0;
    double worst_row_ratio = 0.0;
    for (i_t r = 0; r < c.view.pb.n_constraints; ++r) {
      __float128 activity = 0;
      for (i_t j = c.h_offsets[r]; j < c.h_offsets[r + 1]; ++j) {
        const i_t var            = c.h_variables[j];
        const double coefficient = c.h_coefficients[j];
        const double value       = c.h_best_assignment[var];
        activity += (__float128)coefficient * (__float128)value;
      }

      const f_t lb           = c.h_cstr_lb[r];
      const f_t ub           = c.h_cstr_ub[r];
      const __float128 below = (__float128)lb - activity;
      const __float128 above = activity - (__float128)ub;
      const double excess    = (double)std::max(std::max(below, above), (__float128)0);
      if (excess <= 0.0) continue;

      const double tol   = c.view.get_corrected_tolerance(r, lb, ub);
      const double ratio = tol > 0 ? excess / tol : std::numeric_limits<double>::infinity();
      if (ratio > 1.0) ++rows_over;
      worst_row_ratio = std::max(worst_row_ratio, ratio);
    }

    i_t bounds_over            = 0;
    i_t integers_over          = 0;
    double worst_bound_ratio   = 0.0;
    double worst_integer_ratio = 0.0;
    __float128 objective       = 0;
    for (i_t v = 0; v < c.view.pb.n_variables; ++v) {
      auto bounds      = c.h_var_bounds[v].get();
      const double x   = (double)c.h_best_assignment[v];
      const double out = std::max(
        std::max((double)cuopt::get_lower(bounds) - x, x - (double)cuopt::get_upper(bounds)), 0.0);
      if (out > int_tol) ++bounds_over;
      worst_bound_ratio = std::max(worst_bound_ratio, int_tol > 0 ? out / int_tol : 0.0);

      if (c.view.pb.is_integer_var(v)) {
        const double residual = std::fabs(x - std::round(x));
        if (residual > int_tol) ++integers_over;
        worst_integer_ratio = std::max(worst_integer_ratio, int_tol > 0 ? residual / int_tol : 0.0);
      }
      const double coefficient = c.h_obj_coeffs[v];
      objective += (__float128)coefficient * (__float128)x;
    }

    // Differenced before narrowing; the drift is smaller than a double ulp of the sum.
    const __float128 difference = objective - (__float128)results[k].best_objective;
    const double drift          = (double)(difference < 0 ? -difference : difference);
    const double exact          = (double)objective;
    const double scale          = std::max(std::fabs(exact), 1.0);
    const bool below_bks        = bks_user && exact < bks - bks_slack;
    const bool bad = rows_over > 0 || bounds_over > 0 || integers_over > 0 || below_bks;
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
  std::printf("AUDIT: %d/%d reporting climbers checked, %d invalid, bks %s\n",
              audited,
              crossed,
              invalid,
              bks_user ? std::to_string(*bks_user).c_str()
                       : (cuopt_bench::is_known_infeasible(path) ? "known infeasible" : "unknown"));

  std::printf("\nSUMMARY: %d/%d crossed (%.0f%%)  wall=%.1fs  total_iters=%.0f  agg_iters/s=%.0f\n",
              crossed,
              n_climbers,
              100.0 * crossed / n_climbers,
              wall,
              sum_iters,
              wall > 0 ? sum_iters / wall : 0.0);
  if (crossed > 0) { std::printf("BEST OBJECTIVE: %.10g\n", (double)best_overall); }
  return 0;
}
