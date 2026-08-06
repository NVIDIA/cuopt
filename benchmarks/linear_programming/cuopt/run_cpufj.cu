/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Benchmark-only harness for the CPU feasibility-jump portfolio. Loads an instance, builds one
// climber per portfolio slot from a zero start clamped to variable bounds, and runs them in
// parallel on pinned cores for a fixed wall-clock budget. Reports per-climber crossing, objective
// and throughput.
//
// No presolve: pass an already-presolved instance. The climbers are built from problem_t, so the
// binary fast path is reachable when the instance qualifies.

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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>
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
  std::printf("instance: %s  n_vars=%d n_cstrs=%d nnz=%d\n",
              path.c_str(),
              problem.n_variables,
              problem.n_constraints,
              problem.nnz);

  // Zero start, clamped into the variable bounds. Shared by every climber; diversity comes from
  // the per-climber seed and sampling parameters below.
  mip::solution_t<i_t, f_t> solution(problem);
  thrust::fill(handle.get_thrust_policy(), solution.assignment.begin(), solution.assignment.end(), f_t{0});
  mip::clamp_within_var_bounds<i_t, f_t>(solution.assignment, &problem, &handle);
  handle.sync_stream();

  // Built serially: each climber host-copies the problem off the same stream.
  std::vector<std::atomic<bool>> preemption_flags(n_climbers);
  std::vector<std::unique_ptr<mip::fj_cpu_climber_t<i_t, f_t>>> climbers(n_climbers);
  for (int k = 0; k < n_climbers; ++k) {
    preemption_flags[k].store(false);
    mip::fj_settings_t settings;
    settings.seed = (int)(base_seed + k);
    climbers[k]   = mip::init_fj_cpu_standalone(problem, solution, preemption_flags[k], settings);

    // Portfolio diversification, decorrelated from the value RNG.
    std::mt19937 rng(base_seed + 7919u * k);
    climbers[k]->mtm_viol_samples = std::uniform_int_distribution<i_t>(15, 50)(rng);
    climbers[k]->mtm_sat_samples  = std::uniform_int_distribution<i_t>(10, 30)(rng);
    climbers[k]->nnz_samples      = std::uniform_int_distribution<i_t>(2000, 15000)(rng);
    climbers[k]->perturb_interval = std::uniform_int_distribution<i_t>(50, 500)(rng);
    climbers[k]->log_prefix       = "[climber " + std::to_string(k) + "] ";
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
