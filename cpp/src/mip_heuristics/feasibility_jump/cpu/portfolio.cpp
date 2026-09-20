/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "climber.hpp"
#include "internal.hpp"
#include "problem.hpp"
#include "setup/bounds.hpp"
#include "starts/starts.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::fj_cpu_deleter_t::operator()(fj_cpu_climber_t<i_t, f_t>* ptr) const
{
  delete ptr;
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::create_worker(
  const lp_problem_t<i_t, f_t>& problem,
  const std::vector<simplex::variable_type_t>& variable_types,
  i_t n_structural,
  const std::vector<f_t>& start_assignment,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::string log_prefix,
  int64_t seed,
  int lane)
{
  auto new_climber = init_fj_cpu_from_host_lp(
    problem, variable_types, n_structural, start_assignment, settings, preemption_flag, seed);
  fj_cpu.reset(new_climber.release());
  fj_cpu->log_prefix           = std::move(log_prefix);
  fj_cpu->improvement_callback = improvement_callback;
  fj_cpu->halted               = false;
  preemption_flag              = false;
  is_initialized               = true;
  if (lane >= 0) apply_lane_diversification<i_t, f_t>(*fj_cpu, lane, fj_cpu->settings.seed);
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_async(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;

  auto& fj_ptr = fj_cpu;
#pragma omp task shared(fj_cpu, is_initialized, fj_ptr) firstprivate(time_limit, work_unit_limit) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none) depend(out : fj_ptr)
  {
    if (is_initialized) { cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit); }
  }
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_sync(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;
  cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit);
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::stop()
{
  if (!is_initialized) return;

  preemption_flag = true;

  auto& fj_ptr = fj_cpu;
#pragma omp taskwait depend(in : fj_ptr)
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::send_stop_signal()
{
  preemption_flag = true;
}

template <typename i_t, typename f_t>
void apply_lane_diversification(fj_cpu_climber_t<i_t, f_t>& c, int lane, int64_t base_seed)
{
  cuopt_assert(lane >= 0, "CPUFJ lane must be nonnegative");

  const bool extreme_hub =
    c.problem->max_var_degree > 512 && c.problem->max_var_degree > 64.0 * c.problem->avg_var_degree;
  const bool low_rank_integer_equalities =
    c.n_binary_vars == 0 && c.n_integer_vars == c.problem->n_variables &&
    c.problem->equality_fraction > 0.99 &&
    int64_t{32} * c.problem->n_constraints < c.problem->n_variables;

  c.use_lp_start                   = lane == 4 || lane == 10 || lane == 11 || lane == 14;
  c.lp_start_feasibility_objective = lane == 4 || lane == 10 || lane == 11;
  c.use_deep_lp_pump               = lane == 4;
  c.use_integer_bit_encoding       = lane != 0 && lane != 7 && c.n_binary_vars > 0;
  c.use_lp_polish = lane == 3 || lane == 5 || lane == 8 || lane == 9 || lane == 10 || lane == 11 ||
                    lane == 12 || lane == 13 || lane == 14;
  c.use_precedence_start      = lane % 8 == 0 && !c.low_latency;
  c.use_equality_substitution = lane % 4 == 0 && !c.low_latency;
  c.use_bound_prop            = lane % 2 == 0 && !c.low_latency;

  if (lane == 13 && low_rank_integer_equalities) {
    c.use_lp_start                   = true;
    c.lp_start_feasibility_objective = true;
    c.use_deep_lp_pump               = false;
  }

  {
    phase_timer_t timer(c.t_start);
    switch (lane % 8) {
      case 1: apply_structural_completion_start<i_t, f_t>(c); break;
      case 3: apply_greedy_covering_start<i_t, f_t>(c); break;
      case 4:
        if (!c.use_lp_start) {
          apply_ambiguous_lock_start<i_t, f_t>(c);
          apply_greedy_covering_start<i_t, f_t>(c);
        }
        break;
      case 0:
        if (lane == 8) {
          apply_exact_k_start<i_t, f_t>(c);
          apply_greedy_covering_start<i_t, f_t>(c);
        }
        break;
      default: break;
    }
    if (lane == 10) apply_greedy_covering_start<i_t, f_t>(c);
    if (lane == 12 || lane == 15) apply_structural_completion_start<i_t, f_t>(c);
    if (lane == 11) {
      apply_structural_completion_start<i_t, f_t>(c);
      apply_greedy_covering_start<i_t, f_t>(c);
    }
  }

  std::mt19937 rng(base_seed + 7919u * lane);
  c.mtm_viol_samples = std::uniform_int_distribution<i_t>(10, 80)(rng);
  c.mtm_sat_samples  = std::uniform_int_distribution<i_t>(5, 50)(rng);
  c.nnz_samples      = std::uniform_int_distribution<i_t>(1000, 20000)(rng);
  c.perturb_interval = std::uniform_int_distribution<i_t>(10, 2000)(rng);

  static constexpr double smoothing[8] = {0.0003, 0.0, 0.001, 0.003, 0.0001, 0.0006, 0.002, 0.0003};
  static constexpr int tabu_min[8]     = {3, 1, 5, 3, 2, 6, 4, 3};
  static constexpr int tabu_max[8]     = {13, 7, 21, 13, 10, 25, 17, 13};
  const int policy                     = lane % 8;
  c.settings.parameters.weight_smoothing_probability = smoothing[policy];
  c.settings.parameters.tabu_tenure_min              = tabu_min[policy];
  c.settings.parameters.tabu_tenure_max              = tabu_max[policy];

  if (lane == 7 || lane == 8) {
    c.mtm_viol_samples = 192;
    c.mtm_sat_samples  = 64;
    c.nnz_samples      = 100000;
  }
  if (lane == 0) {
    c.mtm_viol_samples = std::uniform_int_distribution<i_t>(40, 100)(rng);
    c.mtm_sat_samples  = std::uniform_int_distribution<i_t>(20, 60)(rng);
    c.nnz_samples      = std::uniform_int_distribution<i_t>(10000, 30000)(rng);
  }
  if (lane == 12) {
    c.mtm_viol_samples = std::uniform_int_distribution<i_t>(50, 120)(rng);
    c.mtm_sat_samples  = std::uniform_int_distribution<i_t>(25, 70)(rng);
  }
  if (lane == 11) {
    c.mtm_viol_samples = std::uniform_int_distribution<i_t>(30, 100)(rng);
    c.mtm_sat_samples  = std::uniform_int_distribution<i_t>(15, 50)(rng);
  }
  if (lane == 9 && extreme_hub) {
    c.mtm_viol_samples = 8;
    c.mtm_sat_samples  = 3;
    c.nnz_samples      = 2000;
  }
  if (lane == 10) {
    c.use_bound_prop                 = false;
    c.use_lp_start                   = true;
    c.lp_start_feasibility_objective = true;
    c.settings.seed += 224737;
  }
  if (extreme_hub && lane == 5) {
    c.use_lp_start = c.use_deep_lp_pump = true;
    c.lp_start_feasibility_objective    = false;
  }

  static constexpr f_t objective_weight[4] = {2, 8, 32, 1};
  i_t continuous_objective_vars            = 0;
  for (i_t var : c.problem->h_objective_vars)
    continuous_objective_vars += !is_integer_var<i_t, f_t>(c, var);
  const int64_t objective_var_count = c.problem->h_objective_vars.size();
  const bool continuous_objective_model =
    objective_var_count > 0 &&
    int64_t{10} * continuous_objective_vars >= int64_t{9} * objective_var_count &&
    int64_t{10} * objective_var_count >= int64_t{c.problem->n_variables};
  c.h_objective_weight    = !continuous_objective_model ? f_t{0}
                            : lane == 3                 ? f_t{4}
                            : lane == 9                 ? f_t{8}
                            : lane == 15                ? f_t{16}
                                                        : f_t{0};
  c.seed_objective_weight = lane == 1    ? f_t{32}
                            : lane == 4  ? f_t{8}
                            : lane == 5  ? f_t{16}
                            : lane == 15 ? f_t{8}
                                         : objective_weight[lane % 4];
}

template <typename i_t, typename f_t>
void complete_climber_portfolio(std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> first_climber,
                                const std::vector<int64_t>& lane_seed,
                                std::vector<std::atomic<bool>>& preemption_flags,
                                std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>>& climbers,
                                int64_t base_seed,
                                bool low_latency)
{
  const int n_climbers = climbers.size();
  cuopt_assert(n_climbers > 0, "a CPUFJ portfolio needs at least one climber");
  cuopt_assert(preemption_flags.size() == climbers.size(), "preemption flag count mismatch");
  cuopt_assert(lane_seed.size() == climbers.size(), "lane seed count mismatch");

  climbers[0] = std::move(first_climber);
  cuopt_assert(climbers[0] != nullptr, "missing first CPUFJ climber");
  climbers[0]->low_latency = low_latency;
  apply_exact_k_start<i_t, f_t>(*climbers[0]);
  repair_difficult_anchor<i_t, f_t>(*climbers[0]);
  apply_lane_diversification<i_t, f_t>(*climbers[0], 0, base_seed);

#ifdef _OPENMP
#pragma omp parallel for num_threads(std::max(1, n_climbers - 1)) schedule(static)
#endif
  for (int k = 1; k < n_climbers; ++k) {
    fj_settings_t settings;
    settings.seed            = lane_seed[k];
    climbers[k]              = init_fj_cpu_clone(*climbers[0], preemption_flags[k], settings);
    climbers[k]->low_latency = low_latency;
    apply_lane_diversification<i_t, f_t>(*climbers[k], k, base_seed);
  }
}

#if MIP_INSTANTIATE_FLOAT
template struct fj_cpu_worker_t<int, float>;
template void apply_lane_diversification<int, float>(fj_cpu_climber_t<int, float>&, int, int64_t);
template void complete_climber_portfolio<int, float>(
  std::unique_ptr<fj_cpu_climber_t<int, float>>,
  const std::vector<int64_t>&,
  std::vector<std::atomic<bool>>&,
  std::vector<std::unique_ptr<fj_cpu_climber_t<int, float>>>&,
  int64_t,
  bool);
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct fj_cpu_worker_t<int, double>;
template void apply_lane_diversification<int, double>(fj_cpu_climber_t<int, double>&, int, int64_t);
template void complete_climber_portfolio<int, double>(
  std::unique_ptr<fj_cpu_climber_t<int, double>>,
  const std::vector<int64_t>&,
  std::vector<std::atomic<bool>>&,
  std::vector<std::unique_ptr<fj_cpu_climber_t<int, double>>>&,
  int64_t,
  bool);
#endif

}  // namespace cuopt::mathematical_optimization::mip
