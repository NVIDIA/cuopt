/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "population.cuh"

#include <mip_heuristics/local_search/local_search.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <utilities/timer.hpp>

#include <rmm/device_uvector.hpp>

#include <functional>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// LP-guided CPU fix-and-propagate on the root LP point (fix_propagate_host.hpp does the pass):
// fix the integers in the configured order to the configured values with bounds propagation and
// one-level backtracking, evaluate the pattern with a cost-aware fixed-integer LP, push and repair
// it, and hand feasible points to the population. Built on the diversity manager's state for one
// call (diversity_manager_t::run_root_fix_and_propagate).
template <typename i_t, typename f_t>
class fix_and_propagate_t {
 public:
  fix_and_propagate_t(mip_solver_context_t<i_t, f_t>& context,
                      problem_t<i_t, f_t>* problem_ptr,
                      population_t<i_t, f_t>& population,
                      local_search_t<i_t, f_t>& local_search,
                      cuopt::timer_t& timer,
                      const rmm::device_uvector<f_t>& lp_primal,
                      const rmm::device_uvector<f_t>& lp_dual,
                      std::function<bool()> preempted);
  // one pass with the configured strategies within the configured budget; true when a pattern was
  // repaired and added to the population
  bool run();

 private:
  // feasibility jump repair of a complete but infeasible pattern (integers from the pass,
  // continuous from the LP point): a short fixed-integer LP, then FJ on every variable
  struct fj_repair_t {
    bool feasible{false};
    bool published{false};  // the population took it as its new best
    f_t objective{std::numeric_limits<f_t>::infinity()};
    std::vector<f_t> assignment;
    double time{0};
  };
  fj_repair_t fj_repair_pattern(const std::vector<f_t>& assignment, f_t time_limit);
  // Evaluation of one pattern (full host assignment whose integers are the pattern): the integers
  // are fixed and the LP over the remaining continuous variables is solved in up to three stages,
  // at most 6 LP solves, each stage stopping early when it is done or out of time:
  //   1. Ranking LP, 1 PDLP solve (cost-aware, `tolerance` and `time_limit`; cold with presolve, or
  //      warm-started from warm_primal/warm_dual without presolve). The objective of its point,
  //      integers included, ranks the pattern. If that point cannot beat the incumbent, the
  //      evaluation stops here: the next stages rarely improve the objective, so the pattern could
  //      not be published anyway.
  //   2. Pushes, up to 2 PDLP solves warm-started from stage 1 with a frozen large primal
  //      weight and no restarts (capped by max_push_time). PDLP's point usually still violates
  //      some rows; the pushes cut that violation at a nearly unchanged objective, so stage 3
  //      moves the point less.
  //   3. Repair, up to 3 chunks of a zero-objective LP within repair_time_limit, stopping at the
  //      first feasible point: moves the point to feasibility, ignoring the objective. Runs only if
  //      the point is still infeasible.
  // Stages 2 and 3 run only when `repair` is set and the ranking objective is below repair_below. A
  // feasible point is handed to the population.
  struct pattern_eval_t {
    bool usable{false};
    bool optimal{false};
    f_t lp_objective{std::numeric_limits<f_t>::infinity()};
    f_t l2_primal_residual{0};
    f_t repaired_objective{std::numeric_limits<f_t>::infinity()};
    bool published{false};
    int iterations{0};
    double time{0};
    std::vector<f_t> primal, dual;  // stage-1 (ranking LP) primal/dual
    std::vector<f_t> repaired_assignment;
    f_t pushed_lp_objective{std::numeric_limits<f_t>::infinity()};
    f_t pushed_l2_primal_residual{-1};
    double push_time{0};
    bool cannot_beat_incumbent{false};  // stopped after stage 1 (ranking LP)
  };
  pattern_eval_t eval_fixed_pattern(const std::vector<f_t>& assignment,
                                    f_t tolerance,
                                    f_t time_limit,
                                    const std::vector<f_t>* warm_primal,
                                    const std::vector<f_t>* warm_dual,
                                    bool repair,
                                    f_t repair_time_limit,
                                    f_t repair_below  = std::numeric_limits<f_t>::infinity(),
                                    f_t max_push_time = 11.);
  bool check_b_b_preemption() { return preempted(); }

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  population_t<i_t, f_t>& population;
  local_search_t<i_t, f_t>& local_search;
  cuopt::timer_t& timer;
  const rmm::device_uvector<f_t>& lp_optimal_solution;
  const rmm::device_uvector<f_t>& lp_dual_optimal_solution;
  std::function<bool()> preempted;
};

}  // namespace cuopt::mathematical_optimization::mip
