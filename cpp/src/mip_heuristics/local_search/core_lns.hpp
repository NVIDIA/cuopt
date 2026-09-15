/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <branch_and_bound/worker.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <limits>
#include <list>
#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class branch_and_bound_t;

template <typename i_t, typename f_t>
struct core_lns_params_t {
  // Minimum size of the group of variables associated to a control variable
  // for it to be considered a "core" variable
  i_t min_group_size = 2;

  // Maximum number of "core" variables (absolute)
  i_t max_core_size = 5000;

  // Maximum number of "core" variables in term of the total number of binary variables
  f_t max_binary_fraction = 0.10;

  // Minimum set of variables covered by the core to turn on this heuristic.
  f_t min_coverage = 0.25;

  // How many groups a round releases before any evidence has arrived.
  i_t base_radius = 30;

  // Safety rails only. The controller is expected to settle well inside these.
  i_t min_radius = 10;
  i_t max_radius = 120;

  // How far one round's outcome moves the radius.
  i_t radius_step = 5;

  // The destroy operators run concurrently, one sub-MIP each, with this many threads per solve.
  i_t threads_per_solve = 8;
};

template <typename i_t, typename f_t>
class core_lns_t {
 public:
  core_lns_t(branch_and_bound_t<i_t, f_t>* branch_and_bound, i_t num_threads)
    : branch_and_bound_ptr(branch_and_bound), variable_groups_(0, 0, 0)
  {
    num_threads_used_ =
      std::floor(num_threads / params_.threads_per_solve) * params_.threads_per_solve;
  }

  ~core_lns_t() { stop_and_sync(); }

  bool recognize();

  void run(const simplex::lp_problem_t<i_t, f_t>& lp,
           const csr_matrix_t<i_t, f_t>& Arow,
           const std::vector<simplex::variable_type_t>& var_types,
           const std::vector<f_t>& root_solution,
           const std::vector<f_t>& root_edge_norm,
           pseudo_costs_t<i_t, f_t>& pseudo_costs);

  void stop_and_sync()
  {
    halt.store(true, std::memory_order_release);
    for (auto& worker : workers_) {
      diving_worker_t<i_t, f_t>* worker_ptr = worker.get();
#pragma omp taskwait depend(in : *worker_ptr)
    }
  }

  i_t num_threads_used() { return num_threads_used_; }

 private:
  void search(diving_worker_t<i_t, f_t>* worker);
  void create_workers(i_t num_workers,
                      const simplex::lp_problem_t<i_t, f_t>& lp,
                      const csr_matrix_t<i_t, f_t>& Arow,
                      const std::vector<simplex::variable_type_t>& var_types,
                      const std::vector<f_t>& root_solution,
                      const std::vector<f_t>& root_edge_norm,
                      pseudo_costs_t<i_t, f_t>& pseudo_costs);

  // Fix every group outside `released`, propagate, and hand the residual to solve_submip. The
  // outcome needs no return: solve_submip records it against the radius in submip_stats_, which
  // is where next_radius reads it back from.
  void evaluate(diving_worker_t<i_t, f_t>* worker,
                const std::vector<uint8_t>& value,
                const std::vector<uint8_t>* released,
                submip_stats_t& submip_stats);

  // Choose this round's neighbourhood: `radius` groups drawn uniformly from the core.
  void select_groups_to_release(diving_worker_t<i_t, f_t>* worker,
                                const submip_stats_t& submip_stats,
                                i_t radius,
                                std::vector<uint8_t>& released);

  // Counter values as of the previous round. `submip_stats` is per worker and only this worker
  // writes it, so whichever counter moved since these were taken is the outcome of the round that
  // just finished.
  struct round_counts_t {
    i_t success   = 0;
    i_t exhausted = 0;
    i_t truncated = 0;
  };

  // Step the radius by the last round's outcome: an exhausted round searched the whole
  // neighbourhood and found nothing, so widen; a truncated one ran out of budget before it could,
  // so narrow; a round that improved the incumbent leaves it where it is. `previous` is updated
  // in place.
  i_t next_radius(const submip_stats_t& submip_stats, round_counts_t& previous, i_t radius) const;

  branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr;
  std::atomic<int> halt{false};

  std::vector<std::unique_ptr<diving_worker_t<i_t, f_t>>> workers_;
  i_t num_threads_used_;

  core_lns_params_t<i_t, f_t> params_;

  // The core, one group per row: row k holds the variables of group k, with the group's parity
  // for each of them as the value -- variable `j[p]` takes the group's bit XOR `x[p]`.
  csr_matrix_t<i_t, f_t> variable_groups_;  // m = groups, n = variables

  std::vector<f_t> lp_value_;
  std::vector<i_t> ranked_;
  f_t lp_mass_{0};
};

}  // namespace cuopt::mathematical_optimization::mip
