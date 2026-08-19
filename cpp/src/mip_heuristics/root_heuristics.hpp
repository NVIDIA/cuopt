/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/worker.hpp>
#include <dual_simplex/user_problem.hpp>
#include "feasibility_jump/fj_cpu_worker.cuh"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct root_heuristics_t {
  std::vector<simplex::variable_type_t> var_types_;
  csr_matrix_t<i_t, f_t> Arow_;
  std::atomic<int> halt_{false};

  std::unique_ptr<diving_worker_t<i_t, f_t>> submip_worker_;
  fj_cpu_worker_t<i_t, f_t> fj_cpu_worker_;

  root_heuristics_t(const csr_matrix_t<i_t, f_t>& Arow,
                    const std::vector<simplex::variable_type_t>& var_types)
    : submip_worker_(nullptr), var_types_(var_types), Arow_(Arow) {};

  ~root_heuristics_t() { stop(); }

  void stop()
  {
    halt_.store(true, std::memory_order_release);
    fj_cpu_worker_.stop();
    if (submip_worker_) {
      diving_worker_t<i_t, f_t>* worker = submip_worker_.get();
#pragma omp taskwait depend(in : *worker)
    }
  }

  diving_worker_t<i_t, f_t>* create_submip_worker(
    i_t id,
    const simplex::lp_problem_t<i_t, f_t>& lp,
    const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
    f_t root_obj,
    const std::vector<simplex::variable_status_t>& root_vstatus,
    const std::vector<f_t>& sol)
  {
    submip_worker_ =
      std::make_unique<diving_worker_t<i_t, f_t>>(id, lp, Arow_, var_types_, settings);
    submip_worker_->start_node       = mip_node_t<i_t, f_t>(root_obj, root_vstatus);
    submip_worker_->leaf_vstatus     = root_vstatus;
    submip_worker_->leaf_solution.x  = sol;
    submip_worker_->recompute_bounds = false;
    submip_worker_->recompute_basis  = true;
    submip_worker_->search_strategy  = search_strategy_t::RINS;
    submip_worker_->set_active();

    return submip_worker_.get();
  }
};
}  // namespace cuopt::mathematical_optimization::mip
