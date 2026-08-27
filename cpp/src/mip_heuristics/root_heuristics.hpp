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
struct cut_pass_heuristics_t {
  std::vector<simplex::variable_type_t> var_types_;
  csr_matrix_t<i_t, f_t> Arow_;
  std::vector<f_t> root_solution_;
  std::vector<f_t> root_edge_norm_;
  std::atomic<int> halt_;

  std::unique_ptr<diving_worker_t<i_t, f_t>> submip_worker_;
  fj_cpu_worker_t<i_t, f_t> fj_cpu_worker_;

  cut_pass_heuristics_t(const csr_matrix_t<i_t, f_t>& Arow,
                        const std::vector<simplex::variable_type_t>& var_types,
                        const std::vector<f_t>& root_solution,
                        const std::vector<f_t>& root_edge_norm)
    : var_types_(var_types),
      Arow_(Arow),
      root_solution_(root_solution),
      root_edge_norm_(root_edge_norm),
      halt_(false),
      submip_worker_(nullptr) {};

  ~cut_pass_heuristics_t() { stop_and_sync(); }

  void send_stop_signal()
  {
    fj_cpu_worker_.send_stop_signal();
    halt_ = true;
  }

  void stop_and_sync()
  {
    fj_cpu_worker_.stop();
    halt_ = true;

    if (submip_worker_) {
      diving_worker_t<i_t, f_t>* worker = submip_worker_.get();
#pragma omp taskwait depend(in : *worker)
      submip_worker_.reset();
    }
  }

  diving_worker_t<i_t, f_t>* create_submip_worker(
    i_t id,
    const simplex::lp_problem_t<i_t, f_t>& lp,
    const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
    f_t root_obj,
    const std::vector<simplex::variable_status_t>& root_vstatus,
    const std::vector<f_t>& sol,
    search_strategy_t type)
  {
    submip_worker_ = std::make_unique<diving_worker_t<i_t, f_t>>(
      id, lp, Arow_, var_types_, settings, root_solution_, root_edge_norm_);
    submip_worker_->start_node       = mip_node_t<i_t, f_t>(root_obj, root_vstatus);
    submip_worker_->leaf_vstatus     = root_vstatus;
    submip_worker_->leaf_solution.x  = sol;
    submip_worker_->recompute_bounds = false;
    submip_worker_->recompute_basis  = true;
    submip_worker_->search_strategy  = type;
    submip_worker_->set_active();

    return submip_worker_.get();
  }
};

/// \brief Object Representing the heuristics run on the root node.
template <typename i_t, typename f_t>
struct root_heuristics_t {
  // List of the heuristics that run alongside a single cut pass.
  // It holds the workers and all the necessary information.
  //
  // We use the `shared_ptr` here so the object is only destroyed when the task terminates
  // (we declare the `shared_ptr` as firstprivate in the task, so they live until the end the
  // task). In this way, we can send the stop signal, destroy the entry in the list and the
  // object itself will be destroyed when all related tasks ends.
  std::list<std::shared_ptr<cut_pass_heuristics_t<i_t, f_t>>> cut_passes_heuristics_;

  // Count the number of active workers. Same reason as above.
  std::shared_ptr<omp_atomic_t<i_t>> worker_count_;
  i_t max_workers_;

  root_heuristics_t(i_t max_workers)
    : worker_count_(std::make_shared<omp_atomic_t<i_t>>(0)), max_workers_(max_workers)
  {
  }

  ~root_heuristics_t() { stop_and_sync(); }

  void stop_and_sync()
  {
    for (auto& heuristic : cut_passes_heuristics_) {
      heuristic->send_stop_signal();
    }

    for (auto& heuristic : cut_passes_heuristics_) {
      heuristic->stop_and_sync();
    }

    cut_passes_heuristics_.clear();
  }

  std::shared_ptr<cut_pass_heuristics_t<i_t, f_t>> create_new_cut_pass_heuristic(
    i_t cut_pass,
    const csr_matrix_t<i_t, f_t>& Arow,
    const std::vector<simplex::variable_type_t>& var_types,
    const std::vector<f_t>& root_solution,
    const std::vector<f_t>& root_edge_norm)
  {
    // If we already exhausted all threads for the root heuristics, stop workers for the
    // oldest set of heuristics launched. Leave 2 threads for the cut passes and the clique
    // table generation. Add the number of workers that will be launched (1 submip worker +
    // 1 CPU FJ worker).
    i_t clique_table_generation = cut_pass == 0 ? 1 : 0;
    if (*worker_count_ + 3 + clique_table_generation > max_workers_ &&
        !cut_passes_heuristics_.empty()) {
      cut_passes_heuristics_.begin()->get()->send_stop_signal();
      cut_passes_heuristics_.erase(cut_passes_heuristics_.begin());
    }

    return cut_passes_heuristics_.emplace_back(std::make_shared<cut_pass_heuristics_t<i_t, f_t>>(
      Arow, var_types, root_solution, root_edge_norm));
  }
};

}  // namespace cuopt::mathematical_optimization::mip
