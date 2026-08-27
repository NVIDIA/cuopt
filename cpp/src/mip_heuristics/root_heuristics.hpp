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
  pseudo_costs_t<i_t, f_t> pseudo_costs_;
  omp_atomic_t<i_t> active_workers_;
  std::atomic<int> halt_;

  std::unique_ptr<diving_worker_t<i_t, f_t>> submip_worker_;
  std::vector<std::unique_ptr<diving_worker_t<i_t, f_t>>> diving_workers_;
  fj_cpu_worker_t<i_t, f_t> fj_cpu_worker_;

  cut_pass_heuristics_t(const csr_matrix_t<i_t, f_t>& Arow,
                        const std::vector<simplex::variable_type_t>& var_types,
                        const std::vector<f_t>& root_solution,
                        const std::vector<f_t>& root_edge_norm,
                        const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
    : var_types_(var_types),
      Arow_(Arow),
      root_solution_(root_solution),
      root_edge_norm_(root_edge_norm),
      pseudo_costs_(root_solution.size(), settings),
      active_workers_(0),
      halt_(false),
      submip_worker_(nullptr)
  {
    pseudo_costs_.Arow = Arow;
  };

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

    for (auto& worker : diving_workers_) {
      diving_worker_t<i_t, f_t>* w = worker.get();
#pragma omp taskwait depend(in : *w)
      worker.reset();
    }

    diving_workers_.clear();
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
      id, lp, Arow_, var_types_, settings, pseudo_costs_, root_solution_, root_edge_norm_);
    submip_worker_->start_node       = mip_node_t<i_t, f_t>(root_obj, root_vstatus);
    submip_worker_->leaf_vstatus     = root_vstatus;
    submip_worker_->leaf_solution.x  = sol;
    submip_worker_->recompute_bounds = false;
    submip_worker_->recompute_basis  = true;
    submip_worker_->search_strategy  = type;
    submip_worker_->set_active();

    return submip_worker_.get();
  }

  void initialize_pseudocost(const simplex::lp_problem_t<i_t, f_t>& lp,
                             const std::vector<simplex::variable_status_t>& vstatus,
                             const std::vector<i_t>& fractional,
                             const simplex::lp_solution_t<i_t, f_t>& lp_solution,
                             const std::vector<i_t>& basic_list,
                             const std::vector<i_t>& nonbasic_list,
                             simplex::basis_update_mpf_t<i_t, f_t>& basis_factors)
  {
    pseudo_costs_.initialize_with_estimate(
      lp, vstatus, fractional, lp_solution, basic_list, nonbasic_list, basis_factors);
  }

  diving_worker_t<i_t, f_t>* create_diving_worker(
    i_t cut_pass,
    const simplex::lp_problem_t<i_t, f_t>& lp,
    const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
    const mip_node_t<i_t, f_t>& root_node,
    search_strategy_t strategy)
  {
    std::unique_ptr<diving_worker_t<i_t, f_t>>& worker = diving_workers_.emplace_back(
      std::make_unique<diving_worker_t<i_t, f_t>>(diving_workers_.size(),
                                                  lp,
                                                  Arow_,
                                                  var_types_,
                                                  settings,
                                                  pseudo_costs_,
                                                  root_solution_,
                                                  root_edge_norm_));
    worker->start_node      = root_node.detach_copy();
    worker->start_lower     = lp.lower;
    worker->start_upper     = lp.upper;
    worker->search_strategy = strategy;
    worker->set_active();

    return worker.get();
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

  // Keep track of the last diving heuristic used (so we can cycle between them in low thread
  // count systems)
  i_t next_diving_type_;

  root_heuristics_t(i_t max_workers)
    : worker_count_(std::make_shared<omp_atomic_t<i_t>>(0)),
      max_workers_(max_workers),
      next_diving_type_(0)
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

  void stop_old_workers(i_t cut_pass, i_t new_workers)
  {
    if (new_workers <= 0) return;

    // On the first pass we use a thread to generate the clique table
    i_t cut_generation = cut_pass == 0 ? 2 : 1;
    i_t total_workers  = new_workers + worker_count_->load() + cut_generation;
    if (total_workers <= max_workers_) { return; }

    for (i_t i = 0; i < cut_passes_heuristics_.size() - 1; ++i) {
      i_t active = cut_passes_heuristics_[i]->active_workers_;
      if (active > 0 && !cut_passes_heuristics_[i]->halt_.load(std::memory_order_acquire)) {
        cut_passes_heuristics_[i]->send_stop_signal();
        new_workers -= active;
        if (new_workers <= 0) return;
      }
    }
  }

  std::shared_ptr<cut_pass_heuristics_t<i_t, f_t>> create_new_cut_pass_heuristic(
    const csr_matrix_t<i_t, f_t>& Arow,
    const std::vector<simplex::variable_type_t>& var_types,
    const std::vector<f_t>& root_solution,
    const std::vector<f_t>& root_edge_norm,
    const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
  {
    return cut_passes_heuristics_.emplace_back(std::make_shared<cut_pass_heuristics_t<i_t, f_t>>(
      Arow, var_types, root_solution, root_edge_norm, settings));
  }
};

}  // namespace cuopt::mathematical_optimization::mip
