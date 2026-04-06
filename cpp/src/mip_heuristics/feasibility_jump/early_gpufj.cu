/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_gpufj.cuh"

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/solver_context.cuh>
#include <utilities/logger.hpp>

#include <raft/core/error.hpp>

#include <limits>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
early_gpufj_t<i_t, f_t>::early_gpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                                       const mip_solver_settings_t<i_t, f_t>& settings,
                                       early_incumbent_callback_t<f_t> incumbent_callback)
  : early_heuristic_t<i_t, f_t, early_gpufj_t<i_t, f_t>>(
      op_problem, settings.get_tolerances(), std::move(incumbent_callback))
{
  context_ptr_ = std::make_unique<mip_solver_context_t<i_t, f_t>>(
    &this->handle_, this->problem_ptr_.get(), settings);
}

template <typename i_t, typename f_t>
early_gpufj_t<i_t, f_t>::~early_gpufj_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_gpufj_t<i_t, f_t>::start()
{
  if (worker_thread_) { return; }

  this->start_time_ = std::chrono::steady_clock::now();

  fj_settings_t fj_settings;
  fj_settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = std::numeric_limits<int>::max();
  fj_settings.time_limit             = std::numeric_limits<f_t>::infinity();
  fj_settings.iteration_limit        = std::numeric_limits<int>::max();
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;

  fj_ptr_ = std::make_unique<fj_t<i_t, f_t>>(*context_ptr_, fj_settings);

  fj_ptr_->improvement_callback = [this](f_t user_obj, const std::vector<f_t>& h_assignment) {
    f_t solver_obj = this->problem_ptr_->get_solver_obj_from_user_obj(user_obj);
    this->try_update_best(solver_obj, h_assignment);
  };

  worker_thread_ = std::make_unique<std::thread>(&early_gpufj_t::run_worker, this);
}

template <typename i_t, typename f_t>
void early_gpufj_t<i_t, f_t>::run_worker()
{
  RAFT_CUDA_TRY(cudaSetDevice(this->device_id_));
  fj_ptr_->solve(*this->solution_ptr_);
}

template <typename i_t, typename f_t>
void early_gpufj_t<i_t, f_t>::stop()
{
  if (!worker_thread_) { return; }

  context_ptr_->preempt_heuristic_solver_.store(true);

  if (worker_thread_->joinable()) { worker_thread_->join(); }

  CUOPT_LOG_DEBUG("[Early GPU FJ] Stopped, solution_found=%d", this->solution_found_);

  fj_ptr_.reset();
  worker_thread_.reset();
}

#if MIP_INSTANTIATE_FLOAT
template class early_gpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_gpufj_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
