/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/logger.hpp>
#include <math_optimization/solution_writer.hpp>
#include <mip/mip_constants.hpp>
#include <mps_parser/utilities/span.hpp>

#include <raft/util/cudart_utils.hpp>

#include <thrust/scatter.h>
#include <rmm/exec_policy.hpp>

#include <thrust/scatter.h>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t>::pdlp_solver_settings_t(const pdlp_solver_settings_t& other,
                                                         rmm::cuda_stream_view stream_view)
  : tolerances(other.tolerances),
    detect_infeasibility(other.detect_infeasibility),
    strict_infeasibility(other.strict_infeasibility),
    iteration_limit(other.iteration_limit),
    time_limit(other.time_limit),
    pdlp_solver_mode(other.pdlp_solver_mode),
    log_file(other.log_file),
    sol_file(other.sol_file),
    per_constraint_residual(other.per_constraint_residual),
    crossover(other.crossover),
    save_best_primal_so_far(other.save_best_primal_so_far),
    first_primal_feasible(other.first_primal_feasible),
    pdlp_warm_start_data_(other.pdlp_warm_start_data_, stream_view),
    concurrent_halt(other.concurrent_halt)
{
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_optimality_tolerance(f_t eps_optimal)
{
  tolerances.absolute_dual_tolerance   = eps_optimal;
  tolerances.relative_dual_tolerance   = eps_optimal;
  tolerances.absolute_primal_tolerance = eps_optimal;
  tolerances.relative_primal_tolerance = eps_optimal;
  tolerances.absolute_gap_tolerance    = eps_optimal;
  tolerances.relative_gap_tolerance    = eps_optimal;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_initial_primal_solution(
  const f_t* initial_primal_solution, i_t size, rmm::cuda_stream_view stream)
{
  cuopt_expects(initial_primal_solution != nullptr,
                error_type_t::ValidationError,
                "initial_primal_solution cannot be null");

  initial_primal_solution_ = std::make_shared<rmm::device_uvector<f_t>>(size, stream);
  raft::copy(initial_primal_solution_.get()->data(), initial_primal_solution, size, stream);
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_initial_dual_solution(const f_t* initial_dual_solution,
                                                                 i_t size,
                                                                 rmm::cuda_stream_view stream)
{
  cuopt_expects(initial_dual_solution != nullptr,
                error_type_t::ValidationError,
                "initial_dual_solution cannot be null");

  initial_dual_solution_ = std::make_shared<rmm::device_uvector<f_t>>(size, stream);
  raft::copy(initial_dual_solution_.get()->data(), initial_dual_solution, size, stream);
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  pdlp_warm_start_data_t<i_t, f_t>& pdlp_warm_start_data_view,
  const rmm::device_uvector<i_t>& var_mapping,
  const rmm::device_uvector<i_t>& constraint_mapping)
{
  pdlp_warm_start_data_ = std::move(pdlp_warm_start_data_view);

  // A var_mapping was given
  if (var_mapping.size() != 0) {
    // If less variables, scatter using the passed argument and reduce the size of all primal
    // related vectors
    if (var_mapping.size() <
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.size()) {
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.current_primal_solution_.begin(),
                      pdlp_warm_start_data_.current_primal_solution_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.current_primal_solution_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.initial_primal_average_.begin(),
                      pdlp_warm_start_data_.initial_primal_average_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.initial_primal_average_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.current_ATY_.begin(),
                      pdlp_warm_start_data_.current_ATY_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.current_ATY_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.sum_primal_solutions_.begin(),
                      pdlp_warm_start_data_.sum_primal_solutions_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.sum_primal_solutions_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.begin());

      pdlp_warm_start_data_.current_primal_solution_.resize(var_mapping.size(),
                                                            var_mapping.stream());
      pdlp_warm_start_data_.initial_primal_average_.resize(var_mapping.size(),
                                                           var_mapping.stream());
      pdlp_warm_start_data_.current_ATY_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.sum_primal_solutions_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.resize(var_mapping.size(),
                                                                             var_mapping.stream());
    } else if (var_mapping.size() >
               pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.size()) {
      const auto previous_size =
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.size();

      // If more variables just pad with 0s
      pdlp_warm_start_data_.current_primal_solution_.resize(var_mapping.size(),
                                                            var_mapping.stream());
      pdlp_warm_start_data_.initial_primal_average_.resize(var_mapping.size(),
                                                           var_mapping.stream());
      pdlp_warm_start_data_.current_ATY_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.sum_primal_solutions_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.resize(var_mapping.size(),
                                                                             var_mapping.stream());

      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.current_primal_solution_.begin() + previous_size,
                   pdlp_warm_start_data_.current_primal_solution_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.initial_primal_average_.begin() + previous_size,
                   pdlp_warm_start_data_.initial_primal_average_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.current_ATY_.begin() + previous_size,
                   pdlp_warm_start_data_.current_ATY_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.sum_primal_solutions_.begin() + previous_size,
                   pdlp_warm_start_data_.sum_primal_solutions_.end(),
                   f_t(0));
      thrust::fill(
        rmm::exec_policy(var_mapping.stream()),
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.begin() + previous_size,
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.end(),
        f_t(0));
    }
  }

  // A constraint_mapping was given
  if (constraint_mapping.size() != 0) {
    // If less variables, scatter using the passed argument and reduce the size of all dual related
    // vectors
    if (constraint_mapping.size() <
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.size()) {
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.current_dual_solution_.begin(),
                      pdlp_warm_start_data_.current_dual_solution_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.current_dual_solution_.begin());
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.initial_dual_average_.begin(),
                      pdlp_warm_start_data_.initial_dual_average_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.initial_dual_average_.begin());
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.sum_dual_solutions_.begin(),
                      pdlp_warm_start_data_.sum_dual_solutions_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.sum_dual_solutions_.begin());
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.begin());

      pdlp_warm_start_data_.current_dual_solution_.resize(constraint_mapping.size(),
                                                          constraint_mapping.stream());
      pdlp_warm_start_data_.initial_dual_average_.resize(constraint_mapping.size(),
                                                         constraint_mapping.stream());
      pdlp_warm_start_data_.sum_dual_solutions_.resize(constraint_mapping.size(),
                                                       constraint_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.resize(
        constraint_mapping.size(), constraint_mapping.stream());
    } else if (constraint_mapping.size() >
               pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.size()) {
      const auto previous_size =
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.size();

      // If more variables just pad with 0s
      pdlp_warm_start_data_.current_dual_solution_.resize(constraint_mapping.size(),
                                                          constraint_mapping.stream());
      pdlp_warm_start_data_.initial_dual_average_.resize(constraint_mapping.size(),
                                                         constraint_mapping.stream());
      pdlp_warm_start_data_.sum_dual_solutions_.resize(constraint_mapping.size(),
                                                       constraint_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.resize(
        constraint_mapping.size(), constraint_mapping.stream());

      thrust::fill(rmm::exec_policy(constraint_mapping.stream()),
                   pdlp_warm_start_data_.current_dual_solution_.begin() + previous_size,
                   pdlp_warm_start_data_.current_dual_solution_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(constraint_mapping.stream()),
                   pdlp_warm_start_data_.initial_dual_average_.begin() + previous_size,
                   pdlp_warm_start_data_.initial_dual_average_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(constraint_mapping.stream()),
                   pdlp_warm_start_data_.sum_dual_solutions_.begin() + previous_size,
                   pdlp_warm_start_data_.sum_dual_solutions_.end(),
                   f_t(0));
      thrust::fill(
        rmm::exec_policy(constraint_mapping.stream()),
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.begin() + previous_size,
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.end(),
        f_t(0));
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  const f_t* current_primal_solution,
  const f_t* current_dual_solution,
  const f_t* initial_primal_average,
  const f_t* initial_dual_average,
  const f_t* current_ATY,
  const f_t* sum_primal_solutions,
  const f_t* sum_dual_solutions,
  const f_t* last_restart_duality_gap_primal_solution,
  const f_t* last_restart_duality_gap_dual_solution,
  i_t primal_size,
  i_t dual_size,
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
{
  cuopt_expects(current_primal_solution != nullptr,
                error_type_t::ValidationError,
                "current_primal_solution cannot be null");
  cuopt_expects(current_dual_solution != nullptr,
                error_type_t::ValidationError,
                "current_dual_solution cannot be null");
  cuopt_expects(initial_primal_average != nullptr,
                error_type_t::ValidationError,
                "initial_primal_average cannot be null");
  cuopt_expects(initial_dual_average != nullptr,
                error_type_t::ValidationError,
                "initial_dual_average cannot be null");
  cuopt_expects(
    current_ATY != nullptr, error_type_t::ValidationError, "current_ATY cannot be null");
  cuopt_expects(sum_primal_solutions != nullptr,
                error_type_t::ValidationError,
                "sum_primal_solutions cannot be null");
  cuopt_expects(sum_dual_solutions != nullptr,
                error_type_t::ValidationError,
                "sum_dual_solutions cannot be null");
  cuopt_expects(last_restart_duality_gap_primal_solution != nullptr,
                error_type_t::ValidationError,
                "last_restart_duality_gap_primal_solution cannot be null");
  cuopt_expects(last_restart_duality_gap_dual_solution != nullptr,
                error_type_t::ValidationError,
                "last_restart_duality_gap_dual_solution cannot be null");

  pdlp_warm_start_data_view_.current_primal_solution_ =
    cuopt::mps_parser::span<f_t const>(current_primal_solution, primal_size);
  pdlp_warm_start_data_view_.current_dual_solution_ =
    cuopt::mps_parser::span<f_t const>(current_dual_solution, dual_size);
  pdlp_warm_start_data_view_.initial_primal_average_ =
    cuopt::mps_parser::span<f_t const>(initial_primal_average, primal_size);
  pdlp_warm_start_data_view_.initial_dual_average_ =
    cuopt::mps_parser::span<f_t const>(initial_dual_average, dual_size);
  pdlp_warm_start_data_view_.current_ATY_ =
    cuopt::mps_parser::span<f_t const>(current_ATY, primal_size);
  pdlp_warm_start_data_view_.sum_primal_solutions_ =
    cuopt::mps_parser::span<f_t const>(sum_primal_solutions, primal_size);
  pdlp_warm_start_data_view_.sum_dual_solutions_ =
    cuopt::mps_parser::span<f_t const>(sum_dual_solutions, dual_size);
  pdlp_warm_start_data_view_.last_restart_duality_gap_primal_solution_ =
    cuopt::mps_parser::span<f_t const>(last_restart_duality_gap_primal_solution, primal_size);
  pdlp_warm_start_data_view_.last_restart_duality_gap_dual_solution_ =
    cuopt::mps_parser::span<f_t const>(last_restart_duality_gap_dual_solution, dual_size);
  pdlp_warm_start_data_view_.initial_primal_weight_         = initial_primal_weight;
  pdlp_warm_start_data_view_.initial_step_size_             = initial_step_size;
  pdlp_warm_start_data_view_.total_pdlp_iterations_         = total_pdlp_iterations;
  pdlp_warm_start_data_view_.total_pdhg_iterations_         = total_pdhg_iterations;
  pdlp_warm_start_data_view_.last_candidate_kkt_score_      = last_candidate_kkt_score;
  pdlp_warm_start_data_view_.last_restart_kkt_score_        = last_restart_kkt_score;
  pdlp_warm_start_data_view_.sum_solution_weight_           = sum_solution_weight;
  pdlp_warm_start_data_view_.iterations_since_last_restart_ = iterations_since_last_restart;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdlp_solver_settings_t<i_t, f_t>::get_initial_primal_solution()
  const
{
  cuopt_expects(initial_primal_solution_.get() != nullptr,
                error_type_t::ValidationError,
                "Initial primal solution was not set, but accessed!");
  return *initial_primal_solution_.get();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdlp_solver_settings_t<i_t, f_t>::get_initial_dual_solution() const
{
  cuopt_expects(initial_dual_solution_.get() != nullptr,
                error_type_t::ValidationError,
                "Initial dual solution was not set, but accessed!");
  return *initial_dual_solution_.get();
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::has_initial_primal_solution() const
{
  return initial_primal_solution_.get() != nullptr;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::has_initial_dual_solution() const
{
  return initial_dual_solution_.get() != nullptr;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_t<i_t, f_t>& pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data()
  const noexcept
{
  return pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>& pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data()
{
  return pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_view_t<i_t, f_t>&
pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data_view() const noexcept
{
  return pdlp_warm_start_data_view_;
}

#if MIP_INSTANTIATE_FLOAT
template class pdlp_solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pdlp_solver_settings_t<int, double>;
#endif

}  // namespace cuopt::linear_programming
