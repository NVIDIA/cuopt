/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>

#include <mip/mip_constants.hpp>

#include <rmm/device_uvector.hpp>

#include <raft/util/cudart_utils.hpp>

#include <utilities/macros.cuh>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>::pdlp_warm_start_data_t(
  std::vector<f_t> current_primal_solution,
  std::vector<f_t> current_dual_solution,
  std::vector<f_t> initial_primal_average,
  std::vector<f_t> initial_dual_average,
  std::vector<f_t> current_ATY,
  std::vector<f_t> sum_primal_solutions,
  std::vector<f_t> sum_dual_solutions,
  std::vector<f_t> last_restart_duality_gap_primal_solution,
  std::vector<f_t> last_restart_duality_gap_dual_solution,
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
  : current_primal_solution_(std::move(current_primal_solution)),
    current_dual_solution_(std::move(current_dual_solution)),
    initial_primal_average_(std::move(initial_primal_average)),
    initial_dual_average_(std::move(initial_dual_average)),
    current_ATY_(std::move(current_ATY)),
    sum_primal_solutions_(std::move(sum_primal_solutions)),
    sum_dual_solutions_(std::move(sum_dual_solutions)),
    last_restart_duality_gap_primal_solution_(std::move(last_restart_duality_gap_primal_solution)),
    last_restart_duality_gap_dual_solution_(std::move(last_restart_duality_gap_dual_solution)),
    initial_primal_weight_(initial_primal_weight),
    initial_step_size_(initial_step_size),
    total_pdlp_iterations_(total_pdlp_iterations),
    total_pdhg_iterations_(total_pdhg_iterations),
    last_candidate_kkt_score_(last_candidate_kkt_score),
    last_restart_kkt_score_(last_restart_kkt_score),
    sum_solution_weight_(sum_solution_weight),
    iterations_since_last_restart_(iterations_since_last_restart)
{
  check_sizes();
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>::pdlp_warm_start_data_t()
  : current_primal_solution_(),
    current_dual_solution_(),
    initial_primal_average_(),
    initial_dual_average_(),
    current_ATY_(),
    sum_primal_solutions_(),
    sum_dual_solutions_(),
    last_restart_duality_gap_primal_solution_(),
    last_restart_duality_gap_dual_solution_()
{
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>::pdlp_warm_start_data_t(
  const pdlp_warm_start_data_view_t<i_t, f_t>& other)
  : current_primal_solution_(other.current_primal_solution_.size()),
    current_dual_solution_(other.current_dual_solution_.size()),
    initial_primal_average_(other.initial_primal_average_.size()),
    initial_dual_average_(other.initial_dual_average_.size()),
    current_ATY_(other.current_ATY_.size()),
    sum_primal_solutions_(other.sum_primal_solutions_.size()),
    sum_dual_solutions_(other.sum_dual_solutions_.size()),
    last_restart_duality_gap_primal_solution_(
      other.last_restart_duality_gap_primal_solution_.size()),
    last_restart_duality_gap_dual_solution_(other.last_restart_duality_gap_dual_solution_.size()),
    initial_primal_weight_(other.initial_primal_weight_),
    initial_step_size_(other.initial_step_size_),
    total_pdlp_iterations_(other.total_pdlp_iterations_),
    total_pdhg_iterations_(other.total_pdhg_iterations_),
    last_candidate_kkt_score_(other.last_candidate_kkt_score_),
    last_restart_kkt_score_(other.last_restart_kkt_score_),
    sum_solution_weight_(other.sum_solution_weight_),
    iterations_since_last_restart_(other.iterations_since_last_restart_)
{
  // Note: pdlp_warm_start_data_view_t contains device pointers
  // This constructor is used by Cython, so we need to copy from device to host
  // We'll need to add this device-to-host copy logic when we integrate with Cython
  // For now, this creates empty vectors sized correctly
  // TODO: Add device-to-host copy when integrating with Cython interface

  check_sizes();
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>::pdlp_warm_start_data_t(const pdlp_warm_start_data_t& other)
  : current_primal_solution_(other.current_primal_solution_),
    current_dual_solution_(other.current_dual_solution_),
    initial_primal_average_(other.initial_primal_average_),
    initial_dual_average_(other.initial_dual_average_),
    current_ATY_(other.current_ATY_),
    sum_primal_solutions_(other.sum_primal_solutions_),
    sum_dual_solutions_(other.sum_dual_solutions_),
    last_restart_duality_gap_primal_solution_(other.last_restart_duality_gap_primal_solution_),
    last_restart_duality_gap_dual_solution_(other.last_restart_duality_gap_dual_solution_),
    initial_primal_weight_(other.initial_primal_weight_),
    initial_step_size_(other.initial_step_size_),
    total_pdlp_iterations_(other.total_pdlp_iterations_),
    total_pdhg_iterations_(other.total_pdhg_iterations_),
    last_candidate_kkt_score_(other.last_candidate_kkt_score_),
    last_restart_kkt_score_(other.last_restart_kkt_score_),
    sum_solution_weight_(other.sum_solution_weight_),
    iterations_since_last_restart_(other.iterations_since_last_restart_)
{
  check_sizes();
}

template <typename i_t, typename f_t>
void pdlp_warm_start_data_t<i_t, f_t>::check_sizes()
{
  cuopt_assert(current_primal_solution_.size() == initial_primal_average_.size() &&
                 initial_primal_average_.size() == current_ATY_.size() &&
                 current_ATY_.size() == sum_primal_solutions_.size() &&
                 sum_primal_solutions_.size() == last_restart_duality_gap_primal_solution_.size(),
               "All primal vectors should be of same size");
  cuopt_assert(current_dual_solution_.size() == initial_dual_average_.size() &&
                 initial_dual_average_.size() == sum_dual_solutions_.size() &&
                 sum_dual_solutions_.size() == last_restart_duality_gap_dual_solution_.size(),
               "All dual vectors should be of same size");
}

#if MIP_INSTANTIATE_FLOAT
template class pdlp_warm_start_data_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pdlp_warm_start_data_t<int, double>;
#endif
}  // namespace cuopt::linear_programming
