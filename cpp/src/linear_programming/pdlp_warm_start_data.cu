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
  rmm::device_uvector<f_t>& current_primal_solution,
  rmm::device_uvector<f_t>& current_dual_solution,
  rmm::device_uvector<f_t>& initial_primal_average,
  rmm::device_uvector<f_t>& initial_dual_average,
  rmm::device_uvector<f_t>& current_ATY,
  rmm::device_uvector<f_t>& sum_primal_solutions,
  rmm::device_uvector<f_t>& sum_dual_solutions,
  rmm::device_uvector<f_t>& last_restart_duality_gap_primal_solution,
  rmm::device_uvector<f_t>& last_restart_duality_gap_dual_solution,
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
  :  // When initially creating this object, we can't move neither the primal/dual solution nor
     // the average since they might be used as a solution by the solution object, they have to be
     // copied
    current_primal_solution_(current_primal_solution, current_primal_solution.stream()),
    current_dual_solution_(current_dual_solution, current_dual_solution.stream()),
    initial_primal_average_(initial_primal_average, initial_primal_average.stream()),
    initial_dual_average_(initial_dual_average, initial_dual_average.stream()),
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
  : current_primal_solution_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    current_dual_solution_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    initial_primal_average_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    initial_dual_average_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    current_ATY_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    sum_primal_solutions_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    sum_dual_solutions_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    last_restart_duality_gap_primal_solution_{
      rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)},
    last_restart_duality_gap_dual_solution_{rmm::device_uvector<f_t>(0, rmm::cuda_stream_default)}
{
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>::pdlp_warm_start_data_t(
  const pdlp_warm_start_data_view_t<i_t, f_t>& other, rmm::cuda_stream_view stream_view)
  : current_primal_solution_(other.current_primal_solution_.size(), stream_view),
    current_dual_solution_(other.current_dual_solution_.size(), stream_view),
    initial_primal_average_(other.initial_primal_average_.size(), stream_view),
    initial_dual_average_(other.initial_dual_average_.size(), stream_view),
    current_ATY_(other.current_ATY_.size(), stream_view),
    sum_primal_solutions_(other.sum_primal_solutions_.size(), stream_view),
    sum_dual_solutions_(other.sum_dual_solutions_.size(), stream_view),
    last_restart_duality_gap_primal_solution_(
      other.last_restart_duality_gap_primal_solution_.size(), stream_view),
    last_restart_duality_gap_dual_solution_(other.last_restart_duality_gap_dual_solution_.size(),
                                            stream_view),
    initial_primal_weight_(other.initial_primal_weight_),
    initial_step_size_(other.initial_step_size_),
    total_pdlp_iterations_(other.total_pdlp_iterations_),
    total_pdhg_iterations_(other.total_pdhg_iterations_),
    last_candidate_kkt_score_(other.last_candidate_kkt_score_),
    last_restart_kkt_score_(other.last_restart_kkt_score_),
    sum_solution_weight_(other.sum_solution_weight_),
    iterations_since_last_restart_(other.iterations_since_last_restart_)
{
  raft::copy(current_primal_solution_.data(),
             other.current_primal_solution_.data(),
             other.current_primal_solution_.size(),
             stream_view);
  raft::copy(current_dual_solution_.data(),
             other.current_dual_solution_.data(),
             other.current_dual_solution_.size(),
             stream_view);
  raft::copy(initial_primal_average_.data(),
             other.initial_primal_average_.data(),
             other.initial_primal_average_.size(),
             stream_view);
  raft::copy(initial_dual_average_.data(),
             other.initial_dual_average_.data(),
             other.initial_dual_average_.size(),
             stream_view);
  raft::copy(
    current_ATY_.data(), other.current_ATY_.data(), other.current_ATY_.size(), stream_view);
  raft::copy(sum_primal_solutions_.data(),
             other.sum_primal_solutions_.data(),
             other.sum_primal_solutions_.size(),
             stream_view);
  raft::copy(sum_dual_solutions_.data(),
             other.sum_dual_solutions_.data(),
             other.sum_dual_solutions_.size(),
             stream_view);
  raft::copy(last_restart_duality_gap_primal_solution_.data(),
             other.last_restart_duality_gap_primal_solution_.data(),
             other.last_restart_duality_gap_primal_solution_.size(),
             stream_view);
  raft::copy(last_restart_duality_gap_dual_solution_.data(),
             other.last_restart_duality_gap_dual_solution_.data(),
             other.last_restart_duality_gap_dual_solution_.size(),
             stream_view);

  check_sizes();
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>::pdlp_warm_start_data_t(const pdlp_warm_start_data_t& other,
                                                         rmm::cuda_stream_view stream_view)
  : current_primal_solution_(other.current_primal_solution_, stream_view),
    current_dual_solution_(other.current_dual_solution_, stream_view),
    initial_primal_average_(other.initial_primal_average_, stream_view),
    initial_dual_average_(other.initial_dual_average_, stream_view),
    current_ATY_(other.current_ATY_, stream_view),
    sum_primal_solutions_(other.sum_primal_solutions_, stream_view),
    sum_dual_solutions_(other.sum_dual_solutions_, stream_view),
    last_restart_duality_gap_primal_solution_(other.last_restart_duality_gap_primal_solution_,
                                              stream_view),
    last_restart_duality_gap_dual_solution_(other.last_restart_duality_gap_dual_solution_,
                                            stream_view),
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
