/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <linear_programming/restart_strategy/localized_duality_gap_container.hpp>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>

#include <mip/mip_constants.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
localized_duality_gap_container_t<i_t, f_t>::localized_duality_gap_container_t(
  raft::handle_t const* handle_ptr, i_t primal_size, i_t dual_size)
  : primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    lagrangian_value_{handle_ptr->get_stream()},
    lower_bound_value_{handle_ptr->get_stream()},
    upper_bound_value_{handle_ptr->get_stream()},
    distance_traveled_{handle_ptr->get_stream()},
    primal_distance_traveled_{handle_ptr->get_stream()},
    dual_distance_traveled_{handle_ptr->get_stream()},
    normalized_gap_{handle_ptr->get_stream()},
    primal_solution_{static_cast<size_t>(primal_size),
                     handle_ptr->get_stream()},                                // Needed even in kkt
    dual_solution_{static_cast<size_t>(dual_size), handle_ptr->get_stream()},  // Needed even in kkt
    primal_gradient_{is_KKT_restart<i_t, f_t>() ? 0 : static_cast<size_t>(primal_size),
                     handle_ptr->get_stream()},
    dual_gradient_{is_KKT_restart<i_t, f_t>() ? 0 : static_cast<size_t>(dual_size),
                   handle_ptr->get_stream()},
    primal_solution_tr_{is_KKT_restart<i_t, f_t>() ? 0 : static_cast<size_t>(primal_size),
                        handle_ptr->get_stream()},
    dual_solution_tr_{is_KKT_restart<i_t, f_t>() ? 0 : static_cast<size_t>(dual_size),
                      handle_ptr->get_stream()}
{
  RAFT_CUDA_TRY(cudaMemsetAsync(primal_solution_.data(),
                                f_t(0.0),
                                sizeof(f_t) * primal_solution_.size(),
                                handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaMemsetAsync(dual_solution_.data(),
                                f_t(0.0),
                                sizeof(f_t) * dual_solution_.size(),
                                handle_ptr->get_stream()));
}

template <typename i_t, typename f_t>
typename localized_duality_gap_container_t<i_t, f_t>::view_t
localized_duality_gap_container_t<i_t, f_t>::view()
{
  localized_duality_gap_container_t<i_t, f_t>::view_t v{};
  v.primal_size = primal_size_h_;
  v.dual_size   = dual_size_h_;

  v.lagrangian_value         = lagrangian_value_.data();
  v.lower_bound_value        = lower_bound_value_.data();
  v.upper_bound_value        = upper_bound_value_.data();
  v.distance_traveled        = distance_traveled_.data();
  v.primal_distance_traveled = primal_distance_traveled_.data();
  v.dual_distance_traveled   = dual_distance_traveled_.data();
  v.normalized_gap           = normalized_gap_.data();

  v.primal_solution    = primal_solution_.data();
  v.dual_solution      = dual_solution_.data();
  v.primal_solution_tr = primal_solution_tr_.data();
  v.dual_solution_tr   = dual_solution_tr_.data();
  v.primal_gradient    = primal_gradient_.data();
  v.dual_gradient      = dual_gradient_.data();
  return v;
}

#if MIP_INSTANTIATE_FLOAT
template struct localized_duality_gap_container_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template struct localized_duality_gap_container_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
