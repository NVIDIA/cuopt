/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <linear_programming/saddle_point.hpp>
#include <mip/mip_constants.hpp>

#include <thrust/fill.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
saddle_point_state_t<i_t, f_t>::saddle_point_state_t(raft::handle_t const* handle_ptr,
                                                     i_t primal_size,
                                                     i_t dual_size)
  : primal_size_{primal_size},
    dual_size_{dual_size},
    primal_solution_{static_cast<size_t>(primal_size_), handle_ptr->get_stream()},
    dual_solution_{static_cast<size_t>(dual_size_), handle_ptr->get_stream()},
    delta_primal_{static_cast<size_t>(primal_size_), handle_ptr->get_stream()},
    delta_dual_{static_cast<size_t>(dual_size_), handle_ptr->get_stream()},
    primal_gradient_{static_cast<size_t>(primal_size_), handle_ptr->get_stream()},
    dual_gradient_{static_cast<size_t>(dual_size_), handle_ptr->get_stream()},
    current_AtY_{static_cast<size_t>(primal_size_), handle_ptr->get_stream()},
    next_AtY_{static_cast<size_t>(primal_size_), handle_ptr->get_stream()}
{
  EXE_CUOPT_EXPECTS(primal_size > 0, "Size of the primal problem must be larger than 0");
  EXE_CUOPT_EXPECTS(dual_size > 0, "Size of the dual problem must be larger than 0");

  // Starting from all 0
  thrust::fill(
    handle_ptr->get_thrust_policy(), primal_solution_.data(), primal_solution_.end(), f_t(0));
  thrust::fill(
    handle_ptr->get_thrust_policy(), dual_solution_.data(), dual_solution_.end(), f_t(0));

  RAFT_CUDA_TRY(cudaMemsetAsync(
    delta_primal_.data(), 0.0, sizeof(f_t) * primal_size_, handle_ptr->get_stream()));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(delta_dual_.data(), 0.0, sizeof(f_t) * dual_size_, handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_gradient_.data(), 0.0, sizeof(f_t) * primal_size_, handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    dual_gradient_.data(), 0.0, sizeof(f_t) * dual_size_, handle_ptr->get_stream()));

  // No need to 0 init current/next AtY, they are directlty written as result of SpMV
}

template <typename i_t, typename f_t>
void saddle_point_state_t<i_t, f_t>::copy(saddle_point_state_t<i_t, f_t>& other,
                                          rmm::cuda_stream_view stream)
{
  EXE_CUOPT_EXPECTS(this->primal_size_ == other.get_primal_size(),
                    "Size of primal solution must be the same in order to copy");
  EXE_CUOPT_EXPECTS(this->dual_size_ == other.get_dual_size(),
                    "Size of dual solution must be the same in order to copy");

  raft::copy(
    this->primal_solution_.data(), other.get_primal_solution().data(), this->primal_size_, stream);
  raft::copy(
    this->dual_solution_.data(), other.get_dual_solution().data(), this->dual_size_, stream);
}

template <typename i_t, typename f_t>
i_t saddle_point_state_t<i_t, f_t>::get_primal_size() const
{
  return primal_size_;
}

template <typename i_t, typename f_t>
i_t saddle_point_state_t<i_t, f_t>::get_dual_size() const
{
  return dual_size_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_primal_solution()
{
  return primal_solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_dual_solution()
{
  return dual_solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_delta_primal()
{
  return delta_primal_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_delta_dual()
{
  return delta_dual_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_primal_gradient()
{
  return primal_gradient_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_dual_gradient()
{
  return dual_gradient_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_current_AtY()
{
  return current_AtY_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& saddle_point_state_t<i_t, f_t>::get_next_AtY()
{
  return next_AtY_;
}

#if MIP_INSTANTIATE_FLOAT
template class saddle_point_state_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class saddle_point_state_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
