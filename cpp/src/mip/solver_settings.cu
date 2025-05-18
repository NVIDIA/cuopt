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
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <mip/mip_constants.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_initial_solution(const f_t* initial_solution,
                                                           i_t size,
                                                           rmm::cuda_stream_view stream)
{
  cuopt_expects(
    initial_solution != nullptr, error_type_t::ValidationError, "initial_solution cannot be null");
  if (!initial_solution_) {
    initial_solution_ = std::make_shared<rmm::device_uvector<f_t>>(size, stream);
  }

  raft::copy(initial_solution_.get()->data(), initial_solution, size, stream);
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_mip_callback(
  internals::base_solution_callback_t* callback)
{
  mip_callbacks_.push_back(callback);
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& mip_solver_settings_t<i_t, f_t>::get_initial_solution() const
{
  if (!initial_solution_) { throw std::runtime_error("Initial solution has not been set"); }
  return *initial_solution_;
}

template <typename i_t, typename f_t>
bool mip_solver_settings_t<i_t, f_t>::has_initial_solution() const
{
  return initial_solution_.get() != nullptr;
}

template <typename i_t, typename f_t>
const std::vector<internals::base_solution_callback_t*>
mip_solver_settings_t<i_t, f_t>::get_mip_callbacks() const
{
  return mip_callbacks_;
}

template <typename i_t, typename f_t>
typename mip_solver_settings_t<i_t, f_t>::tolerances_t
mip_solver_settings_t<i_t, f_t>::get_tolerances() const noexcept
{
  return tolerances;
}

// Explicit template instantiations for common types
#if MIP_INSTANTIATE_FLOAT
template class mip_solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class mip_solver_settings_t<int, double>;
#endif

}  // namespace cuopt::linear_programming
