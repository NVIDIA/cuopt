/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <mip/mip_constants.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::add_initial_solution(const f_t* initial_solution,
                                                           i_t size,
                                                           rmm::cuda_stream_view stream)
{
  cuopt_expects(
    initial_solution != nullptr, error_type_t::ValidationError, "initial_solution cannot be null");
  initial_solutions.emplace_back(std::make_shared<rmm::device_uvector<f_t>>(size, stream));
  raft::copy(initial_solutions.back()->data(), initial_solution, size, stream);
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_mip_callback(
  internals::base_solution_callback_t* callback, void* user_data)
{
  if (callback == nullptr) { return; }
  callback->set_user_data(user_data);
  mip_callbacks_.push_back(callback);
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
