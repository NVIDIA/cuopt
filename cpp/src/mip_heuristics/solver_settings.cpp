/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Host-only members of mip_solver_settings_t, split out of solver_settings.cu.
//
// Only add_initial_solution() touches the device (it copies into an rmm::device_uvector),
// so it stays in the CUDA TU while these build into the CUDA-free cuopt_client library.
// The gRPC client reaches get_mip_callbacks() via solve_remote's callback handling.
//
// Instantiated per-member rather than with `template class`: the class holds
// device_uvector-backed initial_solutions, so instantiating all of it here would pull
// device code into the client library.

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <mip_heuristics/mip_constants.hpp>

#include <vector>

namespace cuopt::mathematical_optimization {

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

#if MIP_INSTANTIATE_FLOAT
template CUOPT_EXPORT void mip_solver_settings_t<int, float>::set_mip_callback(
  internals::base_solution_callback_t*, void*);
template CUOPT_EXPORT const std::vector<internals::base_solution_callback_t*>
mip_solver_settings_t<int, float>::get_mip_callbacks() const;
template CUOPT_EXPORT mip_solver_settings_t<int, float>::tolerances_t
mip_solver_settings_t<int, float>::get_tolerances() const noexcept;
#endif

#if MIP_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void mip_solver_settings_t<int, double>::set_mip_callback(
  internals::base_solution_callback_t*, void*);
template CUOPT_EXPORT const std::vector<internals::base_solution_callback_t*>
mip_solver_settings_t<int, double>::get_mip_callbacks() const;
template CUOPT_EXPORT mip_solver_settings_t<int, double>::tolerances_t
mip_solver_settings_t<int, double>::get_tolerances() const noexcept;
#endif

}  // namespace cuopt::mathematical_optimization
