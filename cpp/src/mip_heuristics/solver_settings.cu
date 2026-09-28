/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuda/stream>
#include <cuopt/error.hpp>
#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::add_initial_solution(const f_t* initial_solution,
                                                           i_t size,
                                                           cuda::stream_ref stream)
{
  cuopt_expects(
    initial_solution != nullptr, error_type_t::ValidationError, "initial_solution cannot be null");
  initial_solutions.emplace_back(std::make_shared<rmm::device_uvector<f_t>>(size, stream));
  raft::copy(initial_solutions.back()->data(), initial_solution, size, stream);
}

// Explicit template instantiations for common types
#if MIP_INSTANTIATE_FLOAT
template class CUOPT_EXPORT mip_solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class CUOPT_EXPORT mip_solver_settings_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization
