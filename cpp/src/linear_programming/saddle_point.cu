/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>

#include <linear_programming/saddle_point.hpp>
#include <linear_programming/swap_and_resize_helper.cuh>

#include <mip/mip_constants.hpp>

#include <thrust/fill.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
saddle_point_state_t<i_t, f_t>::saddle_point_state_t(raft::handle_t const* handle_ptr,
                                                     const i_t primal_size,
                                                     const i_t dual_size,
                                                     const size_t batch_size)
  : primal_size_{primal_size},
    dual_size_{dual_size},
    primal_solution_{batch_size * primal_size, handle_ptr->get_stream()},
    dual_solution_{batch_size * dual_size, handle_ptr->get_stream()},
    delta_primal_{batch_size * primal_size, handle_ptr->get_stream()},
    delta_dual_{batch_size * dual_size, handle_ptr->get_stream()},
    // Primal gradient is only used in trust region restart mode which does not support batch mode
    primal_gradient_{static_cast<size_t>(primal_size), handle_ptr->get_stream()},
    dual_gradient_{batch_size * dual_size, handle_ptr->get_stream()},
    current_AtY_{batch_size * primal_size, handle_ptr->get_stream()},
    next_AtY_{batch_size * primal_size, handle_ptr->get_stream()}
{
  EXE_CUOPT_EXPECTS(primal_size > 0, "Size of the primal problem must be larger than 0");
  EXE_CUOPT_EXPECTS(dual_size > 0, "Size of the dual problem must be larger than 0");

  // Starting from all 0
  thrust::fill(
    handle_ptr->get_thrust_policy(), primal_solution_.data(), primal_solution_.end(), f_t(0));
  thrust::fill(
    handle_ptr->get_thrust_policy(), dual_solution_.data(), dual_solution_.end(), f_t(0));

  RAFT_CUDA_TRY(cudaMemsetAsync(
    delta_primal_.data(), 0, sizeof(f_t) * delta_primal_.size(), handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    delta_dual_.data(), 0, sizeof(f_t) * delta_dual_.size(), handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_gradient_.data(), 0, sizeof(f_t) * primal_gradient_.size(), handle_ptr->get_stream()));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    dual_gradient_.data(), 0, sizeof(f_t) * dual_gradient_.size(), handle_ptr->get_stream()));

  // No need to 0 init current/next AtY, they are directlty written as result of SpMV
}

template <typename i_t, typename f_t>
void saddle_point_state_t<i_t, f_t>::swap_context(
  const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  [[maybe_unused]] const auto batch_size = static_cast<i_t>(primal_solution_.size() / primal_size_);
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  for (const auto& pair : swap_pairs) {
    cuopt_assert(pair.left < pair.right, "Left swap index must be less than right swap index");
    cuopt_assert(pair.right < batch_size, "Right swap index is out of bounds");
  }

  matrix_swap(primal_solution_, primal_size_, swap_pairs);
  matrix_swap(dual_solution_, dual_size_, swap_pairs);
  matrix_swap(delta_primal_, primal_size_, swap_pairs);
  matrix_swap(delta_dual_, dual_size_, swap_pairs);
  matrix_swap(dual_gradient_, dual_size_, swap_pairs);
  matrix_swap(current_AtY_, primal_size_, swap_pairs);
  matrix_swap(next_AtY_, primal_size_, swap_pairs);
}

template <typename i_t, typename f_t>
void saddle_point_state_t<i_t, f_t>::resize_context(i_t new_size)
{
  [[maybe_unused]] const auto batch_size = static_cast<i_t>(primal_solution_.size() / primal_size_);
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  cuopt_assert(new_size > 0, "New size must be greater than 0");
  cuopt_assert(new_size < batch_size, "New size must be less than batch size");

  primal_solution_.resize(new_size * primal_size_, primal_solution_.stream());
  dual_solution_.resize(new_size * dual_size_, dual_solution_.stream());
  delta_primal_.resize(new_size * primal_size_, delta_primal_.stream());
  delta_dual_.resize(new_size * dual_size_, delta_dual_.stream());
  dual_gradient_.resize(new_size * dual_size_, dual_gradient_.stream());
  current_AtY_.resize(new_size * primal_size_, current_AtY_.stream());
  next_AtY_.resize(new_size * primal_size_, next_AtY_.stream());
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
