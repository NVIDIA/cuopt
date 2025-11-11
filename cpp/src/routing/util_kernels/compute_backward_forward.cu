/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../solution/solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_backward_forward_kernel(
  raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes)
{
  const i_t route_id   = blockIdx.x / 2;
  auto curr_route      = routes[route_id];
  bool compute_forward = (blockIdx.x % 2) == 0;

  if (threadIdx.x == 0) {
    if (compute_forward) {
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward(curr_route);
      curr_route.compute_cost();
    } else {
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward(curr_route);
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_actual_arrival_kernel(
  raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes)
{
  const i_t route_id = blockIdx.x;
  auto curr_route    = routes[route_id];

  if (threadIdx.x == 0) { curr_route.compute_actual_arrival_time(); }
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::compute_backward_forward()
{
  raft::common::nvtx::range fun_scope("compute_backward_forward");
  constexpr i_t TPB = 32;
  if (n_routes) {
    compute_backward_forward_kernel<i_t, f_t, REQUEST>
      <<<n_routes * 2, TPB, 0, sol_handle->get_stream()>>>(view().routes);
    sol_handle->sync_stream();
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::compute_actual_arrival_times()
{
  raft::common::nvtx::range fun_scope("compute_backward_forward");
  constexpr i_t TPB = 32;
  if (n_routes && problem_ptr->dimensions_info.has_dimension(dim_t::TIME))
    compute_actual_arrival_kernel<i_t, f_t, REQUEST>
      <<<n_routes, TPB, 0, sol_handle->get_stream()>>>(view().routes);
}

template void solution_t<int, float, request_t::PDP>::compute_backward_forward();
template void solution_t<int, float, request_t::VRP>::compute_backward_forward();
template void solution_t<int, float, request_t::PDP>::compute_actual_arrival_times();
template void solution_t<int, float, request_t::VRP>::compute_actual_arrival_times();

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
