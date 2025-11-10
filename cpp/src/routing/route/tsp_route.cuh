/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/node.cuh"
#include "../solution/solution_handle.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class tsp_route_t {
 public:
  tsp_route_t(solution_handle_t<i_t, f_t> const* sol_handle_)
    : pred(0, sol_handle_->get_stream()), succ(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero pdp_route_t copy_ctr");
  }

  tsp_route_t(const tsp_route_t& tsp_route, solution_handle_t<i_t, f_t> const* sol_handle_)
    : pred(tsp_route.pred, sol_handle_->get_stream()),
      succ(tsp_route.succ, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("pdp route copy_ctr");
  }

  tsp_route_t& operator=(tsp_route_t&& tsp_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    pred.resize(max_nodes_per_route, stream);
    succ.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return pred.empty(); }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(i_t* shmem, i_t n_nodes_route)
    {
      view_t v;

      v.pred = raft::device_span<NodeInfo<i_t>>{(NodeInfo<i_t>*)shmem, (size_t)n_nodes_route + 1};
      v.succ = raft::device_span<NodeInfo<i_t>>{&v.pred.data()[n_nodes_route + 1],
                                                (size_t)n_nodes_route + 1};

      i_t* sh_ptr = (i_t*)&v.succ.data()[n_nodes_route + 1];
      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<NodeInfo<i_t>> pred;
    raft::device_span<NodeInfo<i_t>> succ;
    NodeInfo<i_t> start;
    NodeInfo<i_t> end;
  };

  view_t view()
  {
    view_t v;
    v.pred = raft::device_span<NodeInfo<i_t>>{pred.data(), pred.size()};
    v.succ = raft::device_span<NodeInfo<i_t>>{succ.data(), succ.size()};

    return v;
  }

  /**
   * @brief Get the shared memory size required to store a request route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI static size_t get_shared_size(i_t route_size)
  {
    // pred, succ
    size_t byte_size = 2 * route_size * sizeof(NodeInfo<i_t>);
    return raft::alignTo(byte_size, sizeof(double));
  }

  // pred ids
  rmm::device_uvector<NodeInfo<i_t>> pred;

  // succ ids
  rmm::device_uvector<NodeInfo<i_t>> succ;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
