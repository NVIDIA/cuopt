/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/copy_helpers.hpp>
#include <utilities/vector_helpers.cuh>
#include "../../solution/solution_handle.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
struct ret_cycles_t {
  ret_cycles_t(size_t max_size, rmm::cuda_stream_view stream_view)
    : paths(max_size, stream_view),
      offsets(max_size, stream_view),
      n_cycles_(0, stream_view),
      curr_iter_n_starts(0, stream_view)
  {
  }

  size_t size() { return n_cycles_.value(n_cycles_.stream()); }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    n_cycles_.set_value_to_zero_async(sol_handle->get_stream());
    async_fill(paths, 0, sol_handle->get_stream());
    async_fill(offsets, 0, sol_handle->get_stream());
    total_cycle_cost = 0.;
  }
  struct host_t {
    std::vector<i_t> paths;
    std::vector<i_t> offsets;
    i_t n_cycles;
  };

  host_t to_host()
  {
    host_t h;
    h.paths    = host_copy(paths);
    h.offsets  = host_copy(offsets);
    h.n_cycles = size();
    return h;
  }

  struct view_t {
    DI void push_back(i_t val) { paths[offsets[*n_cycles_] + curr_cycle_size++] = val; }

    DI void append_cycle(i_t cycle_size)
    {
      *n_cycles_ += 1;
      offsets[*n_cycles_] = offsets[*n_cycles_ - 1] + cycle_size;
    }

    i_t* n_cycles_{nullptr};
    i_t* curr_iter_n_starts{nullptr};
    size_t curr_cycle_size{0};
    raft::device_span<i_t> paths;
    raft::device_span<i_t> offsets;
  };

  view_t view()
  {
    view_t v;
    v.paths              = raft::device_span<i_t>{paths.data(), paths.size()};
    v.offsets            = raft::device_span<i_t>{offsets.data(), offsets.size()};
    v.n_cycles_          = n_cycles_.data();
    v.curr_iter_n_starts = curr_iter_n_starts.data();
    return v;
  }
  rmm::device_uvector<i_t> paths;
  rmm::device_uvector<i_t> offsets;
  rmm::device_scalar<i_t> n_cycles_;
  rmm::device_scalar<i_t> curr_iter_n_starts;
  double total_cycle_cost;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
