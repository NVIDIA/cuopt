/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "routing/structures.hpp"

#include <utilities/cuda_helpers.cuh>

#include <cstddef>

namespace cuopt {
namespace routing {
namespace detail {

// Single uint64 holding the p_score on the 16 MSB bits, and respectivly route id, pickup and
// delivery indices on the reamining 48 LSB bits P_score used filed through atomicMin inside the
// kernel, remaning information allows to execute the move
struct found_sol_t {
  static constexpr uint64_t uninitialized = static_cast<uint64_t>(-1);

  found_sol_t()
    : p_val(static_cast<uint16_t>(-1)),
      route_id(static_cast<uint16_t>(-1)),
      pickup_location(static_cast<uint16_t>(-1)),
      delivery_location(static_cast<uint16_t>(-1))
  {
  }

  HD found_sol_t(uint16_t p_v, uint16_t r_i, uint16_t p_l, uint16_t d_l)
    : p_val(p_v), route_id(r_i), pickup_location(p_l), delivery_location(d_l)
  {
  }

  DI void is_valid(int n_routes, int route_size) const noexcept
  {
    cuopt_assert(route_id < n_routes, "found sol route_id should be inferior to number of routes");
    cuopt_assert(pickup_location < route_size, "found sol pickup_location should be positive");
    cuopt_assert(delivery_location < route_size, "found sol delivery_location should be positive");
  }

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  uint64_t p_val             : 16;
  uint64_t route_id          : 16;  // Can also be used to know fragement_size
  uint64_t pickup_location   : 16;
  uint64_t delivery_location : 16;
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint64_t delivery_location : 16;
  uint64_t pickup_location   : 16;
  uint64_t route_id          : 16;
  uint64_t p_val             : 16;
#endif
};

static_assert(sizeof(found_sol_t) == sizeof(uint64_t));

/**
 * FIXME:: Use view pattern
 * 1. Write a struct for feasible_candidates_t, and move feasible_candidates_data_,
 * feasible_candidats_size, compacted_feasible_candidates_data_ into that struct
 * 2. Implement init(), get_random_candidate() methods
 * 3. feasible_move_t will be feasible_candidates_t::view_t
 */
struct feasible_move_t {
  // Does not own the data
  feasible_move_t(raft::device_span<found_sol_t> data,
                  int* size,
                  const int num_orders,
                  const int break_dims,
                  const int num_routes)
    : data_(data),
      size_(size),
      num_orders_(num_orders),
      break_dims_(break_dims),
      num_routes_(num_routes)
  {
  }

  DI void record(const NodeInfo<> pickup_insertion_node_info,
                 const int route_id,
                 const int insertion_pos,
                 const found_sol_t found_sol)
  {
    // Number of possible insertion positions. This is extreme and achieved only when there is a
    // single route
    const int n_insertion_pos = num_orders_ + break_dims_;
    // any insertion after service node (not including depot or break)
    // are recorded in first norders X (norders + nbreak_dims) memory space,
    // any insertion after depot is recorded in next
    // nroutes (norders + nbreak_dims) memory space
    // any insertion after break node is recorded in the next
    // nroutes (norders + nbreak_dims) memory space
    if (pickup_insertion_node_info.is_depot()) {
      data_[num_orders_ * n_insertion_pos + route_id * n_insertion_pos + insertion_pos] = found_sol;
    } else if (pickup_insertion_node_info.is_break()) {
      data_[num_orders_ * n_insertion_pos + (num_routes_ + route_id) * n_insertion_pos +
            insertion_pos] = found_sol;
    } else {
      data_[pickup_insertion_node_info.node() * n_insertion_pos + insertion_pos] = found_sol;
    }
  }

  raft::device_span<found_sol_t> data_{};
  int* size_{nullptr};

  const int num_orders_{0};
  const int break_dims_{0};
  const int num_routes_{0};
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
