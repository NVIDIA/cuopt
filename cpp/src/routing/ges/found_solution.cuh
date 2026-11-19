/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  feasible_move_t(raft::device_span<found_sol_t> data, int* size) : data_(data), size_(size) {}

  DI void record(const int index, const found_sol_t found_sol)
  {
    cuopt_assert(index < (int)data_.size(), "Candidate count should stay below the table size");
    data_[index] = found_sol;
  }

  raft::device_span<found_sol_t> data_{};
  int* size_{nullptr};
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
