/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t>
struct sliding_tsp_cand_t {
  i_t window_size;
  i_t window_start;
  i_t insertion_pos;
  i_t reverse;
  double selection_delta;

  static constexpr sliding_tsp_cand_t<i_t> init_data()
  {
    return {-1, -1, -1, 0, std::numeric_limits<double>::max()};
  }

  constexpr bool operator()(sliding_tsp_cand_t<i_t> cand1, sliding_tsp_cand_t<i_t> cand2) const
  {
    return cand1.selection_delta < cand2.selection_delta;
  }
};

template <typename i_t>
struct is_sliding_tsp_uinitialized_t {
  static constexpr sliding_tsp_cand_t<i_t> init_data()
  {
    return {-1, -1, -1, 0, std::numeric_limits<double>::max()};
  }

  __device__ bool operator()(const sliding_tsp_cand_t<i_t>& x) { return x.window_size == -1; }
};

template <typename i_t>
struct is_sliding_tsp_initialized_t {
  __device__ bool operator()(const sliding_tsp_cand_t<i_t>& x) { return x.window_size != -1; }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
