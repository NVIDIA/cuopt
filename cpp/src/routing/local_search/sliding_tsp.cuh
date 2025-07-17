/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
