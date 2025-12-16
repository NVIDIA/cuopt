/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../../solution/solution.cuh"
#include "../move_candidates/move_candidates.cuh"
#include "fragment_kernels.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t>
struct search_data_t {
  i_t block_node_id;
  i_t node_id_2;
  // the neighbors of block_node_id that we will use
  raft::device_span<const i_t> nodes_to_consider;
  i_t start_idx_1;
  i_t start_idx_2;
  i_t frag_size_1;
  i_t frag_size_2;
  bool reversed_frag_1;
  bool reversed_frag_2;
  i_t move_type;
  i_t offset;

  DI void print()
  {
    printf(
      "block_node_id %d, \
    node_id_2 %d, \
    start_idx_1 %d, \
    start_idx_2 %d, \
    frag_size_1 %d, \
    frag_size_2 %d, \
    reversed_frag_1 %d, \
    reversed_frag_2 %d, \
    offset %d\n",
      block_node_id,
      node_id_2,
      start_idx_1,
      start_idx_2,
      frag_size_1,
      frag_size_2,
      reversed_frag_1,
      reversed_frag_2,
      offset);
  }
};

template <typename i_t, typename f_t, request_t REQUEST>
bool perform_vrp_search(solution_t<i_t, f_t, REQUEST>& sol,
                        move_candidates_t<i_t, f_t>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
