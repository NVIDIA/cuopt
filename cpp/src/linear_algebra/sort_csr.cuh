/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <cub/cub.cuh>

#include <raft/core/nvtx.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {

namespace mathematical_optimization {

template <typename i_t, typename f_t>
void sort_csr(optimization_problem_t<i_t, f_t>& op_problem)
{
  raft::common::nvtx::range fun_scope("sort_csr");
  auto stream_view = op_problem.get_handle_ptr()->get_stream();
  rmm::device_uvector<std::byte> d_tmp_storage_bytes(0, stream_view);
  size_t tmp_storage_bytes{0};
  auto num_segments = op_problem.get_n_constraints();
  auto num_items    = op_problem.get_nnz();
  cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                      tmp_storage_bytes,
                                      op_problem.get_constraint_matrix_indices().data(),
                                      op_problem.get_constraint_matrix_indices().data(),
                                      op_problem.get_constraint_matrix_values().data(),
                                      op_problem.get_constraint_matrix_values().data(),
                                      num_items,
                                      num_segments,
                                      op_problem.get_constraint_matrix_offsets().data(),
                                      op_problem.get_constraint_matrix_offsets().data() + 1,
                                      stream_view);
  d_tmp_storage_bytes.resize(tmp_storage_bytes, stream_view);
  cub::DeviceSegmentedSort::SortPairs(d_tmp_storage_bytes.data(),
                                      tmp_storage_bytes,
                                      op_problem.get_constraint_matrix_indices().data(),
                                      op_problem.get_constraint_matrix_indices().data(),
                                      op_problem.get_constraint_matrix_values().data(),
                                      op_problem.get_constraint_matrix_values().data(),
                                      num_items,
                                      num_segments,
                                      op_problem.get_constraint_matrix_offsets().data(),
                                      op_problem.get_constraint_matrix_offsets().data() + 1,
                                      stream_view);
  RAFT_CHECK_CUDA(stream_view);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view));
}

}  // namespace mathematical_optimization
}  // namespace cuopt
