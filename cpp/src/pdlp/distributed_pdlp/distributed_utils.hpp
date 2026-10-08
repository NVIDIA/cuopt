/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

// Slices the data to prepare a split from graph partitioning with halo communication.
template <typename i_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> create_rank_data_from_parts(
  const std::vector<int>& parts,
  const std::vector<i_t>& A_row_offsets,
  const std::vector<i_t>& A_col_indices,
  const std::vector<f_t>& A_values,
  const std::vector<i_t>& A_t_row_offsets,
  const std::vector<i_t>& A_t_col_indices,
  const std::vector<f_t>& A_t_values,
  i_t nb_parts,
  i_t nb_cstr,
  i_t nb_vars,
  i_t nnz);

// Narrows shard data built with a wide global index type down to the solver's index type.
// Multi-GPU PDLP partitions a 64-bit problem so that every shard is individually addressable
// in i_t; this is where that assumption is checked. Consumes its input, releasing each wide
// shard as it is converted. index_t == i_t is a valid (copying) no-op.
template <typename i_t, typename index_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> narrow_rank_data(std::vector<rank_data_t<index_t, f_t>>&& wide);

}  // namespace cuopt::mathematical_optimization::pdlp
