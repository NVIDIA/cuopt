/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>

#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct partition_loader_t {
  // Read a Metis-style partition file: one part-id per line (whitespace-tolerant),
  // ASCII integers in [0, nb_parts). Returns a flat vector of length
  // nb_cstr + nb_vars, indexed as in create_rank_data_from_parts (cstrs first, then vars).
  static std::vector<i_t> parse_distributed_pdlp_partition_file(std::string const& file);

  // Write a partition vector to file in the same format parse_... reads back:
  // one part-id per line. Useful for inspecting / reusing a computed partition
  // (e.g. CLI --distributed-pdlp-export-parts).
  static void export_distributed_pdlp_partition_file(std::string const& file,
                                                     std::vector<i_t> const& parts);

  // Slices the data to prepare a split from metis partitionning with halo communication
  static std::vector<rank_data_t<i_t, f_t>> create_rank_data_from_parts(
    const std::vector<i_t>& parts,
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
};

}  // namespace cuopt::linear_programming::detail
