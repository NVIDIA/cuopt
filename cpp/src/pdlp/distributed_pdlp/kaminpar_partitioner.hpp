/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pdlp/distributed_pdlp/partitioner.hpp>

namespace cuopt::linear_programming::detail {

// Multi-threaded k-way partitioner backed by KaMinPar. Builds the same
// constraint/variable bipartite graph as metis_partitioner_t, but runs the
// shared-memory parallel KaMinPar kernel so partitioning scales across all CPU
// cores of a node (set via partitioner_input_t::nb_threads; <= 0 => all
// hardware threads).
template <typename i_t, typename f_t>
class kaminpar_partitioner_t : public partitioner_i<i_t, f_t> {
 public:
  std::vector<i_t> partition(partitioner_input_t<i_t, f_t> const& input) const override;
};

}  // namespace cuopt::linear_programming::detail
