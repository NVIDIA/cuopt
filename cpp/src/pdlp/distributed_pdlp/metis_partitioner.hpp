/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pdlp/distributed_pdlp/partitioner.hpp>

namespace cuopt::linear_programming::detail {

// METIS k-way partitioner on the constraint/variable bipartite graph induced by A.
// Requires partitioner_input_t::A and A_t (or A row_offsets/col_indices only — the
// implementation builds the bipartite adjacency the same way as metis_tests:
// cstr nodes [0, nb_cstr), var nodes [nb_cstr, nb_cstr+nb_vars), edges from A and A_t).
//
// Wire into make_partitioner() once METIS is an optional cuOpt dependency.
template <typename i_t, typename f_t>
class metis_partitioner_t : public partitioner_i<i_t, f_t> {
 public:
  std::vector<i_t> partition(partitioner_input_t<i_t, f_t> const& input) const override;
};

}  // namespace cuopt::linear_programming::detail
