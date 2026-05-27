/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

// Non-owning view of a host CSR matrix (A or A_t).
template <typename i_t, typename f_t>
struct csr_host_view_t {
  std::vector<i_t> const* row_offsets{nullptr};
  std::vector<i_t> const* col_indices{nullptr};
  std::vector<f_t> const* values{nullptr};  // optional; unused by topology-only partitioners
  i_t num_rows{0};
  i_t num_cols{0};
};

// Inputs shared by all distributed-PDLP partitioners.
// Returns a flat vector of length (nb_cstr + nb_vars): constraint part-ids first,
// then variable part-ids, each in [0, nb_parts).
template <typename i_t, typename f_t>
struct partitioner_input_t {
  i_t nb_cstr{0};
  i_t nb_vars{0};
  i_t nb_parts{0};
  // Constraint matrix A (rows = constraints, cols = variables).
  csr_host_view_t<i_t, f_t> A{};
  // Transpose A_t (rows = variables, cols = constraints). Optional for partitioners
  // that build a bipartite graph (e.g. METIS); dummy partitioner ignores both matrices.
  csr_host_view_t<i_t, f_t> A_t{};
};

enum class partitioner_kind_t { Dummy /*, Metis */ };

template <typename i_t, typename f_t>
class partitioner_i {
 public:
  virtual ~partitioner_i() = default;
  virtual std::vector<i_t> partition(partitioner_input_t<i_t, f_t> const& input) const = 0;
};

template <typename i_t, typename f_t>
class dummy_partitioner_t : public partitioner_i<i_t, f_t> {
 public:
  std::vector<i_t> partition(partitioner_input_t<i_t, f_t> const& input) const override;
};

void validate_partition(std::vector<int> const& parts,
                        int nb_cstr,
                        int nb_vars,
                        int nb_parts,
                        char const* context = "partition");

template <typename i_t, typename f_t>
std::unique_ptr<partitioner_i<i_t, f_t>> make_partitioner(partitioner_kind_t kind);

}  // namespace cuopt::linear_programming::detail
