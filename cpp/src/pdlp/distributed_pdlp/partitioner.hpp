/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

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
  // Number of CPU threads the partitioner may use. Only honored by the
  // multi-threaded KaMinPar backend. <= 0 means "auto"
  i_t nb_threads{0};
  // Constraint matrix A (rows = constraints, cols = variables).
  csr_host_view_t<i_t, f_t> A{};
  // Transpose A_t (rows = variables, cols = constraints)
  csr_host_view_t<i_t, f_t> A_t{};
};

// Dummy: round-robin, no graph (single-shard / debugging).
// KaMinPar: multi-threaded KaMinPar (preferred for multi-shard partitioning).
enum class partitioner_kind_t { Dummy, KaMinPar };

template <typename i_t, typename f_t>
class partitioner_i {
 public:
  virtual ~partitioner_i()                                                             = default;
  virtual std::vector<i_t> partition(partitioner_input_t<i_t, f_t> const& input) const = 0;
};

template <typename i_t, typename f_t>
class dummy_partitioner_t : public partitioner_i<i_t, f_t> {
 public:
  std::vector<i_t> partition(partitioner_input_t<i_t, f_t> const& input) const override;
};

// Multi-threaded k-way partitioner backed by KaMinPar. Builds a
// constraint/variable bipartite graph and runs the shared-memory parallel
// KaMinPar kernel so partitioning scales across all CPU cores of a node.
template <typename i_t, typename f_t>
class kaminpar_partitioner_t : public partitioner_i<i_t, f_t> {
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

}  // namespace cuopt::mathematical_optimization::pdlp
