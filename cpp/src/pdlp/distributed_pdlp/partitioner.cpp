/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Plain C++ translation unit (not .cu): this file contains no device code, and
// KaMinPar's public header (<kaminpar.h>) is C++20 host code that pulls in TBB.
// Keeping the whole partitioner implementation out of nvcc avoids
// device-compiler friction.

#include <pdlp/distributed_pdlp/partitioner.hpp>

#include <utilities/logger.hpp>

#include <cuopt/error.hpp>

#include <kaminpar.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <span>
#include <thread>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
std::vector<i_t> dummy_partitioner_t<i_t, f_t>::partition(
  partitioner_input_t<i_t, f_t> const& input) const
{
  cuopt_expects(input.nb_parts > 0,
                error_type_t::ValidationError,
                "dummy_partitioner: nb_parts must be positive");
  cuopt_expects(input.nb_cstr >= 0 && input.nb_vars >= 0,
                error_type_t::ValidationError,
                "dummy_partitioner: invalid problem dimensions");

  const std::size_t nvtx =
    static_cast<std::size_t>(input.nb_cstr) + static_cast<std::size_t>(input.nb_vars);
  std::vector<i_t> parts(nvtx);
  for (std::size_t i = 0; i < nvtx; ++i) {
    parts[i] = static_cast<i_t>(i % static_cast<std::size_t>(input.nb_parts));
  }
  validate_partition(parts,
                     static_cast<int>(input.nb_cstr),
                     static_cast<int>(input.nb_vars),
                     static_cast<int>(input.nb_parts),
                     "dummy_partitioner");
  return parts;
}

// Builds the bipartite constraint/variable graph induced by A and runs the
// multi-threaded KaMinPar k-way kernel.
//   * nodes [0, nb_cstr)              : constraint nodes
//   * nodes [nb_cstr, nb_cstr+nb_vars): variable nodes
//   * each edge is a nnz between a constraint and a variable
template <typename i_t, typename f_t>
std::vector<i_t> kaminpar_partitioner_t<i_t, f_t>::partition(
  partitioner_input_t<i_t, f_t> const& input) const
{
  cuopt_expects(input.nb_parts >= 1,
                error_type_t::ValidationError,
                "kaminpar_partitioner: nb_parts must be >= 1");
  cuopt_expects(input.nb_cstr >= 0 && input.nb_vars >= 0,
                error_type_t::ValidationError,
                "kaminpar_partitioner: invalid problem dimensions");

  // return trivial partition if only one part
  if (input.nb_parts == 1) {
    CUOPT_LOG_INFO("KaMinPar: nb_parts == 1, returning trivial single-block partition");
    return std::vector<i_t>(static_cast<std::size_t>(input.nb_cstr + input.nb_vars), i_t{0});
  }
  cuopt_expects(input.A.row_offsets != nullptr && input.A.col_indices != nullptr,
                error_type_t::ValidationError,
                "kaminpar_partitioner: A.row_offsets and A.col_indices are required");
  cuopt_expects(input.A_t.row_offsets != nullptr && input.A_t.col_indices != nullptr,
                error_type_t::ValidationError,
                "kaminpar_partitioner: A_t.row_offsets and A_t.col_indices are required");

  auto const& A_offsets   = *input.A.row_offsets;
  auto const& A_cols      = *input.A.col_indices;
  auto const& A_t_offsets = *input.A_t.row_offsets;
  auto const& A_t_cols    = *input.A_t.col_indices;

  cuopt_expects(static_cast<i_t>(A_offsets.size()) == input.nb_cstr + 1,
                error_type_t::ValidationError,
                "kaminpar_partitioner: A.row_offsets size mismatch (expected nb_cstr+1)");
  cuopt_expects(static_cast<i_t>(A_t_offsets.size()) == input.nb_vars + 1,
                error_type_t::ValidationError,
                "kaminpar_partitioner: A_t.row_offsets size mismatch (expected nb_vars+1)");
  cuopt_expects(A_cols.size() == A_t_cols.size(),
                error_type_t::ValidationError,
                "kaminpar_partitioner: A and A_t nnz mismatch");

  const i_t nb_cstr = input.nb_cstr;
  const i_t nb_vars = input.nb_vars;
  const i_t nnz     = static_cast<i_t>(A_cols.size());
  const i_t nvtx    = nb_cstr + nb_vars;

  // Resolve thread count: <= 0 => all hardware threads (1 as a last resort).
  int nthreads = input.nb_threads > 0 ? static_cast<int>(input.nb_threads) : 0;
  if (nthreads <= 0) {
    nthreads = static_cast<int>(std::thread::hardware_concurrency());
    if (nthreads <= 0) { nthreads = 1; }
  }

  // Bipartite CSR using KaMinPar index types (EdgeID for offsets, NodeID for neighbours).
  std::vector<kaminpar::shm::EdgeID> xadj(static_cast<std::size_t>(nvtx) + 1);
  std::vector<kaminpar::shm::NodeID> adjncy(2 * static_cast<std::size_t>(nnz));

  // CSR already represents an adjency list of cstr -> variables.
  // Adding the transpose to represent the var -> cstr edges.
  // Casting the types to KaMinPar friendly types
  for (i_t i = 0; i <= nb_cstr; ++i) {
    xadj[i] = static_cast<kaminpar::shm::EdgeID>(A_offsets[i]);
  }
  for (i_t i = 0; i <= nb_vars; ++i) {
    xadj[nb_cstr + i] =
      static_cast<kaminpar::shm::EdgeID>(A_t_offsets[i]) + static_cast<kaminpar::shm::EdgeID>(nnz);
  }
  for (i_t k = 0; k < nnz; ++k) {
    adjncy[k] =
      static_cast<kaminpar::shm::NodeID>(A_cols[k]) + static_cast<kaminpar::shm::NodeID>(nb_cstr);
  }
  for (i_t k = 0; k < nnz; ++k) {
    adjncy[nnz + k] = static_cast<kaminpar::shm::NodeID>(A_t_cols[k]);
  }

  std::vector<kaminpar::shm::BlockID> block_of(static_cast<std::size_t>(nvtx));

  kaminpar::KaMinPar engine(nthreads, kaminpar::shm::create_default_context());
  engine.set_output_level(kaminpar::OutputLevel::QUIET);
  engine.copy_graph(std::span<const kaminpar::shm::EdgeID>(xadj),
                    std::span<const kaminpar::shm::NodeID>(adjncy));
  engine.set_k(static_cast<kaminpar::shm::BlockID>(input.nb_parts));

  // Allow up to 3% number of nodes imbalance. Could be knobbed for maximum performance
  engine.set_uniform_max_block_weights(0.03);

  // The actual partition computation
  auto t0 = std::chrono::high_resolution_clock::now();

  const kaminpar::shm::EdgeWeight edge_cut =
    engine.compute_partition(std::span<kaminpar::shm::BlockID>(block_of));

  auto t1         = std::chrono::high_resolution_clock::now();
  const double dt = std::chrono::duration<double>(t1 - t0).count();

  CUOPT_LOG_INFO(
    "KaMinPar partitioned bipartite graph: nvtx=%d nnz=%d nb_parts=%d nthreads=%d edge_cut=%lld "
    "in %.3fs",
    static_cast<int>(nvtx),
    static_cast<int>(nnz),
    static_cast<int>(input.nb_parts),
    nthreads,
    static_cast<long long>(edge_cut),
    dt);

  std::vector<i_t> parts(static_cast<std::size_t>(nvtx));
  for (i_t i = 0; i < nvtx; ++i) {
    parts[i] = static_cast<i_t>(block_of[i]);
  }

  validate_partition(parts,
                     static_cast<int>(nb_cstr),
                     static_cast<int>(nb_vars),
                     static_cast<int>(input.nb_parts),
                     "kaminpar_partitioner");
  return parts;
}

void validate_partition(
  std::vector<int> const& parts, int nb_cstr, int nb_vars, int nb_parts, char const* context)
{
  const std::size_t expected =
    static_cast<std::size_t>(nb_cstr) + static_cast<std::size_t>(nb_vars);
  cuopt_expects(parts.size() == expected,
                error_type_t::ValidationError,
                "%s: expected %zu part entries (cstrs + vars), got %zu",
                context,
                expected,
                parts.size());
  cuopt_expects(
    nb_parts > 0, error_type_t::ValidationError, "%s: nb_parts must be positive", context);
  if (parts.empty()) { return; }
  const auto [min_it, max_it] = std::minmax_element(parts.begin(), parts.end());
  cuopt_expects(*min_it >= 0,
                error_type_t::ValidationError,
                "%s: partition ids must be non-negative (min=%d)",
                context,
                static_cast<int>(*min_it));
  cuopt_expects(*max_it < nb_parts,
                error_type_t::ValidationError,
                "%s: partition ids must be in [0, %d) (max=%d)",
                context,
                static_cast<int>(nb_parts),
                static_cast<int>(*max_it));
}

template <typename i_t, typename f_t>
std::unique_ptr<partitioner_i<i_t, f_t>> make_partitioner(partitioner_kind_t kind)
{
  switch (kind) {
    case partitioner_kind_t::Dummy: return std::make_unique<dummy_partitioner_t<i_t, f_t>>();
    case partitioner_kind_t::KaMinPar: return std::make_unique<kaminpar_partitioner_t<i_t, f_t>>();
  }
  cuopt_expects(
    false, error_type_t::RuntimeError, "make_partitioner: unsupported partitioner kind");
  return nullptr;
}

template class dummy_partitioner_t<int, double>;
template class kaminpar_partitioner_t<int, double>;
template std::unique_ptr<partitioner_i<int, double>> make_partitioner<int, double>(
  partitioner_kind_t);

}  // namespace cuopt::linear_programming::detail
