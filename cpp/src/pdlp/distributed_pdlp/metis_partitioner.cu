/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/metis_partitioner.hpp>
#include <pdlp/distributed_pdlp/partitioner.hpp>

#include <utilities/logger.hpp>

#include <cuopt/error.hpp>

#include <metis.h>

#include <chrono>
#include <cstddef>
#include <vector>

namespace cuopt::linear_programming::detail {

// Builds the bipartite constraint/variable graph induced by A and runs
// METIS_PartGraphKway to assign each of the (nb_cstr + nb_vars) nodes to a
// part in [0, nb_parts). Layout matches metis_tests:
//   * nodes [0, nb_cstr)              : constraint nodes
//   * nodes [nb_cstr, nb_cstr+nb_vars): variable nodes
//   * undirected edges from each A nonzero (one half via A, one via A_t)
// The output is consumed by partition_loader_t::create_rank_data_from_parts.
template <typename i_t, typename f_t>
std::vector<i_t> metis_partitioner_t<i_t, f_t>::partition(
  partitioner_input_t<i_t, f_t> const& input) const
{
  cuopt_expects(input.nb_parts > 0,
                error_type_t::ValidationError,
                "metis_partitioner: nb_parts must be positive");
  cuopt_expects(input.nb_cstr >= 0 && input.nb_vars >= 0,
                error_type_t::ValidationError,
                "metis_partitioner: invalid problem dimensions");

  cuopt_expects(input.A.row_offsets != nullptr && input.A.col_indices != nullptr,
                error_type_t::ValidationError,
                "metis_partitioner: A.row_offsets and A.col_indices are required");
  cuopt_expects(input.A_t.row_offsets != nullptr && input.A_t.col_indices != nullptr,
                error_type_t::ValidationError,
                "metis_partitioner: A_t.row_offsets and A_t.col_indices are required");

  auto const& A_offsets   = *input.A.row_offsets;
  auto const& A_cols      = *input.A.col_indices;
  auto const& A_t_offsets = *input.A_t.row_offsets;
  auto const& A_t_cols    = *input.A_t.col_indices;

  cuopt_expects(static_cast<i_t>(A_offsets.size()) == input.nb_cstr + 1,
                error_type_t::ValidationError,
                "metis_partitioner: A.row_offsets size mismatch (expected nb_cstr+1)");
  cuopt_expects(static_cast<i_t>(A_t_offsets.size()) == input.nb_vars + 1,
                error_type_t::ValidationError,
                "metis_partitioner: A_t.row_offsets size mismatch (expected nb_vars+1)");
  cuopt_expects(A_cols.size() == A_t_cols.size(),
                error_type_t::ValidationError,
                "metis_partitioner: A and A_t nnz mismatch");

  const i_t nb_cstr = input.nb_cstr;
  const i_t nb_vars = input.nb_vars;
  const i_t nnz     = static_cast<i_t>(A_cols.size());
  const i_t nvtx    = nb_cstr + nb_vars;

  // Bipartite CSR. Same construction as metis_tests/src/main.cpp:
  //   xadj   has length nvtx + 1
  //   adjncy has length 2 * nnz (each A nonzero contributes one half-edge
  //          from cstr side via A and one half-edge from var side via A_t)
  std::vector<idx_t> xadj(nvtx + 1);
  std::vector<idx_t> adjncy(2 * static_cast<std::size_t>(nnz));

  // cstr-side row offsets: A_offsets[0..nb_cstr] (no shift).
  for (i_t i = 0; i <= nb_cstr; ++i) { xadj[i] = static_cast<idx_t>(A_offsets[i]); }
  // var-side row offsets: A_t_offsets[0..nb_vars], shifted by +nnz so that
  // they index into the second half of adjncy.
  for (i_t i = 0; i <= nb_vars; ++i) {
    xadj[nb_cstr + i] = static_cast<idx_t>(A_t_offsets[i]) + static_cast<idx_t>(nnz);
  }

  // cstr-side neighbours: A_cols[i] shifted by +nb_cstr to index into the
  // variable node block.
  for (i_t k = 0; k < nnz; ++k) {
    adjncy[k] = static_cast<idx_t>(A_cols[k]) + static_cast<idx_t>(nb_cstr);
  }
  // var-side neighbours: A_t_cols[i] already in [0, nb_cstr).
  for (i_t k = 0; k < nnz; ++k) {
    adjncy[nnz + k] = static_cast<idx_t>(A_t_cols[k]);
  }

  idx_t metis_options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(metis_options);
  metis_options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

  idx_t metis_nvtx = static_cast<idx_t>(nvtx);
  idx_t ncon       = 1;
  idx_t nparts     = static_cast<idx_t>(input.nb_parts);
  idx_t objval     = 0;
  std::vector<idx_t> metis_parts(nvtx);

  auto t0 = std::chrono::high_resolution_clock::now();
  const int status = METIS_PartGraphKway(&metis_nvtx,
                                         &ncon,
                                         xadj.data(),
                                         adjncy.data(),
                                         /*vwgt=*/nullptr,
                                         /*vsize=*/nullptr,
                                         /*adjwgt=*/nullptr,
                                         &nparts,
                                         /*tpwgts=*/nullptr,
                                         /*ubvec=*/nullptr,
                                         metis_options,
                                         &objval,
                                         metis_parts.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  const double dt = std::chrono::duration<double>(t1 - t0).count();
  cuopt_expects(status == METIS_OK,
                error_type_t::RuntimeError,
                "METIS_PartGraphKway failed (status=%d)",
                status);
  CUOPT_LOG_INFO(
    "METIS partitioned bipartite graph: nvtx=%d nnz=%d nb_parts=%d edge_cut=%lld in %.3fs",
    static_cast<int>(nvtx),
    static_cast<int>(nnz),
    static_cast<int>(input.nb_parts),
    static_cast<long long>(objval),
    dt);

  std::vector<i_t> parts(static_cast<std::size_t>(nvtx));
  for (i_t i = 0; i < nvtx; ++i) { parts[i] = static_cast<i_t>(metis_parts[i]); }

  validate_partition(parts,
                     static_cast<int>(nb_cstr),
                     static_cast<int>(nb_vars),
                     static_cast<int>(input.nb_parts),
                     "metis_partitioner");
  return parts;
}

template class metis_partitioner_t<int, double>;

}  // namespace cuopt::linear_programming::detail
