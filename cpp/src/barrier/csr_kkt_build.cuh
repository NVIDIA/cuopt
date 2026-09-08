/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <barrier/device_sparse_matrix.cuh>
#include <barrier/second_order_cone_kernels.cuh>

#include <raft/core/nvtx.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <vector>

namespace cuopt::mathematical_optimization::barrier {

constexpr int augmented_csr_block_size = 256;

// Cone -> augmented-KKT-CSR assembly data.
template <std::integral i_t, std::floating_point f_t>
struct cone_kkt_data_t {
  explicit cone_kkt_data_t(rmm::cuda_stream_view stream)
    : sparse_ids_by_cone(0, stream),
      dense_ids_by_cone(0, stream),
      dense_cone_entry_rank(0, stream),
      dense_block_offsets(0, stream),
      dense_cone_ids(0, stream),
      cone_csr_indices(0, stream),
      cone_Q_values(0, stream),
      dense_cone_diag_csr_indices(0, stream),
      sparse_hessian_diag(0, stream),
      sparse_hessian_Q(0, stream),
      sparse_exp_v_col(0, stream),
      sparse_exp_u_col(0, stream),
      sparse_exp_v_row(0, stream),
      sparse_exp_u_row(0, stream),
      sparse_expansion_D(0, stream),
      sparse_Hs_diag(0, stream)
  {
  }

  // Routing metadata (build_augmented_csr_metadata).
  rmm::device_uvector<i_t> sparse_ids_by_cone;
  rmm::device_uvector<i_t> dense_ids_by_cone;
  rmm::device_uvector<i_t> dense_cone_entry_rank;
  rmm::device_uvector<size_t> dense_block_offsets;
  rmm::device_uvector<i_t> dense_cone_ids;
  i_t dense_soc_kkt_nnz{0};

  // CSR positions / baked-in values (fill_augmented_csr_row_kernel).
  rmm::device_uvector<i_t> cone_csr_indices;
  rmm::device_uvector<f_t> cone_Q_values;
  rmm::device_uvector<i_t> dense_cone_diag_csr_indices;
  rmm::device_uvector<i_t> sparse_hessian_diag;
  rmm::device_uvector<f_t> sparse_hessian_Q;
  rmm::device_uvector<i_t> sparse_exp_v_col;
  rmm::device_uvector<i_t> sparse_exp_u_col;
  rmm::device_uvector<i_t> sparse_exp_v_row;
  rmm::device_uvector<i_t> sparse_exp_u_row;
  rmm::device_uvector<i_t> sparse_expansion_D;
  rmm::device_uvector<f_t> sparse_Hs_diag;  // recomputed every refactorization
};

// Device-usable view of the cone-augmented-CSR buffers (plain spans, safe to pass to kernels).
// augmented_diagonal_indices is the one non-cone field: it indexes the diagonal of every
// augmented-matrix row (linear, dual, and cone alike), but is populated in the same
// fill/refactor-scatter pass as the cone fields below, so it's threaded through here rather than
// via a separate parameter.
template <std::integral i_t, std::floating_point f_t>
struct cone_kkt_views_t {
  raft::device_span<i_t> augmented_diagonal_indices;
  raft::device_span<i_t> cone_csr_indices;
  raft::device_span<f_t> cone_Q_values;
  raft::device_span<i_t> dense_cone_diag_csr_indices;
  raft::device_span<i_t> sparse_hessian_diag;
  raft::device_span<f_t> sparse_hessian_Q;
  raft::device_span<i_t> sparse_exp_v_col;
  raft::device_span<i_t> sparse_exp_u_col;
  raft::device_span<i_t> sparse_exp_v_row;
  raft::device_span<i_t> sparse_exp_u_row;
  raft::device_span<i_t> sparse_expansion_D;
};

template <std::integral i_t, std::floating_point f_t>
cone_kkt_views_t<i_t, f_t> make_cone_kkt_views(cone_kkt_data_t<i_t, f_t>& d,
                                               rmm::device_uvector<i_t>& augmented_diagonal_indices)
{
  return cone_kkt_views_t<i_t, f_t>{cuopt::make_span(augmented_diagonal_indices),
                                    cuopt::make_span(d.cone_csr_indices),
                                    cuopt::make_span(d.cone_Q_values),
                                    cuopt::make_span(d.dense_cone_diag_csr_indices),
                                    cuopt::make_span(d.sparse_hessian_diag),
                                    cuopt::make_span(d.sparse_hessian_Q),
                                    cuopt::make_span(d.sparse_exp_v_col),
                                    cuopt::make_span(d.sparse_exp_u_col),
                                    cuopt::make_span(d.sparse_exp_v_row),
                                    cuopt::make_span(d.sparse_exp_u_row),
                                    cuopt::make_span(d.sparse_expansion_D)};
}

// Per-call read-only spans describing the cone <-> row-element mapping, sourced from cone_data_t
// (cones()) inside form_augmented(). Distinct from cone_kkt_views_t above: this struct
// is entirely read-only classification input (not mutable write-targets), and is genuinely all
// cone data (no non-cone fields).
template <std::integral i_t, std::floating_point f_t>
struct sparse_cone_views_t {
  raft::device_span<const i_t> element_cone_ids{};
  raft::device_span<const size_t> cone_offsets{};
  raft::device_span<const i_t> sparse_cone_ids{};
  raft::device_span<const i_t> sparse_entry_offsets{};
  size_t n_sparse_cone_entries{0};
};

template <std::integral i_t>
__global__ void scatter_sparse_ids_by_cone_kernel(raft::device_span<i_t> sparse_ids_by_cone,
                                                  raft::device_span<const i_t> sparse_cone_ids,
                                                  i_t n_sparse)
{
  const i_t s = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s >= n_sparse) { return; }
  sparse_ids_by_cone[sparse_cone_ids[s]] = s;
}

template <std::integral i_t>
__global__ void build_dense_ids_by_cone_kernel(raft::device_span<i_t> dense_ids_by_cone,
                                               raft::device_span<const i_t> cone_is_sparse,
                                               raft::device_span<const i_t> dense_prefix,
                                               i_t n_cones)
{
  const i_t k = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= n_cones) { return; }
  dense_ids_by_cone[k] = cone_is_sparse[k] ? i_t(-1) : dense_prefix[k];
}

template <std::integral i_t>
__global__ void compact_dense_cone_ids_kernel(raft::device_span<i_t> dense_cone_ids,
                                              raft::device_span<const i_t> dense_prefix,
                                              raft::device_span<const i_t> cone_is_sparse,
                                              i_t n_cones)
{
  const i_t k = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= n_cones || cone_is_sparse[k]) { return; }
  dense_cone_ids[dense_prefix[k]] = k;
}

template <std::integral i_t>
__global__ void build_dense_block_sizes_kernel(raft::device_span<size_t> dense_block_sizes,
                                               raft::device_span<const i_t> dense_cone_ids,
                                               raft::device_span<const size_t> cone_offsets,
                                               i_t n_dense)
{
  const i_t d = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (d >= n_dense) { return; }
  const i_t k          = dense_cone_ids[d];
  const size_t q       = cone_offsets[k + 1] - cone_offsets[k];
  dense_block_sizes[d] = q * q;
}

template <std::integral i_t>
__global__ void build_dense_cone_entry_rank_kernel(raft::device_span<i_t> dense_cone_entry_rank,
                                                   raft::device_span<const i_t> element_cone_ids,
                                                   raft::device_span<const i_t> cone_is_sparse,
                                                   raft::device_span<const i_t> dense_entry_prefix,
                                                   i_t n_cone_entries)
{
  const i_t idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= n_cone_entries) { return; }
  const i_t cone             = element_cone_ids[idx];
  dense_cone_entry_rank[idx] = cone_is_sparse[cone] ? i_t(-1) : dense_entry_prefix[idx];
}

// Count nnz of a cone primal row's Hessian part. The Q entries before and after
// the cone column block [cone_col_start, cone_col_end) are counted identically
// for sparse and dense cones; only the in-block contribution differs:
//   - dense : a full q_k-wide dense block (in-block Q entries are subsumed by it).
//   - sparse: the 3 structural entries (diagonal + two expansion columns v, u),
//             plus the in-block off-diagonal Q entries.
template <std::integral i_t, std::floating_point f_t>
__device__ i_t count_q_cone(const csc_view_t<i_t, f_t>& Q,
                            i_t row,
                            i_t cone_col_start,
                            i_t cone_col_end,
                            i_t q_k,
                            bool is_sparse,
                            i_t nnzQ)
{
  i_t count = is_sparse ? i_t(3) : q_k;
  if (nnzQ == 0) { return count; }
  i_t qp          = Q.col_start[row];
  const i_t q_end = Q.col_start[row + 1];
  // Q entries left of the cone block.
  for (; qp < q_end && Q.i[qp] < cone_col_start; ++qp) {
    ++count;
  }
  // In-block Q entries: dense subsumes them in the q_k columns, so they add
  // nothing; sparse counts the off-diagonals and skips the diagonal (col == row).
  for (; qp < q_end && Q.i[qp] < cone_col_end; ++qp) {
    if (is_sparse && Q.i[qp] != row) { ++count; }
  }
  // Q entries right of the cone block.
  count += q_end - qp;

  return count;
}

template <std::integral i_t, std::floating_point f_t>
__device__ i_t count_primal_row_nnz_device(i_t row,
                                           i_t cone_start,
                                           i_t m_c,
                                           i_t nnzQ,
                                           const csc_view_t<i_t, f_t>& A,
                                           const csc_view_t<i_t, f_t>& Q,
                                           raft::device_span<const i_t> element_cone_ids,
                                           raft::device_span<const size_t> cone_offsets,
                                           raft::device_span<const i_t> sparse_ids_by_cone)
{
  i_t count              = 0;
  const bool is_cone_row = (row >= cone_start) && (row < cone_start + m_c);
  if (is_cone_row) {
    const i_t local_idx      = row - cone_start;
    const i_t k              = element_cone_ids[local_idx];
    const i_t q_k            = static_cast<i_t>(cone_offsets[k + 1] - cone_offsets[k]);
    const i_t cone_col_start = cone_start + static_cast<i_t>(cone_offsets[k]);
    const i_t cone_col_end   = cone_col_start + q_k;
    const bool is_sparse     = sparse_ids_by_cone[k] >= 0;
    count += count_q_cone(Q, row, cone_col_start, cone_col_end, q_k, is_sparse, nnzQ);
  } else if (nnzQ == 0) {
    ++count;
  } else {
    const i_t q_col_beg = Q.col_start[row];
    const i_t q_col_end = Q.col_start[row + 1];
    count += q_col_end - q_col_beg;
    bool has_diagonal = false;
    for (i_t qp = q_col_beg; qp < q_col_end; ++qp) {
      if (Q.i[qp] == row) {
        has_diagonal = true;
        break;
      }
    }
    if (!has_diagonal) { ++count; }
  }
  count += A.col_start[row + 1] - A.col_start[row];
  return count;
}

template <std::integral i_t, std::floating_point f_t>
__global__ void count_augmented_row_nnz_kernel(i_t factorization_size,
                                               i_t n,
                                               i_t m,
                                               i_t cone_start,
                                               i_t m_c,
                                               i_t nnzQ,
                                               csc_view_t<i_t, f_t> A,
                                               csc_view_t<i_t, f_t> Q,
                                               csc_view_t<i_t, f_t> AT,
                                               sparse_cone_views_t<i_t, f_t> cone_views,
                                               raft::device_span<const i_t> sparse_ids_by_cone,
                                               raft::device_span<i_t> row_nnz)
{
  const i_t row = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= factorization_size) { return; }

  if (row < n) {
    row_nnz[row] = count_primal_row_nnz_device(row,
                                               cone_start,
                                               m_c,
                                               nnzQ,
                                               A,
                                               Q,
                                               cone_views.element_cone_ids,
                                               cone_views.cone_offsets,
                                               sparse_ids_by_cone);
    return;
  }

  if (row < n + m) {
    const i_t l  = row - n;
    row_nnz[row] = AT.col_start[l + 1] - AT.col_start[l] + 1;
    return;
  }

  const i_t exp_row = row - (n + m);
  const i_t s       = exp_row / 2;
  const i_t k       = cone_views.sparse_cone_ids[s];
  const i_t q_k     = static_cast<i_t>(cone_views.cone_offsets[k + 1] - cone_views.cone_offsets[k]);
  row_nnz[row]      = q_k + 1;
}

template <std::integral i_t, std::floating_point f_t>
__global__ void fill_augmented_csr_row_kernel(i_t factorization_size,
                                              i_t n,
                                              i_t m,
                                              i_t p,
                                              i_t cone_start,
                                              i_t m_c,
                                              i_t nnzQ,
                                              f_t dual_perturb,
                                              f_t primal_perturb,
                                              csc_view_t<i_t, f_t> A,
                                              csc_view_t<i_t, f_t> Q,
                                              csc_view_t<i_t, f_t> AT,
                                              raft::device_span<const f_t> diag,
                                              raft::device_span<const i_t> row_start,
                                              raft::device_span<i_t> j,
                                              raft::device_span<f_t> x,
                                              sparse_cone_views_t<i_t, f_t> cone_views,
                                              raft::device_span<const i_t> sparse_ids_by_cone,
                                              raft::device_span<const i_t> dense_ids_by_cone,
                                              raft::device_span<const size_t> dense_block_offsets,
                                              raft::device_span<const i_t> dense_cone_entry_rank,
                                              cone_kkt_views_t<i_t, f_t> views)
{
  const i_t row = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= factorization_size) { return; }

  i_t q = row_start[row];

  if (row < n) {
    const bool is_cone_row = (row >= cone_start) && (row < cone_start + m_c);

    // Sparse-cone expansion columns are the largest column indices in the row
    // (>= n + m), so to keep the row column-sorted they are emitted after the
    // A^T block below. exp_v_col >= 0 marks extra columns to emit.
    size_t exp_flat_idx = 0;
    i_t exp_v_col       = -1;
    i_t exp_u_col       = 0;

    if (is_cone_row) {
      const i_t local_idx = row - cone_start;
      const i_t k         = cone_views.element_cone_ids[local_idx];
      const i_t local_r =
        static_cast<i_t>(static_cast<size_t>(local_idx) - cone_views.cone_offsets[k]);
      const i_t q_k = static_cast<i_t>(cone_views.cone_offsets[k + 1] - cone_views.cone_offsets[k]);
      const i_t cone_col_start = cone_start + static_cast<i_t>(cone_views.cone_offsets[k]);
      const i_t sparse_idx     = sparse_ids_by_cone[k];

      if (sparse_idx >= 0) {
        // Column-sorted Hessian region: Q entries with column < row, then the
        // diagonal (folding Q's diagonal if present), then Q entries with
        // column > row.
        i_t qp          = Q.col_start[row];
        const i_t q_end = Q.col_start[row + 1];

        for (; qp < q_end && Q.i[qp] < row; ++qp) {
          j[q]   = Q.i[qp];
          x[q++] = -Q.x[qp];
        }

        f_t q_contrib = f_t(0);
        if (qp < q_end && Q.i[qp] == row) {
          q_contrib = Q.x[qp];
          ++qp;
        }
        const size_t flat_idx               = cone_views.sparse_entry_offsets[sparse_idx] + local_r;
        views.sparse_hessian_diag[flat_idx] = q;
        views.sparse_hessian_Q[flat_idx]    = q_contrib;
        views.augmented_diagonal_indices[row] = q;
        j[q]                                  = row;
        x[q++]                                = -dual_perturb - q_contrib;

        for (; qp < q_end; ++qp) {
          j[q]   = Q.i[qp];
          x[q++] = -Q.x[qp];
        }

        // Defer the expansion columns (largest columns) until after the A^T block.
        exp_flat_idx = flat_idx;
        exp_v_col    = n + m + 2 * sparse_idx;
        exp_u_col    = n + m + 2 * sparse_idx + 1;
      } else {
        const i_t dense_idx  = dense_ids_by_cone[k];
        const i_t block_base = static_cast<i_t>(dense_block_offsets[dense_idx]) + local_r * q_k;

        if (nnzQ > 0) {
          i_t qp          = Q.col_start[row];
          const i_t q_end = Q.col_start[row + 1];
          for (; qp < q_end && Q.i[qp] < cone_col_start; ++qp) {
            j[q]   = Q.i[qp];
            x[q++] = -Q.x[qp];
          }
          for (i_t c = 0; c < q_k; ++c) {
            const i_t col         = cone_col_start + c;
            f_t q_contrib         = f_t(0);
            const f_t initial_val = (c == local_r) ? f_t(-dual_perturb) : f_t(0);
            if (qp < q_end && Q.i[qp] == col) {
              q_contrib = Q.x[qp];
              ++qp;
            }
            views.cone_csr_indices[block_base + c] = q;
            views.cone_Q_values[block_base + c]    = q_contrib;
            if (col == row) {
              views.augmented_diagonal_indices[row] = q;
              const i_t dense_rank                  = dense_cone_entry_rank[local_idx];
              if (dense_rank >= 0) { views.dense_cone_diag_csr_indices[dense_rank] = q; }
            }
            j[q]   = col;
            x[q++] = initial_val - q_contrib;
          }
          for (; qp < q_end; ++qp) {
            j[q]   = Q.i[qp];
            x[q++] = -Q.x[qp];
          }
        } else {
          for (i_t c = 0; c < q_k; ++c) {
            const i_t col                          = cone_col_start + c;
            const f_t initial_val                  = (c == local_r) ? f_t(-dual_perturb) : f_t(0);
            views.cone_csr_indices[block_base + c] = q;
            views.cone_Q_values[block_base + c]    = f_t(0);
            if (col == row) {
              views.augmented_diagonal_indices[row] = q;
              const i_t dense_rank                  = dense_cone_entry_rank[local_idx];
              if (dense_rank >= 0) { views.dense_cone_diag_csr_indices[dense_rank] = q; }
            }
            j[q]   = col;
            x[q++] = initial_val;
          }
        }
      }
    } else if (nnzQ == 0) {
      views.augmented_diagonal_indices[row] = q;
      j[q]                                  = row;
      x[q++]                                = -diag[row] - dual_perturb;
    } else {
      // Column-sorted: Q entries with column < row, the diagonal in place
      // (folding Q's diagonal if present), then Q entries with column > row.
      i_t qp          = Q.col_start[row];
      const i_t q_end = Q.col_start[row + 1];
      for (; qp < q_end && Q.i[qp] < row; ++qp) {
        j[q]   = Q.i[qp];
        x[q++] = -Q.x[qp];
      }
      f_t q_diag = f_t(0);
      if (qp < q_end && Q.i[qp] == row) {
        q_diag = Q.x[qp];
        ++qp;
      }
      views.augmented_diagonal_indices[row] = q;
      j[q]                                  = row;
      x[q++]                                = -q_diag - diag[row] - dual_perturb;
      for (; qp < q_end; ++qp) {
        j[q]   = Q.i[qp];
        x[q++] = -Q.x[qp];
      }
    }

    // A^T block columns in [n, n + m).
    const i_t col_beg = A.col_start[row];
    const i_t col_end = A.col_start[row + 1];
    for (i_t p_idx = col_beg; p_idx < col_end; ++p_idx) {
      j[q]   = A.i[p_idx] + n;
      x[q++] = A.x[p_idx];
    }

    // Sparse-cone expansion columns (columns >= n + m).
    if (exp_v_col >= 0) {
      views.sparse_exp_v_col[exp_flat_idx] = q;
      j[q]                                 = exp_v_col;
      x[q++]                               = f_t(0);
      views.sparse_exp_u_col[exp_flat_idx] = q;
      j[q]                                 = exp_u_col;
      x[q++]                               = f_t(0);
    }
    return;
  }

  // Fill A row and the corresponding augmented diagonal entry.
  if (row < n + m) {
    const i_t l       = row - n;
    const i_t col_beg = AT.col_start[l];
    const i_t col_end = AT.col_start[l + 1];
    for (i_t idx = col_beg; idx < col_end; ++idx) {
      j[q]   = AT.i[idx];
      x[q++] = AT.x[idx];
    }
    views.augmented_diagonal_indices[row] = q;
    j[q]                                  = row;
    x[q++]                                = primal_perturb;
    return;
  }

  // Fill the expansion column and D slot.
  const i_t exp_row = row - (n + m);
  const i_t s       = exp_row / 2;
  const i_t k       = cone_views.sparse_cone_ids[s];
  const i_t q_k     = static_cast<i_t>(cone_views.cone_offsets[k + 1] - cone_views.cone_offsets[k]);
  const i_t cone_col_start = cone_start + static_cast<i_t>(cone_views.cone_offsets[k]);
  const size_t flat_base   = cone_views.sparse_entry_offsets[s];

  raft::device_span<i_t> exp_row_idx =
    (exp_row % 2 == 0) ? views.sparse_exp_v_row : views.sparse_exp_u_row;
  for (i_t jj = 0; jj < q_k; ++jj) {
    exp_row_idx[flat_base + jj] = q;
    j[q]                        = cone_col_start + jj;
    x[q++]                      = f_t(0);
  }
  views.sparse_expansion_D[exp_row] = q;
  j[q]                              = n + m + exp_row;
  x[q++]                            = f_t(0);
}

template <std::integral i_t, std::floating_point f_t>
void build_augmented_csr_metadata(const cone_data_t<i_t, f_t>& cones,
                                  cone_kkt_data_t<i_t, f_t>& metadata,
                                  rmm::cuda_stream_view stream)
{
  raft::common::nvtx::range scope("Barrier: augmented: device CSR metadata");
  const i_t n_cones  = cones.n_cones;
  const i_t n_dense  = cones.n_dense_cones();
  const i_t n_sparse = cones.n_sparse_cones;
  const size_t m_c   = cones.n_cone_entries;

  metadata.sparse_ids_by_cone.resize(n_cones, stream);
  thrust::fill(rmm::exec_policy(stream),
               metadata.sparse_ids_by_cone.begin(),
               metadata.sparse_ids_by_cone.end(),
               i_t(-1));
  if (n_sparse > 0) {
    const size_t grid = raft::ceildiv<size_t>(n_sparse, augmented_csr_block_size);
    scatter_sparse_ids_by_cone_kernel<i_t><<<grid, augmented_csr_block_size, 0, stream.get()>>>(
      cuopt::make_span(metadata.sparse_ids_by_cone),
      cuopt::make_span(cones.sparse_cone_ids),
      n_sparse);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  metadata.dense_ids_by_cone.resize(n_cones, stream);
  metadata.dense_cone_ids.resize(n_dense, stream);
  metadata.dense_block_offsets.resize(static_cast<size_t>(n_dense) + 1, stream);

  if (n_dense > 0) {
    rmm::device_uvector<i_t> is_dense_cone(n_cones, stream);
    thrust::transform(rmm::exec_policy(stream),
                      cones.cone_is_sparse.begin(),
                      cones.cone_is_sparse.end(),
                      is_dense_cone.begin(),
                      [] __device__(i_t is_sparse) { return is_sparse ? i_t(0) : i_t(1); });

    rmm::device_uvector<i_t> dense_prefix(n_cones, stream);
    thrust::exclusive_scan(
      rmm::exec_policy(stream), is_dense_cone.begin(), is_dense_cone.end(), dense_prefix.begin());

    const size_t grid = raft::ceildiv<size_t>(n_cones, augmented_csr_block_size);
    build_dense_ids_by_cone_kernel<i_t><<<grid, augmented_csr_block_size, 0, stream.get()>>>(
      cuopt::make_span(metadata.dense_ids_by_cone),
      cuopt::make_span(cones.cone_is_sparse),
      cuopt::make_span(dense_prefix),
      n_cones);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    compact_dense_cone_ids_kernel<i_t><<<grid, augmented_csr_block_size, 0, stream.get()>>>(
      cuopt::make_span(metadata.dense_cone_ids),
      cuopt::make_span(dense_prefix),
      cuopt::make_span(cones.cone_is_sparse),
      n_cones);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    rmm::device_uvector<size_t> dense_block_sizes(n_dense, stream);
    const size_t dense_grid = raft::ceildiv<size_t>(n_dense, augmented_csr_block_size);
    build_dense_block_sizes_kernel<i_t><<<dense_grid, augmented_csr_block_size, 0, stream.get()>>>(
      cuopt::make_span(dense_block_sizes),
      cuopt::make_span(metadata.dense_cone_ids),
      cuopt::make_span(cones.cone_offsets),
      n_dense);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    thrust::exclusive_scan(rmm::exec_policy(stream),
                           dense_block_sizes.begin(),
                           dense_block_sizes.end(),
                           metadata.dense_block_offsets.begin());
    // exclusive_scan writes only n_dense entries; the final offset (total block nnz) is not
    // produced by the scan, so compute and store it explicitly before reading it back.
    const size_t total_block_nnz = metadata.dense_block_offsets.element(n_dense - 1, stream) +
                                   dense_block_sizes.element(n_dense - 1, stream);
    metadata.dense_block_offsets.set_element_async(n_dense, total_block_nnz, stream);
    metadata.dense_soc_kkt_nnz = static_cast<i_t>(total_block_nnz);
  } else {
    metadata.dense_soc_kkt_nnz = 0;
    metadata.dense_block_offsets.set_element_to_zero_async(0, stream);
  }

  metadata.dense_cone_entry_rank.resize(m_c, stream);
  if (m_c > 0) {
    rmm::device_uvector<i_t> is_dense_entry(m_c, stream);
    // Each cone entry is dense iff its owning cone is dense. Map entry -> cone via
    // element_cone_ids, then look up cone_is_sparse[cone]. NOTE: this must be a unary
    // transform over element_cone_ids (length m_c); zipping directly against
    // cone_is_sparse.begin() would read cone_is_sparse (length n_cones < m_c) out of bounds
    // and index it by entry position instead of by cone.
    const i_t* cone_is_sparse_ptr = cones.cone_is_sparse.data();
    thrust::transform(rmm::exec_policy(stream),
                      cones.element_cone_ids.begin(),
                      cones.element_cone_ids.end(),
                      is_dense_entry.begin(),
                      [cone_is_sparse_ptr] __device__(i_t cone) {
                        return cone_is_sparse_ptr[cone] ? i_t(0) : i_t(1);
                      });

    rmm::device_uvector<i_t> dense_entry_prefix(m_c, stream);
    thrust::exclusive_scan(rmm::exec_policy(stream),
                           is_dense_entry.begin(),
                           is_dense_entry.end(),
                           dense_entry_prefix.begin());

    const size_t entry_grid = raft::ceildiv<size_t>(m_c, augmented_csr_block_size);
    build_dense_cone_entry_rank_kernel<i_t>
      <<<entry_grid, augmented_csr_block_size, 0, stream.get()>>>(
        cuopt::make_span(metadata.dense_cone_entry_rank),
        cuopt::make_span(cones.element_cone_ids),
        cuopt::make_span(cones.cone_is_sparse),
        cuopt::make_span(dense_entry_prefix),
        static_cast<i_t>(m_c));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

template <std::integral i_t, std::floating_point f_t>
i_t build_augmented_csr_on_device(i_t n,
                                  i_t m,
                                  i_t p,
                                  i_t cone_start,
                                  i_t m_c,
                                  i_t nnzQ,
                                  f_t dual_perturb,
                                  f_t primal_perturb,
                                  device_csc_matrix_t<i_t, f_t>& A,
                                  device_csc_matrix_t<i_t, f_t>& Q,
                                  device_csc_matrix_t<i_t, f_t>& AT,
                                  raft::device_span<const f_t> diag,
                                  sparse_cone_views_t<i_t, f_t> cone_views,
                                  cone_kkt_data_t<i_t, f_t>& cone_data,
                                  rmm::device_uvector<i_t>& augmented_diagonal_indices,
                                  device_csr_matrix_t<i_t, f_t>& device_augmented,
                                  rmm::cuda_stream_view stream)
{
  const i_t factorization_size       = n + m + p;
  const csc_view_t<i_t, f_t> A_view  = A.view();
  const csc_view_t<i_t, f_t> Q_view  = Q.view();
  const csc_view_t<i_t, f_t> AT_view = AT.view();

  rmm::device_uvector<i_t> row_nnz(factorization_size, stream);
  {
    raft::common::nvtx::range scope("Barrier: augmented: device CSR count");
    const size_t grid = raft::ceildiv<size_t>(factorization_size, augmented_csr_block_size);
    count_augmented_row_nnz_kernel<i_t, f_t><<<grid, augmented_csr_block_size, 0, stream.get()>>>(
      factorization_size,
      n,
      m,
      cone_start,
      m_c,
      nnzQ,
      A_view,
      Q_view,
      AT_view,
      cone_views,
      cuopt::make_span(cone_data.sparse_ids_by_cone),
      cuopt::make_span(row_nnz));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  i_t total_nnz = 0;
  {
    raft::common::nvtx::range scope("Barrier: augmented: device CSR scan");
    device_augmented.m = factorization_size;
    device_augmented.n = factorization_size;
    device_augmented.row_start.resize(static_cast<size_t>(factorization_size) + 1, stream);

    // Inclusive scan of the per-row counts into row_start[1..], with row_start[0] = 0.
    device_augmented.row_start.set_element_to_zero_async(0, stream);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           row_nnz.begin(),
                           row_nnz.end(),
                           device_augmented.row_start.begin() + 1);

    total_nnz               = device_augmented.row_start.element(factorization_size, stream);
    device_augmented.nz_max = total_nnz;
    device_augmented.j.resize(total_nnz, stream);
    device_augmented.x.resize(total_nnz, stream);
  }

  // Resize a buffer and initialize every element to a sentinel in one shot.
  // A size of 0 makes both the resize and the fill no-ops.
  auto resize_and_fill = [&](auto& buf, size_t size, auto value) {
    buf.resize(size, stream);
    thrust::fill(rmm::exec_policy(stream), buf.begin(), buf.end(), value);
  };

  resize_and_fill(augmented_diagonal_indices, factorization_size, i_t(-1));
  resize_and_fill(cone_data.cone_csr_indices, cone_data.dense_soc_kkt_nnz, i_t(-1));
  resize_and_fill(cone_data.cone_Q_values, cone_data.dense_soc_kkt_nnz, f_t(0));

  const size_t n_sparse_entries = cone_views.n_sparse_cone_entries;
  resize_and_fill(cone_data.sparse_hessian_diag, n_sparse_entries, i_t(-1));
  resize_and_fill(cone_data.sparse_hessian_Q, n_sparse_entries, f_t(0));
  resize_and_fill(cone_data.sparse_exp_v_col, n_sparse_entries, i_t(-1));
  resize_and_fill(cone_data.sparse_exp_u_col, n_sparse_entries, i_t(-1));
  resize_and_fill(cone_data.sparse_exp_v_row, n_sparse_entries, i_t(-1));
  resize_and_fill(cone_data.sparse_exp_u_row, n_sparse_entries, i_t(-1));
  resize_and_fill(cone_data.sparse_expansion_D, p, i_t(-1));

  const i_t n_dense_entries =
    static_cast<i_t>(m_c) - static_cast<i_t>(cone_views.n_sparse_cone_entries);
  resize_and_fill(
    cone_data.dense_cone_diag_csr_indices, std::max<i_t>(0, n_dense_entries), i_t(-1));

  {
    raft::common::nvtx::range scope("Barrier: augmented: device CSR fill");
    auto views        = make_cone_kkt_views(cone_data, augmented_diagonal_indices);
    const size_t grid = raft::ceildiv<size_t>(factorization_size, augmented_csr_block_size);
    fill_augmented_csr_row_kernel<i_t, f_t><<<grid, augmented_csr_block_size, 0, stream.get()>>>(
      factorization_size,
      n,
      m,
      p,
      cone_start,
      m_c,
      nnzQ,
      dual_perturb,
      primal_perturb,
      A_view,
      Q_view,
      AT_view,
      diag,
      cuopt::make_span(device_augmented.row_start),
      cuopt::make_span(device_augmented.j),
      cuopt::make_span(device_augmented.x),
      cone_views,
      cuopt::make_span(cone_data.sparse_ids_by_cone),
      cuopt::make_span(cone_data.dense_ids_by_cone),
      cuopt::make_span(cone_data.dense_block_offsets),
      cuopt::make_span(cone_data.dense_cone_entry_rank),
      views);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  return total_nnz;
}

}  // namespace cuopt::mathematical_optimization::barrier
