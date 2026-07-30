/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <barrier/device_sparse_matrix.cuh>

#include <linear_algebra/sparse_matrix.hpp>

#include <pdlp/cusparse_view.hpp>

#include <cusparse_v2.h>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

// Lightweight cuSparse view over a sparse matrix descriptor. The dense vectors
// are owned by the caller, which allows many x/y pairs to share one matrix view.
namespace cuopt::mathematical_optimization::barrier {

template <typename i_t, typename f_t>
class cusparse_view_t {
 public:
  // Copy CSC -> owned CSR + CSC-transpose, with preprocess. Supports forward and transpose SpMV.
  // TMP matrix data should already be on the GPU and in CSR not CSC
  cusparse_view_t(raft::handle_t const* handle_ptr, const csc_matrix_t<i_t, f_t>& A);
  ~cusparse_view_t();

  pdlp::cusparse_dn_vec_descr_wrapper_t<f_t> create_vector(rmm::device_uvector<f_t> const& vec);

  template <typename AllocatorA, typename AllocatorB>
  void spmv(f_t alpha,
            const std::vector<f_t, AllocatorA>& x,
            f_t beta,
            std::vector<f_t, AllocatorB>& y);
  void spmv(f_t alpha, rmm::device_uvector<f_t> const& x, f_t beta, rmm::device_uvector<f_t>& y);
  void spmv(f_t alpha,
            pdlp::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
            f_t beta,
            pdlp::cusparse_dn_vec_descr_wrapper_t<f_t> const& y);
  template <typename AllocatorA, typename AllocatorB>
  void transpose_spmv(f_t alpha,
                      const std::vector<f_t, AllocatorA>& x,
                      f_t beta,
                      std::vector<f_t, AllocatorB>& y);
  void transpose_spmv(f_t alpha,
                      rmm::device_uvector<f_t> const& x,
                      f_t beta,
                      rmm::device_uvector<f_t>& y);
  void transpose_spmv(f_t alpha,
                      pdlp::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
                      f_t beta,
                      pdlp::cusparse_dn_vec_descr_wrapper_t<f_t> const& y);

  raft::handle_t const* handle_ptr_{nullptr};

 private:
  void init_spmv_buffer_and_preprocess(cusparseSpMatDescr_t mat,
                                       cusparseDnVecDescr_t x,
                                       cusparseDnVecDescr_t y,
                                       rmm::device_buffer& buffer,
                                       i_t rows);

  rmm::device_uvector<i_t> A_offsets_;
  rmm::device_uvector<i_t> A_indices_;
  rmm::device_uvector<f_t> A_data_;
  cusparseSpMatDescr_t A_{nullptr};
  rmm::device_uvector<i_t> A_T_offsets_;
  rmm::device_uvector<i_t> A_T_indices_;
  rmm::device_uvector<f_t> A_T_data_;
  cusparseSpMatDescr_t A_T_{nullptr};
  rmm::device_buffer spmv_buffer_;
  rmm::device_buffer spmv_buffer_transpose_;
  rmm::device_scalar<f_t> d_one_;
  rmm::device_scalar<f_t> d_minus_one_;
  rmm::device_scalar<f_t> d_zero_;
  i_t rows_{0};
};
}  // namespace cuopt::mathematical_optimization::barrier
