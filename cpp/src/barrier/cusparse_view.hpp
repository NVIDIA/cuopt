/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/sparse_matrix.hpp>

#include <linear_programming/cusparse_view.hpp>

#include <cusparse_v2.h>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

// Lightweight cuSparse view
// Only owns data linked to the associated matrix
// Associated dense vector should be owned by the calling object
// This allows handling many different X Y vector along with one common matrix
namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class cusparse_view_t {
 public:
  // TMP matrix data should already be on the GPU and in CSR not CSC
  cusparse_view_t(raft::handle_t const* handle_ptr, const csc_matrix_t<i_t, f_t>& A);
  ~cusparse_view_t();

  detail::cusparse_dn_vec_descr_wrapper_t<f_t> create_vector(rmm::device_uvector<f_t> const& vec);

  template <typename AllocatorA, typename AllocatorB>
  void spmv(f_t alpha,
            const std::vector<f_t, AllocatorA>& x,
            f_t beta,
            std::vector<f_t, AllocatorB>& y);
  void spmv(f_t alpha, rmm::device_uvector<f_t> const& x, f_t beta, rmm::device_uvector<f_t>& y);
  void spmv(f_t alpha,
            detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
            f_t beta,
            detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& y);
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
                      detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
                      f_t beta,
                      detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& y);

  raft::handle_t const* handle_ptr_{nullptr};

 private:
  rmm::device_uvector<i_t> A_offsets_;
  rmm::device_uvector<i_t> A_indices_;
  rmm::device_uvector<f_t> A_data_;
  cusparseSpMatDescr_t A_;
  rmm::device_uvector<i_t> A_T_offsets_;
  rmm::device_uvector<i_t> A_T_indices_;
  rmm::device_uvector<f_t> A_T_data_;
  cusparseSpMatDescr_t A_T_;
  rmm::device_buffer spmv_buffer_;
  rmm::device_buffer spmv_buffer_transpose_;
  rmm::device_scalar<f_t> d_one_;
  rmm::device_scalar<f_t> d_minus_one_;
  rmm::device_scalar<f_t> d_zero_;
};
}  // namespace cuopt::linear_programming::dual_simplex
