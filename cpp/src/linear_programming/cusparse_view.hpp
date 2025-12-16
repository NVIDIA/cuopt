/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <linear_programming/saddle_point.hpp>

#include <mip/problem/problem.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>

#include <cusparse_v2.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class cusparse_sp_mat_descr_wrapper_t {
 public:
  cusparse_sp_mat_descr_wrapper_t();
  ~cusparse_sp_mat_descr_wrapper_t();

  cusparse_sp_mat_descr_wrapper_t(const cusparse_sp_mat_descr_wrapper_t& other);

  cusparse_sp_mat_descr_wrapper_t& operator=(const cusparse_sp_mat_descr_wrapper_t& other) = delete;

  void create(int64_t m, int64_t n, int64_t nnz, i_t* offsets, i_t* indices, f_t* values);

  operator cusparseSpMatDescr_t() const;

 private:
  cusparseSpMatDescr_t descr_;
  bool need_destruction_;
};

template <typename f_t>
class cusparse_dn_vec_descr_wrapper_t {
 public:
  cusparse_dn_vec_descr_wrapper_t();
  ~cusparse_dn_vec_descr_wrapper_t();

  cusparse_dn_vec_descr_wrapper_t(const cusparse_dn_vec_descr_wrapper_t& other);
  cusparse_dn_vec_descr_wrapper_t& operator=(cusparse_dn_vec_descr_wrapper_t&& other);
  cusparse_dn_vec_descr_wrapper_t& operator=(const cusparse_dn_vec_descr_wrapper_t& other) = delete;

  void create(int64_t size, f_t* values);

  operator cusparseDnVecDescr_t() const;

 private:
  cusparseDnVecDescr_t descr_;
  bool need_destruction_;
};

template <typename i_t, typename f_t>
class cusparse_view_t {
 public:
  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  saddle_point_state_t<i_t, f_t>& current_saddle_point_state,
                  rmm::device_uvector<f_t>& _tmp_primal,
                  rmm::device_uvector<f_t>& _tmp_dual,
                  rmm::device_uvector<f_t>& _potential_next_dual_solution,
                  rmm::device_uvector<f_t>& _reflected_primal_solution);

  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  rmm::device_uvector<f_t>& _primal_solution,
                  rmm::device_uvector<f_t>& _dual_solution,
                  rmm::device_uvector<f_t>& _tmp_primal,
                  rmm::device_uvector<f_t>& _tmp_dual,
                  rmm::device_uvector<f_t>& _potential_next_primal,
                  rmm::device_uvector<f_t>& _potential_next_dual,
                  const rmm::device_uvector<f_t>& _A_T,
                  const rmm::device_uvector<i_t>& _A_T_offsets,
                  const rmm::device_uvector<i_t>& _A_T_indices);

  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  const cusparse_view_t<i_t, f_t>& existing_cusparse_view,
                  f_t* _primal_solution,
                  f_t* _dual_solution,
                  f_t* _primal_gradient,
                  f_t* _dual_gradient);

  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const rmm::device_uvector<f_t>&,  // Empty just to init the const&
                  const rmm::device_uvector<i_t>&   // Empty just to init the const&
  );

  raft::handle_t const* handle_ptr_{nullptr};

  // cusparse view of linear program
  cusparse_sp_mat_descr_wrapper_t<i_t, f_t> A;
  cusparse_sp_mat_descr_wrapper_t<i_t, f_t> A_T;
  cusparse_dn_vec_descr_wrapper_t<f_t> c;

  // cusparse view of solutions
  cusparse_dn_vec_descr_wrapper_t<f_t> primal_solution;
  cusparse_dn_vec_descr_wrapper_t<f_t> dual_solution;

  // cusparse view of gradients
  cusparse_dn_vec_descr_wrapper_t<f_t> primal_gradient;
  cusparse_dn_vec_descr_wrapper_t<f_t> dual_gradient;

  // cusparse view of At * Y computation
  cusparse_dn_vec_descr_wrapper_t<f_t>
    current_AtY;  // Only used at very first iteration and after each restart to average
  cusparse_dn_vec_descr_wrapper_t<f_t>
    next_AtY;  // Next value is swapped out with current after each valid PDHG
               // step to save the first AtY SpMV in compute next primal
  cusparse_dn_vec_descr_wrapper_t<f_t> potential_next_dual_solution;

  // cusparse view of auxiliary space needed for some spmv computations
  cusparse_dn_vec_descr_wrapper_t<f_t> tmp_primal;
  cusparse_dn_vec_descr_wrapper_t<f_t> tmp_dual;

  // reuse buffers for cusparse spmv
  rmm::device_uvector<uint8_t> buffer_non_transpose;
  rmm::device_uvector<uint8_t> buffer_transpose;

  // Only when using reflection
  cusparse_dn_vec_descr_wrapper_t<f_t> reflected_primal_solution;

  // Ref to the A_T found in either
  // Initial problem, we use it to have an unscaled A_T
  // PDLP copy of the problem which holds the scaled version
  // This works under the assumption that while PDLP is optimizing a problem, the original problem
  // is never modified by anyone (including MIP)
  const rmm::device_uvector<f_t>& A_T_;
  const rmm::device_uvector<i_t>& A_T_offsets_;
  const rmm::device_uvector<i_t>& A_T_indices_;

  // original A non-transpose matrix
  const rmm::device_uvector<f_t>& A_;
  const rmm::device_uvector<i_t>& A_offsets_;
  const rmm::device_uvector<i_t>& A_indices_;
};
}  // namespace cuopt::linear_programming::detail
