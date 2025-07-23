/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <linear_programming/saddle_point.hpp>

#include <mip/problem/problem.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cusparse_v2.h>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class cusparse_view_t {
 public:
  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  saddle_point_state_t<i_t, f_t>& current_saddle_point_state,
                  rmm::device_uvector<f_t>& _tmp_primal,
                  rmm::device_uvector<f_t>& _tmp_dual,
                  rmm::device_uvector<f_t>& _potential_next_dual_solution);

  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  rmm::device_uvector<f_t>& _primal_solution,
                  rmm::device_uvector<f_t>& _dual_solution,
                  rmm::device_uvector<f_t>& _tmp_primal,
                  rmm::device_uvector<f_t>& _tmp_dual,
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
  cusparseSpMatDescr_t A;
  cusparseSpMatDescr_t A_T;
  cusparseDnVecDescr_t c;

  // cusparse view of solutions
  cusparseDnVecDescr_t primal_solution;
  cusparseDnVecDescr_t dual_solution;

  // cusparse view of gradients
  cusparseDnVecDescr_t primal_gradient;
  cusparseDnVecDescr_t dual_gradient;

  // cusparse view of At * Y computation
  cusparseDnVecDescr_t
    current_AtY;  // Only used at very first iteration and after each restart to average
  cusparseDnVecDescr_t next_AtY;  // Next value is swapped out with current after each valid PDHG
                                  // step to save the first AtY SpMV in compute next primal
  cusparseDnVecDescr_t potential_next_dual_solution;

  // cusparse view of auxiliary space needed for some spmv computations
  cusparseDnVecDescr_t tmp_primal;
  cusparseDnVecDescr_t tmp_dual;

  // reuse buffers for cusparse spmv
  rmm::device_uvector<uint8_t> buffer_non_transpose;
  rmm::device_uvector<uint8_t> buffer_transpose;

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
