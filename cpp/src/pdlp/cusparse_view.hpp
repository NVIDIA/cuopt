/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <pdlp/pdlp_climber_strategy.hpp>
#include <pdlp/saddle_point.hpp>

#include <mip_heuristics/problem/problem.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>

#include <cusparse_v2.h>

#include <memory>
#include <type_traits>

#define CUDA_VER_13_2_UP (CUDART_VERSION >= 13020)

namespace cuopt::linear_programming::detail {

// ---------------------------------------------------------------------------
// Deleters and unique_ptr aliases for cuSPARSE opaque handles.
//
// Each cuSPARSE handle (cusparseSpMatDescr_t etc.) is a typedef for a pointer
// to an opaque struct. We use std::remove_pointer_t to feed unique_ptr the
// pointee type so that unique_ptr<...>::pointer matches the cuSPARSE handle.
// ---------------------------------------------------------------------------

struct cusparse_sp_mat_deleter {
  void operator()(cusparseSpMatDescr_t descr) const noexcept
  {
    if (descr) { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descr)); }
  }
};

struct cusparse_dn_vec_deleter {
  void operator()(cusparseDnVecDescr_t descr) const noexcept
  {
    if (descr) { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(descr)); }
  }
};

struct cusparse_dn_mat_deleter {
  void operator()(cusparseDnMatDescr_t descr) const noexcept
  {
    if (descr) { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr)); }
  }
};

using cusparse_sp_mat_uptr =
  std::unique_ptr<std::remove_pointer_t<cusparseSpMatDescr_t>, cusparse_sp_mat_deleter>;
using cusparse_dn_vec_uptr =
  std::unique_ptr<std::remove_pointer_t<cusparseDnVecDescr_t>, cusparse_dn_vec_deleter>;
using cusparse_dn_mat_uptr =
  std::unique_ptr<std::remove_pointer_t<cusparseDnMatDescr_t>, cusparse_dn_mat_deleter>;

// Borrowed views: identical to the raw cuSPARSE handle types but the alias makes the non-owning
// intent explicit at API boundaries. Pair with the *_uptr aliases above:
//   _uptr  -> owns the descriptor; the destructor calls cusparseDestroy*
//   _view  -> non-owning, just the raw handle, lifetime managed elsewhere
using cusparse_sp_mat_descr_view = cusparseSpMatDescr_t;
using cusparse_dn_vec_descr_view = cusparseDnVecDescr_t;
using cusparse_dn_mat_descr_view = cusparseDnMatDescr_t;

// Factory functions replacing the old `wrapper.create(...)` two-phase init.

template <typename i_t, typename f_t>
inline cusparse_sp_mat_uptr make_csr(
  int64_t m, int64_t n, int64_t nnz, i_t* offsets, i_t* indices, f_t* values)
{
  cusparseSpMatDescr_t descr{nullptr};
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatecsr(&descr, m, n, nnz, offsets, indices, values));
  return cusparse_sp_mat_uptr{descr};
}

template <typename f_t>
inline cusparse_dn_vec_uptr make_dnvec(int64_t size, f_t* values)
{
  cusparseDnVecDescr_t descr{nullptr};
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(&descr, size, values));
  return cusparse_dn_vec_uptr{descr};
}

template <typename f_t>
inline cusparse_dn_mat_uptr make_dnmat(
  int64_t row, int64_t col, int64_t ld, f_t* values, cusparseOrder_t order)
{
  cusparseDnMatDescr_t descr{nullptr};
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednmat(&descr, row, col, ld, values, order));
  return cusparse_dn_mat_uptr{descr};
}

#if CUDA_VER_13_2_UP
// ---------------------------------------------------------------------------
// SpMVOp descriptor and plan deleters.
//
// The cusparseSpMVOp_{create,destroy}{Descr,Plan} symbols may not be present
// in the runtime cuSPARSE (the compiled CUDA version may differ from the one
// at runtime), so destruction is dispatched through dlsym. The deleters below
// resolve the destroy symbol at first use and cache it via a function-local
// static.
// ---------------------------------------------------------------------------

struct cusparse_spmvop_descr_deleter {
  void operator()(cusparseSpMVOpDescr_t descr) const noexcept;
};

struct cusparse_spmvop_plan_deleter {
  void operator()(cusparseSpMVOpPlan_t plan) const noexcept;
};

using cusparse_spmvop_descr_uptr =
  std::unique_ptr<std::remove_pointer_t<cusparseSpMVOpDescr_t>, cusparse_spmvop_descr_deleter>;
using cusparse_spmvop_plan_uptr =
  std::unique_ptr<std::remove_pointer_t<cusparseSpMVOpPlan_t>, cusparse_spmvop_plan_deleter>;

// Factories. `make_spmvop_descr` resolves cusparseSpMVOp_createDescr via dlsym.
cusparse_spmvop_descr_uptr make_spmvop_descr(cusparseHandle_t handle,
                                             cusparseOperation_t opA,
                                             cusparse_sp_mat_descr_view matA,
                                             cusparse_dn_vec_descr_view vecX,
                                             cusparse_dn_vec_descr_view vecY,
                                             cusparse_dn_vec_descr_view vecZ,
                                             cudaDataType computeType,
                                             rmm::device_uvector<uint8_t>& buffer);

// `make_spmvop_plan` passes nullptr/0 for ltoIRBuf/ltoIRSize so cuSPARSE JITs
// internally; cuOpt does not supply user-provided LTO IR.
cusparse_spmvop_plan_uptr make_spmvop_plan(cusparseHandle_t handle,
                                           cusparseSpMVOpDescr_t descr);
#endif

template <typename i_t, typename f_t>
class cusparse_view_t {
 public:
  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  saddle_point_state_t<i_t, f_t>& current_saddle_point_state,
                  rmm::device_uvector<f_t>& _tmp_primal,
                  rmm::device_uvector<f_t>& _tmp_dual,
                  rmm::device_uvector<f_t>& _potential_next_dual_solution,
                  rmm::device_uvector<f_t>& _reflected_primal_solution,
                  const std::vector<pdlp_climber_strategy_t>& climber_strategies,
                  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params,
                  bool enable_mixed_precision_spmv);

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
                  const rmm::device_uvector<i_t>& _A_T_indices,
                  const std::vector<pdlp_climber_strategy_t>& climber_strategies,
                  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params);

  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const problem_t<i_t, f_t>& op_problem,
                  const cusparse_view_t<i_t, f_t>& existing_cusparse_view,
                  f_t* _primal_solution,
                  f_t* _dual_solution,
                  f_t* _primal_gradient,
                  f_t* _dual_gradient);

  cusparse_view_t(raft::handle_t const* handle_ptr,
                  const rmm::device_uvector<f_t>&,               // Empty just to init the const&
                  const rmm::device_uvector<i_t>&,               // Empty just to init the const&
                  const std::vector<pdlp_climber_strategy_t>&);  // Empty just to init the const&

  const bool batch_mode_{false};

  raft::handle_t const* handle_ptr_{nullptr};

  // cusparse view of linear program
  cusparse_sp_mat_uptr A;
  cusparse_sp_mat_uptr A_T;
  cusparse_dn_vec_uptr c;

  // cusparse view of solutions
  cusparse_dn_vec_uptr primal_solution;
  cusparse_dn_vec_uptr dual_solution;

  // cusparse view of gradients
  cusparse_dn_vec_uptr primal_gradient;
  cusparse_dn_vec_uptr dual_gradient;

  // cusparse view of batch gradients
  cusparse_dn_mat_uptr batch_dual_gradients;

  // cusparse view of batch solutions
  cusparse_dn_mat_uptr batch_primal_solutions;
  cusparse_dn_mat_uptr batch_dual_solutions;
  cusparse_dn_mat_uptr batch_potential_next_dual_solution;
  cusparse_dn_mat_uptr batch_next_AtYs;
  cusparse_dn_mat_uptr batch_tmp_duals;
  cusparse_dn_mat_uptr batch_reflected_primal_solutions;
  cusparse_dn_mat_uptr batch_delta_primal_solutions;
  cusparse_dn_mat_uptr batch_delta_dual_solutions;

  // cusparse view of At * Y batch computation
  cusparse_dn_mat_uptr batch_current_AtYs;

  // cusparse view of auxillirary space needed for some spmm computations
  cusparse_dn_mat_uptr batch_tmp_primals;

  // cusparse view of At * Y computation
  cusparse_dn_vec_uptr current_AtY;  // Only used at very first iteration and after each restart to
                                     // average
  cusparse_dn_vec_uptr next_AtY;     // Next value is swapped out with current after each valid PDHG
                                     // step to save the first AtY SpMV in compute next primal
  cusparse_dn_vec_uptr potential_next_dual_solution;

  // cusparse view of auxiliary space needed for some spmv computations
  cusparse_dn_vec_uptr tmp_primal;
  cusparse_dn_vec_uptr tmp_dual;

  // reuse buffers for cusparse spmv
  rmm::device_uvector<uint8_t> buffer_non_transpose;
  rmm::device_uvector<uint8_t> buffer_transpose;

  // SpMVOp buffers for A and A_T
  rmm::device_uvector<uint8_t> buffer_non_transpose_spmvop{0, handle_ptr_->get_stream()};
  rmm::device_uvector<uint8_t> buffer_transpose_spmvop{0, handle_ptr_->get_stream()};

#if CUDA_VER_13_2_UP
  // SpMVOp descriptors and plans for A and A_T (descr before plan so dtor destroys plan first)
  cusparse_spmvop_descr_uptr spmv_op_descr_A_;
  cusparse_spmvop_plan_uptr spmv_op_plan_A_;
  cusparse_spmvop_descr_uptr spmv_op_descr_A_t_;
  cusparse_spmvop_plan_uptr spmv_op_plan_A_t_;
#endif
  // reuse buffers for cusparse spmm
  rmm::device_uvector<uint8_t> buffer_transpose_batch;
  rmm::device_uvector<uint8_t> buffer_non_transpose_batch;
  rmm::device_uvector<uint8_t> buffer_transpose_batch_row_row_;
  rmm::device_uvector<uint8_t> buffer_non_transpose_batch_row_row_;
  // Only when using reflection
  cusparse_dn_vec_uptr reflected_primal_solution;

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

  const std::vector<pdlp_climber_strategy_t>& climber_strategies_;

  // Mixed precision SpMV support (FP32 matrix with FP64 vectors/compute)
  // Only used when mixed_precision_enabled_ is true and f_t = double
  rmm::device_uvector<float> A_float_;                       // FP32 copy of A values
  rmm::device_uvector<float> A_T_float_;                     // FP32 copy of A_T values
  cusparse_sp_mat_uptr A_mixed_;                             // FP32 matrix descriptor for A
  cusparse_sp_mat_uptr A_T_mixed_;                           // FP32 matrix descriptor for A_T
  rmm::device_uvector<uint8_t> buffer_non_transpose_mixed_;  // SpMV buffer for mixed precision A
  rmm::device_uvector<uint8_t> buffer_transpose_mixed_;      // SpMV buffer for mixed precision A_T
  bool mixed_precision_enabled_{false};

  // Update FP32 matrix copies after scaling (must be called after scale_problem())
  void update_mixed_precision_matrices();

  // Redirects the cuSPARSE CSR structure pointers from op_problem_scaled_ to the original problem
  // so the duplicated row/column buffers can be freed.
  void redirect_cusparse_csr_structure_pointers(const problem_t<i_t, f_t>& original_problem);
  // Creates SpMVOp plans. Must be called after scale_problem() so plans use the scaled matrix.
  void create_spmv_op_plans(bool is_reflected);
};

// Mixed precision SpMV: FP32 matrix with FP64 vectors and FP64 compute type
void mixed_precision_spmv(cusparseHandle_t handle,
                          cusparseOperation_t opA,
                          const double* alpha,
                          cusparse_sp_mat_descr_view matA,  // FP32 matrix
                          cusparse_dn_vec_descr_view vecX,  // FP64 vector
                          const double* beta,
                          cusparse_dn_vec_descr_view vecY,  // FP64 vector
                          cusparseSpMVAlg_t alg,
                          void* externalBuffer,
                          cudaStream_t stream);

size_t mixed_precision_spmv_buffersize(cusparseHandle_t handle,
                                       cusparseOperation_t opA,
                                       const double* alpha,
                                       cusparse_sp_mat_descr_view matA,  // FP32 matrix
                                       cusparse_dn_vec_descr_view vecX,  // FP64 vector
                                       const double* beta,
                                       cusparse_dn_vec_descr_view vecY,  // FP64 vector
                                       cusparseSpMVAlg_t alg,
                                       cudaStream_t stream);

#if CUDA_VER_12_4_UP
void mixed_precision_spmv_preprocess(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     const double* alpha,
                                     cusparse_sp_mat_descr_view matA,  // FP32 matrix
                                     cusparse_dn_vec_descr_view vecX,  // FP64 vector
                                     const double* beta,
                                     cusparse_dn_vec_descr_view vecY,  // FP64 vector
                                     cusparseSpMVAlg_t alg,
                                     void* externalBuffer,
                                     cudaStream_t stream);
#endif

#if CUDA_VER_12_4_UP
template <
  typename T,
  typename std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>* = nullptr>
void my_cusparsespmm_preprocess(cusparseHandle_t handle,
                                cusparseOperation_t opA,
                                cusparseOperation_t opB,
                                const T* alpha,
                                cusparse_sp_mat_descr_view matA,
                                cusparse_dn_mat_descr_view matB,
                                const T* beta,
                                cusparse_dn_mat_descr_view matC,
                                cusparseSpMMAlg_t alg,
                                void* externalBuffer,
                                cudaStream_t stream);
#endif

bool is_cusparse_runtime_mixed_precision_supported();

// False if cuda version < 13.2 or runtime cuSPARSE does not export SpMVOp symbols. True otherwise.
bool is_cusparse_runtime_spmvop_supported();

#if CUDA_VER_13_2_UP
// Dispatches to the runtime cusparseSpMVOp via dlsym so callers (e.g., pdhg.cu) never
// reference the symbol statically. Caller must have verified
// is_cusparse_runtime_spmvop_supported().
void cusparse_spmvop_run(cusparseHandle_t handle,
                         cusparseSpMVOpPlan_t plan,
                         const void* alpha,
                         const void* beta,
                         cusparse_dn_vec_descr_view vecX,
                         cusparse_dn_vec_descr_view vecY,
                         cusparse_dn_vec_descr_view vecZ,
                         cudaStream_t stream);
#endif

}  // namespace cuopt::linear_programming::detail
