/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <utilities/macros.cuh>

#include <mip_heuristics/mip_constants.hpp>
#include <pdlp/cusparse_view.hpp>
#include <pdlp/pdlp_climber_strategy.hpp>
#include <pdlp/utils.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/sparse/linalg/transpose.cuh>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cub/cub.cuh>

struct double_to_float_functor {
  __host__ __device__ float operator()(double val) const { return static_cast<float>(val); }
};

namespace cuopt::linear_programming::detail {

// All inline factories and aliases for SpMat/DnVec/DnMat live in the header.
// Deleter operator() bodies for SpMVOpDescr/SpMVOpPlan are defined further down
// because they need dlsym-resolved cuSPARSE symbols.

#if CUDA_VER_12_4_UP
struct dynamic_load_runtime {
  static void* get_cusparse_runtime_handle()
  {
    auto close_cudart = [](void* handle) { ::dlclose(handle); };
    auto open_cudart  = []() {
      ::dlerror();
      int major_version;
      RAFT_CUSPARSE_TRY(cusparseGetProperty(libraryPropertyType_t::MAJOR_VERSION, &major_version));
      const std::string libname_ver_o = "libcusparse.so." + std::to_string(major_version) + ".0";
      const std::string libname_ver   = "libcusparse.so." + std::to_string(major_version);
      const std::string libname       = "libcusparse.so";

      auto ptr = ::dlopen(libname_ver_o.c_str(), RTLD_LAZY);
      if (!ptr) { ptr = ::dlopen(libname_ver.c_str(), RTLD_LAZY); }
      if (!ptr) { ptr = ::dlopen(libname.c_str(), RTLD_LAZY); }
      if (ptr) { return ptr; }

      EXE_CUOPT_FAIL("Unable to dlopen cusparse");
    };
    static std::unique_ptr<void, decltype(close_cudart)> cudart_handle{open_cudart(), close_cudart};
    return cudart_handle.get();
  }

  template <typename... Args>
  using function_sig = std::add_pointer_t<cusparseStatus_t(Args...)>;

  template <typename signature>
  static std::optional<signature> function(const char* func_name)
  {
    auto* runtime = get_cusparse_runtime_handle();
    auto* handle  = ::dlsym(runtime, func_name);
    if (!handle) { return std::nullopt; }
    auto* function_ptr = reinterpret_cast<signature>(handle);
    return std::optional<signature>(function_ptr);
  }
};

template <typename... Args>
using cusparse_sig = dynamic_load_runtime::function_sig<Args...>;

using cusparseSpMV_preprocess_sig = cusparse_sig<cusparseHandle_t,
                                                 cusparseOperation_t,
                                                 const void*,
                                                 cusparseConstSpMatDescr_t,
                                                 cusparseConstDnVecDescr_t,
                                                 const void*,
                                                 cusparseDnVecDescr_t,
                                                 cudaDataType,
                                                 cusparseSpMVAlg_t,
                                                 void*>;

// This is tmp until it's added to raft
template <
  typename T,
  typename std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>* = nullptr>
void my_cusparsespmv_preprocess(cusparseHandle_t handle,
                                cusparseOperation_t opA,
                                const T* alpha,
                                cusparseConstSpMatDescr_t matA,
                                cusparseConstDnVecDescr_t vecX,
                                const T* beta,
                                cusparseDnVecDescr_t vecY,
                                cusparseSpMVAlg_t alg,
                                void* externalBuffer,
                                cudaStream_t stream)
{
  auto constexpr float_type = []() constexpr {
    if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
      return CUDA_R_64F;
    }
  }();

  // There can be a missmatch between compiled CUDA version and the runtime CUDA version
  // Since cusparse is only available post >= 12.4 we need to use dlsym to make sure the symbol is
  // present at runtime
  static const auto func =
    dynamic_load_runtime::function<cusparseSpMV_preprocess_sig>("cusparseSpMV_preprocess");
  if (func.has_value()) {
    RAFT_CUSPARSE_TRY(cusparseSetStream(handle, stream));
    RAFT_CUSPARSE_TRY(
      (*func)(handle, opA, alpha, matA, vecX, beta, vecY, float_type, alg, externalBuffer));
  }
}
#endif

// TODO add proper checking
#if CUDA_VER_12_4_UP
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>*>
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
                                cudaStream_t stream)
{
  auto constexpr float_type = []() constexpr {
    if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
      return CUDA_R_64F;
    }
  }();
  RAFT_CUSPARSE_TRY(cusparseSetStream(handle, stream));
  RAFT_CUSPARSE_TRY(cusparseSpMM_preprocess(
    handle, opA, opB, alpha, matA, matB, beta, matC, float_type, alg, externalBuffer));
}
#endif

#if CUDA_VER_13_2_UP
// SpMVOp symbols. Resolved at runtime via dlsym, because the runtime minor version might not match
// the compiled minor version. We can go back to direct linking once CUDA 14 is adopted
using cusparseSpMVOp_destroyDescr_sig = cusparse_sig<cusparseSpMVOpDescr_t>;
using cusparseSpMVOp_destroyPlan_sig  = cusparse_sig<cusparseSpMVOpPlan_t>;
using cusparseSpMVOp_bufferSize_sig   = cusparse_sig<cusparseHandle_t,
                                                     cusparseOperation_t,
                                                     cusparseSpMatDescr_t,
                                                     cusparseDnVecDescr_t,
                                                     cusparseDnVecDescr_t,
                                                     cusparseDnVecDescr_t,
                                                     cudaDataType,
                                                     size_t*>;
using cusparseSpMVOp_createDescr_sig  = cusparse_sig<cusparseHandle_t,
                                                     cusparseSpMVOpDescr_t*,
                                                     cusparseOperation_t,
                                                     cusparseSpMatDescr_t,
                                                     cusparseDnVecDescr_t,
                                                     cusparseDnVecDescr_t,
                                                     cusparseDnVecDescr_t,
                                                     cudaDataType,
                                                     void*>;
using cusparseSpMVOp_createPlan_sig =
  cusparse_sig<cusparseHandle_t, cusparseSpMVOpDescr_t, cusparseSpMVOpPlan_t*, char*, size_t>;
using cusparseSpMVOp_sig = cusparse_sig<cusparseHandle_t,
                                        cusparseSpMVOpPlan_t,
                                        const void*,
                                        const void*,
                                        cusparseDnVecDescr_t,
                                        cusparseDnVecDescr_t,
                                        cusparseDnVecDescr_t>;

// dlsym-resolved create/destroy. The *_destroy_sig variants are reused by the
// deleters; the *_create_sig variants are reused by make_spmvop_{descr,plan}.

void cusparse_spmvop_descr_deleter_t::operator()(cusparseSpMVOpDescr_t descr) const noexcept
{
  if (!descr) { return; }
  static const auto fn =
    dynamic_load_runtime::function<cusparseSpMVOp_destroyDescr_sig>("cusparseSpMVOp_destroyDescr");
  if (fn.has_value()) { RAFT_CUSPARSE_TRY_NO_THROW((*fn)(descr)); }
}

void cusparse_spmvop_plan_deleter_t::operator()(cusparseSpMVOpPlan_t plan) const noexcept
{
  if (!plan) { return; }
  static const auto fn =
    dynamic_load_runtime::function<cusparseSpMVOp_destroyPlan_sig>("cusparseSpMVOp_destroyPlan");
  if (fn.has_value()) { RAFT_CUSPARSE_TRY_NO_THROW((*fn)(plan)); }
}

cusparse_spmvop_descr_uptr make_spmvop_descr(cusparseHandle_t handle,
                                             cusparseOperation_t opA,
                                             cusparse_sp_mat_descr_view matA,
                                             cusparse_dn_vec_descr_view vecX,
                                             cusparse_dn_vec_descr_view vecY,
                                             cusparse_dn_vec_descr_view vecZ,
                                             cudaDataType computeType,
                                             rmm::device_uvector<uint8_t>& buffer)
{
  static const auto fn =
    dynamic_load_runtime::function<cusparseSpMVOp_createDescr_sig>("cusparseSpMVOp_createDescr");
  cusparseSpMVOpDescr_t descr{nullptr};
  RAFT_CUSPARSE_TRY(
    (*fn)(handle, &descr, opA, matA, vecX, vecY, vecZ, computeType, buffer.data()));
  return cusparse_spmvop_descr_uptr{descr};
}

cusparse_spmvop_plan_uptr make_spmvop_plan(cusparseHandle_t handle, cusparseSpMVOpDescr_t descr)
{
  static const auto fn =
    dynamic_load_runtime::function<cusparseSpMVOp_createPlan_sig>("cusparseSpMVOp_createPlan");
  cusparseSpMVOpPlan_t plan{nullptr};
  // cuOpt does not supply user-provided LTO IR; pass nullptr/0 so cuSPARSE JITs internally.
  RAFT_CUSPARSE_TRY((*fn)(handle, descr, &plan, /*ltoIRBuf=*/nullptr, /*ltoIRSize=*/0));
  return cusparse_spmvop_plan_uptr{plan};
}

void cusparse_spmvop_run(cusparseHandle_t handle,
                         cusparseSpMVOpPlan_t plan,
                         const void* alpha,
                         const void* beta,
                         cusparse_dn_vec_descr_view vecX,
                         cusparse_dn_vec_descr_view vecY,
                         cusparse_dn_vec_descr_view vecZ,
                         cudaStream_t stream)
{
  static const auto func = dynamic_load_runtime::function<cusparseSpMVOp_sig>("cusparseSpMVOp");
  RAFT_CUSPARSE_TRY(cusparseSetStream(handle, stream));
  RAFT_CUSPARSE_TRY((*func)(handle, plan, alpha, beta, vecX, vecY, vecZ));
}
#endif

// This cstr is used in pdhg, step size strategy and in cuPDLPx infeasible detection
// A_T is owned by the scaled problem
// It was already transposed in the scaled_problem version
template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(
  raft::handle_t const* handle_ptr,
  const problem_t<i_t, f_t>& op_problem_scaled,
  saddle_point_state_t<i_t, f_t>& current_saddle_point_state,
  rmm::device_uvector<f_t>& _tmp_primal,
  rmm::device_uvector<f_t>& _tmp_dual,
  rmm::device_uvector<f_t>& _potential_next_dual_solution,
  rmm::device_uvector<f_t>& _reflected_primal_solution,
  const std::vector<pdlp_climber_strategy_t>& climber_strategies,
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params,
  bool enable_mixed_precision_spmv)
  : batch_mode_(climber_strategies.size() > 1),
    handle_ptr_(handle_ptr),
    A_T_{op_problem_scaled.reverse_coefficients},
    A_T_offsets_{op_problem_scaled.reverse_offsets},
    A_T_indices_{op_problem_scaled.reverse_constraints},
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    buffer_non_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_batch{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch{0, handle_ptr->get_stream()},
    buffer_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    A_{op_problem_scaled.coefficients},
    A_offsets_{op_problem_scaled.offsets},
    A_indices_{op_problem_scaled.variables},
    climber_strategies_(climber_strategies),
    A_float_{0, handle_ptr->get_stream()},
    A_T_float_{0, handle_ptr->get_stream()},
    buffer_non_transpose_mixed_{0, handle_ptr->get_stream()},
    buffer_transpose_mixed_{0, handle_ptr->get_stream()},
    mixed_precision_enabled_{false}
{
  raft::common::nvtx::range fun_scope("Initializing cuSparse view");

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "PDHG cusparse view init" << std::endl;
#endif

  // setup cusparse view
  A = make_csr<i_t, f_t>(op_problem_scaled.n_constraints,
                         op_problem_scaled.n_variables,
                         op_problem_scaled.nnz,
                         const_cast<i_t*>(op_problem_scaled.offsets.data()),
                         const_cast<i_t*>(op_problem_scaled.variables.data()),
                         const_cast<f_t*>(op_problem_scaled.coefficients.data()));

  A_T = make_csr<i_t, f_t>(op_problem_scaled.n_variables,
                           op_problem_scaled.n_constraints,
                           op_problem_scaled.nnz,
                           const_cast<i_t*>(A_T_offsets_.data()),
                           const_cast<i_t*>(A_T_indices_.data()),
                           const_cast<f_t*>(A_T_.data()));

  c = make_dnvec<f_t>(op_problem_scaled.n_variables,
                      const_cast<f_t*>(op_problem_scaled.objective_coefficients.data()));

  primal_solution = make_dnvec<f_t>(op_problem_scaled.n_variables,
                                    current_saddle_point_state.get_primal_solution().data());
  dual_solution   = make_dnvec<f_t>(op_problem_scaled.n_constraints,
                                  current_saddle_point_state.get_dual_solution().data());

  if (batch_mode_) {
    [[maybe_unused]] const bool is_cupdlpx = is_cupdlpx_restart<i_t, f_t>(hyper_params);
    cuopt_assert(is_cupdlpx, "Batch mode only supported with cuPDLPx restart");
    batch_dual_solutions = make_dnmat<f_t>(op_problem_scaled.n_constraints,
                                           climber_strategies.size(),
                                           climber_strategies.size(),
                                           current_saddle_point_state.get_dual_solution().data(),
                                           CUSPARSE_ORDER_ROW);
    batch_current_AtYs   = make_dnmat<f_t>(op_problem_scaled.n_variables,
                                         climber_strategies.size(),
                                         climber_strategies.size(),
                                         current_saddle_point_state.get_current_AtY().data(),
                                         CUSPARSE_ORDER_ROW);
    batch_potential_next_dual_solution =
      make_dnmat<f_t>(op_problem_scaled.n_constraints,
                      climber_strategies.size(),
                      op_problem_scaled.n_constraints,
                      _potential_next_dual_solution.data(),
                      CUSPARSE_ORDER_COL);
    batch_next_AtYs = make_dnmat<f_t>(op_problem_scaled.n_variables,
                                      climber_strategies.size(),
                                      op_problem_scaled.n_variables,
                                      current_saddle_point_state.get_next_AtY().data(),
                                      CUSPARSE_ORDER_COL);
    cuopt_assert(_reflected_primal_solution.size() > 0, "Reflected primal solution empty");
    batch_reflected_primal_solutions = make_dnmat<f_t>(op_problem_scaled.n_variables,
                                                       climber_strategies.size(),
                                                       climber_strategies.size(),
                                                       _reflected_primal_solution.data(),
                                                       CUSPARSE_ORDER_ROW);
    batch_dual_gradients = make_dnmat<f_t>(op_problem_scaled.n_constraints,
                                           climber_strategies.size(),
                                           climber_strategies.size(),
                                           current_saddle_point_state.get_dual_gradient().data(),
                                           CUSPARSE_ORDER_ROW);
  }

  // Necessary even in non batch mode (because of infeasiblity detection)
  batch_delta_primal_solutions =
    make_dnmat<f_t>(op_problem_scaled.n_variables,
                    climber_strategies.size(),
                    op_problem_scaled.n_variables,
                    current_saddle_point_state.get_delta_primal().data(),
                    CUSPARSE_ORDER_COL);
  batch_delta_dual_solutions = make_dnmat<f_t>(op_problem_scaled.n_constraints,
                                               climber_strategies.size(),
                                               op_problem_scaled.n_constraints,
                                               current_saddle_point_state.get_delta_dual().data(),
                                               CUSPARSE_ORDER_COL);
  batch_tmp_duals            = make_dnmat<f_t>(op_problem_scaled.n_constraints,
                                    climber_strategies.size(),
                                    op_problem_scaled.n_constraints,
                                    _tmp_dual.data(),
                                    CUSPARSE_ORDER_COL);
  batch_tmp_primals          = make_dnmat<f_t>(op_problem_scaled.n_variables,
                                      climber_strategies.size(),
                                      op_problem_scaled.n_variables,
                                      _tmp_primal.data(),
                                      CUSPARSE_ORDER_COL);

  primal_gradient = make_dnvec<f_t>(
    current_saddle_point_state.get_primal_gradient().size(),  // It is 0 in cupdlpx
    current_saddle_point_state.get_primal_gradient().data());
  dual_gradient = make_dnvec<f_t>(op_problem_scaled.n_constraints,
                                  current_saddle_point_state.get_dual_gradient().data());

  current_AtY = make_dnvec<f_t>(op_problem_scaled.n_variables,
                                current_saddle_point_state.get_current_AtY().data());
  next_AtY    = make_dnvec<f_t>(op_problem_scaled.n_variables,
                             current_saddle_point_state.get_next_AtY().data());

  potential_next_dual_solution =
    make_dnvec<f_t>(op_problem_scaled.n_constraints, _potential_next_dual_solution.data());

  tmp_primal = make_dnvec<f_t>(op_problem_scaled.n_variables, _tmp_primal.data());
  tmp_dual   = make_dnvec<f_t>(op_problem_scaled.n_constraints, _tmp_dual.data());
  if (hyper_params.use_reflected_primal_dual) {
    cuopt_assert(_reflected_primal_solution.size() > 0, "Reflected primal solution empty");
    reflected_primal_solution =
      make_dnvec<f_t>(op_problem_scaled.n_variables, _reflected_primal_solution.data());
  }

  const rmm::device_scalar<f_t> alpha{1, handle_ptr->get_stream()};
  const rmm::device_scalar<f_t> beta{0, handle_ptr->get_stream()};
  size_t buffer_size_non_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A.get(),
                                                  c.get(),
                                                  beta.data(),
                                                  dual_solution.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_non_transpose,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose.resize(buffer_size_non_transpose, handle_ptr->get_stream());

  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T.get(),
                                                  dual_solution.get(),
                                                  beta.data(),
                                                  c.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_transpose,
                                                  handle_ptr->get_stream()));

  buffer_transpose.resize(buffer_size_transpose, handle_ptr->get_stream());

  // We need it even in non batch mode since we also use SpMM in infeasibility detection
  size_t buffer_size_transpose_batch = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmm_bufferSize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T.get(),
                                                  batch_delta_dual_solutions.get(),
                                                  beta.data(),
                                                  batch_tmp_primals.get(),
                                                  CUSPARSE_SPMM_CSR_ALG3,
                                                  &buffer_size_transpose_batch,
                                                  handle_ptr->get_stream()));

  buffer_transpose_batch.resize(buffer_size_transpose_batch, handle_ptr->get_stream());
  size_t buffer_size_non_transpose_batch = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmm_bufferSize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A.get(),
                                                  batch_delta_primal_solutions.get(),
                                                  beta.data(),
                                                  batch_tmp_duals.get(),
                                                  CUSPARSE_SPMM_CSR_ALG3,
                                                  &buffer_size_non_transpose_batch,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose_batch.resize(buffer_size_non_transpose_batch, handle_ptr->get_stream());

  // In row row the buffer size may be different
  if (batch_mode_) {
    size_t buffer_size_transpose_batch_row_row = 0;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(
      handle_ptr_->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A_T.get(),
      batch_dual_solutions.get(),
      beta.data(),
      batch_current_AtYs.get(),
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      &buffer_size_transpose_batch_row_row,
      handle_ptr->get_stream()));
    buffer_transpose_batch_row_row_.resize(buffer_size_transpose_batch_row_row,
                                           handle_ptr->get_stream());
    size_t buffer_size_non_transpose_batch_row_row = 0;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(
      handle_ptr_->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A.get(),
      batch_reflected_primal_solutions.get(),
      beta.data(),
      batch_dual_gradients.get(),
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      &buffer_size_non_transpose_batch_row_row,
      handle_ptr->get_stream()));
    buffer_non_transpose_batch_row_row_.resize(buffer_size_non_transpose_batch_row_row,
                                               handle_ptr->get_stream());
  }

#if CUDA_VER_12_4_UP
  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A.get(),
                             c.get(),
                             beta.data(),
                             dual_solution.get(),
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_non_transpose.data(),
                             handle_ptr->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T.get(),
                             dual_solution.get(),
                             beta.data(),
                             c.get(),
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_transpose.data(),
                             handle_ptr->get_stream());
  my_cusparsespmm_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T.get(),
                             batch_delta_dual_solutions.get(),
                             beta.data(),
                             batch_tmp_primals.get(),
                             CUSPARSE_SPMM_CSR_ALG3,
                             buffer_transpose_batch.data(),
                             handle_ptr->get_stream());

  my_cusparsespmm_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A.get(),
                             batch_delta_primal_solutions.get(),
                             beta.data(),
                             batch_tmp_duals.get(),
                             CUSPARSE_SPMM_CSR_ALG3,
                             buffer_non_transpose_batch.data(),
                             handle_ptr->get_stream());
  if (batch_mode_) {
    my_cusparsespmm_preprocess(
      handle_ptr_->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A_T.get(),
      batch_dual_solutions.get(),
      beta.data(),
      batch_current_AtYs.get(),
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      buffer_transpose_batch_row_row_.data(),
      handle_ptr->get_stream());
    my_cusparsespmm_preprocess(
      handle_ptr_->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A.get(),
      batch_reflected_primal_solutions.get(),
      beta.data(),
      batch_dual_gradients.get(),
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      buffer_non_transpose_batch_row_row_.data(),
      handle_ptr->get_stream());
  }
#endif

  if constexpr (std::is_same_v<f_t, double>) {
    if (enable_mixed_precision_spmv && !batch_mode_) {
      mixed_precision_enabled_ = true;

      A_float_.resize(op_problem_scaled.nnz, handle_ptr->get_stream());
      A_T_float_.resize(op_problem_scaled.nnz, handle_ptr->get_stream());

      RAFT_CUDA_TRY(cub::DeviceTransform::Transform(op_problem_scaled.coefficients.data(),
                                                    A_float_.data(),
                                                    op_problem_scaled.nnz,
                                                    double_to_float_functor{},
                                                    handle_ptr->get_stream().value()));

      RAFT_CUDA_TRY(cub::DeviceTransform::Transform(A_T_.data(),
                                                    A_T_float_.data(),
                                                    op_problem_scaled.nnz,
                                                    double_to_float_functor{},
                                                    handle_ptr->get_stream().value()));

      A_mixed_   = make_csr<i_t, float>(op_problem_scaled.n_constraints,
                                      op_problem_scaled.n_variables,
                                      op_problem_scaled.nnz,
                                      const_cast<i_t*>(op_problem_scaled.offsets.data()),
                                      const_cast<i_t*>(op_problem_scaled.variables.data()),
                                      A_float_.data());
      A_T_mixed_ = make_csr<i_t, float>(op_problem_scaled.n_variables,
                                        op_problem_scaled.n_constraints,
                                        op_problem_scaled.nnz,
                                        const_cast<i_t*>(A_T_offsets_.data()),
                                        const_cast<i_t*>(A_T_indices_.data()),
                                        A_T_float_.data());

      const rmm::device_scalar<double> alpha_d{1.0, handle_ptr->get_stream()};
      const rmm::device_scalar<double> beta_d{0.0, handle_ptr->get_stream()};

      size_t buffer_size_non_transpose_mixed =
        mixed_precision_spmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        alpha_d.data(),
                                        A_mixed_.get(),
                                        c.get(),
                                        beta_d.data(),
                                        dual_solution.get(),
                                        CUSPARSE_SPMV_CSR_ALG2,
                                        handle_ptr->get_stream());
      buffer_non_transpose_mixed_.resize(buffer_size_non_transpose_mixed, handle_ptr->get_stream());

      size_t buffer_size_transpose_mixed =
        mixed_precision_spmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        alpha_d.data(),
                                        A_T_mixed_.get(),
                                        dual_solution.get(),
                                        beta_d.data(),
                                        c.get(),
                                        CUSPARSE_SPMV_CSR_ALG2,
                                        handle_ptr->get_stream());
      buffer_transpose_mixed_.resize(buffer_size_transpose_mixed, handle_ptr->get_stream());

#if CUDA_VER_12_4_UP
      mixed_precision_spmv_preprocess(handle_ptr_->get_cusparse_handle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      alpha_d.data(),
                                      A_mixed_.get(),
                                      c.get(),
                                      beta_d.data(),
                                      dual_solution.get(),
                                      CUSPARSE_SPMV_CSR_ALG2,
                                      buffer_non_transpose_mixed_.data(),
                                      handle_ptr->get_stream());

      mixed_precision_spmv_preprocess(handle_ptr_->get_cusparse_handle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      alpha_d.data(),
                                      A_T_mixed_.get(),
                                      dual_solution.get(),
                                      beta_d.data(),
                                      c.get(),
                                      CUSPARSE_SPMV_CSR_ALG2,
                                      buffer_transpose_mixed_.data(),
                                      handle_ptr->get_stream());
#endif
    }
  }
}

// Used by pdlp object for current and average termination condition
// A_T is owned by the problem object and is transposed by the problem
template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(
  raft::handle_t const* handle_ptr,
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
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
  : batch_mode_(climber_strategies.size() > 1),
    handle_ptr_(handle_ptr),
    A_T_{_A_T},
    A_T_offsets_{_A_T_offsets},
    A_T_indices_{_A_T_indices},
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    buffer_non_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_batch{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch{0, handle_ptr->get_stream()},
    buffer_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    A_{op_problem.coefficients},
    A_offsets_{op_problem.offsets},
    A_indices_{op_problem.variables},
    climber_strategies_(climber_strategies),
    A_float_{0, handle_ptr->get_stream()},
    A_T_float_{0, handle_ptr->get_stream()},
    buffer_non_transpose_mixed_{0, handle_ptr->get_stream()},
    buffer_transpose_mixed_{0, handle_ptr->get_stream()},
    mixed_precision_enabled_{false}
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "PDLP cusparse view init" << std::endl;
#endif

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr_->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));

  // setup cusparse view
  A = make_csr<i_t, f_t>(op_problem.n_constraints,
                         op_problem.n_variables,
                         op_problem.nnz,
                         const_cast<i_t*>(op_problem.offsets.data()),
                         const_cast<i_t*>(op_problem.variables.data()),
                         const_cast<f_t*>(op_problem.coefficients.data()));

  A_T = make_csr<i_t, f_t>(op_problem.n_variables,
                           op_problem.n_constraints,
                           op_problem.nnz,
                           const_cast<i_t*>(A_T_offsets_.data()),
                           const_cast<i_t*>(A_T_indices_.data()),
                           const_cast<f_t*>(A_T_.data()));

  c = make_dnvec<f_t>(op_problem.n_variables,
                      const_cast<f_t*>(op_problem.objective_coefficients.data()));

  if (!hyper_params.use_adaptive_step_size_strategy) {
    primal_solution = make_dnvec<f_t>(op_problem.n_variables, _potential_next_primal.data());
    dual_solution   = make_dnvec<f_t>(op_problem.n_constraints, _potential_next_dual.data());
  } else {
    primal_solution = make_dnvec<f_t>(op_problem.n_variables, _primal_solution.data());
    dual_solution   = make_dnvec<f_t>(op_problem.n_constraints, _dual_solution.data());
  }

  tmp_primal = make_dnvec<f_t>(op_problem.n_variables, _tmp_primal.data());
  tmp_dual   = make_dnvec<f_t>(op_problem.n_constraints, _tmp_dual.data());

  if (batch_mode_) {
    [[maybe_unused]] const bool is_cupdlpx = is_cupdlpx_restart<i_t, f_t>(hyper_params);
    cuopt_assert(is_cupdlpx, "Batch mode only supported with cuPDLPx restart");
    batch_primal_solutions = make_dnmat<f_t>(op_problem.n_variables,
                                             climber_strategies.size(),
                                             op_problem.n_variables,
                                             _potential_next_primal.data(),
                                             CUSPARSE_ORDER_COL);
    batch_dual_solutions   = make_dnmat<f_t>(op_problem.n_constraints,
                                           climber_strategies.size(),
                                           op_problem.n_constraints,
                                           _potential_next_dual.data(),
                                           CUSPARSE_ORDER_COL);
    batch_tmp_duals        = make_dnmat<f_t>(op_problem.n_constraints,
                                      climber_strategies.size(),
                                      op_problem.n_constraints,
                                      _tmp_dual.data(),
                                      CUSPARSE_ORDER_COL);
    batch_tmp_primals      = make_dnmat<f_t>(op_problem.n_variables,
                                        climber_strategies.size(),
                                        op_problem.n_variables,
                                        _tmp_primal.data(),
                                        CUSPARSE_ORDER_COL);
  }

  const rmm::device_scalar<f_t> alpha{1, handle_ptr->get_stream()};
  const rmm::device_scalar<f_t> beta{1, handle_ptr->get_stream()};
  size_t buffer_size_non_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A.get(),
                                                  c.get(),
                                                  beta.data(),
                                                  dual_solution.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_non_transpose,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose.resize(buffer_size_non_transpose, handle_ptr->get_stream());

  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T.get(),
                                                  dual_solution.get(),
                                                  beta.data(),
                                                  c.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_transpose,
                                                  handle_ptr->get_stream()));

  buffer_transpose.resize(buffer_size_transpose, handle_ptr->get_stream());

  if (batch_mode_) {
    size_t buffer_size_transpose_batch = 0;
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmm_bufferSize(handle_ptr_->get_cusparse_handle(),
                                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    alpha.data(),
                                                    A_T.get(),
                                                    batch_dual_solutions.get(),
                                                    beta.data(),
                                                    batch_tmp_primals.get(),
                                                    CUSPARSE_SPMM_CSR_ALG3,
                                                    &buffer_size_transpose_batch,
                                                    handle_ptr->get_stream()));
    buffer_transpose_batch.resize(buffer_size_transpose_batch, handle_ptr->get_stream());
    size_t buffer_size_non_transpose_batch = 0;
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmm_bufferSize(handle_ptr_->get_cusparse_handle(),
                                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    alpha.data(),
                                                    A.get(),
                                                    batch_primal_solutions.get(),
                                                    beta.data(),
                                                    batch_tmp_duals.get(),
                                                    CUSPARSE_SPMM_CSR_ALG3,
                                                    &buffer_size_non_transpose_batch,
                                                    handle_ptr->get_stream()));
    buffer_non_transpose_batch.resize(buffer_size_non_transpose_batch, handle_ptr->get_stream());
  }

#if CUDA_VER_12_4_UP
  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A.get(),
                             c.get(),
                             beta.data(),
                             dual_solution.get(),
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_non_transpose.data(),
                             handle_ptr->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T.get(),
                             dual_solution.get(),
                             beta.data(),
                             c.get(),
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_transpose.data(),
                             handle_ptr->get_stream());

  if (batch_mode_) {
    my_cusparsespmm_preprocess(handle_ptr_->get_cusparse_handle(),
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               alpha.data(),
                               A.get(),
                               batch_primal_solutions.get(),
                               beta.data(),
                               batch_tmp_duals.get(),
                               CUSPARSE_SPMM_CSR_ALG3,
                               buffer_non_transpose_batch.data(),
                               handle_ptr->get_stream());

    my_cusparsespmm_preprocess(handle_ptr_->get_cusparse_handle(),
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               alpha.data(),
                               A_T.get(),
                               batch_dual_solutions.get(),
                               beta.data(),
                               batch_tmp_primals.get(),
                               CUSPARSE_SPMM_CSR_ALG3,
                               buffer_transpose_batch.data(),
                               handle_ptr->get_stream());
  }
#endif
}

// Constructor used 3 times in restart strategy for the duality gaps (trust region restart).
// All cuSPARSE descriptors are recreated (no borrowing) to avoid the post-CUDA-12.4 segfault
// observed when copying cusparseSpMatDescr_t handles. tmp_primal/tmp_dual underlying buffer
// pointers are recovered from existing_cusparse_view via cusparseDnVecGetValues; the buffers
// themselves are owned by the parent pdhg_solver_t and outlive every cusparse_view_t.
template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(
  raft::handle_t const* handle_ptr,
  const problem_t<i_t, f_t>& op_problem,  // Just used for the sizes
  const cusparse_view_t<i_t, f_t>& existing_cusparse_view,
  f_t* _primal_solution,  // Solutions of each duality gap container
  f_t* _dual_solution,
  f_t* _primal_gradient,
  f_t* _dual_gradient)
  : handle_ptr_(handle_ptr),
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    buffer_non_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_batch{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch{0, handle_ptr->get_stream()},
    buffer_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    A_T_{existing_cusparse_view.A_T_},                  // Need to be init but not used
    A_T_offsets_{existing_cusparse_view.A_T_offsets_},  // Need to be init but not used
    A_T_indices_{existing_cusparse_view.A_T_indices_},  // Need to be init but not used
    A_{existing_cusparse_view.A_},
    A_offsets_{existing_cusparse_view.A_offsets_},
    A_indices_{existing_cusparse_view.A_indices_},
    climber_strategies_(existing_cusparse_view.climber_strategies_),
    A_float_{0, handle_ptr->get_stream()},
    A_T_float_{0, handle_ptr->get_stream()},
    buffer_non_transpose_mixed_{0, handle_ptr->get_stream()},
    buffer_transpose_mixed_{0, handle_ptr->get_stream()},
    mixed_precision_enabled_{false}
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Restart Strategy cusparse view init" << std::endl;
#endif

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr_->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));

  // All descriptors are reinstantiated below from the saved data pointers (a copy of the
  // cusparseSpMatDescr_t handles segfaults under CUDA 12.4+).
  A = make_csr<i_t, f_t>(op_problem.n_constraints,
                         op_problem.n_variables,
                         op_problem.nnz,
                         const_cast<i_t*>(A_offsets_.data()),
                         const_cast<i_t*>(A_indices_.data()),
                         const_cast<f_t*>(A_.data()));

  A_T = make_csr<i_t, f_t>(op_problem.n_variables,
                           op_problem.n_constraints,
                           op_problem.nnz,
                           const_cast<i_t*>(existing_cusparse_view.A_T_offsets_.data()),
                           const_cast<i_t*>(existing_cusparse_view.A_T_indices_.data()),
                           const_cast<f_t*>(existing_cusparse_view.A_T_.data()));

  c = make_dnvec<f_t>(op_problem.n_variables,
                      const_cast<f_t*>(op_problem.objective_coefficients.data()));

  primal_solution = make_dnvec<f_t>(op_problem.n_variables, _primal_solution);
  dual_solution   = make_dnvec<f_t>(op_problem.n_constraints, _dual_solution);

  primal_gradient = make_dnvec<f_t>(op_problem.n_variables, _primal_gradient);
  dual_gradient   = make_dnvec<f_t>(op_problem.n_constraints, _dual_gradient);

  // Recover the underlying pdhg-owned scratch buffers from the existing view's descriptors so
  // the new descriptors point at the same memory without plumbing the rmm::device_uvector refs
  // through pdlp_restart_strategy_t.
  void* tmp_primal_data{nullptr};
  void* tmp_dual_data{nullptr};
  RAFT_CUSPARSE_TRY(
    cusparseDnVecGetValues(existing_cusparse_view.tmp_primal.get(), &tmp_primal_data));
  RAFT_CUSPARSE_TRY(cusparseDnVecGetValues(existing_cusparse_view.tmp_dual.get(), &tmp_dual_data));
  tmp_primal = make_dnvec<f_t>(op_problem.n_variables, static_cast<f_t*>(tmp_primal_data));
  tmp_dual   = make_dnvec<f_t>(op_problem.n_constraints, static_cast<f_t*>(tmp_dual_data));

  const rmm::device_scalar<f_t> alpha{1, handle_ptr->get_stream()};
  const rmm::device_scalar<f_t> beta{1, handle_ptr->get_stream()};
  size_t buffer_size_non_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A.get(),
                                                  c.get(),
                                                  beta.data(),
                                                  dual_solution.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_non_transpose,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose.resize(buffer_size_non_transpose, handle_ptr->get_stream());

  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T.get(),
                                                  dual_solution.get(),
                                                  beta.data(),
                                                  c.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_transpose,
                                                  handle_ptr->get_stream()));

  buffer_transpose.resize(buffer_size_transpose, handle_ptr->get_stream());

#if CUDA_VER_12_4_UP
  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A.get(),
                             c.get(),
                             beta.data(),
                             dual_solution.get(),
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_non_transpose.data(),
                             handle_ptr->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T.get(),
                             dual_solution.get(),
                             beta.data(),
                             c.get(),
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_transpose.data(),
                             handle_ptr->get_stream());
#endif
}

// Empty constructor used in kkt restart to save memory
template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(
  raft::handle_t const* handle_ptr,
  const rmm::device_uvector<f_t>& dummy_float,  // Empty just to init the const&
  const rmm::device_uvector<i_t>& dummy_int,    // Empty just to init the const&
  const std::vector<pdlp_climber_strategy_t>& climber_strategies)
  : handle_ptr_(handle_ptr),
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    buffer_non_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_spmvop{0, handle_ptr->get_stream()},
    buffer_transpose_batch{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch{0, handle_ptr->get_stream()},
    buffer_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    buffer_non_transpose_batch_row_row_{0, handle_ptr->get_stream()},
    A_T_(dummy_float),
    A_T_offsets_(dummy_int),
    A_T_indices_(dummy_int),
    A_(dummy_float),
    A_offsets_(dummy_int),
    A_indices_(dummy_int),
    climber_strategies_(climber_strategies),
    A_float_{0, handle_ptr->get_stream()},
    A_T_float_{0, handle_ptr->get_stream()},
    buffer_non_transpose_mixed_{0, handle_ptr->get_stream()},
    buffer_transpose_mixed_{0, handle_ptr->get_stream()},
    mixed_precision_enabled_{false}
{
}

// Update FP32 matrix copies after scaling (must be called after scale_problem())
template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::update_mixed_precision_matrices()
{
  if constexpr (std::is_same_v<f_t, double>) {
    if (!mixed_precision_enabled_) { return; }

    RAFT_CUDA_TRY(cub::DeviceTransform::Transform(A_.data(),
                                                  A_float_.data(),
                                                  A_.size(),
                                                  double_to_float_functor{},
                                                  handle_ptr_->get_stream().value()));

    RAFT_CUDA_TRY(cub::DeviceTransform::Transform(A_T_.data(),
                                                  A_T_float_.data(),
                                                  A_T_.size(),
                                                  double_to_float_functor{},
                                                  handle_ptr_->get_stream().value()));

    handle_ptr_->get_stream().synchronize();
  }
}

// Redirects the cuSPARSE CSR structure pointers from op_problem_scaled_ to the original problem
// so the duplicated row/column buffers can be freed.
template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::redirect_cusparse_csr_structure_pointers(
  const problem_t<i_t, f_t>& original_problem)
{
  RAFT_CUSPARSE_TRY(cusparseCsrSetPointers(A.get(),
                                           const_cast<i_t*>(original_problem.offsets.data()),
                                           const_cast<i_t*>(original_problem.variables.data()),
                                           const_cast<f_t*>(A_.data())));

  RAFT_CUSPARSE_TRY(
    cusparseCsrSetPointers(A_T.get(),
                           const_cast<i_t*>(original_problem.reverse_offsets.data()),
                           const_cast<i_t*>(original_problem.reverse_constraints.data()),
                           const_cast<f_t*>(A_T_.data())));

  if constexpr (std::is_same_v<f_t, double>) {
    if (mixed_precision_enabled_) {
      RAFT_CUSPARSE_TRY(cusparseCsrSetPointers(A_mixed_.get(),
                                               const_cast<i_t*>(original_problem.offsets.data()),
                                               const_cast<i_t*>(original_problem.variables.data()),
                                               A_float_.data()));

      RAFT_CUSPARSE_TRY(
        cusparseCsrSetPointers(A_T_mixed_.get(),
                               const_cast<i_t*>(original_problem.reverse_offsets.data()),
                               const_cast<i_t*>(original_problem.reverse_constraints.data()),
                               A_T_float_.data()));
    }
  }
}

// Mixed precision SpMV implementation: FP32 matrix with FP64 vectors and FP64 compute type
size_t mixed_precision_spmv_buffersize(cusparseHandle_t handle,
                                       cusparseOperation_t opA,
                                       const double* alpha,
                                       cusparse_sp_mat_descr_view matA,  // FP32 matrix
                                       cusparse_dn_vec_descr_view vecX,  // FP64 vector
                                       const double* beta,
                                       cusparse_dn_vec_descr_view vecY,  // FP64 vector
                                       cusparseSpMVAlg_t alg,
                                       cudaStream_t stream)
{
  size_t bufferSize = 0;
  RAFT_CUSPARSE_TRY(cusparseSetStream(handle, stream));
  RAFT_CUSPARSE_TRY(cusparseSpMV_bufferSize(
    handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, alg, &bufferSize));
  return bufferSize;
}

void mixed_precision_spmv(cusparseHandle_t handle,
                          cusparseOperation_t opA,
                          const double* alpha,
                          cusparse_sp_mat_descr_view matA,  // FP32 matrix
                          cusparse_dn_vec_descr_view vecX,  // FP64 vector
                          const double* beta,
                          cusparse_dn_vec_descr_view vecY,  // FP64 vector
                          cusparseSpMVAlg_t alg,
                          void* externalBuffer,
                          cudaStream_t stream)
{
  RAFT_CUSPARSE_TRY(cusparseSetStream(handle, stream));
  RAFT_CUSPARSE_TRY(
    cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, alg, externalBuffer));
}

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
                                     cudaStream_t stream)
{
  static const auto func =
    dynamic_load_runtime::function<cusparseSpMV_preprocess_sig>("cusparseSpMV_preprocess");
  if (func.has_value()) {
    RAFT_CUSPARSE_TRY(cusparseSetStream(handle, stream));
    RAFT_CUSPARSE_TRY(
      (*func)(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, alg, externalBuffer));
  }
}
#endif

bool is_cusparse_runtime_mixed_precision_supported()
{
  int major = 0, minor = 0;
  auto status = cusparseGetProperty(libraryPropertyType_t::MAJOR_VERSION, &major);
  if (status != CUSPARSE_STATUS_SUCCESS) return false;
  status = cusparseGetProperty(libraryPropertyType_t::MINOR_VERSION, &minor);
  if (status != CUSPARSE_STATUS_SUCCESS) return false;
  return (major > 12) || (major == 12 && minor >= 5);
}

bool is_cusparse_runtime_spmvop_supported()
{
#if CUDA_VER_13_2_UP
  // Probe the runtimme to ensure cusparseSpMVOp is supported
  static const bool supported =
    dynamic_load_runtime::function<cusparseSpMVOp_sig>("cusparseSpMVOp").has_value();
  return supported;
#else
  return false;
#endif
}

// Creates SpMVOp plans. Must be called after scale_problem() so plans use the scaled matrix.
template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::create_spmv_op_plans(bool is_reflected)
{
#if CUDA_VER_13_2_UP
  if (!is_cusparse_runtime_spmvop_supported() || !(std::is_same_v<f_t, double>)) { return; }
  static const auto buffer_size =
    dynamic_load_runtime::function<cusparseSpMVOp_bufferSize_sig>("cusparseSpMVOp_bufferSize");
  RAFT_CUSPARSE_TRY(
    cusparseSetStream(handle_ptr_->get_cusparse_handle(), handle_ptr_->get_stream()));
  // Prepare buffers for At_y SpMVOp
  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY((*buffer_size)(handle_ptr_->get_cusparse_handle(),
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   A_T.get(),
                                   dual_solution.get(),
                                   current_AtY.get(),
                                   current_AtY.get(),
                                   CUDA_R_64F,
                                   &buffer_size_transpose));
  buffer_transpose_spmvop.resize(buffer_size_transpose, handle_ptr_->get_stream());

  spmv_op_descr_A_t_ = make_spmvop_descr(handle_ptr_->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         A_T.get(),
                                         dual_solution.get(),
                                         current_AtY.get(),
                                         current_AtY.get(),
                                         CUDA_R_64F,
                                         buffer_transpose_spmvop);

  spmv_op_plan_A_t_ =
    make_spmvop_plan(handle_ptr_->get_cusparse_handle(), spmv_op_descr_A_t_.get());

  // Only prepare buffers for A_x if we are using reflected_halpern
  if (is_reflected) {
    size_t buffer_size_non_transpose = 0;
    RAFT_CUSPARSE_TRY((*buffer_size)(handle_ptr_->get_cusparse_handle(),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     A.get(),
                                     reflected_primal_solution.get(),
                                     dual_gradient.get(),
                                     dual_gradient.get(),
                                     CUDA_R_64F,
                                     &buffer_size_non_transpose));
    buffer_non_transpose_spmvop.resize(buffer_size_non_transpose, handle_ptr_->get_stream());

    spmv_op_descr_A_ = make_spmvop_descr(handle_ptr_->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         A.get(),
                                         reflected_primal_solution.get(),
                                         dual_gradient.get(),
                                         dual_gradient.get(),
                                         CUDA_R_64F,
                                         buffer_non_transpose_spmvop);

    spmv_op_plan_A_ = make_spmvop_plan(handle_ptr_->get_cusparse_handle(), spmv_op_descr_A_.get());
  }
#endif
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template class cusparse_view_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class cusparse_view_t<int, double>;
#endif

#if CUDA_VER_12_4_UP
#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template void my_cusparsespmm_preprocess<float>(cusparseHandle_t,
                                                cusparseOperation_t,
                                                cusparseOperation_t,
                                                const float*,
                                                const cusparseSpMatDescr_t,
                                                const cusparseDnMatDescr_t,
                                                const float*,
                                                const cusparseDnMatDescr_t,
                                                cusparseSpMMAlg_t,
                                                void*,
                                                cudaStream_t);
#endif
#if MIP_INSTANTIATE_DOUBLE
template void my_cusparsespmm_preprocess<double>(cusparseHandle_t,
                                                 cusparseOperation_t,
                                                 cusparseOperation_t,
                                                 const double*,
                                                 const cusparseSpMatDescr_t,
                                                 const cusparseDnMatDescr_t,
                                                 const double*,
                                                 const cusparseDnMatDescr_t,
                                                 cusparseSpMMAlg_t,
                                                 void*,
                                                 cudaStream_t);
#endif
#endif

}  // namespace cuopt::linear_programming::detail
