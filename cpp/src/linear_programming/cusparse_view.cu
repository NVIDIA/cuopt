/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>

#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/transpose.cuh>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

namespace cuopt::linear_programming::detail {

// cusparse_sp_mat_descr_wrapper_t implementation
template <typename i_t, typename f_t>
cusparse_sp_mat_descr_wrapper_t<i_t, f_t>::cusparse_sp_mat_descr_wrapper_t()
  : need_destruction_(false)
{
}

template <typename i_t, typename f_t>
cusparse_sp_mat_descr_wrapper_t<i_t, f_t>::~cusparse_sp_mat_descr_wrapper_t()
{
  if (need_destruction_) { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descr_)); }
}

template <typename i_t, typename f_t>
cusparse_sp_mat_descr_wrapper_t<i_t, f_t>::cusparse_sp_mat_descr_wrapper_t(
  const cusparse_sp_mat_descr_wrapper_t& other)
  : descr_(other.descr_), need_destruction_(false)
{
}

template <typename i_t, typename f_t>
void cusparse_sp_mat_descr_wrapper_t<i_t, f_t>::create(
  int64_t m, int64_t n, int64_t nnz, i_t* offsets, i_t* indices, f_t* values)
{
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatecsr(&descr_, m, n, nnz, offsets, indices, values));
  need_destruction_ = true;
}

template <typename i_t, typename f_t>
cusparse_sp_mat_descr_wrapper_t<i_t, f_t>::operator cusparseSpMatDescr_t() const
{
  return descr_;
}

// cusparse_dn_vec_descr_wrapper_t implementation
template <typename f_t>
cusparse_dn_vec_descr_wrapper_t<f_t>::cusparse_dn_vec_descr_wrapper_t() : need_destruction_(false)
{
}

template <typename f_t>
cusparse_dn_vec_descr_wrapper_t<f_t>::~cusparse_dn_vec_descr_wrapper_t()
{
  if (need_destruction_) { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(descr_)); }
}

template <typename f_t>
cusparse_dn_vec_descr_wrapper_t<f_t>::cusparse_dn_vec_descr_wrapper_t(
  const cusparse_dn_vec_descr_wrapper_t& other)
  : descr_(other.descr_), need_destruction_(false)
{
}

template <typename f_t>
cusparse_dn_vec_descr_wrapper_t<f_t>& cusparse_dn_vec_descr_wrapper_t<f_t>::operator=(
  cusparse_dn_vec_descr_wrapper_t<f_t>&& other)
{
  if (need_destruction_) { RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(descr_)); }
  descr_                  = other.descr_;
  other.need_destruction_ = false;
  return *this;
}

template <typename f_t>
void cusparse_dn_vec_descr_wrapper_t<f_t>::create(int64_t size, f_t* values)
{
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(&descr_, size, values));
  need_destruction_ = true;
}

template <typename f_t>
cusparse_dn_vec_descr_wrapper_t<f_t>::operator cusparseDnVecDescr_t() const
{
  return descr_;
}

#define CUDA_VER_12_4_UP (CUDART_VERSION >= 12040)

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

// This cstr is used in pdhg
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
  rmm::device_uvector<f_t>& _reflected_primal_solution)
  : handle_ptr_(handle_ptr),
    A{},
    A_T{},
    c{},
    primal_solution{},
    dual_solution{},
    primal_gradient{},
    dual_gradient{},
    current_AtY{},
    next_AtY{},
    potential_next_dual_solution{},
    tmp_primal{},
    tmp_dual{},
    A_T_{op_problem_scaled.reverse_coefficients},
    A_T_offsets_{op_problem_scaled.reverse_offsets},
    A_T_indices_{op_problem_scaled.reverse_constraints},
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    A_{op_problem_scaled.coefficients},
    A_offsets_{op_problem_scaled.offsets},
    A_indices_{op_problem_scaled.variables}
{
  raft::common::nvtx::range fun_scope("Initializing cuSparse view");

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "PDHG cusparse view init" << std::endl;
#endif

  // setup cusparse view
  A.create(op_problem_scaled.n_constraints,
           op_problem_scaled.n_variables,
           op_problem_scaled.nnz,
           const_cast<i_t*>(op_problem_scaled.offsets.data()),
           const_cast<i_t*>(op_problem_scaled.variables.data()),
           const_cast<f_t*>(op_problem_scaled.coefficients.data()));

  A_T.create(op_problem_scaled.n_variables,
             op_problem_scaled.n_constraints,
             op_problem_scaled.nnz,
             const_cast<i_t*>(A_T_offsets_.data()),
             const_cast<i_t*>(A_T_indices_.data()),
             const_cast<f_t*>(A_T_.data()));

  c.create(op_problem_scaled.n_variables,
           const_cast<f_t*>(op_problem_scaled.objective_coefficients.data()));

  primal_solution.create(op_problem_scaled.n_variables,
                         current_saddle_point_state.get_primal_solution().data());
  dual_solution.create(op_problem_scaled.n_constraints,
                       current_saddle_point_state.get_dual_solution().data());

  primal_gradient.create(op_problem_scaled.n_variables,
                         current_saddle_point_state.get_primal_gradient().data());
  dual_gradient.create(op_problem_scaled.n_constraints,
                       current_saddle_point_state.get_dual_gradient().data());

  current_AtY.create(op_problem_scaled.n_variables,
                     current_saddle_point_state.get_current_AtY().data());
  next_AtY.create(op_problem_scaled.n_variables, current_saddle_point_state.get_next_AtY().data());

  potential_next_dual_solution.create(op_problem_scaled.n_constraints,
                                      _potential_next_dual_solution.data());

  tmp_primal.create(op_problem_scaled.n_variables, _tmp_primal.data());
  tmp_dual.create(op_problem_scaled.n_constraints, _tmp_dual.data());
  if (pdlp_hyper_params::use_reflected_primal_dual) {
    cuopt_assert(_reflected_primal_solution.size() > 0, "Reflected primal solution empty");
    reflected_primal_solution.create(op_problem_scaled.n_variables,
                                     _reflected_primal_solution.data());
  }

  const rmm::device_scalar<f_t> alpha{1, handle_ptr->get_stream()};
  const rmm::device_scalar<f_t> beta{1, handle_ptr->get_stream()};
  size_t buffer_size_non_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A,
                                                  c,
                                                  beta.data(),
                                                  dual_solution,
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_non_transpose,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose.resize(buffer_size_non_transpose, handle_ptr->get_stream());

  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T,
                                                  dual_solution,
                                                  beta.data(),
                                                  c,
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_transpose,
                                                  handle_ptr->get_stream()));

  buffer_transpose.resize(buffer_size_transpose, handle_ptr->get_stream());

#if CUDA_VER_12_4_UP
  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A,
                             c,
                             beta.data(),
                             dual_solution,
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_non_transpose.data(),
                             handle_ptr->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T,
                             dual_solution,
                             beta.data(),
                             c,
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_transpose.data(),
                             handle_ptr->get_stream());
#endif
}

// Used by pdlp object for current and average termination condition
// A_T is owned by the problem object and is transposed by the problem
template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(raft::handle_t const* handle_ptr,
                                           const problem_t<i_t, f_t>& op_problem,
                                           rmm::device_uvector<f_t>& _primal_solution,
                                           rmm::device_uvector<f_t>& _dual_solution,
                                           rmm::device_uvector<f_t>& _tmp_primal,
                                           rmm::device_uvector<f_t>& _tmp_dual,
                                           rmm::device_uvector<f_t>& _potential_next_primal,
                                           rmm::device_uvector<f_t>& _potential_next_dual,
                                           const rmm::device_uvector<f_t>& _A_T,
                                           const rmm::device_uvector<i_t>& _A_T_offsets,
                                           const rmm::device_uvector<i_t>& _A_T_indices)
  : handle_ptr_(handle_ptr),
    A{},
    A_T{},
    c{},
    primal_solution{},
    dual_solution{},
    primal_gradient{},
    dual_gradient{},
    tmp_primal{},
    tmp_dual{},
    A_T_{_A_T},
    A_T_offsets_{_A_T_offsets},
    A_T_indices_{_A_T_indices},
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    A_{op_problem.coefficients},
    A_offsets_{op_problem.offsets},
    A_indices_{op_problem.variables}
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "PDLP cusparse view init" << std::endl;
#endif

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr_->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));

  // setup cusparse view
  A.create(op_problem.n_constraints,
           op_problem.n_variables,
           op_problem.nnz,
           const_cast<i_t*>(op_problem.offsets.data()),
           const_cast<i_t*>(op_problem.variables.data()),
           const_cast<f_t*>(op_problem.coefficients.data()));

  A_T.create(op_problem.n_variables,
             op_problem.n_constraints,
             op_problem.nnz,
             const_cast<i_t*>(A_T_offsets_.data()),
             const_cast<i_t*>(A_T_indices_.data()),
             const_cast<f_t*>(A_T_.data()));

  c.create(op_problem.n_variables, const_cast<f_t*>(op_problem.objective_coefficients.data()));

  if (!pdlp_hyper_params::use_adaptive_step_size_strategy) {
    primal_solution.create(op_problem.n_variables, _potential_next_primal.data());
    dual_solution.create(op_problem.n_constraints, _potential_next_dual.data());
  } else {
    primal_solution.create(op_problem.n_variables, _primal_solution.data());
    dual_solution.create(op_problem.n_constraints, _dual_solution.data());
  }

  tmp_primal.create(op_problem.n_variables, _tmp_primal.data());
  tmp_dual.create(op_problem.n_constraints, _tmp_dual.data());

  const rmm::device_scalar<f_t> alpha{1, handle_ptr->get_stream()};
  const rmm::device_scalar<f_t> beta{1, handle_ptr->get_stream()};
  size_t buffer_size_non_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A,
                                                  c,
                                                  beta.data(),
                                                  dual_solution,
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_non_transpose,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose.resize(buffer_size_non_transpose, handle_ptr->get_stream());

  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T,
                                                  dual_solution,
                                                  beta.data(),
                                                  c,
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_transpose,
                                                  handle_ptr->get_stream()));

  buffer_transpose.resize(buffer_size_transpose, handle_ptr->get_stream());

#if CUDA_VER_12_4_UP
  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A,
                             c,
                             beta.data(),
                             dual_solution,
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_non_transpose.data(),
                             handle_ptr->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T,
                             dual_solution,
                             beta.data(),
                             c,
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_transpose.data(),
                             handle_ptr->get_stream());
#endif
}

// Constructor used 3 times in restart strategy for trust region restart
template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(
  raft::handle_t const* handle_ptr,
  const problem_t<i_t, f_t>& op_problem,  // Just used for the sizes
  const cusparse_view_t<i_t, f_t>& existing_cusparse_view,
  f_t* _primal_solution,
  f_t* _dual_solution,
  f_t* _primal_gradient,
  f_t* _dual_gradient)
  : handle_ptr_(handle_ptr),
    c(existing_cusparse_view.c),
    primal_solution{},
    dual_solution{},
    primal_gradient{},
    dual_gradient{},
    tmp_primal(existing_cusparse_view.tmp_primal),
    tmp_dual(existing_cusparse_view.tmp_dual),
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    A_T_{existing_cusparse_view.A_T_},                  // Need to be init but not used
    A_T_offsets_{existing_cusparse_view.A_T_offsets_},  // Need to be init but not used
    A_T_indices_{existing_cusparse_view.A_T_indices_},  // Need to be init but not used
    A_{existing_cusparse_view.A_},
    A_offsets_{existing_cusparse_view.A_offsets_},
    A_indices_{existing_cusparse_view.A_indices_}
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Restart Strategy cusparse view init" << std::endl;
#endif

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr_->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));

  // Need to reinstanciate the cuSparse views
  // Copying them from the existing cuSparse view is a bad practice and creates segfault post
  // CUDA 12.4 Using the saved pointer of the existing cusparse view to make sure we capture the
  // correct pointer
  A.create(op_problem.n_constraints,
           op_problem.n_variables,
           op_problem.nnz,
           const_cast<i_t*>(A_offsets_.data()),
           const_cast<i_t*>(A_indices_.data()),
           const_cast<f_t*>(A_.data()));

  A_T.create(op_problem.n_variables,
             op_problem.n_constraints,
             op_problem.nnz,
             const_cast<i_t*>(existing_cusparse_view.A_T_offsets_.data()),
             const_cast<i_t*>(existing_cusparse_view.A_T_indices_.data()),
             const_cast<f_t*>(existing_cusparse_view.A_T_.data()));

  primal_solution.create(op_problem.n_variables, _primal_solution);
  dual_solution.create(op_problem.n_constraints, _dual_solution);

  primal_gradient.create(op_problem.n_variables, _primal_gradient);
  dual_gradient.create(op_problem.n_constraints, _dual_gradient);

  const rmm::device_scalar<f_t> alpha{1, handle_ptr->get_stream()};
  const rmm::device_scalar<f_t> beta{1, handle_ptr->get_stream()};
  size_t buffer_size_non_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A,
                                                  c,
                                                  beta.data(),
                                                  dual_solution,
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_non_transpose,
                                                  handle_ptr->get_stream()));
  buffer_non_transpose.resize(buffer_size_non_transpose, handle_ptr->get_stream());

  size_t buffer_size_transpose = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  alpha.data(),
                                                  A_T,
                                                  dual_solution,
                                                  beta.data(),
                                                  c,
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &buffer_size_transpose,
                                                  handle_ptr->get_stream()));

  buffer_transpose.resize(buffer_size_transpose, handle_ptr->get_stream());

#if CUDA_VER_12_4_UP
  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A,
                             c,
                             beta.data(),
                             dual_solution,
                             CUSPARSE_SPMV_CSR_ALG2,
                             buffer_non_transpose.data(),
                             handle_ptr->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             alpha.data(),
                             A_T,
                             dual_solution,
                             beta.data(),
                             c,
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
  const rmm::device_uvector<i_t>& dummy_int     // Empty just to init the const&
  )
  : handle_ptr_(handle_ptr),
    buffer_non_transpose{0, handle_ptr->get_stream()},
    buffer_transpose{0, handle_ptr->get_stream()},
    A_T_(dummy_float),
    A_T_offsets_(dummy_int),
    A_T_indices_(dummy_int),
    A_(dummy_float),
    A_offsets_(dummy_int),
    A_indices_(dummy_int)
{
}

#if MIP_INSTANTIATE_FLOAT
template class cusparse_sp_mat_descr_wrapper_t<int, float>;
template class cusparse_dn_vec_descr_wrapper_t<float>;
template class cusparse_view_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class cusparse_sp_mat_descr_wrapper_t<int, double>;
template class cusparse_dn_vec_descr_wrapper_t<double>;
template class cusparse_view_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
