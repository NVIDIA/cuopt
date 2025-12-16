/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/dense_vector.hpp>
#include <dual_simplex/pinned_host_allocator.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include "dual_simplex/cusparse_view.hpp"

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <cuopt/error.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/transpose.cuh>

#include <dlfcn.h>

namespace cuopt::linear_programming::dual_simplex {

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

static cusparseSpMVAlg_t get_spmv_alg(int num_rows)
{
  // The older version of ALG2 has a bug with single row matrices
  if (num_rows == 1 && __CUDACC_VER_MAJOR__ < 13) { return CUSPARSE_SPMV_CSR_ALG1; }
  return CUSPARSE_SPMV_CSR_ALG2;
}

template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::cusparse_view_t(raft::handle_t const* handle_ptr,
                                           const csc_matrix_t<i_t, f_t>& A)
  : handle_ptr_(handle_ptr),
    A_offsets_(0, handle_ptr->get_stream()),
    A_indices_(0, handle_ptr->get_stream()),
    A_data_(0, handle_ptr->get_stream()),
    A_T_offsets_(0, handle_ptr->get_stream()),
    A_T_indices_(0, handle_ptr->get_stream()),
    A_T_data_(0, handle_ptr->get_stream()),
    spmv_buffer_(0, handle_ptr->get_stream()),
    d_one_(f_t(1), handle_ptr->get_stream()),
    d_minus_one_(f_t(-1), handle_ptr->get_stream()),
    d_zero_(f_t(0), handle_ptr->get_stream())
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  // TMP matrix data should already be on the GPU
  constexpr bool debug = false;
  if (debug) { printf("A hash: %zu\n", A.hash()); }
  csr_matrix_t<i_t, f_t> A_csr(A.m, A.n, 1);
  A.to_compressed_row(A_csr);
  i_t rows                        = A_csr.m;
  i_t cols                        = A_csr.n;
  i_t nnz                         = A_csr.x.size();
  const std::vector<i_t>& offsets = A_csr.row_start;
  const std::vector<i_t>& indices = A_csr.j;
  const std::vector<f_t>& data    = A_csr.x;

  A_offsets_ = device_copy(offsets, handle_ptr->get_stream());
  A_indices_ = device_copy(indices, handle_ptr->get_stream());
  A_data_    = device_copy(data, handle_ptr->get_stream());

  A_T_offsets_ = device_copy(A.col_start, handle_ptr->get_stream());
  A_T_indices_ = device_copy(A.i, handle_ptr->get_stream());
  A_T_data_    = device_copy(A.x, handle_ptr->get_stream());

  cusparseCreateCsr(&A_,
                    rows,
                    cols,
                    nnz,
                    A_offsets_.data(),
                    A_indices_.data(),
                    A_data_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);

  cusparseCreateCsr(&A_T_,
                    cols,
                    rows,
                    nnz,
                    A_T_offsets_.data(),
                    A_T_indices_.data(),
                    A_T_data_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);

  // Tmp just to init the buffer size and preprocess
  cusparseDnVecDescr_t x;
  cusparseDnVecDescr_t y;
  rmm::device_uvector<f_t> d_x(cols, handle_ptr_->get_stream());
  rmm::device_uvector<f_t> d_y(rows, handle_ptr_->get_stream());
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(&x, d_x.size(), d_x.data()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(&y, d_y.size(), d_y.data()));

  size_t buffer_size_spmv = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  d_one_.data(),
                                                  A_,
                                                  x,
                                                  d_one_.data(),
                                                  y,
                                                  get_spmv_alg(A_offsets_.size() - 1),
                                                  &buffer_size_spmv,
                                                  handle_ptr_->get_stream()));
  spmv_buffer_.resize(buffer_size_spmv, handle_ptr_->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             d_one_.data(),
                             A_,
                             x,
                             d_one_.data(),
                             y,
                             get_spmv_alg(A_offsets_.size() - 1),
                             spmv_buffer_.data(),
                             handle_ptr->get_stream());

  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  d_one_.data(),
                                                  A_T_,
                                                  y,
                                                  d_one_.data(),
                                                  x,
                                                  get_spmv_alg(A_T_offsets_.size() - 1),
                                                  &buffer_size_spmv,
                                                  handle_ptr_->get_stream()));
  spmv_buffer_transpose_.resize(buffer_size_spmv, handle_ptr_->get_stream());

  my_cusparsespmv_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             d_one_.data(),
                             A_T_,
                             y,
                             d_one_.data(),
                             x,
                             get_spmv_alg(A_T_offsets_.size() - 1),
                             spmv_buffer_transpose_.data(),
                             handle_ptr->get_stream());
  RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(x));
  RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(y));
}

template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>::~cusparse_view_t()
{
  CUOPT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(A_));
  CUOPT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(A_T_));
}

template <typename i_t, typename f_t>
detail::cusparse_dn_vec_descr_wrapper_t<f_t> cusparse_view_t<i_t, f_t>::create_vector(
  rmm::device_uvector<f_t> const& vec)
{
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> descr;
  descr.create(vec.size(), const_cast<f_t*>(vec.data()));
  return descr;
}

template <typename i_t, typename f_t>
template <typename AllocatorA, typename AllocatorB>
void cusparse_view_t<i_t, f_t>::spmv(f_t alpha,
                                     const std::vector<f_t, AllocatorA>& x,
                                     f_t beta,
                                     std::vector<f_t, AllocatorB>& y)
{
  auto d_x = device_copy(x, handle_ptr_->get_stream());
  auto d_y = device_copy(y, handle_ptr_->get_stream());
  spmv(alpha, d_x, beta, d_y);
  y = cuopt::host_copy<f_t, AllocatorB>(d_y, handle_ptr_->get_stream());
}

template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::spmv(f_t alpha,
                                     const rmm::device_uvector<f_t>& x,
                                     f_t beta,
                                     rmm::device_uvector<f_t>& y)
{
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> x_cusparse = create_vector(x);
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> y_cusparse = create_vector(y);
  spmv(alpha, x_cusparse, beta, y_cusparse);
}

template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::spmv(f_t alpha,
                                     detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
                                     f_t beta,
                                     detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& y)
{
  // Would be simpler if we could pass host data direclty but other cusparse calls with the same
  // handler depend on device data
  cuopt_assert(alpha == f_t(1) || alpha == f_t(-1), "Only alpha 1 or -1 supported");
  cuopt_assert(beta == f_t(1) || beta == f_t(-1) || beta == f_t(0),
               "Only beta 1 or -1 or 0 supported");
  rmm::device_scalar<f_t>* d_beta = &d_one_;
  if (beta == f_t(0))
    d_beta = &d_zero_;
  else if (beta == f_t(-1))
    d_beta = &d_minus_one_;
  raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     (alpha == 1) ? d_one_.data() : d_minus_one_.data(),
                                     A_,
                                     x,
                                     d_beta->data(),
                                     y,
                                     get_spmv_alg(A_offsets_.size() - 1),
                                     (f_t*)spmv_buffer_.data(),
                                     handle_ptr_->get_stream());
}

template <typename i_t, typename f_t>
template <typename AllocatorA, typename AllocatorB>
void cusparse_view_t<i_t, f_t>::transpose_spmv(f_t alpha,
                                               const std::vector<f_t, AllocatorA>& x,
                                               f_t beta,
                                               std::vector<f_t, AllocatorB>& y)
{
  auto d_x = device_copy(x, handle_ptr_->get_stream());
  auto d_y = device_copy(y, handle_ptr_->get_stream());
  transpose_spmv(alpha, d_x, beta, d_y);
  y = cuopt::host_copy<f_t, AllocatorB>(d_y, handle_ptr_->get_stream());
}

template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::transpose_spmv(f_t alpha,
                                               const rmm::device_uvector<f_t>& x,
                                               f_t beta,
                                               rmm::device_uvector<f_t>& y)
{
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> x_cusparse = create_vector(x);
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> y_cusparse = create_vector(y);
  transpose_spmv(alpha, x_cusparse, beta, y_cusparse);
}

template <typename i_t, typename f_t>
void cusparse_view_t<i_t, f_t>::transpose_spmv(
  f_t alpha,
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
  f_t beta,
  detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& y)
{
  // Would be simpler if we could pass host data direct;y but other cusparse calls with the same
  // handler depend on device data
  cuopt_assert(alpha == f_t(1) || alpha == f_t(-1), "Only alpha 1 or -1 supported");
  cuopt_assert(beta == f_t(1) || beta == f_t(-1) || beta == f_t(0),
               "Only beta 1 or -1 or 0 supported");
  rmm::device_scalar<f_t>* d_beta = &d_one_;
  if (beta == f_t(0))
    d_beta = &d_zero_;
  else if (beta == f_t(-1))
    d_beta = &d_minus_one_;
  raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     (alpha == 1) ? d_one_.data() : d_minus_one_.data(),
                                     A_T_,
                                     x,
                                     d_beta->data(),
                                     y,
                                     get_spmv_alg(A_T_offsets_.size() - 1),
                                     (f_t*)spmv_buffer_transpose_.data(),
                                     handle_ptr_->get_stream());
}

template class cusparse_view_t<int, double>;
template void
cusparse_view_t<int, double>::spmv<PinnedHostAllocator<double>, PinnedHostAllocator<double>>(
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template void
cusparse_view_t<int, double>::spmv<PinnedHostAllocator<double>, std::allocator<double>>(
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

template void
cusparse_view_t<int, double>::spmv<std::allocator<double>, PinnedHostAllocator<double>>(
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template void cusparse_view_t<int, double>::spmv<std::allocator<double>, std::allocator<double>>(
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

template void cusparse_view_t<int, double>::transpose_spmv<PinnedHostAllocator<double>,
                                                           PinnedHostAllocator<double>>(
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template void
cusparse_view_t<int, double>::transpose_spmv<PinnedHostAllocator<double>, std::allocator<double>>(
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

template void
cusparse_view_t<int, double>::transpose_spmv<std::allocator<double>, PinnedHostAllocator<double>>(
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template void
cusparse_view_t<int, double>::transpose_spmv<std::allocator<double>, std::allocator<double>>(
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

}  // namespace cuopt::linear_programming::dual_simplex
