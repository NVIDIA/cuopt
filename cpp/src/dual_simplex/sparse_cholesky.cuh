/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "dual_simplex/dense_vector.hpp"
#include "dual_simplex/device_sparse_matrix.cuh"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "dual_simplex/sparse_matrix.hpp"
#include "dual_simplex/tic_toc.hpp"

#include <cuda_runtime.h>
#include <utilities/driver_helpers.cuh>

#include <raft/common/nvtx.hpp>

#include "cudss.h"

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class sparse_cholesky_base_t {
 public:
  virtual ~sparse_cholesky_base_t()                                                 = default;
  virtual i_t analyze(const csc_matrix_t<i_t, f_t>& A_in)                           = 0;
  virtual i_t factorize(const csc_matrix_t<i_t, f_t>& A_in)                         = 0;
  virtual i_t analyze(device_csr_matrix_t<i_t, f_t>& A_in)                          = 0;
  virtual i_t factorize(device_csr_matrix_t<i_t, f_t>& A_in)                        = 0;
  virtual i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) = 0;
  virtual i_t solve(rmm::device_uvector<f_t>& b, rmm::device_uvector<f_t>& x)       = 0;
  virtual void set_positive_definite(bool positive_definite)                        = 0;
};

#define CUDSS_EXAMPLE_FREE \
  do {                     \
  } while (0)

#define CUDA_CALL_AND_CHECK(call, msg)                                                 \
  do {                                                                                 \
    cuda_error = call;                                                                 \
    if (cuda_error != cudaSuccess) {                                                   \
      printf("FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
      CUDSS_EXAMPLE_FREE;                                                              \
      return -1;                                                                       \
    }                                                                                  \
  } while (0);

#define CUDA_CALL_AND_CHECK_EXIT(call, msg)                                            \
  do {                                                                                 \
    cuda_error = call;                                                                 \
    if (cuda_error != cudaSuccess) {                                                   \
      printf("FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
      CUDSS_EXAMPLE_FREE;                                                              \
      exit(-1);                                                                        \
    }                                                                                  \
  } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                      \
  do {                                                               \
    status = call;                                                   \
    if (status != CUDSS_STATUS_SUCCESS) {                            \
      printf(                                                        \
        "FAILED: CUDSS call ended unsuccessfully with status = %d, " \
        "details: " #msg "\n",                                       \
        status);                                                     \
      CUDSS_EXAMPLE_FREE;                                            \
      return -2;                                                     \
    }                                                                \
  } while (0);

#define CUDSS_CALL_AND_CHECK_EXIT(call, status, msg)                 \
  do {                                                               \
    status = call;                                                   \
    if (status != CUDSS_STATUS_SUCCESS) {                            \
      printf(                                                        \
        "FAILED: CUDSS call ended unsuccessfully with status = %d, " \
        "details: " #msg "\n",                                       \
        status);                                                     \
      CUDSS_EXAMPLE_FREE;                                            \
      exit(-2);                                                      \
    }                                                                \
  } while (0);

// RMM pool fragmentation makes the workspace size smaller than the actual free space on the GPU
// Use cudaMallocAsync instead of the RMM pool until we reduce our memory footprint/fragmentation.
// TODO: Still use RMM for smaller problems to benefit from their allocation optimizations.
template <typename mem_pool_t>
int cudss_device_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream)
{
  int status = cudaMallocAsync(ptr, size, stream);
  if (status != cudaSuccess) { throw raft::cuda_error("Cuda error in cudss_device_alloc"); }
  return status;
}

template <typename mem_pool_t>
int cudss_device_dealloc(void* ctx, void* ptr, size_t size, cudaStream_t stream)
{
  int status = cudaFreeAsync(ptr, stream);
  if (status != cudaSuccess) { throw raft::cuda_error("Cuda error in cudss_device_dealloc"); }
  return status;
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Function to compute a combined hash for an array of doubles
template <typename i_t, typename f_t>
std::size_t compute_hash(const dense_vector_t<i_t, f_t>& arr)
{
  std::size_t seed = arr.size();
  for (const auto& i : arr) {
    hash_combine(seed, i);
  }
  return seed;
}

template <typename f_t>
std::size_t compute_hash(const f_t* arr, size_t size)
{
  std::size_t seed = size;
  for (size_t i = 0; i < size; i++) {
    hash_combine(seed, arr[i]);
  }
  return seed;
}

template <typename i_t, typename f_t>
class sparse_cholesky_cudss_t : public sparse_cholesky_base_t<i_t, f_t> {
 public:
  sparse_cholesky_cudss_t(raft::handle_t const* handle_ptr,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          i_t size)
    : handle_ptr_(handle_ptr),
      n(size),
      nnz(-1),
      first_factor(true),
      positive_definite(true),
      A_created(false),
      settings_(settings),
      stream(handle_ptr->get_stream())
  {
    int major, minor, patch;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL, &patch);
    settings.log.printf("cuDSS Version               : %d.%d.%d\n", major, minor, patch);

    cuda_error = cudaSuccess;
    status     = CUDSS_STATUS_SUCCESS;

    if (CUDART_VERSION >= 13000 && settings_.concurrent_halt != nullptr) {
      cuGetErrorString_func = cuopt::detail::get_driver_entry_point("cuGetErrorString");
      // 1. Set up the GPU resources
      CUdevResource initial_device_GPU_resources = {};
      auto cuDeviceGetDevResource_func =
        cuopt::detail::get_driver_entry_point("cuDeviceGetDevResource");
      CU_CHECK(reinterpret_cast<decltype(::cuDeviceGetDevResource)*>(cuDeviceGetDevResource_func)(
                 handle_ptr_->get_device(), &initial_device_GPU_resources, CU_DEV_RESOURCE_TYPE_SM),
               reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));

#ifdef DEBUG
      settings.log.printf(
        "   Initial GPU resources retrieved via "
        "cuDeviceGetDevResource() have type "
        "%d and SM count %d\n",
        initial_device_GPU_resources.type,
        initial_device_GPU_resources.sm.smCount);
#endif

      // 2. Partition the GPU resources
      auto total_SMs   = initial_device_GPU_resources.sm.smCount;
      auto barrier_sms = raft::alignTo(static_cast<i_t>(total_SMs * 0.75f), 8);
      CUdevResource resource;
      auto cuDevSmResourceSplitByCount_func =
        cuopt::detail::get_driver_entry_point("cuDevSmResourceSplitByCount");
      auto n_groups  = 1u;
      auto use_flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;  // or 0
      CU_CHECK(
        reinterpret_cast<decltype(::cuDevSmResourceSplitByCount)*>(
          cuDevSmResourceSplitByCount_func)(
          &resource, &n_groups, &initial_device_GPU_resources, nullptr, use_flags, barrier_sms),
        reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
#ifdef DEBUG
      settings.log.printf(
        "   Resources were split into %d resource groups (had "
        "requested %d) with %d SMs each (had "
        "requested % d)\n",
        n_groups,
        n_groups,
        resource.sm.smCount,
        barrier_sms);
#endif
      // 3. Create the resource descriptor
      auto constexpr const n_resource_desc = 1;
      CUdevResourceDesc resource_desc;
      auto cuDevResourceGenerateDesc_func =
        cuopt::detail::get_driver_entry_point("cuDevResourceGenerateDesc");
      CU_CHECK(reinterpret_cast<decltype(::cuDevResourceGenerateDesc)*>(
                 cuDevResourceGenerateDesc_func)(&resource_desc, &resource, n_resource_desc),
               reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
#ifdef DEBUG
      settings.log.printf(
        "   For the resource descriptor of barrier green context "
        "we will combine %d resources of "
        "%d SMs each\n",
        n_resource_desc,
        resource.sm.smCount);
#endif

      // Only perform this if CUDA version is more than 13
      // (all resource splitting and descriptor creation already
      // above) No additional code needed here as the logic is
      // already guarded above.
      // 4. Create the green context and stream for that green
      // context CUstream barrier_green_ctx_stream;
      i_t stream_priority;
      cudaStream_t cuda_stream    = handle_ptr_->get_stream();
      cudaError_t priority_result = cudaStreamGetPriority(cuda_stream, &stream_priority);
      RAFT_CUDA_TRY(priority_result);
      auto cuGreenCtxCreate_func = cuopt::detail::get_driver_entry_point("cuGreenCtxCreate");
      CU_CHECK(reinterpret_cast<decltype(::cuGreenCtxCreate)*>(cuGreenCtxCreate_func)(
                 &barrier_green_ctx,
                 resource_desc,
                 handle_ptr_->get_device(),
                 CU_GREEN_CTX_DEFAULT_STREAM),
               reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
      auto cuGreenCtxStreamCreate_func =
        cuopt::detail::get_driver_entry_point("cuGreenCtxStreamCreate");
      CU_CHECK(reinterpret_cast<decltype(::cuGreenCtxStreamCreate)*>(cuGreenCtxStreamCreate_func)(
                 &stream, barrier_green_ctx, CU_STREAM_NON_BLOCKING, stream_priority),
               reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
    }

    CUDSS_CALL_AND_CHECK_EXIT(cudssCreate(&handle), status, "cudssCreate");
    CUDSS_CALL_AND_CHECK_EXIT(cudssSetStream(handle, stream), status, "cudaStreamCreate");

    mem_handler.ctx          = reinterpret_cast<void*>(handle_ptr_->get_workspace_resource());
    mem_handler.device_alloc = cudss_device_alloc<rmm::mr::device_memory_resource>;
    mem_handler.device_free  = cudss_device_dealloc<rmm::mr::device_memory_resource>;

    CUDSS_CALL_AND_CHECK_EXIT(
      cudssSetDeviceMemHandler(handle, &mem_handler), status, "cudssSetDeviceMemHandler");

    const char* cudss_mt_lib_file = nullptr;
    char* env_value               = std::getenv("CUDSS_THREADING_LIB");
    if (env_value != nullptr) {
      cudss_mt_lib_file = env_value;
    } else if (CUDSS_MT_LIB_FILE_NAME != nullptr) {
      cudss_mt_lib_file = CUDSS_MT_LIB_FILE_NAME;
    }

    if (cudss_mt_lib_file != nullptr) {
      settings.log.printf("cuDSS Threading layer       : %s\n", cudss_mt_lib_file);
      CUDSS_CALL_AND_CHECK_EXIT(
        cudssSetThreadingLayer(handle, cudss_mt_lib_file), status, "cudssSetThreadingLayer");
    }

    CUDSS_CALL_AND_CHECK_EXIT(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK_EXIT(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

#if CUDSS_VERSION_MAJOR >= 0 && CUDSS_VERSION_MINOR >= 7
    if (settings_.concurrent_halt != nullptr) {
      CUDSS_CALL_AND_CHECK_EXIT(cudssDataSet(handle,
                                             solverData,
                                             CUDSS_DATA_USER_HOST_INTERRUPT,
                                             (void*)settings_.concurrent_halt,
                                             sizeof(int)),
                                status,
                                "cudssDataSet for interrupt");
    }

    if (settings_.cudss_deterministic) {
      settings_.log.printf("cuDSS solve mode            : deterministic\n");
      int32_t deterministic = 1;
      CUDSS_CALL_AND_CHECK_EXIT(
        cudssConfigSet(
          solverConfig, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic, sizeof(int32_t)),
        status,
        "cudssConfigSet for deterministic mode");
    }
#endif

#if USE_ITERATIVE_REFINEMENT
    int32_t ir_n_steps = 2;
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS, &ir_n_steps, sizeof(int32_t)),
      status,
      "cudssConfigSet for ir n steps");
#endif

#if USE_MATCHING
    settings_.log.printf("Using matching\n");
    int32_t use_matching = 1;
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssConfigSet(solverConfig, CUDSS_CONFIG_USE_MATCHING, &use_matching, sizeof(int32_t)),
      status,
      "cudssConfigSet for use matching");
#endif

    // Device pointers
    csr_offset_d  = nullptr;
    csr_columns_d = nullptr;
    csr_values_d  = nullptr;
    x_values_d    = nullptr;
    b_values_d    = nullptr;
    CUDA_CALL_AND_CHECK_EXIT(cudaMallocAsync(&x_values_d, n * sizeof(f_t), stream),
                             "cudaMalloc for x_values");
    CUDA_CALL_AND_CHECK_EXIT(cudaMallocAsync(&b_values_d, n * sizeof(f_t), stream),
                             "cudaMalloc for b_values");

    i_t ldb = n;
    i_t ldx = n;
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixCreateDn(&cudss_b, n, 1, ldb, b_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
      status,
      "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixCreateDn(&cudss_x, n, 1, ldx, x_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
      status,
      "cudssMatrixCreateDn for x");
    handle_ptr_->get_stream().synchronize();
  }

  ~sparse_cholesky_cudss_t() override
  {
    cudaFreeAsync(csr_values_d, stream);
    cudaFreeAsync(csr_columns_d, stream);
    cudaFreeAsync(csr_offset_d, stream);

    cudaFreeAsync(x_values_d, stream);
    cudaFreeAsync(b_values_d, stream);
    if (A_created) {
      CUDSS_CALL_AND_CHECK_EXIT(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    }

    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixDestroy(cudss_x), status, "cudssMatrixDestroy for cudss_x");
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixDestroy(cudss_b), status, "cudssMatrixDestroy for cudss_b");
    CUDSS_CALL_AND_CHECK_EXIT(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK_EXIT(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK_EXIT(cudssDestroy(handle), status, "cudssDestroy");
    CUDA_CALL_AND_CHECK_EXIT(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
#if CUDART_VERSION >= 13000
    if (settings_.concurrent_halt != nullptr) {
      auto cuStreamDestroy_func = cuopt::detail::get_driver_entry_point("cuStreamDestroy");
      CU_CHECK(reinterpret_cast<decltype(::cuStreamDestroy)*>(cuStreamDestroy_func)(stream),
               reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
      auto cuGreenCtxDestroy_func = cuopt::detail::get_driver_entry_point("cuGreenCtxDestroy");
      CU_CHECK(
        reinterpret_cast<decltype(::cuGreenCtxDestroy)*>(cuGreenCtxDestroy_func)(barrier_green_ctx),
        reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
      handle_ptr_->get_stream().synchronize();
    }
#endif
  }

  i_t analyze(device_csr_matrix_t<i_t, f_t>& Arow) override
  {
    raft::common::nvtx::range fun_scope("Barrier: cuDSS Analyze");

#ifdef WRITE_MATRIX_MARKET
    {
      csr_matrix_t<i_t, f_t> Arow_host = Arow.to_host(Arow.row_start.stream());
      csc_matrix_t<i_t, f_t> A_col(Arow_host.m, Arow_host.n, 1);
      Arow_host.to_compressed_col(A_col);
      FILE* fid = fopen("A_to_factorize.mtx", "w");
      settings_.log.printf("writing matrix matrix\n");
      A_col.write_matrix_market(fid);
      settings_.log.printf("finished\n");
      fclose(fid);
    }
#endif

    nnz               = Arow.row_start.element(Arow.m, Arow.row_start.stream());
    const f_t density = static_cast<f_t>(nnz) / (static_cast<f_t>(n) * static_cast<f_t>(n));

    if (first_factor &&
        ((settings_.ordering == -1 && density >= 0.05) || settings_.ordering == 1) && n > 1) {
      settings_.log.printf("Reordering algorithm        : AMD\n");
      // Tell cuDSS to use AMD
      cudssAlgType_t reorder_alg = CUDSS_ALG_3;
      CUDSS_CALL_AND_CHECK_EXIT(
        cudssConfigSet(
          solverConfig, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(cudssAlgType_t)),
        status,
        "cudssConfigSet for reordering alg");
    }

    if (!first_factor) {
      raft::common::nvtx::range fun_scope("Barrier: cuDSS Analyze : Destroy");
      CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    }

    {
      raft::common::nvtx::range fun_scope("Barrier: cuDSS Analyze : cudssMatrixCreateCsr");
      CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateCsr(&A,
                             n,
                             n,
                             nnz,
                             Arow.row_start.data(),
                             nullptr,
                             Arow.j.data(),
                             Arow.x.data(),
                             CUDA_R_32I,
                             CUDA_R_64F,
                             positive_definite ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_SYMMETRIC,
                             CUDSS_MVIEW_FULL,
                             CUDSS_BASE_ZERO),
        status,
        "cudssMatrixCreateCsr");
      A_created = true;
    }

    // Perform symbolic analysis
    f_t start_symbolic = tic();
    f_t start_symbolic_factor;

    {
      raft::common::nvtx::range fun_scope("Barrier: cuDSS Analyze : CUDSS_PHASE_ANALYSIS");
      status =
        cudssExecute(handle, CUDSS_PHASE_REORDERING, solverConfig, solverData, A, cudss_x, cudss_b);
      if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }
      if (status != CUDSS_STATUS_SUCCESS) {
        settings_.log.printf(
          "FAILED: CUDSS call ended unsuccessfully with status = %d, details: cuDSSExecute for "
          "reordering\n",
          status);
        return -1;
      }
      f_t reordering_time = toc(start_symbolic);
      settings_.log.printf("Reordering time             : %.2fs\n", reordering_time);
      start_symbolic_factor = tic();

      status = cudssExecute(
        handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b);
      if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }
      if (status != CUDSS_STATUS_SUCCESS) {
        settings_.log.printf(
          "FAILED: CUDSS call ended unsuccessfully with status = %d, details: cuDSSExecute for "
          "symbolic factorization\n",
          status);
        return -1;
      }
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    f_t symbolic_factorization_time = toc(start_symbolic_factor);
    settings_.log.printf("Symbolic factorization time : %.2fs\n", symbolic_factorization_time);
    settings_.log.printf("Total symbolic time         : %.2fs\n", toc(start_symbolic));
    int64_t lu_nz       = 0;
    size_t size_written = 0;
    CUDSS_CALL_AND_CHECK(
      cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nz, sizeof(int64_t), &size_written),
      status,
      "cudssDataGet for LU_NNZ");
    settings_.log.printf("Symbolic nonzeros in factor : %.2e\n", static_cast<f_t>(lu_nz) / 2.0);
    // TODO: Is there any way to get nonzeros in the factors?
    // TODO: Is there any way to get flops for the factorization?
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    handle_ptr_->get_stream().synchronize();

    return 0;
  }
  i_t factorize(device_csr_matrix_t<i_t, f_t>& Arow) override
  {
    raft::common::nvtx::range fun_scope("Factorize: cuDSS");

// #define PRINT_MATRIX_NORM
#ifdef PRINT_MATRIX_NORM
    cudaStreamSynchronize(stream);
    csr_matrix_t<i_t, f_t> Arow_host = Arow.to_host(Arow.row_start.stream());
    csc_matrix_t<i_t, f_t> A_col(Arow_host.m, Arow_host.n, 1);
    Arow_host.to_compressed_col(A_col);
    settings_.log.printf(
      "before factorize || A to factor|| = %.16e hash: %zu\n", A_col.norm1(), A_col.hash());
    cudaStreamSynchronize(stream);
#endif
    // csr_matrix_t<i_t, f_t> Arow;
    // A_in.to_compressed_row(Arow);

    auto d_nnz = Arow.row_start.element(Arow.m, Arow.row_start.stream());
    if (nnz != d_nnz) {
      settings_.log.printf("Error: nnz %d != A_in.col_start[A_in.n] %d\n", nnz, d_nnz);
      exit(1);
    }

    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(A, Arow.x.data()), status, "cudssMatrixSetValues for A");

    f_t start_numeric = tic();
    status            = cudssExecute(
      handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b);
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }
    if (status != CUDSS_STATUS_SUCCESS) {
      settings_.log.printf(
        "FAILED: CUDSS call ended unsuccessfully with status = %d, details: cuDSSExecute for "
        "factorization\n",
        status);
      return -1;
    }

#ifdef TIME_FACTORIZATION
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
#endif

    f_t numeric_time = toc(start_numeric);
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }

    int info;
    size_t sizeWritten = 0;
    CUDSS_CALL_AND_CHECK(
      cudssDataGet(handle, solverData, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten),
      status,
      "cudssDataGet for info");

    handle_ptr_->get_stream().synchronize();
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    if (info != 0) {
      settings_.log.printf("Factorization failed info %d\n", info);
      return -1;
    }

    if (first_factor) {
      settings_.log.debug("Factorization time          : %.2fs\n", numeric_time);
      first_factor = false;
    }
    if (status != CUDSS_STATUS_SUCCESS) {
      settings_.log.printf("cuDSS Factorization failed\n");
      return -1;
    }
    return 0;
  }

  i_t analyze(const csc_matrix_t<i_t, f_t>& A_in) override
  {
    csr_matrix_t<i_t, f_t> Arow(A_in.n, A_in.m, A_in.col_start[A_in.n]);

#ifdef WRITE_MATRIX_MARKET
    FILE* fid = fopen("A.mtx", "w");
    A_in.write_matrix_market(fid);
    fclose(fid);
    settings_.log.printf("Wrote A.mtx\n");
#endif
    A_in.to_compressed_row(Arow);

#ifdef CHECK_MATRIX
    settings_.log.printf("Checking matrices\n");
    A_in.check_matrix();
    Arow.check_matrix();
    settings_.log.printf("Finished checking matrices\n");
#endif
    if (A_in.n != n) {
      printf("Analyze input does not match size %d != %d\n", A_in.n, n);
      exit(1);
    }

    nnz = A_in.col_start[A_in.n];

    CUDA_CALL_AND_CHECK(cudaMallocAsync(&csr_offset_d, (n + 1) * sizeof(i_t), stream),
                        "cudaMalloc for csr_offset");
    CUDA_CALL_AND_CHECK(cudaMallocAsync(&csr_columns_d, nnz * sizeof(i_t), stream),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMallocAsync(&csr_values_d, nnz * sizeof(f_t), stream),
                        "cudaMalloc for csr_values");

    CUDA_CALL_AND_CHECK(
      cudaMemcpyAsync(
        csr_offset_d, Arow.row_start.data(), (n + 1) * sizeof(i_t), cudaMemcpyHostToDevice, stream),
      "cudaMemcpy for csr_offset");
    CUDA_CALL_AND_CHECK(
      cudaMemcpyAsync(
        csr_columns_d, Arow.j.data(), nnz * sizeof(i_t), cudaMemcpyHostToDevice, stream),
      "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(
      cudaMemcpyAsync(
        csr_values_d, Arow.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice, stream),
      "cudaMemcpy for csr_values");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    if (!first_factor) {
      CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
      A_created = false;
    }

    CUDSS_CALL_AND_CHECK(
      cudssMatrixCreateCsr(&A,
                           n,
                           n,
                           nnz,
                           csr_offset_d,
                           nullptr,
                           csr_columns_d,
                           csr_values_d,
                           CUDA_R_32I,
                           CUDA_R_64F,
                           positive_definite ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_SYMMETRIC,
                           CUDSS_MVIEW_FULL,
                           CUDSS_BASE_ZERO),
      status,
      "cudssMatrixCreateCsr");
    A_created = true;

    // Perform symbolic analysis
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }
    f_t start_analysis = tic();
    CUDSS_CALL_AND_CHECK(
      cudssExecute(handle, CUDSS_PHASE_REORDERING, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for reordering");

    f_t reorder_time = toc(start_analysis);
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }

    f_t start_symbolic = tic();

    CUDSS_CALL_AND_CHECK(
      cudssExecute(
        handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for symbolic factorization");

    f_t symbolic_time = toc(start_symbolic);
    f_t analysis_time = toc(start_analysis);
    settings_.log.printf("Symbolic factorization time : %.2fs\n", symbolic_time);
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) {
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      handle_ptr_->get_stream().synchronize();
      return -2;
    }
    int64_t lu_nz       = 0;
    size_t size_written = 0;
    CUDSS_CALL_AND_CHECK(
      cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nz, sizeof(int64_t), &size_written),
      status,
      "cudssDataGet for LU_NNZ");
    settings_.log.printf("Symbolic nonzeros in factor : %.2e\n", static_cast<f_t>(lu_nz) / 2.0);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    handle_ptr_->get_stream().synchronize();
    // TODO: Is there any way to get nonzeros in the factors?
    // TODO: Is there any way to get flops for the factorization?

    return 0;
  }
  i_t factorize(const csc_matrix_t<i_t, f_t>& A_in) override
  {
    csr_matrix_t<i_t, f_t> Arow(A_in.n, A_in.m, A_in.col_start[A_in.n]);
    A_in.to_compressed_row(Arow);

    if (A_in.n != n) { settings_.log.printf("Error A in n %d != size %d\n", A_in.n, n); }

    if (nnz != A_in.col_start[A_in.n]) {
      settings_.log.printf(
        "Error: nnz %d != A_in.col_start[A_in.n] %d\n", nnz, A_in.col_start[A_in.n]);
      exit(1);
    }

    CUDA_CALL_AND_CHECK(
      cudaMemcpyAsync(
        csr_values_d, Arow.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice, stream),
      "cudaMemcpy for csr_values");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    handle_ptr_->get_stream().synchronize();

    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(A, csr_values_d), status, "cudssMatrixSetValues for A");

    f_t start_numeric = tic();
    CUDSS_CALL_AND_CHECK(
      cudssExecute(
        handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for factorization");

    f_t numeric_time = toc(start_numeric);
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }

    int info;
    size_t sizeWritten = 0;
    CUDSS_CALL_AND_CHECK(
      cudssDataGet(handle, solverData, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten),
      status,
      "cudssDataGet for info");
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    handle_ptr_->get_stream().synchronize();
    if (info != 0) {
      settings_.log.printf("Factorization failed info %d\n", info);
      return -1;
    }

    if (first_factor) {
      settings_.log.debug("Factorization time          : %.2fs\n", numeric_time);
      first_factor = false;
    }
    if (status != CUDSS_STATUS_SUCCESS) {
      settings_.log.printf("cuDSS Factorization failed\n");
      return -1;
    }
    return 0;
  }

  i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) override
  {
    auto d_b = cuopt::device_copy(b, handle_ptr_->get_stream());
    auto d_x = cuopt::device_copy(x, handle_ptr_->get_stream());
    handle_ptr_->get_stream().synchronize();

    i_t out = solve(d_b, d_x);

    raft::copy(x.data(), d_x.data(), d_x.size(), handle_ptr_->get_stream());
    // Sync so that data is on the host
    handle_ptr_->get_stream().synchronize();

    for (i_t i = 0; i < n; i++) {
      if (x[i] != x[i]) { return -1; }
    }

    return out;
  }

  i_t solve(rmm::device_uvector<f_t>& b, rmm::device_uvector<f_t>& x) override
  {
    handle_ptr_->get_stream().synchronize();
    if (static_cast<i_t>(b.size()) != n) {
      settings_.log.printf("Error: b.size() %d != n %d\n", b.size(), n);
      exit(1);
    }
    if (static_cast<i_t>(x.size()) != n) {
      settings_.log.printf("Error: x.size() %d != n %d\n", x.size(), n);
      exit(1);
    }

    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(cudss_b, b.data()), status, "cudssMatrixSetValues for b");
    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(cudss_x, x.data()), status, "cudssMatrixSetValues for x");

    i_t ldb = n;
    i_t ldx = n;
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixCreateDn(&cudss_b, n, 1, ldb, b.data(), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
      status,
      "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixCreateDn(&cudss_x, n, 1, ldx, x.data(), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
      status,
      "cudssMatrixCreateDn for x");

    status = cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, cudss_x, cudss_b);
    if (settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1) { return -2; }
    if (status != CUDSS_STATUS_SUCCESS) {
      settings_.log.printf(
        "FAILED: CUDSS call ended unsuccessfully with status = %d, details: cuDSSExecute for "
        "solve\n",
        status);
      return -1;
    }

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    handle_ptr_->get_stream().synchronize();

#ifdef PRINT_RHS_AND_SOLUTION_HASH
    dense_vector_t<i_t, f_t> b_host(n);
    dense_vector_t<i_t, f_t> x_host(n);
    raft::copy(b_host.data(), b.data(), n, stream);
    raft::copy(x_host.data(), x.data(), n, stream);
    cudaStreamSynchronize(stream);
    settings_.log.printf("RHS norm %.16e, hash: %zu, Solution norm %.16e, hash: %zu\n",
                         vector_norm2<i_t, f_t>(b_host),
                         compute_hash(b_host),
                         vector_norm2<i_t, f_t>(x_host),
                         compute_hash(x_host));
#endif

    return 0;
  }

  void set_positive_definite(bool positive_definite) override
  {
    this->positive_definite = positive_definite;
  }

 private:
  raft::handle_t const* handle_ptr_;
  i_t n;
  i_t nnz;
  bool first_factor;
  bool positive_definite;
  cudaError_t cuda_error;
  cudssStatus_t status;
  // rmm::cuda_stream_view stream;
  cudssHandle_t handle;
  cudssDeviceMemHandler_t mem_handler;
  cudssConfig_t solverConfig;
  cudssData_t solverData;
  bool A_created;
  cudssMatrix_t A;
  cudssMatrix_t cudss_x;
  cudssMatrix_t cudss_b;
  i_t* csr_offset_d;
  i_t* csr_columns_d;
  f_t* csr_values_d;
  f_t* x_values_d;
  f_t* b_values_d;

  const simplex_solver_settings_t<i_t, f_t>& settings_;
  CUgreenCtx barrier_green_ctx;
  CUstream stream;
  void* cuGetErrorString_func;
};

}  // namespace cuopt::linear_programming::dual_simplex
