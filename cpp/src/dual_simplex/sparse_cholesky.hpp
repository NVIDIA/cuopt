/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "dual_simplex/simplex_solver_settings.hpp"
#include "dual_simplex/sparse_matrix.hpp"
#include "dual_simplex/tic_toc.hpp"

#include <cuda_runtime.h>

#include "cudss.h"

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class sparse_cholesky_base_t {
 public:
  virtual ~sparse_cholesky_base_t()                                                 = default;
  virtual i_t analyze(const csc_matrix_t<i_t, f_t>& A_in)                           = 0;
  virtual i_t factorize(const csc_matrix_t<i_t, f_t>& A_in)                         = 0;
  virtual i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) = 0;
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

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                               \
  do {                                                                                        \
    status = call;                                                                            \
    if (status != CUDSS_STATUS_SUCCESS) {                                                     \
      printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", \
             status);                                                                         \
      CUDSS_EXAMPLE_FREE;                                                                     \
      return -2;                                                                              \
    }                                                                                         \
  } while (0);

#define CUDSS_CALL_AND_CHECK_EXIT(call, status, msg)                                          \
  do {                                                                                        \
    status = call;                                                                            \
    if (status != CUDSS_STATUS_SUCCESS) {                                                     \
      printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", \
             status);                                                                         \
      CUDSS_EXAMPLE_FREE;                                                                     \
      exit(-2);                                                                               \
    }                                                                                         \
  } while (0);

template <typename i_t, typename f_t>
class sparse_cholesky_cudss_t : public sparse_cholesky_base_t<i_t, f_t> {
 public:
  sparse_cholesky_cudss_t(const simplex_solver_settings_t<i_t, f_t>& settings, i_t size)
    : n(size), nnz(-1), first_factor(true), positive_definite(true), settings_(settings)
  {
    int major, minor, patch;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL, &patch);
    settings.log.printf("CUDSS Version %d.%d.%d\n", major, minor, patch);

    cuda_error = cudaSuccess;
    status     = CUDSS_STATUS_SUCCESS;
    CUDA_CALL_AND_CHECK_EXIT(cudaStreamCreate(&stream), "cudaStreamCreate");
    CUDSS_CALL_AND_CHECK_EXIT(cudssCreate(&handle), status, "cudssCreate");
    CUDSS_CALL_AND_CHECK_EXIT(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK_EXIT(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

#ifdef USE_AMD
    // Tell cuDSS to use AMD
    cudssAlgType_t reorder_alg = CUDSS_ALG_3;
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssConfigSet(
        solverConfig, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(cudssAlgType_t)),
      status,
      "cudssConfigSet for reordering alg");
#endif

#if 0
        int32_t ir_n_steps = 2;
        CUDSS_CALL_AND_CHECK_EXIT(cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS,
                                          &ir_n_steps, sizeof(int32_t)), status, "cudssConfigSet for ir n steps");
#endif

    // Device pointers
    csr_offset_d  = nullptr;
    csr_columns_d = nullptr;
    csr_values_d  = nullptr;
    x_values_d    = nullptr;
    b_values_d    = nullptr;
    CUDA_CALL_AND_CHECK_EXIT(cudaMalloc(&x_values_d, n * sizeof(f_t)), "cudaMalloc for x_values");
    CUDA_CALL_AND_CHECK_EXIT(cudaMalloc(&b_values_d, n * sizeof(f_t)), "cudaMalloc for b_values");

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
  }
  ~sparse_cholesky_cudss_t() override
  {
    cudaFree(csr_values_d);
    cudaFree(csr_columns_d);
    cudaFree(csr_offset_d);

    cudaFree(x_values_d);
    cudaFree(b_values_d);

    CUDSS_CALL_AND_CHECK_EXIT(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");

    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixDestroy(cudss_x), status, "cudssMatrixDestroy for cudss_x");
    CUDSS_CALL_AND_CHECK_EXIT(
      cudssMatrixDestroy(cudss_b), status, "cudssMatrixDestroy for cudss_b");
    CUDSS_CALL_AND_CHECK_EXIT(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK_EXIT(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK_EXIT(cudssDestroy(handle), status, "cudssDestroy");
    CUDA_CALL_AND_CHECK_EXIT(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    CUDA_CALL_AND_CHECK_EXIT(cudaStreamDestroy(stream), "cudaStreamDestroy");
  }
  i_t analyze(const csc_matrix_t<i_t, f_t>& A_in) override
  {
    csr_matrix_t<i_t, f_t> Arow;
#ifdef WRITE_MATRIX_MARKET
    FILE* fid = fopen("A.mtx", "w");
    A_in.write_matrix_market(fid);
    fclose(fid);
#endif
    A_in.to_compressed_row(Arow);

    nnz = A_in.col_start[A_in.n];

    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offset_d, (n + 1) * sizeof(i_t)),
                        "cudaMalloc for csr_offset");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(i_t)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(f_t)), "cudaMalloc for csr_values");

    CUDA_CALL_AND_CHECK(
      cudaMemcpy(
        csr_offset_d, Arow.row_start.data(), (n + 1) * sizeof(i_t), cudaMemcpyHostToDevice),
      "cudaMemcpy for csr_offset");
    CUDA_CALL_AND_CHECK(
      cudaMemcpy(csr_columns_d, Arow.j.data(), nnz * sizeof(i_t), cudaMemcpyHostToDevice),
      "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(
      cudaMemcpy(csr_values_d, Arow.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice),
      "cudaMemcpy for csr_values");

    if (!first_factor) {
      CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
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

    // Perform symbolic analysis
    f_t start_analysis = tic();

    CUDSS_CALL_AND_CHECK(
      cudssExecute(handle, CUDSS_PHASE_REORDERING, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for reordering");

    f_t reorder_time = toc(start_analysis);
    settings_.log.printf("Reordering time %.2fs\n", reorder_time);

    f_t start_symbolic = tic();

    CUDSS_CALL_AND_CHECK(
      cudssExecute(handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for symbolic factorization");

    f_t symbolic_time = toc(start_symbolic);
    f_t analysis_time = toc(start_analysis);
    settings_.log.printf("Symbolic factorization time %.2fs\n", symbolic_time);
    settings_.log.printf("Symbolic time %.2fs\n", analysis_time);
    int64_t lu_nz       = 0;
    size_t size_written = 0;
    CUDSS_CALL_AND_CHECK(
      cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nz, sizeof(int64_t), &size_written),
      status,
      "cudssDataGet for LU_NNZ");
    settings_.log.printf("Symbolic nonzeros in factor %e\n", static_cast<f_t>(lu_nz) / 2.0);
    // TODO: Is there any way to get nonzeros in the factors?
    // TODO: Is there any way to get flops for the factorization?

    return 0;
  }
  i_t factorize(const csc_matrix_t<i_t, f_t>& A_in) override
  {
    csr_matrix_t<i_t, f_t> Arow;
    A_in.to_compressed_row(Arow);

    if (nnz != A_in.col_start[A_in.n]) {
      settings_.log.printf("Error: nnz %d != A_in.col_start[A_in.n] %d\n", nnz, A_in.col_start[A_in.n]);
      exit(1);
    }

    CUDA_CALL_AND_CHECK(
      cudaMemcpy(csr_values_d, Arow.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice),
      "cudaMemcpy for csr_values");

    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(A, csr_values_d), status, "cudssMatrixSetValues for A");

    f_t start_numeric = tic();
    CUDSS_CALL_AND_CHECK(
      cudssExecute(
        handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for factorization");

    f_t numeric_time = toc(start_numeric);

    int info;
    size_t sizeWritten = 0;
    CUDSS_CALL_AND_CHECK(
      cudssDataGet(handle, solverData, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten),
      status,
      "cudssDataGet for info");
    if (info != 0) {
      settings_.log.printf("Factorization failed info %d\n", info);
      return -1;
    }

    if (first_factor) {
      settings_.log.printf("Factor time %.2fs\n", numeric_time);
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
    if (b.size() != n) {
      settings_.log.printf("Error: b.size() %d != n %d\n", b.size(), n);
      exit(1);
    }
    if (x.size() != n) {
      settings_.log.printf("Error: x.size() %d != n %d\n", x.size(), n);
      exit(1);
    }
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy for b_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_d, x.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy for x_values");

    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(cudss_b, b_values_d), status, "cudssMatrixSetValues for b");
    CUDSS_CALL_AND_CHECK(
      cudssMatrixSetValues(cudss_x, x_values_d), status, "cudssMatrixSetValues for x");

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

    CUDSS_CALL_AND_CHECK(
      cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, cudss_x, cudss_b),
      status,
      "cudssExecute for solve");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    CUDA_CALL_AND_CHECK(cudaMemcpy(x.data(), x_values_d, n * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x");

    for (i_t i = 0; i < n; i++) {
      if (x[i] != x[i]) { return -1; }
    }

    return 0;
  }

  void set_positive_definite(bool positive_definite) override
  {
    this->positive_definite = positive_definite;
  }

 private:
  i_t n;
  i_t nnz;
  bool first_factor;
  bool positive_definite;
  cudaError_t cuda_error;
  cudssStatus_t status;
  cudaStream_t stream;
  cudssHandle_t handle;
  cudssConfig_t solverConfig;
  cudssData_t solverData;
  cudssMatrix_t A;
  cudssMatrix_t cudss_x;
  cudssMatrix_t cudss_b;
  i_t* csr_offset_d;
  i_t* csr_columns_d;
  f_t* csr_values_d;
  f_t* x_values_d;
  f_t* b_values_d;

  const simplex_solver_settings_t<i_t, f_t>& settings_;
};

}  // namespace cuopt::linear_programming::dual_simplex
