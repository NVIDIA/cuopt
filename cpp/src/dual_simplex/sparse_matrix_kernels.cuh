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

#include <dual_simplex/cusparse_info.hpp>
#include <dual_simplex/device_sparse_matrix.cuh>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void initialize_cusparse_data(raft::handle_t const* handle,
                              device_csr_matrix_t<i_t, f_t>& A,
                              device_csc_matrix_t<i_t, f_t>& DAT,
                              device_csr_matrix_t<i_t, f_t>& ADAT,
                              cusparse_info_t<i_t, f_t>& cusparse_data)
{
  auto A_nnz         = A.nz_max;
  auto DAT_nnz       = DAT.nz_max;
  f_t chunk_fraction = 0.15;

  // Create matrix descriptors
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
    &cusparse_data.matA_descr, A.m, A.n, A_nnz, A.row_start.data(), A.j.data(), A.x.data()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&cusparse_data.matDAT_descr,
                                                            DAT.n,
                                                            DAT.m,
                                                            DAT_nnz,
                                                            DAT.col_start.data(),
                                                            DAT.i.data(),
                                                            DAT.x.data()));

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&cusparse_data.matADAT_descr,
                                                            ADAT.m,
                                                            ADAT.n,
                                                            0,
                                                            ADAT.row_start.data(),
                                                            ADAT.j.data(),
                                                            ADAT.x.data()));
  RAFT_CUSPARSE_TRY(cusparseSpGEMM_createDescr(&cusparse_data.spgemm_descr));

  // Buffer size
  size_t buffer_size;
  RAFT_CUSPARSE_TRY(cusparseSpGEMM_workEstimation(handle->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  cusparse_data.alpha.data(),
                                                  cusparse_data.matA_descr,
                                                  cusparse_data.matDAT_descr,
                                                  cusparse_data.beta.data(),
                                                  cusparse_data.matADAT_descr,
                                                  CUDA_R_64F,
                                                  CUSPARSE_SPGEMM_ALG3,
                                                  cusparse_data.spgemm_descr,
                                                  &buffer_size,
                                                  nullptr));
  cusparse_data.buffer_size.resize(buffer_size, handle->get_stream());

  RAFT_CUSPARSE_TRY(cusparseSpGEMM_workEstimation(handle->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  cusparse_data.alpha.data(),
                                                  cusparse_data.matA_descr,
                                                  cusparse_data.matDAT_descr,
                                                  cusparse_data.beta.data(),
                                                  cusparse_data.matADAT_descr,
                                                  CUDA_R_64F,
                                                  CUSPARSE_SPGEMM_ALG3,
                                                  cusparse_data.spgemm_descr,
                                                  &buffer_size,
                                                  cusparse_data.buffer_size.data()));

  int64_t num_prods;
  RAFT_CUSPARSE_TRY(cusparseSpGEMM_getNumProducts(cusparse_data.spgemm_descr, &num_prods));

  size_t buffer_size_3_size;
  RAFT_CUSPARSE_TRY(cusparseSpGEMM_estimateMemory(handle->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  cusparse_data.alpha.data(),
                                                  cusparse_data.matA_descr,
                                                  cusparse_data.matDAT_descr,
                                                  cusparse_data.beta.data(),
                                                  cusparse_data.matADAT_descr,
                                                  CUDA_R_64F,
                                                  CUSPARSE_SPGEMM_ALG3,
                                                  cusparse_data.spgemm_descr,
                                                  chunk_fraction,
                                                  &buffer_size_3_size,
                                                  nullptr,
                                                  nullptr));
  cusparse_data.buffer_size_3.resize(buffer_size_3_size, handle->get_stream());

  RAFT_CUSPARSE_TRY(cusparseSpGEMM_estimateMemory(handle->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  cusparse_data.alpha.data(),
                                                  cusparse_data.matA_descr,
                                                  cusparse_data.matDAT_descr,
                                                  cusparse_data.beta.data(),
                                                  cusparse_data.matADAT_descr,
                                                  CUDA_R_64F,
                                                  CUSPARSE_SPGEMM_ALG3,
                                                  cusparse_data.spgemm_descr,
                                                  chunk_fraction,
                                                  &buffer_size_3_size,
                                                  cusparse_data.buffer_size_3.data(),
                                                  &cusparse_data.buffer_size_2_size));
  cusparse_data.buffer_size_3.resize(0, handle->get_stream());
  cusparse_data.buffer_size_2.resize(cusparse_data.buffer_size_2_size, handle->get_stream());
}

template <typename i_t, typename f_t>
void multiply_kernels(raft::handle_t const* handle,
                      device_csr_matrix_t<i_t, f_t>& A,
                      device_csc_matrix_t<i_t, f_t>& DAT,
                      device_csr_matrix_t<i_t, f_t>& ADAT,
                      cusparse_info_t<i_t, f_t>& cusparse_data)
{
  RAFT_CUSPARSE_TRY(
    cusparseSpGEMM_compute(handle->get_cusparse_handle(),
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           cusparse_data.alpha.data(),
                           cusparse_data.matA_descr,    // non-const descriptor supported
                           cusparse_data.matDAT_descr,  // non-const descriptor supported
                           cusparse_data.beta.data(),
                           cusparse_data.matADAT_descr,
                           CUDA_R_64F,
                           CUSPARSE_SPGEMM_ALG3,
                           cusparse_data.spgemm_descr,
                           &cusparse_data.buffer_size_2_size,
                           cusparse_data.buffer_size_2.data()));

  // get matrix C non-zero entries C_nnz1
  int64_t ADAT_num_rows, ADAT_num_cols, ADAT_nnz1;
  RAFT_CUSPARSE_TRY(
    cusparseSpMatGetSize(cusparse_data.matADAT_descr, &ADAT_num_rows, &ADAT_num_cols, &ADAT_nnz1));
  ADAT.resize_to_nnz(ADAT_nnz1, handle->get_stream());

  thrust::fill(rmm::exec_policy(handle->get_stream()), ADAT.x.begin(), ADAT.x.end(), 0.0);

  // update matC with the new pointers
  RAFT_CUSPARSE_TRY(cusparseCsrSetPointers(
    cusparse_data.matADAT_descr, ADAT.row_start.data(), ADAT.j.data(), ADAT.x.data()));

  RAFT_CUSPARSE_TRY(cusparseSpGEMM_copy(handle->get_cusparse_handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        cusparse_data.alpha.data(),
                                        cusparse_data.matA_descr,
                                        cusparse_data.matDAT_descr,
                                        cusparse_data.beta.data(),
                                        cusparse_data.matADAT_descr,
                                        CUDA_R_64F,
                                        CUSPARSE_SPGEMM_ALG3,
                                        cusparse_data.spgemm_descr));

  handle->sync_stream();
}

}  // namespace cuopt::linear_programming::dual_simplex
