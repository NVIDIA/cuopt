/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/mip_constants.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>
#include <utilities/vector_helpers.cuh>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/transpose.cuh>
#include "cusparse.h"

#include <cub/cub.cuh>
#include "conditional_bound_strengthening.cuh"

#include <unordered_set>
namespace cuopt::linear_programming::detail {

constexpr size_t max_pair_per_row = 100;

template <typename i_t, typename f_t>
conditional_bound_strengthening_t<i_t, f_t>::conditional_bound_strengthening_t(
  problem_t<i_t, f_t>& problem)
  : constraint_pairs(0, problem.handle_ptr->get_stream()),
    locks_per_constraint(0, problem.handle_ptr->get_stream())
{
}

template <typename i_t, typename f_t>
void conditional_bound_strengthening_t<i_t, f_t>::resize(problem_t<i_t, f_t>& problem)
{
  constraint_pairs.resize(0, problem.handle_ptr->get_stream());
  locks_per_constraint.resize(2 * problem.n_constraints, problem.handle_ptr->get_stream());
  // FIXME:: for now fall back on to cpu if there is not enough memory. A better way to do this
  // is computing by chunks, i.e. subset of rows at a time
  try {
    select_constraint_pairs_device(problem);
  } catch (std::bad_alloc& e) {
    select_constraint_pairs_host(problem);
  }
  async_fill(locks_per_constraint, 0, problem.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void conditional_bound_strengthening_t<i_t, f_t>::update_constraint_bounds(
  problem_t<i_t, f_t>& problem, bound_presolve_t<i_t, f_t>& bounds_update)
{
  // initialize constraint pairs
  resize(problem);
  // Update the constraint bounds based on activity if activities are tighter than original bounds
  bounds_update.calc_and_set_updated_constraint_bounds(problem);
  // Update constraint bounds using knapsack solves
  solve(problem);
}

void check_cusparse_status(cusparseStatus_t status)
{
  if (status != CUSPARSE_STATUS_SUCCESS) { throw std::bad_alloc(); }
}

template <typename i_t, typename f_t>
void spgemm_cusparse([[maybe_unused]] rmm::device_uvector<i_t>& offsetsA,
                     [[maybe_unused]] rmm::device_uvector<i_t>& colsA,
                     [[maybe_unused]] rmm::device_uvector<f_t>& valsA,
                     [[maybe_unused]] rmm::device_uvector<i_t>& offsetsB,
                     [[maybe_unused]] rmm::device_uvector<i_t>& colsB,
                     [[maybe_unused]] rmm::device_uvector<f_t>& valsB,
                     [[maybe_unused]] rmm::device_uvector<i_t>& offsetsC,
                     [[maybe_unused]] rmm::device_uvector<i_t>& colsC,
                     [[maybe_unused]] rmm::device_uvector<f_t>& valsC)
{
#if CUDART_VERSION >= 12000
  auto stream = offsetsA.stream();
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseSetStream(handle, stream);

  int m    = offsetsA.size() - 1;
  int n    = offsetsB.size() - 1;
  int nnzA = colsA.size();
  int nnzB = colsB.size();

  offsetsC.resize(m + 1, stream);

  cusparseSpMatDescr_t matA, matB, matC;
  cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG1;
  rmm::device_buffer dBuffer1(0, stream), dBuffer2(0, stream), dBuffer3(0, stream);

  float alpha              = 1.0f;
  float beta               = 0.0f;
  cusparseOperation_t opA  = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB  = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;

  check_cusparse_status(cusparseCreateCsr(&matA,
                                          m,
                                          n,
                                          nnzA,
                                          offsetsA.data(),
                                          colsA.data(),
                                          valsA.data(),
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_32F));
  check_cusparse_status(cusparseCreateCsr(&matB,
                                          n,
                                          m,
                                          nnzB,
                                          offsetsB.data(),
                                          colsB.data(),
                                          valsB.data(),
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_32F));
  check_cusparse_status(cusparseCreateCsr(&matC,
                                          m,
                                          m,
                                          0,
                                          offsetsC.data(),
                                          NULL,
                                          NULL,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          CUDA_R_32F));

  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);

  size_t bufferSize1 = 0;
  // ask bufferSize1 bytes for external memory
  check_cusparse_status(cusparseSpGEMM_workEstimation(handle,
                                                      opA,
                                                      opB,
                                                      &alpha,
                                                      matA,
                                                      matB,
                                                      &beta,
                                                      matC,
                                                      computeType,
                                                      alg,
                                                      spgemmDesc,
                                                      &bufferSize1,
                                                      NULL));
  dBuffer1.resize(bufferSize1, stream);

  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  check_cusparse_status(cusparseSpGEMM_workEstimation(handle,
                                                      opA,
                                                      opB,
                                                      &alpha,
                                                      matA,
                                                      matB,
                                                      &beta,
                                                      matC,
                                                      computeType,
                                                      alg,
                                                      spgemmDesc,
                                                      &bufferSize1,
                                                      dBuffer1.data()));

  size_t bufferSize2 = 0;
  // ask bufferSize2 bytes for external memory
  check_cusparse_status(cusparseSpGEMM_compute(handle,
                                               opA,
                                               opB,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               computeType,
                                               alg,
                                               spgemmDesc,
                                               &bufferSize2,
                                               NULL));
  dBuffer2.resize(bufferSize2, stream);

  // compute the intermediate product of A * B
  check_cusparse_status(cusparseSpGEMM_compute(handle,
                                               opA,
                                               opB,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               computeType,
                                               alg,
                                               spgemmDesc,
                                               &bufferSize2,
                                               dBuffer2.data()));

  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  check_cusparse_status(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1));
  colsC.resize(C_nnz1, stream);
  valsC.resize(C_nnz1, stream);

  check_cusparse_status(cusparseCsrSetPointers(matC, offsetsC.data(), colsC.data(), valsC.data()));

  check_cusparse_status(cusparseSpGEMM_copy(
    handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, alg, spgemmDesc));
  stream.synchronize();

  cusparseSpGEMM_destroyDescr(spgemmDesc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
#else
  throw std::bad_alloc();
#endif
}

// #define DEBUG_COND_BOUNDS_PROP

template <typename i_t, typename f_t>
void conditional_bound_strengthening_t<i_t, f_t>::select_constraint_pairs_host(
  problem_t<i_t, f_t>& problem)
{
#ifdef DEBUG_COND_BOUNDS_PROP
  auto start_time = std::chrono::high_resolution_clock::now();
#endif
  auto variables = cuopt::host_copy(problem.variables);
  auto offsets   = cuopt::host_copy(problem.offsets);

  auto reverse_constraints = cuopt::host_copy(problem.reverse_constraints);
  auto reverse_offsets     = cuopt::host_copy(problem.reverse_offsets);

  std::vector<int2> constraint_pairs_h(max_pair_per_row * problem.n_constraints, {-1, -1});
  std::unordered_set<int> cnstr_pair;

#pragma omp parallel for private(cnstr_pair)
  for (int cnstr = 0; cnstr < problem.n_constraints; ++cnstr) {
    for (int jj = offsets[cnstr]; jj < offsets[cnstr + 1]; ++jj) {
      int var = variables[jj];
      for (int kk = reverse_offsets[var]; kk < reverse_offsets[var + 1]; ++kk) {
        if (reverse_constraints[kk] != cnstr) { cnstr_pair.insert(reverse_constraints[kk]); }
        if (cnstr_pair.size() == max_pair_per_row) { break; }
      }
      // FIXME: have a better mechanism instead
      if (cnstr_pair.size() == max_pair_per_row) { break; }
    }

    int counter = 0;
    for (auto& temp : cnstr_pair) {
      constraint_pairs_h[cnstr * max_pair_per_row + counter++] = {cnstr, temp};
    }
    cnstr_pair.clear();
  }

  constraint_pairs = cuopt::device_copy(constraint_pairs_h, problem.handle_ptr->get_stream());

#ifdef DEBUG_COND_BOUNDS_PROP
  auto end_time = std::chrono::high_resolution_clock::now();
  std::cout << "Time for constructing pairs:: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << " ms" << std::endl;
#endif
}

template <typename i_t, typename f_t>
void conditional_bound_strengthening_t<i_t, f_t>::select_constraint_pairs_device(
  problem_t<i_t, f_t>& problem)
{
#ifdef DEBUG_COND_BOUNDS_PROP
  auto start_time = std::chrono::high_resolution_clock::now();
#endif
  auto stream = problem.variables.stream();
  rmm::device_uvector<float> valsA(problem.variables.size(), stream),
    valsB(problem.variables.size(), stream), valsC(0, stream);
  // fill valsA, valsB with 1
  thrust::fill(problem.handle_ptr->get_thrust_policy(), valsA.begin(), valsA.end(), 1.f);
  thrust::fill(problem.handle_ptr->get_thrust_policy(), valsB.begin(), valsB.end(), 1.f);

  rmm::device_uvector<i_t> offsetsC(0, stream), colsC(0, stream);
  std::vector<i_t> offsets_h, cols_h;

  spgemm_cusparse(problem.offsets,
                  problem.variables,
                  valsA,
                  problem.reverse_offsets,
                  problem.reverse_constraints,
                  valsB,
                  offsetsC,
                  colsC,
                  valsC);
  std::vector<int2> constraint_pairs_h;
  offsets_h = cuopt::host_copy(offsetsC);
  cols_h    = cuopt::host_copy(colsC);

  constraint_pairs_h.reserve(max_pair_per_row * problem.n_constraints);
  for (int i = 0; i < problem.n_constraints; ++i) {
    int cnstr_i = i;
    int cnt     = 0;
    for (int jj = offsets_h[i]; jj < offsets_h[i + 1]; ++jj) {
      int cnstr_j = cols_h[jj];
      cuopt_expects(cnstr_j < problem.n_constraints && cnstr_j >= 0,
                    error_type_t::RuntimeError,
                    "Constraint index should be in range");

      if (cnstr_i != cnstr_j) {
        constraint_pairs_h.push_back({cnstr_i, cnstr_j});
        cnt++;
      }
      if (cnt == max_pair_per_row) { break; }
    }
  }

  constraint_pairs = cuopt::device_copy(constraint_pairs_h, problem.handle_ptr->get_stream());

#ifdef DEBUG_COND_BOUNDS_PROP
  auto end_time = std::chrono::high_resolution_clock::now();

  std::cout << "Time for constructing pairs:: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << " ms" << std::endl;
#endif
}

struct custom_less {
  template <typename DataType>
  __device__ bool operator()(const DataType& lhs, const DataType& rhs)
  {
    return lhs < rhs;
  }
};

template <typename i_t, typename f_t, int TPB>
__device__ f_t knapsack_solve(raft::device_span<f_t> c,
                              raft::device_span<f_t> a,
                              raft::device_span<f_t> lb,
                              raft::device_span<f_t> ub,
                              raft::device_span<var_t> vtypes,
                              const f_t a_l,
                              const f_t a_u,
                              raft::device_span<f_t> x,
                              raft::device_span<i_t> sorted_indices,
                              f_t feasibility_tol)
{
  i_t tid       = threadIdx.x;
  f_t ax        = 0.;
  i_t unbounded = 0;
  // assume that the row size is less than TPB
  if (tid < a.size()) {
    x[tid] = c[tid] > 0 ? lb[tid] : ub[tid];
    ax     = a[tid] * x[tid];

    unbounded = (ub[tid] == std::numeric_limits<f_t>::infinity() && c[tid] <= 0.) ||
                (lb[tid] == -std::numeric_limits<f_t>::infinity() && c[tid] >= 0.);
  }

  cuopt_assert(a.size() == c.size(), "objective and constraint should have same size");
  cuopt_assert(a.size() == lb.size(), "constraint and lower bounds should have same size");
  cuopt_assert(a.size() == ub.size(), "constraint and upper bounds should have same size");
  cuopt_assert(a.size() == x.size(), "constraint and solution should have same size");
  cuopt_assert(a.size() == sorted_indices.size(), "constraint and indices should have same size");

  __shared__ f_t shmem_for_sum[raft::WarpSize];
  bool any_unbounded = raft::blockReduce(unbounded, (char*)shmem_for_sum) > 0;
  if (any_unbounded) { return -std::numeric_limits<f_t>::infinity(); }

  __syncthreads();

  f_t w_init = raft::blockReduce(ax, (char*)shmem_for_sum);

  cuopt_assert(std::isfinite(w_init), "w should be finite");

  if (w_init > a_u + feasibility_tol || w_init < a_l - feasibility_tol) {
    __syncthreads();
    // Do cub sort and store the indices
    // compute sorted indices;
    double fact = w_init > a_u ? -1. : 1.;

    using BlockMergeSort = cub::BlockMergeSort<f_t, TPB, 1, i_t>;
    __shared__ typename BlockMergeSort::TempStorage temp_storage;

    f_t thread_val[1];
    i_t sorted_id[1];

    thread_val[0] = tid < a.size() ? fact * c[tid] / a[tid] : 1e10;
    sorted_id[0]  = tid;

    BlockMergeSort(temp_storage).Sort(thread_val, sorted_id, custom_less());

    if (tid < a.size()) { sorted_indices[tid] = sorted_id[0]; }

    __syncthreads();

    if (threadIdx.x == 0) {
      // FIXME:: Instead of using just thread id 0, we can do a cumulative sum of maximum changes of
      // each variable, and find the index where we switch from infeasibility to feasibility
      int running_index = 0;
      while (w_init > a_u + feasibility_tol || w_init < a_l - feasibility_tol) {
        f_t req_change = w_init > a_u ? (a_u - w_init) : (a_l - w_init);
        f_t dxi        = 0;

        for (; running_index < a.size(); ++running_index) {
          int i  = sorted_indices[running_index];
          f_t ai = a[i];
          f_t xi = x[i];
          f_t li = lb[i];
          f_t ui = ub[i];
          if ((xi == li && fact * ai > 0.) || (xi == ui && fact * ai < 0.)) {
            f_t temp = req_change / ai;
            dxi      = min(temp, ui - li);
            dxi      = max(dxi, li - ui);
            cuopt_assert(std::isfinite(dxi), "solution change should be finite");
            // Using integer logic is making things worse, FIXME:: Need to re-check this logic
#if 0
            if (var_t::INTEGER == vtypes[i]) {
              // assume li and ui are integers for vtypes, so dxi should also be an integer
              if (xi == li) {
                dxi = ceil(dxi); dxi = min(dxi, ui - li);
              } else {
                dxi = floor(dxi); dxi = max(dxi, li - ui);
              }
            }
#endif
            break;
          }
        }

        if (running_index == a.size()) { break; }

        cuopt_assert(running_index < sorted_indices.size(), "index should be in range");
        int best_index = sorted_indices[running_index];
        cuopt_assert(best_index < a.size() && best_index >= 0, "best index should be in range");
        w_init += a[best_index] * dxi;
        x[best_index] += dxi;

        running_index++;
      }
    }
  }

  __syncthreads();
  f_t cx = 0.;
  if (tid < a.size()) { cx = c[tid] * x[tid]; }

  f_t best_cost = raft::blockReduce(cx, (char*)shmem_for_sum);
  return best_cost;
}

template <typename inT, typename outT>
DI thrust::tuple<raft::device_span<outT>, inT*> create_shared_span(inT* shmem, size_t sz)
{
  auto vec = raft::device_span<outT>{(outT*)shmem, sz};

  return {vec, (inT*)&(vec.data()[sz])};
}

template <typename i_t>
DI i_t binary_lookup(raft::device_span<i_t> arr, i_t begin, i_t end, i_t val)
{
  i_t low  = begin;
  i_t high = end - 1;

  if (arr[low] > val || arr[high] < val) { return -1; }

  while (low <= high) {
    i_t mid = (low + high) >> 1;
    if (arr[mid] == val) {
      return mid;
    } else if (arr[mid] < val) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return -1;
}

template <typename i_t, typename f_t, int TPB>
__global__ void update_constraint_bounds_kernel(typename problem_t<i_t, f_t>::view_t pb,
                                                raft::device_span<int2> constraint_pairs,
                                                raft::device_span<i_t> lock_per_constraint)
{
  auto constraint_pair = constraint_pairs[blockIdx.x];
  int constr_i         = get_lower(constraint_pair);
  if (constr_i == -1) { return; }

  int constr_j = get_upper(constraint_pair);

  // FIXME:: for now handle only the constraints that fit in shared
  i_t offset_j                  = pb.offsets[constr_j];
  i_t offset_i                  = pb.offsets[constr_i];
  i_t n_variables_in_constraint = pb.offsets[constr_j + 1] - pb.offsets[constr_j];
  i_t n_variables_in_cost       = pb.offsets[constr_i + 1] - pb.offsets[constr_i];

  if (n_variables_in_constraint > TPB) { return; }

  extern __shared__ i_t shmem_temp[];

  i_t* shmem = &(shmem_temp[0]);

  // 5 doubles and 1 int
  raft::device_span<i_t> index_buffer;
  raft::device_span<f_t> c, a, lb, ub, x;
  raft::device_span<var_t> vtypes;
  thrust::tie(c, shmem)            = create_shared_span<i_t, f_t>(shmem, n_variables_in_constraint);
  thrust::tie(a, shmem)            = create_shared_span<i_t, f_t>(shmem, n_variables_in_constraint);
  thrust::tie(lb, shmem)           = create_shared_span<i_t, f_t>(shmem, n_variables_in_constraint);
  thrust::tie(ub, shmem)           = create_shared_span<i_t, f_t>(shmem, n_variables_in_constraint);
  thrust::tie(x, shmem)            = create_shared_span<i_t, f_t>(shmem, n_variables_in_constraint);
  thrust::tie(index_buffer, shmem) = create_shared_span<i_t, i_t>(shmem, n_variables_in_constraint);
  thrust::tie(vtypes, shmem) = create_shared_span<i_t, var_t>(shmem, n_variables_in_constraint);

  // The bounds could be updated by other blocks, so we have to get new bounds,
  // each thread might get different values for bounds, so here we lock by thread 0 and
  // get the values into shared memory before copying it to thread variables
  __shared__ f_t sh_a_l, sh_a_u;
  if (threadIdx.x == 0) {
    // FIXME:: Do we need lock here?
    acquire_lock(&lock_per_constraint[2 * constr_j]);
    sh_a_l = pb.constraint_lower_bounds[constr_j];
    release_lock(&lock_per_constraint[2 * constr_j]);

    acquire_lock(&lock_per_constraint[2 * constr_j + 1]);
    sh_a_u = pb.constraint_upper_bounds[constr_j];
    release_lock(&lock_per_constraint[2 * constr_j + 1]);
  }

  __syncthreads();

  const auto feasibility_tol = pb.tolerances.absolute_tolerance;
  if (sh_a_u - sh_a_l < feasibility_tol) { return; }

  i_t tid = threadIdx.x;
  if (tid < n_variables_in_constraint) {
    i_t variable_j = pb.variables[offset_j + tid];
    a[tid]         = pb.coefficients[offset_j + tid];
    auto bounds    = pb.variable_bounds[variable_j];
    lb[tid]        = get_lower(bounds);
    ub[tid]        = get_upper(bounds);
    vtypes[tid]    = pb.variable_types[variable_j];

    c[tid] = 0.;

    i_t ii =
      binary_lookup(pb.variables, pb.offsets[constr_i], pb.offsets[constr_i + 1], variable_j);
    if (ii >= 0) {
      c[tid] = pb.coefficients[ii];
      cuopt_assert(variable_j == pb.variables[ii], "binary lookup should be correct");
    }
  }

  f_t min_activity_if_not_participating = 0.;
  f_t max_activity_if_not_participating = 0.;

  for (i_t index = tid; index < n_variables_in_cost; index += blockDim.x) {
    i_t variable_i = pb.variables[offset_i + index];
    i_t jj =
      binary_lookup(pb.variables, pb.offsets[constr_j], pb.offsets[constr_j + 1], variable_i);

    if (jj < 0) {
      f_t coeff = pb.coefficients[offset_i + index];

      auto bounds = pb.variable_bounds[variable_i];
      f_t li      = get_lower(bounds);
      f_t ui      = get_upper(bounds);
      min_activity_if_not_participating += (coeff > 0. ? coeff * li : coeff * ui);
      max_activity_if_not_participating += (coeff > 0. ? coeff * ui : coeff * li);
    }
  }

  __shared__ f_t shmem_for_sum[raft::WarpSize];
  f_t min_activity_of_not_participating =
    raft::blockReduce(min_activity_if_not_participating, (char*)shmem_for_sum);
  __syncthreads();
  f_t max_activity_of_not_participating =
    raft::blockReduce(max_activity_if_not_participating, (char*)shmem_for_sum);
  __syncthreads();

  f_t a_l = sh_a_l;
  f_t a_u = sh_a_u;
  // solve minimization problem
  f_t min_activity_of_participating =
    knapsack_solve<i_t, f_t, TPB>(c, a, lb, ub, vtypes, a_l, a_u, x, index_buffer, feasibility_tol);

  __syncthreads();
  // solve maximization problem
  if (tid < a.size()) { c[tid] *= -1.; }

  __syncthreads();
  f_t max_activity_of_participating = -knapsack_solve<i_t, f_t, TPB>(
    c, a, lb, ub, vtypes, a_l, a_u, x, index_buffer, feasibility_tol);

  if (tid == 0) {
    f_t lower_bound = min_activity_of_participating + min_activity_of_not_participating;
    f_t upper_bound = max_activity_of_participating + max_activity_of_not_participating;

    if (std::isfinite(lower_bound) && lower_bound > pb.constraint_lower_bounds[constr_i]) {
      acquire_lock(&lock_per_constraint[2 * constr_i]);
      if (lower_bound > pb.constraint_lower_bounds[constr_i]) {
        pb.constraint_lower_bounds[constr_i] =
          cuda::std::min(lower_bound, pb.constraint_upper_bounds[constr_i]);
      }
      release_lock(&lock_per_constraint[2 * constr_i]);
    }

    if (std::isfinite(upper_bound) && upper_bound < pb.constraint_upper_bounds[constr_i]) {
      acquire_lock(&lock_per_constraint[2 * constr_i + 1]);
      if (upper_bound < pb.constraint_upper_bounds[constr_i]) {
        pb.constraint_upper_bounds[constr_i] =
          cuda::std::max(upper_bound, pb.constraint_lower_bounds[constr_i]);
      }
      release_lock(&lock_per_constraint[2 * constr_i + 1]);
    }
  }
}

template <typename i_t>
struct len_from_offset {
  __host__ __device__ i_t operator()(const thrust::tuple<i_t, i_t> val) const
  {
    return thrust::get<1>(val) - thrust::get<0>(val);
  }
};

// Ideally this should be precomputed and stored in the problem, but that also means we need to
// update it every time the problem is modified, so we will compute it here for now
template <typename i_t>
i_t get_max_row_size(rmm::device_uvector<i_t>& offsets, rmm::cuda_stream_view stream_view)
{
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(offsets.begin(), offsets.begin() + 1));
  auto end   = thrust::make_zip_iterator(thrust::make_tuple(offsets.end() - 1, offsets.end()));

  i_t max_row_size = thrust::transform_reduce(
    rmm::exec_policy(stream_view), begin, end, len_from_offset<i_t>{}, 0, thrust::maximum<i_t>());
  return max_row_size;
}

template <typename i_t, typename f_t>
void conditional_bound_strengthening_t<i_t, f_t>::solve(problem_t<i_t, f_t>& problem)
{
  constexpr int TPB = 128;
  size_t n_blocks   = constraint_pairs.size();

  if (n_blocks == 0) { return; }
  int max_row_size = get_max_row_size(problem.offsets, problem.handle_ptr->get_stream());
  max_row_size     = std::min(TPB, max_row_size);
  size_t sh_size =
    raft::alignTo(5 * sizeof(f_t) + sizeof(i_t) + sizeof(var_t), sizeof(i_t)) * max_row_size;

#ifdef DEBUG_COND_BOUNDS_PROP
  auto old_lb_h = cuopt::host_copy(problem.constraint_lower_bounds);
  auto old_ub_h = cuopt::host_copy(problem.constraint_upper_bounds);

  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (!set_shmem_of_kernel(update_constraint_bounds_kernel<i_t, f_t, TPB>, sh_size)) { return; }

  update_constraint_bounds_kernel<i_t, f_t, TPB><<<n_blocks, TPB, sh_size>>>(
    problem.view(), cuopt::make_span(constraint_pairs), cuopt::make_span(locks_per_constraint));

  RAFT_CHECK_CUDA(problem.handle_ptr->get_stream());
  problem.handle_ptr->sync_stream();

#ifdef DEBUG_COND_BOUNDS_PROP
  auto end_time = std::chrono::high_resolution_clock::now();

  double time_for_presolve =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  auto new_lb_h = cuopt::host_copy(problem.constraint_lower_bounds);
  auto new_ub_h = cuopt::host_copy(problem.constraint_upper_bounds);

  int num_improvements = 0;
  int num_new_equality = 0;
  double reduced_gap   = 0.;
  for (size_t i = 0; i < new_lb_h.size(); ++i) {
    if (new_lb_h[i] > old_lb_h[i] + 1e-3 || new_ub_h[i] + 1e-3 < old_ub_h[i]) {
      num_improvements++;

      double old_gap = (old_ub_h[i] - old_lb_h[i]);
      double new_gap = (new_ub_h[i] - new_lb_h[i]);
      reduced_gap += (std::isfinite(old_gap) ? (old_gap - new_gap) / old_gap : 1.0);
    }
    if (fabs(new_ub_h[i] - new_lb_h[i]) < 1e-6 && fabs(old_ub_h[i] - old_lb_h[i]) > 1e-6) {
      num_new_equality++;
    }
  }

  double avg_reduced_gap = reduced_gap / problem.n_constraints;

  std::cout << "Num constraints:: " << problem.n_constraints
            << ", num pairs:: " << constraint_pairs.size()
            << ", num_improvements:: " << num_improvements
            << ", perc :: " << 100. * num_improvements / problem.n_constraints
            << ", avg reduced gap :: " << avg_reduced_gap << ", time = " << time_for_presolve
            << std::endl;
#endif
}

#if MIP_INSTANTIATE_FLOAT
template class conditional_bound_strengthening_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class conditional_bound_strengthening_t<int, double>;
#endif
}  // namespace cuopt::linear_programming::detail
