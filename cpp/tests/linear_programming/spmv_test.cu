/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <linear_programming/pdhg.hpp>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/solve.cuh>
#include <linear_programming/spmv.cuh>
#include <linear_programming/utils.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mps_parser.hpp>
#include "utilities/pdlp_test_utilities.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/pdlp/evo_settings.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip/problem/problem.cuh>
#include <mps_parser/parser.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

namespace cuopt::linear_programming::test {

constexpr int bench_iter_count = 100;

struct timeit{
  cudaEvent_t beg, end;
  float *milliseconds;
  rmm::cuda_stream_view stream;
  timeit(float *ms, rmm::cuda_stream_view stream_) :
  milliseconds(ms),
  stream(stream_){
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg, stream);
  }
  ~timeit() {
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    *milliseconds = 0;
    cudaEventElapsedTime(milliseconds, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }
};

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

template <typename i_t, typename f_t>
std::tuple<std::vector<i_t>, std::vector<i_t>, std::vector<i_t>> display_degree_dist(
  detail::problem_t<i_t, f_t>& problem, bool disp_cnst)
{
  rmm::device_uvector<i_t>& offsets = disp_cnst ? problem.offsets : problem.reverse_offsets;
  rmm::device_uvector<i_t> degrees(offsets.size() - 1, problem.handle_ptr->get_stream());
  thrust::transform(problem.handle_ptr->get_thrust_policy(),
                    offsets.begin() + 1,
                    offsets.end(),
                    offsets.begin(),
                    degrees.begin(),
                    thrust::minus<i_t>{});
  thrust::sort(problem.handle_ptr->get_thrust_policy(), degrees.begin(), degrees.end());
  // auto count = thrust::count_if(handle_.get_thrust_policy(),
  //                  degrees.begin(), degrees.end(),
  //                  [] __device__ (auto i) {
  //                  return i == 1;
  //                  });
  // std::cout<<"single variable constraint count : "<<count<<"\n";
  std::vector<i_t> lb_dist;
  std::vector<i_t> ub_dist;
  std::vector<i_t> count_dist;
  for (i_t i = 0; i < 32; ++i) {
    auto lb    = i == 0 ? 0 : std::pow(2, i - 1);
    auto ub    = i == 0 ? 1 : std::pow(2, i);
    auto count = thrust::count_if(problem.handle_ptr->get_thrust_policy(),
                                  degrees.begin(),
                                  degrees.end(),
                                  [lb, ub] __device__(auto i) { return (lb < i) && (i <= ub); });
    if (count != 0) {
      lb_dist.push_back(lb);
      ub_dist.push_back(ub);
      count_dist.push_back(count);
    }
  }
  return std::make_tuple(std::move(lb_dist), std::move(ub_dist), std::move(count_dist));
}

template <typename i_t, typename f_t>
void display_vars_dist(detail::problem_t<i_t, f_t>& problem)
{
  auto [vars_lb_dist, vars_ub_dist, vars_count_dist] = display_degree_dist(problem, false);
  // display max degree
  auto max_count = vars_count_dist[0];
  auto max_bin   = 0;
  for (size_t i = 0; i < vars_count_dist.size(); ++i) {
    if (max_count < vars_count_dist[i]) {
      max_count = vars_count_dist[i];
      max_bin   = i;
    }
  }
  std::cout << "\nvars dist max count ";
  std::cout << vars_ub_dist[max_bin] << " " << max_count << "\n";
  for (size_t i = 0; i < vars_lb_dist.size(); ++i) {
    std::cout << vars_lb_dist[i] << " < degree <= " << vars_ub_dist[i] << "\t" << vars_count_dist[i]
              << " " << ((vars_count_dist[i] * vars_ub_dist[i]) + 31) / 32 << "\n";
  }
}

template <typename i_t, typename f_t>
void display_cnst_dist(detail::problem_t<i_t, f_t>& problem)
{
  auto [cnst_lb_dist, cnst_ub_dist, cnst_count_dist] = display_degree_dist(problem, true);
  // display max degree
  auto max_count = cnst_count_dist[0];
  auto max_bin   = 0;
  for (size_t i = 0; i < cnst_count_dist.size(); ++i) {
    if (max_count < cnst_count_dist[i]) {
      max_count = cnst_count_dist[i];
      max_bin   = i;
    }
  }
  std::cout << "\ncnst dist max count ";
  std::cout << cnst_ub_dist[max_bin] << " " << max_count << "\n";
  for (size_t i = 0; i < cnst_lb_dist.size(); ++i) {
    std::cout << cnst_lb_dist[i] << " < degree <= " << cnst_ub_dist[i] << "\t" << cnst_count_dist[i]
              << " " << ((cnst_count_dist[i] * cnst_ub_dist[i]) + 31) / 32 << "\n";
  }
}

#if 0
void test_spmv_functor(std::string path)
{
  const raft::handle_t handle_{};
  auto stream = handle_.get_stream();
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  // auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  // check_problem_representation(op_problem);
  init_handler(op_problem.get_handle_ptr());
  detail::problem_t<int, double> problem(op_problem);

  std::cout << problem.n_constraints << " ";
  std::cout << problem.n_variables << " ";
  std::cout << problem.nnz << " ";

  display_cnst_dist(problem);
  //display_vars_dist(problem);

  rmm::device_uvector<double> ax_in(problem.n_variables, stream);
  rmm::device_uvector<double> ax_out(problem.n_constraints, stream);
  rmm::device_uvector<double> aty_in(problem.n_constraints, stream);
  rmm::device_uvector<double> aty_out(problem.n_variables, stream);
  rmm::device_uvector<double> aty_n_in(problem.n_constraints, stream);
  rmm::device_uvector<double> aty_n_out(problem.n_variables, stream);

  thrust::fill(handle_.get_thrust_policy(), ax_in.begin(), ax_in.end(), 1);
  thrust::fill(handle_.get_thrust_policy(), ax_out.begin(), ax_out.end(), 0);

  detail::spmv_t<int, double> spmv(problem,
      make_span(ax_in),    make_span(ax_out),
      make_span(aty_in),   make_span(aty_out),
      make_span(aty_n_in), make_span(aty_n_out));
  spmv.Ax(&handle_, make_span(ax_in), make_span(ax_out));
  auto ref_out = host_copy(ax_out);
  auto gld_out = cusparse_call(problem);
}
#endif

template <typename cusp_view_t>
std::pair<std::vector<double>, float> cusparse_call_ax(const raft::handle_t* handle_ptr,
                                     detail::problem_t<int, double>& problem,
                                     cusp_view_t& view,
                                     rmm::device_uvector<double>& x,
                                     rmm::device_uvector<double>& ax)
{
  cusparseDnVecDescr_t cusp_x;
  cusparseDnVecDescr_t cusp_ax;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusp_x, problem.n_variables, x.data()));
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusp_ax, problem.n_constraints, ax.data()));

  rmm::device_scalar<double> sc_1(1.0, handle_ptr->get_stream());
  rmm::device_scalar<double> sc_0(0.0, handle_ptr->get_stream());

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
  cudaEventRecord(beg, handle_ptr->get_stream());
  for (size_t i = 0; i < bench_iter_count; ++i) {
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       sc_1.data(),  // 1
                                                       view.A,
                                                       cusp_x,
                                                       sc_0.data(),  // 0
                                                       cusp_ax,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (double*)view.buffer_non_transpose.data(),
                                                       handle_ptr->get_stream()));
  }
  cudaEventRecord(end, handle_ptr->get_stream());
  cudaEventSynchronize(end);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, beg, end);
  return std::make_pair(host_copy(ax), milliseconds);
}

template <typename cusp_view_t>
std::pair<std::vector<double>, float> cusparse_call_aty(const raft::handle_t* handle_ptr,
                                      detail::problem_t<int, double>& problem,
                                      cusp_view_t& view,
                                      rmm::device_uvector<double>& y,
                                      rmm::device_uvector<double>& aty)
{
  cusparseDnVecDescr_t cusp_y;
  cusparseDnVecDescr_t cusp_aty;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusp_y, problem.n_constraints, y.data()));
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusp_aty, problem.n_variables, aty.data()));

  rmm::device_scalar<double> sc_1(1.0, handle_ptr->get_stream());
  rmm::device_scalar<double> sc_0(0.0, handle_ptr->get_stream());

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
  cudaEventRecord(beg, handle_ptr->get_stream());
  for (size_t i = 0; i < bench_iter_count; ++i) {
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       sc_1.data(),  // 1
                                                       view.A_T,
                                                       cusp_y,
                                                       sc_0.data(),  // 0
                                                       cusp_aty,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (double*)view.buffer_non_transpose.data(),
                                                       handle_ptr->get_stream()));
  }
  cudaEventRecord(end, handle_ptr->get_stream());
  cudaEventSynchronize(end);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, beg, end);
  return std::make_pair(host_copy(aty), milliseconds);
}

template <typename f_t>
bool test_eq(const raft::handle_t* handle_ptr,
             std::vector<f_t>& h_res,
             std::vector<f_t>& h_gld,
             f_t tolerance)
{
  bool is_match = true;
  for (size_t i = 0; i < h_res.size(); ++i) {
    if (abs(h_res[i] - 2 * h_gld[i]) > tolerance) {
      std::cout << "\nmismatch " << i << "\t" << h_res[i] << "\t" << 2 * h_gld[i] << "\n";
      is_match = false;
    }
  }
  if (is_match) { std::cout << "matched\n"; }
  return is_match;
}

void test_spmv_functor(std::string path)
{
  const raft::handle_t handle_{};
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  // auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  // check_problem_representation(op_problem);
  init_handler(op_problem.get_handle_ptr());
  detail::problem_t<int, double> problem(op_problem);

  // scale problem
  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints);
  scaling.scale_problem();

  std::cout << "n_cnst " << problem.n_constraints << " ";
  std::cout << "n_vars " << problem.n_variables << " ";
  std::cout << "nnz " << problem.nnz << "\n";
  RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));

  detail::spmv_t<int, double> spmv(problem);
  spmv.setup(problem, true);
  // test ax
  {
    // Ax input
    rmm::device_uvector<double> x(problem.n_variables, handle_.get_stream());

    // Ax output
    rmm::device_uvector<double> cusp_ax(problem.n_constraints, handle_.get_stream());
    rmm::device_uvector<double> ax(problem.n_constraints, handle_.get_stream());

    thrust::fill(handle_.get_thrust_policy(), x.begin(), x.end(), 1);
    thrust::fill(handle_.get_thrust_policy(), cusp_ax.begin(), cusp_ax.end(), 0);
    thrust::fill(handle_.get_thrust_policy(), ax.begin(), ax.end(), 0);

    std::cerr<<"call cusp ax\n";
    auto [gld_res, gld_time] =
    cusparse_call_ax(&handle_, problem, pdhg_solver.get_cusparse_view(), x, cusp_ax);

    float ax_time;
    {
      std::cerr<<"call cust ax\n";
      cudaEvent_t beg, end;
      cudaEventCreate(&beg);
      cudaEventCreate(&end);

      //RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
      cudaEventRecord(beg, handle_.get_stream());
      for (size_t i = 0; i < bench_iter_count; ++i) {
        spmv.Ax(
          handle_.get_stream(),
          make_span(x),
          make_span(ax),
          [] __device__(int idx, double x, raft::device_span<double> output) { output[idx] = 2 * x; });
      }
      //RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
      cudaEventRecord(end, handle_.get_stream());
      cudaEventSynchronize(end);
      ax_time = 0;
      cudaEventElapsedTime(&ax_time, beg, end);
      RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    }
    std::cerr<<"call host_copy\n";
    auto ref_res = host_copy(ax);

    std::cout<<"speedup "<<gld_time/ax_time<<"\n";
    test_eq(&handle_, ref_res, gld_res, 1e-5);
  }
  {
    // Ax input
    rmm::device_uvector<double> x(problem.n_constraints, handle_.get_stream());

    // Ax output
    rmm::device_uvector<double> cusp_ax(problem.n_variables, handle_.get_stream());
    rmm::device_uvector<double> ax(problem.n_variables, handle_.get_stream());

    thrust::fill(handle_.get_thrust_policy(), x.begin(), x.end(), 1);
    thrust::fill(handle_.get_thrust_policy(), cusp_ax.begin(), cusp_ax.end(), 0);
    thrust::fill(handle_.get_thrust_policy(), ax.begin(), ax.end(), 0);

    auto [gld_res, gld_time] =
      cusparse_call_aty(&handle_, problem, pdhg_solver.get_cusparse_view(), x, cusp_ax);

    rmm::device_scalar<double> param_0(3.0, handle_.get_stream());
    rmm::device_scalar<double> param_1(1.0, handle_.get_stream());
    float aty_time;
    {
      cudaEvent_t beg, end;
      cudaEventCreate(&beg);
      cudaEventCreate(&end);

      //RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
      cudaEventRecord(beg, handle_.get_stream());
      for (size_t i = 0; i < bench_iter_count; ++i) {
        spmv.ATy(
          handle_.get_stream(),
          make_span(x),
          make_span(ax),
          [p_0 = param_0.data(), p_1 = param_1.data()] __device__(
            int idx, double x, raft::device_span<double> output) { output[idx] = (*p_0 - *p_1) * x; });
      }
      //RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
      cudaEventRecord(end, handle_.get_stream());
      cudaEventSynchronize(end);
      aty_time = 0;
      cudaEventElapsedTime(&aty_time, beg, end);
    }
    auto ref_res = host_copy(ax);

    std::cout<<"speedup "<<gld_time/aty_time<<"\n";
    test_eq(&handle_, ref_res, gld_res, 1e-5);
  }
}

TEST(mip_solve, test_lb)
{
  //std::string mps_folder_path, curr_file;
  //std::cin >> mps_folder_path;
  //std::cin >> curr_file;
  //std::cout << "\n\nrun_file " << curr_file << " ";
  //auto file = mps_folder_path + "/" + curr_file;

  std::string file;
  std::cin >> file;
  test_spmv_functor(file);
}

}  // namespace cuopt::linear_programming::test
