/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/problem/load_balanced_problem.cuh>
#include <mip/presolve/lb_bounds_presolve.cuh>
#include <mip/presolve/lb_problem.cuh>
#include <mps_parser/parser.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>
#include <utilities/timer.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>> select_k_random(
  detail::problem_t<int, double>& problem, int sample_size)
{
  auto seed = std::random_device{}();
  std::cout << "Tested with seed " << seed << "\n";
  problem.compute_n_integer_vars();
  auto v_bnds     = host_copy(problem.variable_bounds);
  auto int_var_id = host_copy(problem.integer_indices);
  int_var_id.erase(
    std::remove_if(int_var_id.begin(),
                   int_var_id.end(),
                   [v_bnds](auto id) {
                     auto v_bnd = v_bnds[id];
                     return !(std::isfinite(get_lower(v_bnd)) && std::isfinite(get_upper(v_bnd)));
                   }),
    int_var_id.end());
  sample_size = std::min(sample_size, static_cast<int>(int_var_id.size()));
  std::vector<int> random_int_vars;
  std::mt19937 m{seed};
  std::sample(
    int_var_id.begin(), int_var_id.end(), std::back_inserter(random_int_vars), sample_size, m);
  std::vector<double> probe_0(sample_size);
  std::vector<double> probe_1(sample_size);
  for (int i = 0; i < static_cast<int>(random_int_vars.size()); ++i) {
    if (i % 2) {
      probe_0[i] = get_lower(v_bnds[random_int_vars[i]]);
      probe_1[i] = get_upper(v_bnds[random_int_vars[i]]);
    } else {
      probe_1[i] = get_lower(v_bnds[random_int_vars[i]]);
      probe_0[i] = get_upper(v_bnds[random_int_vars[i]]);
    }
  }
  return std::make_tuple(std::move(random_int_vars), std::move(probe_0), std::move(probe_1));
}

std::pair<std::vector<thrust::pair<int, double>>, std::vector<thrust::pair<int, double>>>
convert_probe_tuple(std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>& probe)
{
  std::vector<thrust::pair<int, double>> probe_first;
  std::vector<thrust::pair<int, double>> probe_second;
  for (size_t i = 0; i < std::get<0>(probe).size(); ++i) {
    probe_first.emplace_back(thrust::make_pair(std::get<0>(probe)[i], std::get<1>(probe)[i]));
    probe_second.emplace_back(thrust::make_pair(std::get<0>(probe)[i], std::get<2>(probe)[i]));
  }
  return std::make_pair(std::move(probe_first), std::move(probe_second));
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
bounds_probe_results(detail::bound_presolve_t<int, double>& bnd_prb_0,
                     detail::bound_presolve_t<int, double>& bnd_prb_1,
                     detail::problem_t<int, double>& problem,
                     const std::pair<std::vector<thrust::pair<int, double>>,
                                     std::vector<thrust::pair<int, double>>>& probe)
{
  auto& probe_first  = std::get<0>(probe);
  auto& probe_second = std::get<1>(probe);
  rmm::device_uvector<double> b_lb_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_lb_1(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_1(problem.n_variables, problem.handle_ptr->get_stream());
  bnd_prb_0.solve(problem, probe_first);
  bnd_prb_0.set_updated_bounds(problem.handle_ptr, make_span(b_lb_0), make_span(b_ub_0));
  bnd_prb_1.solve(problem, probe_second);
  bnd_prb_1.set_updated_bounds(problem.handle_ptr, make_span(b_lb_1), make_span(b_ub_1));

  auto h_lb_0 = host_copy(b_lb_0);
  auto h_ub_0 = host_copy(b_ub_0);
  auto h_lb_1 = host_copy(b_lb_1);
  auto h_ub_1 = host_copy(b_ub_1);
  return std::make_tuple(
    std::move(h_lb_0), std::move(h_ub_0), std::move(h_lb_1), std::move(h_ub_1));
}

void bench(std::string path)
{
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  const raft::handle_t handle_{};
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  detail::problem_t<int, double> problem(op_problem);
  mip_solver_settings_t<int, double> default_settings{};
  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);
  detail::mip_solver_t<int, double> solver(problem, default_settings, scaling, cuopt::timer_t(0));
  detail::bound_presolve_t<int, double> bnd_prs(solver.context);
  detail::lb_problem_t<int, double> lb_problem(problem);
  detail::lb_bound_presolve_t<int, double> lb_prs(solver.context, lb_problem);

  detail::load_balanced_problem_t<int, double> lb_problem_old(problem);
  detail::load_balanced_bounds_presolve_t<int, double> lb_prs_old(lb_problem_old, solver.context);

  //detail::bound_presolve_t<int, double> bnd_prb(solver.context);
  const int warm_up_iter = 20;
  for (int i = 0; i < warm_up_iter; ++i) {
    bnd_prs.calculate_activity(problem);
    bnd_prs.calculate_bounds_update(problem);
    lb_prs.calculate_constraint_slack_iter(lb_problem, &handle_);
    lb_prs.calculate_bounds_update(lb_problem, &handle_);
    lb_prs_old.calculate_constraint_slack_iter(&handle_);
    lb_prs_old.update_bounds_from_slack(&handle_);
  }

  float bnd_lb_time     = 0;
  float new_lb_time     = 0;
  float old_lb_time     = 0;
  const int iter_limit = 50;
  {
    bnd_prs.copy_input_bounds(problem);
    bnd_prs.upd.init_changed_constraints(&handle_);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, handle_.get_stream());

    for (int i = 0; i < iter_limit; ++i) {
      bnd_prs.calculate_activity(problem);
      bnd_prs.calculate_bounds_update(problem);
    }

    cudaEventRecord(stop, handle_.get_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bnd_lb_time, start, stop);
  }
  {
    lb_prs.copy_input_bounds(lb_problem, &handle_);
    lb_prs.upd.init_changed_constraints(&handle_);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, handle_.get_stream());

    for (int i = 0; i < iter_limit; ++i) {
      lb_prs.calculate_constraint_slack_iter(lb_problem, &handle_);
      lb_prs.calculate_bounds_update(lb_problem, &handle_);
    }

    cudaEventRecord(stop, handle_.get_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&new_lb_time, start, stop);
  }
  {
    lb_prs_old.copy_input_bounds(lb_problem_old);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, handle_.get_stream());

    for (int i = 0; i < iter_limit; ++i) {
      lb_prs_old.calculate_constraint_slack_iter(&handle_);
      lb_prs_old.update_bounds_from_slack(&handle_);
    }

    cudaEventRecord(stop, handle_.get_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&old_lb_time, start, stop);
  }
  std::cout<<path<<" ";
  std::cout<<"n_cnst "<<problem.n_constraints<<" n_var "<<problem.n_variables<<" nnz "<<problem.nnz<<"\t";
  std::cout << "speedup over old lb " << old_lb_time / new_lb_time << "\n";
  std::cout << "speedup over bnd prs " << bnd_lb_time / new_lb_time << "\n";
}

void test_multi_probe(std::string path)
{
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  const raft::handle_t handle_{};
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  detail::problem_t<int, double> problem(op_problem);
  mip_solver_settings_t<int, double> default_settings{};
  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);
  detail::mip_solver_t<int, double> solver(problem, default_settings, scaling, cuopt::timer_t(0));
  detail::lb_problem_t<int, double> lb_problem(problem);
  detail::lb_bound_presolve_t<int, double> lb_prs(solver.context, lb_problem);

  detail::bound_presolve_t<int, double> bnd_prb(solver.context);

  auto probe_tuple       = select_k_random(problem, 100);
  auto bounds_probe_vals = convert_probe_tuple(probe_tuple);
  {
    auto& probe_first = std::get<0>(bounds_probe_vals);
    bnd_prb.solve(problem, probe_first);
    rmm::device_uvector<double> b_lb(problem.n_variables, problem.handle_ptr->get_stream());
    rmm::device_uvector<double> b_ub(problem.n_variables, problem.handle_ptr->get_stream());
    bnd_prb.set_updated_bounds(problem.handle_ptr, make_span(b_lb), make_span(b_ub));

    auto h_lb = host_copy(b_lb);
    auto h_ub = host_copy(b_ub);

    lb_prs.solve(lb_problem, probe_first);

    auto bnds = host_copy(lb_prs.upd.vars_bnd);
    for (int i = 0; i < (int)h_lb.size(); ++i) {
      EXPECT_DOUBLE_EQ(get_lower(bnds[i]), h_lb[i]);
      EXPECT_DOUBLE_EQ(get_upper(bnds[i]), h_ub[i]);
    }
  }
}

//TEST(presolve, multi_probe)
//{
//  std::vector<std::string> test_instances = {
//    //"mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
//    "mip/50v-10-free-bound.mps",
//    "mip/neos5-free-bound.mps"};
//  for (const auto& test_instance : test_instances) {
//    auto path = make_path_absolute(test_instance);
//    test_multi_probe(path);
//  }
//}

TEST(presolve, multi_probe)
{
  std::vector<std::string> test_instances = {
    "/home/aatish/rapids/mip_files/miplib/mas74.mps",
    "/home/aatish/rapids/mip_files/miplib/neos17.mps",
    "/home/aatish/rapids/mip_files/miplib/neos-3402454-bohle.mps",
    "/home/aatish/rapids/mip_files/miplib/sing44.mps",
    "/home/aatish/rapids/mip_files/miplib/square41.mps",
    "/home/aatish/rapids/mip_files/miplib/square47.mps",
    "/home/aatish/rapids/mip_files/miplib/supportcase19.mps",
    "/home/aatish/rapids/mip_files/miplib/s100.mps",
    "/home/aatish/rapids/mip_files/miplib/s250r10.mps",
    "/home/aatish/rapids/mip_files/miplib/roi5alpha10n8.mps",
    "/home/aatish/rapids/mip_files/miplib/neos-5114902-kasavu.mps",
    "/home/aatish/rapids/mip_files/miplib/neos-4647030-tutaki.mps",
    "/home/aatish/rapids/mip_files/miplib/timtab1.mps"};
  for (const auto& test_instance : test_instances) {
    std::cout << "Running: " << test_instance << std::endl;
    bench(test_instance);
  }
}

}  // namespace cuopt::linear_programming::test
