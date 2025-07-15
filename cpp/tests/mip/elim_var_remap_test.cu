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

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <linear_programming/pdlp.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

void setup_pdlp(rmm::cuda_stream_view stream_view)
{
  detail::set_adaptive_step_size_hyper_parameters(stream_view);
  detail::set_restart_hyper_parameters(stream_view);
  detail::set_pdlp_hyper_parameters(stream_view);
}

std::vector<int> select_k_random(int population_size, int sample_size)
{
  std::vector<int> pop(population_size - 1);
  std::iota(pop.begin(), pop.end(), 0);
  std::vector<int> random_vals;
  auto seed = std::random_device{}();
  std::cout << "Tested with seed " << seed << "\n";
  std::sample(
    pop.begin(), pop.end(), std::back_inserter(random_vals), sample_size, std::mt19937{seed});
  return random_vals;
}

void test_elim_var_remap(std::string test_instance)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;
  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  trivial_presolve(problem);

  // golden assignment vector
  rmm::device_uvector<double> full_assignment(problem.n_variables, handle_.get_stream());
  thrust::sequence(handle_.get_thrust_policy(), full_assignment.begin(), full_assignment.end());

  // Simulate bounds presolve :
  // select k random variables and fix their bounds
  // set the golden assignment of fixed variables so that we can detect if
  // the post processing of problem correctly fixes them internally
  auto fixed_vars = select_k_random(problem.n_variables - 1, 5);
  for (auto& v : fixed_vars) {
    double v_val = -v - 1;
    problem.variable_lower_bounds.set_element(v, v_val, handle_.get_stream());
    problem.variable_upper_bounds.set_element(v, v_val, handle_.get_stream());
    full_assignment.set_element(v, v_val, handle_.get_stream());
  }
  // Set free var assignments to 0
  if (op_problem.get_n_variables() < problem.n_variables) {
    thrust::fill(handle_.get_thrust_policy(),
                 full_assignment.begin() + op_problem.get_n_variables(),
                 full_assignment.begin() + problem.n_variables,
                 0);
  }

  detail::problem_t<int, double> sub_problem(problem);
  trivial_presolve(sub_problem);

  // check if number of variables is updated correctly due to trivial presolve
  EXPECT_EQ(sub_problem.n_variables, problem.n_variables - fixed_vars.size());
  detail::solution_t<int, double> sol(sub_problem);

  // Copy all unfixed variable assignments
  auto iter = thrust::remove_copy_if(handle_.get_thrust_policy(),
                                     full_assignment.begin(),
                                     full_assignment.end(),
                                     sol.assignment.begin(),
                                     [] __device__(auto a) { return a < 0; });

  sol.assignment.resize(iter - sol.assignment.begin(), handle_.get_stream());

  sub_problem.post_process_solution(sol);

  auto golden_full_assignment       = host_copy(full_assignment);
  auto fixed_sub_problem_assignment = host_copy(sol.assignment);

  EXPECT_EQ(op_problem.get_n_variables(), fixed_sub_problem_assignment.size());

  for (int i = 0; i < op_problem.get_n_variables(); ++i) {
    EXPECT_NEAR(golden_full_assignment[i], fixed_sub_problem_assignment[i], 1e-2);
  }
}

void test_elim_var_solution(std::string test_instance)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;
  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  setup_pdlp(handle_.get_stream());
  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> standardized_problem(op_problem);
  standardized_problem.preprocess_problem();
  trivial_presolve(standardized_problem);
  detail::problem_t<int, double> sub_problem(standardized_problem);

  mip_solver_settings_t<int, double> default_settings{};

  detail::solution_t<int, double> solution_1(standardized_problem);
  detail::relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit = 120.;
  lp_settings.tolerance  = default_settings.tolerances.absolute_tolerance;
  // run the problem through pdlp
  auto result_1 = detail::get_relaxed_lp_solution(standardized_problem, solution_1, lp_settings);
  solution_1.compute_feasibility();
  // the solution might not be feasible per row as we are getting the result of pdlp
  bool sol_1_feasible = (int)result_1.get_termination_status() == CUOPT_TERIMINATION_STATUS_OPTIMAL;
  EXPECT_EQ((int)result_1.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  standardized_problem.post_process_solution(solution_1);
  auto opt_sol_1 = solution_1.get_solution(sol_1_feasible, solver_stats_t<int, double>{});
  test_objective_sanity(
    mps_problem, opt_sol_1.get_solution(), opt_sol_1.get_objective_value(), 1e-3);
  test_constraint_sanity_per_row(
    mps_problem, opt_sol_1.get_solution(), 1e-3, default_settings.tolerances.relative_tolerance);

  auto fixed_vars = select_k_random(standardized_problem.n_variables - 1, 5);
  for (auto& v : fixed_vars) {
    double v_val = opt_sol_1.get_solution().element(v, handle_.get_stream());
    sub_problem.variable_lower_bounds.set_element(v, v_val, handle_.get_stream());
    sub_problem.variable_upper_bounds.set_element(v, v_val, handle_.get_stream());
  }

  handle_.sync_stream();

  trivial_presolve(sub_problem);

  detail::solution_t<int, double> solution_2(sub_problem);
  detail::relaxed_lp_settings_t lp_settings_2;
  lp_settings_2.time_limit = 120.;
  lp_settings_2.tolerance  = default_settings.tolerances.absolute_tolerance;
  // run the problem through pdlp
  auto result_2 = detail::get_relaxed_lp_solution(sub_problem, solution_2, lp_settings_2);
  solution_2.compute_feasibility();
  bool sol_2_feasible = (int)result_2.get_termination_status() == CUOPT_TERIMINATION_STATUS_OPTIMAL;
  EXPECT_EQ((int)result_2.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  sub_problem.post_process_solution(solution_2);
  auto opt_sol_2 = solution_2.get_solution(sol_2_feasible, solver_stats_t<int, double>{});
  test_objective_sanity(
    mps_problem, opt_sol_2.get_solution(), opt_sol_2.get_objective_value(), 1e-3);
  test_constraint_sanity_per_row(
    mps_problem, opt_sol_2.get_solution(), 1e-3, default_settings.tolerances.relative_tolerance);

  EXPECT_NEAR(opt_sol_1.get_objective_value(), opt_sol_2.get_objective_value(), 1e-1f);
}

TEST(mip_solve, elim_var_remap_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
  for (const auto& test_instance : test_instances) {
    test_elim_var_remap(test_instance);
  }
}

TEST(mip_solve, elim_var_remap_solution_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
  for (const auto& test_instance : test_instances) {
    test_elim_var_solution(test_instance);
  }
}

}  // namespace cuopt::linear_programming::test
