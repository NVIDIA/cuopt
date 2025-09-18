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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>
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

#include <cstdint>
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

void test_bounds_standardization_test(std::string test_instance)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;
  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  setup_pdlp(handle_.get_stream());
  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> standardized_problem(op_problem);
  detail::problem_t<int, double> original_problem(op_problem);
  standardized_problem.preprocess_problem();
  detail::trivial_presolve(standardized_problem);
  detail::solution_t<int, double> solution_1(standardized_problem);

  mip_solver_settings_t<int, double> default_settings{};
  detail::relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit              = 120.;
  lp_settings.tolerance               = default_settings.tolerances.absolute_tolerance;
  lp_settings.per_constraint_residual = false;

  // run the problem through pdlp
  auto result_1 = detail::get_relaxed_lp_solution(standardized_problem, solution_1, lp_settings);
  solution_1.compute_feasibility();
  bool sol_1_feasible = (int)result_1.get_termination_status() == CUOPT_TERIMINATION_STATUS_OPTIMAL;
  // the problem might not be feasible in terms of per constraint residual
  // only consider the pdlp results
  EXPECT_TRUE(sol_1_feasible);
  standardized_problem.post_process_solution(solution_1);
  solution_1.problem_ptr = &original_problem;
  auto optimization_prob_solution =
    solution_1.get_solution(sol_1_feasible, solver_stats_t<int, double>{});
  test_objective_sanity(problem,
                        optimization_prob_solution.get_solution(),
                        optimization_prob_solution.get_objective_value());
  test_constraint_sanity_per_row(problem,
                                 optimization_prob_solution.get_solution(),
                                 1e-3,
                                 default_settings.tolerances.relative_tolerance);

  // now do relax the problem before passing it to the problem, so that bounds standardization is
  // not applied
  op_problem.set_problem_category(problem_category_t::LP);
  auto settings             = pdlp_solver_settings_t<int, double>{};
  settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable1;
  settings.set_optimality_tolerance(1e-4);
  settings.tolerances.relative_primal_tolerance = 1e-6;
  settings.tolerances.relative_dual_tolerance   = 1e-6;
  auto result_2                                 = solve_lp(op_problem, settings);
  EXPECT_NEAR(result_1.get_objective_value(), result_2.get_objective_value(), 1e-2f);
}

TEST(mip_solve, bounds_standardization_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
  for (const auto& test_instance : test_instances) {
    test_bounds_standardization_test(test_instance);
  }
}

}  // namespace cuopt::linear_programming::test
