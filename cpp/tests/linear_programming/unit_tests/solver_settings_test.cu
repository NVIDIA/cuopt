/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {

TEST(SolverSettingsTest, TestSetGet)
{
  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> solver_settings =
    cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double>{};

  const double tolerance_value = 1e-5;

  // Setting tolerances
  solver_settings.tolerances.absolute_dual_tolerance     = tolerance_value;
  solver_settings.tolerances.relative_dual_tolerance     = tolerance_value;
  solver_settings.tolerances.absolute_primal_tolerance   = tolerance_value;
  solver_settings.tolerances.relative_primal_tolerance   = tolerance_value;
  solver_settings.tolerances.absolute_gap_tolerance      = tolerance_value;
  solver_settings.tolerances.relative_gap_tolerance      = tolerance_value;
  solver_settings.tolerances.primal_infeasible_tolerance = tolerance_value;
  solver_settings.tolerances.dual_infeasible_tolerance   = tolerance_value;

  EXPECT_FALSE(solver_settings.per_constraint_residual);
  solver_settings.per_constraint_residual = true;

  EXPECT_NEAR(solver_settings.tolerances.absolute_dual_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.relative_dual_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.absolute_primal_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.relative_primal_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.absolute_gap_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.relative_gap_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.primal_infeasible_tolerance, 1e-5, 1e-10);
  EXPECT_NEAR(solver_settings.tolerances.dual_infeasible_tolerance, 1e-5, 1e-10);

  solver_settings.detect_infeasibility = true;
  EXPECT_TRUE(solver_settings.detect_infeasibility);

  // To avoid the "," inside the macros which are interpreted as extra parameters
  auto Stable3 = cuopt::mathematical_optimization::pdlp_solver_mode_t::Stable3;
  auto Fast1   = cuopt::mathematical_optimization::pdlp_solver_mode_t::Fast1;
  EXPECT_EQ(solver_settings.pdlp_solver_mode, Stable3);
  solver_settings.pdlp_solver_mode = Fast1;
  EXPECT_EQ(solver_settings.pdlp_solver_mode, Fast1);

  EXPECT_TRUE(solver_settings.per_constraint_residual);

  EXPECT_FALSE(solver_settings.save_best_primal_so_far);
  solver_settings.save_best_primal_so_far = true;
  EXPECT_TRUE(solver_settings.save_best_primal_so_far);

  EXPECT_FALSE(solver_settings.first_primal_feasible);
  solver_settings.first_primal_feasible = true;
  EXPECT_TRUE(solver_settings.first_primal_feasible);

  EXPECT_EQ(solver_settings.postsolve_info, -1);
  solver_settings.postsolve_info = 1;
  EXPECT_EQ(solver_settings.postsolve_info, 1);

  EXPECT_EQ(solver_settings.barrier_presolve_bound_free_variables, -1);
  solver_settings.barrier_presolve_bound_free_variables = 0;
  EXPECT_EQ(solver_settings.barrier_presolve_bound_free_variables, 0);
  solver_settings.barrier_presolve_bound_free_variables = 1;
  EXPECT_EQ(solver_settings.barrier_presolve_bound_free_variables, 1);
}

TEST(SolverSettingsTest, warm_start_smaller_vector)
{
  const raft::handle_t handle_{};

  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> solver_settings =
    cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double>{};

  std::vector<double> primal      = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> dual        = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> primal_mapping = {1, 0};     // Only two variables and 0 - 1 swapped
  std::vector<int> dual_mapping   = {0, 2, 1};  // Only three constraints and  1 - 2 swapped

  std::vector<double> primal_expected = {1.0, 0.0};
  std::vector<double> dual_expected   = {0.0, 2.0, 1.0};

  rmm::device_uvector<double> current_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> initial_primal_average =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> current_ATY = cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> sum_primal_solutions =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());

  rmm::device_uvector<double> current_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> initial_dual_average = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> sum_dual_solutions   = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());

  rmm::device_uvector<int> d_primal_mapping =
    cuopt::device_copy(primal_mapping, handle_.get_stream());
  rmm::device_uvector<int> d_dual_mapping = cuopt::device_copy(dual_mapping, handle_.get_stream());

  pdlp_warm_start_data_t<int, double> warm_start_data =
    pdlp_warm_start_data_t<int, double>(current_primal_solution,
                                        current_dual_solution,
                                        initial_primal_average,
                                        initial_dual_average,
                                        current_ATY,
                                        sum_primal_solutions,
                                        sum_dual_solutions,
                                        last_restart_duality_gap_primal_solution,
                                        last_restart_duality_gap_dual_solution,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1);
  solver_settings.set_pdlp_warm_start_data(warm_start_data, d_primal_mapping, d_dual_mapping);

  auto stream = handle_.get_stream();
  std::vector<double> h_current_primal_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_primal_solution_, stream);
  std::vector<double> h_initial_primal_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_primal_average_, stream);
  std::vector<double> h_current_ATY =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_ATY_, stream);
  std::vector<double> h_sum_primal_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_primal_solutions_, stream);
  std::vector<double> h_last_restart_duality_gap_primal_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_, stream);

  EXPECT_EQ(h_current_primal_solution.size(), primal_expected.size());
  EXPECT_EQ(h_initial_primal_average.size(), primal_expected.size());
  EXPECT_EQ(h_current_ATY.size(), primal_expected.size());
  EXPECT_EQ(h_sum_primal_solutions.size(), primal_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution.size(), primal_expected.size());

  EXPECT_EQ(h_current_primal_solution, primal_expected);
  EXPECT_EQ(h_initial_primal_average, primal_expected);
  EXPECT_EQ(h_current_ATY, primal_expected);
  EXPECT_EQ(h_sum_primal_solutions, primal_expected);
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution, primal_expected);

  std::vector<double> h_current_dual_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_dual_solution_, stream);
  std::vector<double> h_initial_dual_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_dual_average_, stream);
  std::vector<double> h_sum_dual_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_dual_solutions_, stream);
  std::vector<double> h_last_restart_duality_gap_dual_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_, stream);

  EXPECT_EQ(h_current_dual_solution.size(), dual_expected.size());
  EXPECT_EQ(h_initial_dual_average.size(), dual_expected.size());
  EXPECT_EQ(h_sum_dual_solutions.size(), dual_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution.size(), dual_expected.size());

  EXPECT_EQ(h_current_dual_solution, dual_expected);
  EXPECT_EQ(h_initial_dual_average, dual_expected);
  EXPECT_EQ(h_sum_dual_solutions, dual_expected);
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution, dual_expected);
}

TEST(SolverSettingsTest, warm_start_bigger_vector)
{
  const raft::handle_t handle_{};

  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> solver_settings =
    cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double>{};

  std::vector<double> primal      = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> dual        = {0.0, 1.0, 2.0};
  std::vector<int> primal_mapping = {0, 1, 2, 3, 4, 5};  // Only two variables and 0 - 1 swapped
  std::vector<int> dual_mapping   = {
    0, 1, 2, 3, 4, 5, 6};  // Only three constraints and  1 - 2 swapped

  std::vector<double> primal_expected = {0.0, 1.0, 2.0, 3.0, 0.0, 0.0};
  std::vector<double> dual_expected   = {0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0};

  rmm::device_uvector<double> current_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> initial_primal_average =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> current_ATY = cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> sum_primal_solutions =
    cuopt::device_copy(primal, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_primal_solution =
    cuopt::device_copy(primal, handle_.get_stream());

  rmm::device_uvector<double> current_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> initial_dual_average = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> sum_dual_solutions   = cuopt::device_copy(dual, handle_.get_stream());
  rmm::device_uvector<double> last_restart_duality_gap_dual_solution =
    cuopt::device_copy(dual, handle_.get_stream());

  rmm::device_uvector<int> d_primal_mapping =
    cuopt::device_copy(primal_mapping, handle_.get_stream());
  rmm::device_uvector<int> d_dual_mapping = cuopt::device_copy(dual_mapping, handle_.get_stream());

  pdlp_warm_start_data_t<int, double> warm_start_data =
    pdlp_warm_start_data_t<int, double>(current_primal_solution,
                                        current_dual_solution,
                                        initial_primal_average,
                                        initial_dual_average,
                                        current_ATY,
                                        sum_primal_solutions,
                                        sum_dual_solutions,
                                        last_restart_duality_gap_primal_solution,
                                        last_restart_duality_gap_dual_solution,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1);
  solver_settings.set_pdlp_warm_start_data(warm_start_data, d_primal_mapping, d_dual_mapping);

  auto stream = handle_.get_stream();
  std::vector<double> h_current_primal_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_primal_solution_, stream);
  std::vector<double> h_initial_primal_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_primal_average_, stream);
  std::vector<double> h_current_ATY =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_ATY_, stream);
  std::vector<double> h_sum_primal_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_primal_solutions_, stream);
  std::vector<double> h_last_restart_duality_gap_primal_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_, stream);

  EXPECT_EQ(h_current_primal_solution.size(), primal_expected.size());
  EXPECT_EQ(h_initial_primal_average.size(), primal_expected.size());
  EXPECT_EQ(h_current_ATY.size(), primal_expected.size());
  EXPECT_EQ(h_sum_primal_solutions.size(), primal_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution.size(), primal_expected.size());

  EXPECT_EQ(h_current_primal_solution, primal_expected);
  EXPECT_EQ(h_initial_primal_average, primal_expected);
  EXPECT_EQ(h_current_ATY, primal_expected);
  EXPECT_EQ(h_sum_primal_solutions, primal_expected);
  EXPECT_EQ(h_last_restart_duality_gap_primal_solution, primal_expected);

  std::vector<double> h_current_dual_solution =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().current_dual_solution_, stream);
  std::vector<double> h_initial_dual_average =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().initial_dual_average_, stream);
  std::vector<double> h_sum_dual_solutions =
    cuopt::host_copy(solver_settings.get_pdlp_warm_start_data().sum_dual_solutions_, stream);
  std::vector<double> h_last_restart_duality_gap_dual_solution = cuopt::host_copy(
    solver_settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_, stream);

  EXPECT_EQ(h_current_dual_solution.size(), dual_expected.size());
  EXPECT_EQ(h_initial_dual_average.size(), dual_expected.size());
  EXPECT_EQ(h_sum_dual_solutions.size(), dual_expected.size());
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution.size(), dual_expected.size());

  EXPECT_EQ(h_current_dual_solution, dual_expected);
  EXPECT_EQ(h_initial_dual_average, dual_expected);
  EXPECT_EQ(h_sum_dual_solutions, dual_expected);
  EXPECT_EQ(h_last_restart_duality_gap_dual_solution, dual_expected);
}

// =============================================================================
// solver_settings_t<i_t, f_t> (the CUDA-free wrapper split across
// math_optimization/solver_settings.cpp and solver_settings_gpu.cu)
// =============================================================================
//
// These exercise every member that solver_settings_gpu.cu explicitly instantiates.
// A member with a missing explicit instantiation compiles and links this test binary
// fine (cuopt_static resolves it internally), but disappears from libcuopt.so's
// exported symbols -- the failure mode described in the PR that introduced this split.
// See ci/checks or `nm -D --defined-only libcuopt.so` for the linkage-level check.

TEST(SolverSettingsWrapperTest, InitialPdlpPrimalAndDualSolution)
{
  const raft::handle_t handle_{};
  auto stream = handle_.get_stream();

  cuopt::mathematical_optimization::solver_settings_t<int, double> settings{};

  std::vector<double> primal = {1.0, 2.0, 3.0};
  std::vector<double> dual   = {4.0, 5.0};

  rmm::device_uvector<double> d_primal = cuopt::device_copy(primal, stream);
  rmm::device_uvector<double> d_dual   = cuopt::device_copy(dual, stream);

  settings.set_initial_pdlp_primal_solution(
    d_primal.data(), static_cast<int>(primal.size()), stream);
  settings.set_initial_pdlp_dual_solution(d_dual.data(), static_cast<int>(dual.size()), stream);

  EXPECT_EQ(cuopt::host_copy(settings.get_initial_pdlp_primal_solution(), stream), primal);
  EXPECT_EQ(cuopt::host_copy(settings.get_initial_pdlp_dual_solution(), stream), dual);
}

TEST(SolverSettingsWrapperTest, AddInitialMipSolution)
{
  const raft::handle_t handle_{};
  auto stream = handle_.get_stream();

  cuopt::mathematical_optimization::solver_settings_t<int, double> settings{};

  std::vector<double> initial_solution   = {1.0, 0.0, 1.0};
  rmm::device_uvector<double> d_solution = cuopt::device_copy(initial_solution, stream);

  settings.add_initial_mip_solution(
    d_solution.data(), static_cast<int>(initial_solution.size()), stream);

  ASSERT_EQ(settings.get_mip_settings().initial_solutions.size(), 1u);
  EXPECT_EQ(cuopt::host_copy(*settings.get_mip_settings().initial_solutions[0], stream),
            initial_solution);
}

TEST(SolverSettingsWrapperTest, SetPdlpWarmStartDataRawPointers)
{
  cuopt::mathematical_optimization::solver_settings_t<int, double> settings{};

  std::vector<double> current_primal_solution             = {0.1, 0.2, 0.3};
  std::vector<double> current_dual_solution               = {0.4, 0.5};
  std::vector<double> initial_primal_average              = {0.6, 0.7, 0.8};
  std::vector<double> initial_dual_average                = {0.9, 1.0};
  std::vector<double> current_ATY                         = {1.1, 1.2, 1.3};
  std::vector<double> sum_primal_solutions                = {1.4, 1.5, 1.6};
  std::vector<double> sum_dual_solutions                  = {1.7, 1.8};
  std::vector<double> last_restart_duality_gap_primal_sol = {1.9, 2.0, 2.1};
  std::vector<double> last_restart_duality_gap_dual_sol   = {2.2, 2.3};

  settings.set_pdlp_warm_start_data(
    current_primal_solution.data(),
    current_dual_solution.data(),
    initial_primal_average.data(),
    initial_dual_average.data(),
    current_ATY.data(),
    sum_primal_solutions.data(),
    sum_dual_solutions.data(),
    last_restart_duality_gap_primal_sol.data(),
    last_restart_duality_gap_dual_sol.data(),
    /*primal_size=*/static_cast<int>(current_primal_solution.size()),
    /*dual_size=*/static_cast<int>(current_dual_solution.size()),
    /*initial_primal_weight=*/1.5,
    /*initial_step_size=*/0.01,
    /*total_pdlp_iterations=*/10,
    /*total_pdhg_iterations=*/20,
    /*last_candidate_kkt_score=*/1e-3,
    /*last_restart_kkt_score=*/1e-4,
    /*sum_solution_weight=*/5.0,
    /*iterations_since_last_restart=*/7);

  const auto& view = settings.get_pdlp_warm_start_data_view();

  auto as_vector = [](auto span) { return std::vector<double>(span.begin(), span.end()); };
  EXPECT_EQ(as_vector(view.current_primal_solution_), current_primal_solution);
  EXPECT_EQ(as_vector(view.current_dual_solution_), current_dual_solution);
  EXPECT_EQ(as_vector(view.initial_primal_average_), initial_primal_average);
  EXPECT_EQ(as_vector(view.initial_dual_average_), initial_dual_average);
  EXPECT_EQ(as_vector(view.current_ATY_), current_ATY);
  EXPECT_EQ(as_vector(view.sum_primal_solutions_), sum_primal_solutions);
  EXPECT_EQ(as_vector(view.sum_dual_solutions_), sum_dual_solutions);
  EXPECT_EQ(as_vector(view.last_restart_duality_gap_primal_solution_),
            last_restart_duality_gap_primal_sol);
  EXPECT_EQ(as_vector(view.last_restart_duality_gap_dual_solution_),
            last_restart_duality_gap_dual_sol);

  EXPECT_DOUBLE_EQ(view.initial_primal_weight_, 1.5);
  EXPECT_DOUBLE_EQ(view.initial_step_size_, 0.01);
  EXPECT_EQ(view.total_pdlp_iterations_, 10);
  EXPECT_EQ(view.total_pdhg_iterations_, 20);
  EXPECT_DOUBLE_EQ(view.last_candidate_kkt_score_, 1e-3);
  EXPECT_DOUBLE_EQ(view.last_restart_kkt_score_, 1e-4);
  EXPECT_DOUBLE_EQ(view.sum_solution_weight_, 5.0);
  EXPECT_EQ(view.iterations_since_last_restart_, 7);
}

TEST(SolverSettingsWrapperTest, MipCallbackRegistrationAndTolerances)
{
  cuopt::mathematical_optimization::solver_settings_t<int, double> settings{};

  EXPECT_TRUE(settings.get_mip_callbacks().empty());

  internals::get_solution_callback_t* null_callback = nullptr;
  settings.set_mip_callback(null_callback, nullptr);
  EXPECT_TRUE(settings.get_mip_callbacks().empty()) << "A null callback must not be registered";

  // get_tolerances() default-constructs a tolerances_t; verify it round-trips through
  // the wrapper -> mip_solver_settings_t split introduced by the host/device separation.
  auto tolerances = settings.get_mip_settings().get_tolerances();
  EXPECT_DOUBLE_EQ(tolerances.absolute_tolerance,
                   mip_solver_settings_t<int, double>::tolerances_t{}.absolute_tolerance);
}

}  // namespace cuopt::mathematical_optimization
