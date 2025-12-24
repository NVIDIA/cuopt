/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/data_model_view.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cstdlib>

namespace cuopt::linear_programming {

// Test fixture that manages environment variables
class RemoteSolveTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Save original env vars if they exist
    const char* host = std::getenv("CUOPT_REMOTE_HOST");
    const char* port = std::getenv("CUOPT_REMOTE_PORT");
    original_host_   = host ? host : "";
    original_port_   = port ? port : "";
    host_was_set_    = (host != nullptr);
    port_was_set_    = (port != nullptr);
  }

  void TearDown() override
  {
    // Restore original env vars
    if (host_was_set_) {
      setenv("CUOPT_REMOTE_HOST", original_host_.c_str(), 1);
    } else {
      unsetenv("CUOPT_REMOTE_HOST");
    }
    if (port_was_set_) {
      setenv("CUOPT_REMOTE_PORT", original_port_.c_str(), 1);
    } else {
      unsetenv("CUOPT_REMOTE_PORT");
    }
  }

  void enable_remote_solve()
  {
    setenv("CUOPT_REMOTE_HOST", "localhost", 1);
    setenv("CUOPT_REMOTE_PORT", "5000", 1);
  }

  void disable_remote_solve()
  {
    unsetenv("CUOPT_REMOTE_HOST");
    unsetenv("CUOPT_REMOTE_PORT");
  }

 private:
  std::string original_host_;
  std::string original_port_;
  bool host_was_set_;
  bool port_was_set_;
};

// White-box test: GPU data + remote solve enabled
// This tests the edge case where data is on GPU but user wants remote solve
TEST_F(RemoteSolveTest, gpu_data_with_remote_solve_enabled)
{
  const raft::handle_t handle_{};

  // Create a simple 2x2 LP problem directly on GPU
  // minimize: 0.2*x1 + 0.1*x2
  // subject to: 3*x1 + 4*x2 <= 5.4
  //             2.7*x1 + 10.1*x2 <= 4.9
  //             x1, x2 >= 0

  // Allocate GPU memory for problem data
  rmm::device_uvector<double> A_values(4, handle_.get_stream());
  rmm::device_uvector<int> A_indices(4, handle_.get_stream());
  rmm::device_uvector<int> A_offsets(3, handle_.get_stream());
  rmm::device_uvector<double> constraint_bounds(2, handle_.get_stream());
  rmm::device_uvector<double> objective_coeffs(2, handle_.get_stream());
  rmm::device_uvector<double> var_lower(2, handle_.get_stream());
  rmm::device_uvector<double> var_upper(2, handle_.get_stream());

  // Host data
  std::vector<double> h_A_values  = {3.0, 4.0, 2.7, 10.1};
  std::vector<int> h_A_indices    = {0, 1, 0, 1};
  std::vector<int> h_A_offsets    = {0, 2, 4};
  std::vector<double> h_bounds    = {5.4, 4.9};
  std::vector<double> h_obj       = {0.2, 0.1};
  std::vector<double> h_var_lower = {0.0, 0.0};
  std::vector<double> h_var_upper = {1e20, 1e20};

  // Copy to GPU
  raft::copy(A_values.data(), h_A_values.data(), 4, handle_.get_stream());
  raft::copy(A_indices.data(), h_A_indices.data(), 4, handle_.get_stream());
  raft::copy(A_offsets.data(), h_A_offsets.data(), 3, handle_.get_stream());
  raft::copy(constraint_bounds.data(), h_bounds.data(), 2, handle_.get_stream());
  raft::copy(objective_coeffs.data(), h_obj.data(), 2, handle_.get_stream());
  raft::copy(var_lower.data(), h_var_lower.data(), 2, handle_.get_stream());
  raft::copy(var_upper.data(), h_var_upper.data(), 2, handle_.get_stream());
  handle_.sync_stream();

  // Create a data_model_view_t pointing to GPU data
  data_model_view_t<int, double> view;
  view.set_maximize(false);
  view.set_csr_constraint_matrix(A_values.data(), 4, A_indices.data(), 4, A_offsets.data(), 3);
  view.set_constraint_bounds(constraint_bounds.data(), 2);
  view.set_objective_coefficients(objective_coeffs.data(), 2);
  view.set_variable_lower_bounds(var_lower.data(), 2);
  view.set_variable_upper_bounds(var_upper.data(), 2);
  view.set_is_device_memory(true);  // Mark as GPU data

  // Enable remote solve
  enable_remote_solve();

  // Verify remote solve is enabled
  ASSERT_TRUE(is_remote_solve_enabled());

  // Call solve_lp with GPU view + remote solve enabled
  // This should trigger the GPU->CPU copy path and return the "not implemented" error
  pdlp_solver_settings_t<int, double> settings;
  auto solution = solve_lp(&handle_, view, settings);

  // Since remote solve is not yet implemented, we expect an error status
  // The key thing is that we didn't crash and the GPU->CPU copy worked
  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::NumericalError);
}

// Control test: GPU data + local solve (no remote)
TEST_F(RemoteSolveTest, gpu_data_with_local_solve)
{
  const raft::handle_t handle_{};

  // Same problem setup as above
  rmm::device_uvector<double> A_values(4, handle_.get_stream());
  rmm::device_uvector<int> A_indices(4, handle_.get_stream());
  rmm::device_uvector<int> A_offsets(3, handle_.get_stream());
  rmm::device_uvector<double> constraint_lower(2, handle_.get_stream());
  rmm::device_uvector<double> constraint_upper(2, handle_.get_stream());
  rmm::device_uvector<double> objective_coeffs(2, handle_.get_stream());
  rmm::device_uvector<double> var_lower(2, handle_.get_stream());
  rmm::device_uvector<double> var_upper(2, handle_.get_stream());

  std::vector<double> h_A_values         = {3.0, 4.0, 2.7, 10.1};
  std::vector<int> h_A_indices           = {0, 1, 0, 1};
  std::vector<int> h_A_offsets           = {0, 2, 4};
  std::vector<double> h_constraint_lower = {-1e20, -1e20};  // -inf (no lower bound)
  std::vector<double> h_constraint_upper = {5.4, 4.9};      // <= constraints
  std::vector<double> h_obj              = {0.2, 0.1};
  std::vector<double> h_var_lower        = {0.0, 0.0};
  std::vector<double> h_var_upper        = {1e20, 1e20};

  raft::copy(A_values.data(), h_A_values.data(), 4, handle_.get_stream());
  raft::copy(A_indices.data(), h_A_indices.data(), 4, handle_.get_stream());
  raft::copy(A_offsets.data(), h_A_offsets.data(), 3, handle_.get_stream());
  raft::copy(constraint_lower.data(), h_constraint_lower.data(), 2, handle_.get_stream());
  raft::copy(constraint_upper.data(), h_constraint_upper.data(), 2, handle_.get_stream());
  raft::copy(objective_coeffs.data(), h_obj.data(), 2, handle_.get_stream());
  raft::copy(var_lower.data(), h_var_lower.data(), 2, handle_.get_stream());
  raft::copy(var_upper.data(), h_var_upper.data(), 2, handle_.get_stream());
  handle_.sync_stream();

  data_model_view_t<int, double> view;
  view.set_maximize(false);
  view.set_csr_constraint_matrix(A_values.data(), 4, A_indices.data(), 4, A_offsets.data(), 3);
  view.set_constraint_lower_bounds(constraint_lower.data(), 2);
  view.set_constraint_upper_bounds(constraint_upper.data(), 2);
  view.set_objective_coefficients(objective_coeffs.data(), 2);
  view.set_variable_lower_bounds(var_lower.data(), 2);
  view.set_variable_upper_bounds(var_upper.data(), 2);
  view.set_is_device_memory(true);

  // Disable remote solve
  disable_remote_solve();

  // Verify remote solve is disabled
  ASSERT_FALSE(is_remote_solve_enabled());

  // Call solve_lp - should solve locally
  pdlp_solver_settings_t<int, double> settings;
  auto solution = solve_lp(&handle_, view, settings);

  // Should succeed with optimal status
  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
}

// Test: CPU data + remote solve enabled
TEST_F(RemoteSolveTest, cpu_data_with_remote_solve_enabled)
{
  const raft::handle_t handle_{};

  // Host data (CPU)
  std::vector<double> h_A_values  = {3.0, 4.0, 2.7, 10.1};
  std::vector<int> h_A_indices    = {0, 1, 0, 1};
  std::vector<int> h_A_offsets    = {0, 2, 4};
  std::vector<double> h_bounds    = {5.4, 4.9};
  std::vector<double> h_obj       = {0.2, 0.1};
  std::vector<double> h_var_lower = {0.0, 0.0};
  std::vector<double> h_var_upper = {1e20, 1e20};

  // Create view pointing to CPU data
  data_model_view_t<int, double> view;
  view.set_maximize(false);
  view.set_csr_constraint_matrix(
    h_A_values.data(), 4, h_A_indices.data(), 4, h_A_offsets.data(), 3);
  view.set_constraint_bounds(h_bounds.data(), 2);
  view.set_objective_coefficients(h_obj.data(), 2);
  view.set_variable_lower_bounds(h_var_lower.data(), 2);
  view.set_variable_upper_bounds(h_var_upper.data(), 2);
  view.set_is_device_memory(false);  // CPU data

  // Enable remote solve
  enable_remote_solve();
  ASSERT_TRUE(is_remote_solve_enabled());

  // Should go to remote path (and return not implemented error)
  pdlp_solver_settings_t<int, double> settings;
  auto solution = solve_lp(&handle_, view, settings);

  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::NumericalError);
}

// Test: CPU data + local solve
TEST_F(RemoteSolveTest, cpu_data_with_local_solve)
{
  const raft::handle_t handle_{};

  std::vector<double> h_A_values         = {3.0, 4.0, 2.7, 10.1};
  std::vector<int> h_A_indices           = {0, 1, 0, 1};
  std::vector<int> h_A_offsets           = {0, 2, 4};
  std::vector<double> h_constraint_lower = {-1e20, -1e20};  // -inf (no lower bound)
  std::vector<double> h_constraint_upper = {5.4, 4.9};      // <= constraints
  std::vector<double> h_obj              = {0.2, 0.1};
  std::vector<double> h_var_lower        = {0.0, 0.0};
  std::vector<double> h_var_upper        = {1e20, 1e20};

  data_model_view_t<int, double> view;
  view.set_maximize(false);
  view.set_csr_constraint_matrix(
    h_A_values.data(), 4, h_A_indices.data(), 4, h_A_offsets.data(), 3);
  view.set_constraint_lower_bounds(h_constraint_lower.data(), 2);
  view.set_constraint_upper_bounds(h_constraint_upper.data(), 2);
  view.set_objective_coefficients(h_obj.data(), 2);
  view.set_variable_lower_bounds(h_var_lower.data(), 2);
  view.set_variable_upper_bounds(h_var_upper.data(), 2);
  view.set_is_device_memory(false);

  disable_remote_solve();
  ASSERT_FALSE(is_remote_solve_enabled());

  // Should copy to GPU and solve locally
  pdlp_solver_settings_t<int, double> settings;
  auto solution = solve_lp(&handle_, view, settings);

  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
}

}  // namespace cuopt::linear_programming
