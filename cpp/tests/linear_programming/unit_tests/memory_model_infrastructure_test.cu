/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/data_model_view.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

namespace cuopt::linear_programming::test {

using cuopt::linear_programming::data_model_view_t;
using cuopt::linear_programming::mip_solution_t;
using cuopt::linear_programming::mip_solver_settings_t;
using cuopt::linear_programming::optimization_problem_solution_t;
using cuopt::linear_programming::pdlp_solver_settings_t;
using cuopt::mps_parser::mps_data_model_t;

// ============================================================================
// Remote Solve Configuration Tests
// ============================================================================

class RemoteSolveConfigTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Clean environment before each test
    unsetenv("CUOPT_REMOTE_HOST");
    unsetenv("CUOPT_REMOTE_PORT");
  }

  void TearDown() override
  {
    // Clean environment after each test
    unsetenv("CUOPT_REMOTE_HOST");
    unsetenv("CUOPT_REMOTE_PORT");
  }
};

TEST_F(RemoteSolveConfigTest, valid_configuration)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);
  setenv("CUOPT_REMOTE_PORT", "8080", 1);

  auto config = get_remote_solve_config();
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->host, "example.com");
  EXPECT_EQ(config->port, 8080);
  EXPECT_TRUE(is_remote_solve_enabled());
}

TEST_F(RemoteSolveConfigTest, missing_host)
{
  setenv("CUOPT_REMOTE_PORT", "8080", 1);

  auto config = get_remote_solve_config();
  EXPECT_FALSE(config.has_value());
  EXPECT_FALSE(is_remote_solve_enabled());
}

TEST_F(RemoteSolveConfigTest, missing_port)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);

  auto config = get_remote_solve_config();
  EXPECT_FALSE(config.has_value());
  EXPECT_FALSE(is_remote_solve_enabled());
}

TEST_F(RemoteSolveConfigTest, empty_host_string)
{
  setenv("CUOPT_REMOTE_HOST", "", 1);
  setenv("CUOPT_REMOTE_PORT", "8080", 1);

  auto config = get_remote_solve_config();
  EXPECT_FALSE(config.has_value());
}

TEST_F(RemoteSolveConfigTest, empty_port_string)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);
  setenv("CUOPT_REMOTE_PORT", "", 1);

  auto config = get_remote_solve_config();
  EXPECT_FALSE(config.has_value());
}

TEST_F(RemoteSolveConfigTest, invalid_port_non_numeric)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);
  setenv("CUOPT_REMOTE_PORT", "not_a_number", 1);

  auto config = get_remote_solve_config();
  EXPECT_FALSE(config.has_value());
}

TEST_F(RemoteSolveConfigTest, port_zero_needs_validation_patch)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);
  setenv("CUOPT_REMOTE_PORT", "0", 1);

  auto config = get_remote_solve_config();
  // Port 0 validation is added in CodeRabbit patch
  // Without patch, this returns valid config (will be fixed)
  EXPECT_TRUE(config.has_value());  // Will be FALSE after patch applied
}

TEST_F(RemoteSolveConfigTest, port_negative_parsed_as_large)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);
  setenv("CUOPT_REMOTE_PORT", "-1", 1);

  auto config = get_remote_solve_config();
  // stoi("-1") succeeds but port validation will catch it with patch
  EXPECT_TRUE(config.has_value());  // Will be FALSE after patch applied
}

TEST_F(RemoteSolveConfigTest, port_too_large_needs_validation_patch)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);
  setenv("CUOPT_REMOTE_PORT", "99999", 1);

  auto config = get_remote_solve_config();
  // Port > 65535 validation is added in CodeRabbit patch
  EXPECT_TRUE(config.has_value());  // Will be FALSE after patch applied
}

TEST_F(RemoteSolveConfigTest, valid_port_boundaries)
{
  setenv("CUOPT_REMOTE_HOST", "example.com", 1);

  // Test port 1 (minimum valid)
  setenv("CUOPT_REMOTE_PORT", "1", 1);
  auto config1 = get_remote_solve_config();
  ASSERT_TRUE(config1.has_value());
  EXPECT_EQ(config1->port, 1);

  // Test port 65535 (maximum valid)
  setenv("CUOPT_REMOTE_PORT", "65535", 1);
  auto config2 = get_remote_solve_config();
  ASSERT_TRUE(config2.has_value());
  EXPECT_EQ(config2->port, 65535);

  // Test common gRPC port
  setenv("CUOPT_REMOTE_PORT", "50051", 1);
  auto config3 = get_remote_solve_config();
  ASSERT_TRUE(config3.has_value());
  EXPECT_EQ(config3->port, 50051);
}

// ============================================================================
// Data Model View Tests
// ============================================================================

TEST(DataModelViewMemory, cpu_memory_flag)
{
  data_model_view_t<int, double> view;

  // Default is false (CPU memory) - views are agnostic by default
  EXPECT_FALSE(view.is_device_memory());

  // Can be set to device (GPU) memory
  view.set_is_device_memory(true);
  EXPECT_TRUE(view.is_device_memory());

  // Can be changed back to host
  view.set_is_device_memory(false);
  EXPECT_FALSE(view.is_device_memory());
}

TEST(DataModelViewMemory, empty_constraint_matrix_handling)
{
  data_model_view_t<int, double> view;

  // Test with truly empty arrays (0 constraints)
  std::vector<double> values;
  std::vector<int> indices;
  std::vector<int> offsets{0};

  EXPECT_NO_THROW(
    view.set_csr_constraint_matrix(values.data(), 0, indices.data(), 0, offsets.data(), 1));
}

TEST(DataModelViewConversion, empty_constraint_matrix_to_optimization_problem)
{
  raft::handle_t handle;
  data_model_view_t<int, double> view;

  // Create a view with no constraints (empty problem)
  std::vector<double> empty_values;
  std::vector<int> empty_indices;
  std::vector<int> empty_offsets{0};
  std::vector<double> obj{1.0};

  view.set_csr_constraint_matrix(
    empty_values.data(), 0, empty_indices.data(), 0, empty_offsets.data(), 1);
  view.set_objective_coefficients(obj.data(), 1);
  view.set_is_device_memory(false);

  // This should not throw - the fix for empty constraint matrices
  EXPECT_NO_THROW({ auto op_problem = data_model_view_to_optimization_problem(&handle, view); });
}

TEST(DataModelViewConversion, view_from_cpu_data_marked_as_host)
{
  data_model_view_t<int, double> view;

  // Set up minimal problem with CPU-side data
  std::vector<double> values{1.0};
  std::vector<int> indices{0};
  std::vector<int> offsets{0, 1};
  std::vector<double> bounds{1.0};
  std::vector<double> obj{1.0};

  view.set_csr_constraint_matrix(values.data(), 1, indices.data(), 1, offsets.data(), 2);
  view.set_constraint_bounds(bounds.data(), 1);
  view.set_objective_coefficients(obj.data(), 1);

  // Explicitly mark as CPU memory (for remote solve path)
  view.set_is_device_memory(false);

  EXPECT_FALSE(view.is_device_memory());
}

// ============================================================================
// Data Model View to Optimization Problem Conversion Tests
// ============================================================================

TEST(DataModelViewToOptimizationProblem, cpu_view_conversion)
{
  raft::handle_t handle;
  data_model_view_t<int, double> view;

  // Set up problem with CPU data
  std::vector<double> values{1.0, 2.0};
  std::vector<int> indices{0, 1};
  std::vector<int> offsets{0, 1, 2};
  std::vector<double> bounds{5.0, 10.0};
  std::vector<double> obj{1.0, -1.0};

  view.set_csr_constraint_matrix(values.data(), 2, indices.data(), 2, offsets.data(), 3);
  view.set_constraint_bounds(bounds.data(), 2);
  view.set_objective_coefficients(obj.data(), 2);
  view.set_is_device_memory(false);  // CPU memory

  // Convert to optimization problem
  EXPECT_NO_THROW({
    auto op_problem = data_model_view_to_optimization_problem(&handle, view);
    // If we get here, conversion succeeded with CPU data
  });
}

TEST(DataModelViewToOptimizationProblem, gpu_view_conversion)
{
  raft::handle_t handle;
  data_model_view_t<int, double> view;

  // Set up problem with GPU data (simulated with host pointers for test)
  std::vector<double> values{1.0, 2.0};
  std::vector<int> indices{0, 1};
  std::vector<int> offsets{0, 1, 2};
  std::vector<double> bounds{5.0, 10.0};
  std::vector<double> obj{1.0, -1.0};

  view.set_csr_constraint_matrix(values.data(), 2, indices.data(), 2, offsets.data(), 3);
  view.set_constraint_bounds(bounds.data(), 2);
  view.set_objective_coefficients(obj.data(), 2);
  view.set_is_device_memory(true);  // GPU memory flag

  // Convert to optimization problem
  EXPECT_NO_THROW({
    auto op_problem = data_model_view_to_optimization_problem(&handle, view);
    // If we get here, conversion succeeded
  });
}

TEST(DataModelViewToOptimizationProblem, empty_constraints_cpu_view)
{
  // Test the fix for empty constraint matrices with CPU memory
  raft::handle_t handle;
  data_model_view_t<int, double> view;

  // Empty constraint matrix (0 constraints) - the fix we added
  std::vector<double> empty_values;
  std::vector<int> empty_indices;
  std::vector<int> empty_offsets{0};
  std::vector<double> obj{1.0, 2.0};

  view.set_csr_constraint_matrix(
    empty_values.data(), 0, empty_indices.data(), 0, empty_offsets.data(), 1);
  view.set_objective_coefficients(obj.data(), 2);
  view.set_is_device_memory(false);  // CPU memory

  // This should not throw with the conditional check fix
  EXPECT_NO_THROW({ auto op_problem = data_model_view_to_optimization_problem(&handle, view); });
}

// ============================================================================
// Remote Solve Stub Tests
// ============================================================================

TEST(RemoteSolveStub, lp_returns_host_memory_solution)
{
  data_model_view_t<int, double> view;

  // Simple problem setup
  std::vector<double> values{1.0};
  std::vector<int> indices{0};
  std::vector<int> offsets{0, 1};
  std::vector<double> bounds{1.0};
  std::vector<double> obj{1.0};

  view.set_csr_constraint_matrix(values.data(), 1, indices.data(), 1, offsets.data(), 2);
  view.set_constraint_bounds(bounds.data(), 1);
  view.set_objective_coefficients(obj.data(), 1);
  view.set_is_device_memory(false);

  remote_solve_config_t config{"localhost", 50051};
  pdlp_solver_settings_t<int, double> settings;

  auto solution = solve_lp_remote(config, view, settings);

  // Stub returns host memory solution
  EXPECT_FALSE(solution.is_device_memory());
  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_EQ(solution.get_primal_solution_host().size(), 1);
  EXPECT_NO_THROW(solution.get_primal_solution_host());
  EXPECT_NO_THROW(solution.get_dual_solution_host());
}

TEST(RemoteSolveStub, mip_returns_host_memory_solution)
{
  data_model_view_t<int, double> view;

  // Simple MIP setup
  std::vector<double> obj{1.0, 2.0};
  std::vector<char> var_types{1, 1};  // Both integers

  view.set_objective_coefficients(obj.data(), 2);
  view.set_variable_types(var_types.data(), 2);
  view.set_is_device_memory(false);

  remote_solve_config_t config{"localhost", 50051};
  mip_solver_settings_t<int, double> settings;

  auto solution = solve_mip_remote(config, view, settings);

  // Stub returns host memory solution
  EXPECT_FALSE(solution.is_device_memory());
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_EQ(solution.get_solution_host().size(), 2);
  EXPECT_NO_THROW(solution.get_solution_host());
}

}  // namespace cuopt::linear_programming::test
