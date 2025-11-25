/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>

#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/problem/problem.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

cuopt::mps_parser::mps_data_model_t<int, double> read_from_mps(const std::string& file,
                                                               bool fixed_mps_format = true)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  return cuopt::mps_parser::parse_mps<int, double>(rel_file, fixed_mps_format);
}

TEST(optimization_problem_t, good_mps_file_1)
{
  const raft::handle_t handle_{};
  auto op_problem = read_from_mps("linear_programming/good-mps-1.mps");
  handle_.sync_stream(handle_.get_stream());
  ASSERT_EQ(int(2), op_problem.get_n_variables());
  ASSERT_EQ(int(2), op_problem.get_n_constraints());
  ASSERT_EQ(int(4), op_problem.get_nnz());
  ASSERT_EQ(op_problem.get_constraint_matrix_values().size(),
            op_problem.get_constraint_matrix_indices().size());
  ASSERT_EQ(std::size_t(3), op_problem.get_constraint_matrix_offsets().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_constraint_bounds().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_objective_coefficients().size());
  ASSERT_EQ(false, op_problem.get_sense());
  auto h_A = op_problem.get_constraint_matrix_values();
  EXPECT_EQ(3., h_A[0]);
  EXPECT_EQ(4., h_A[1]);
  EXPECT_EQ(2.7, h_A[2]);
  EXPECT_EQ(10.1, h_A[3]);
  auto h_A_indices = op_problem.get_constraint_matrix_indices();
  EXPECT_EQ(int(0), h_A_indices[0]);
  EXPECT_EQ(int(1), h_A_indices[1]);
  EXPECT_EQ(int(0), h_A_indices[2]);
  EXPECT_EQ(int(1), h_A_indices[3]);
  auto h_A_offsets = op_problem.get_constraint_matrix_offsets();
  EXPECT_EQ(int(0), h_A_offsets[0]);
  EXPECT_EQ(int(2), h_A_offsets[1]);
  EXPECT_EQ(int(4), h_A_offsets[2]);
  auto h_b = op_problem.get_constraint_bounds();
  EXPECT_EQ(5.4, h_b[0]);
  EXPECT_EQ(4.9, h_b[1]);
  auto h_c = op_problem.get_objective_coefficients();
  EXPECT_EQ(0.2, h_c[0]);
  EXPECT_EQ(0.1, h_c[1]);
}

TEST(optimization_problem_t, good_mps_file_comments)
{
  const raft::handle_t handle_{};
  auto op_problem = read_from_mps("linear_programming/good-mps-1-comments.mps");
  handle_.sync_stream(handle_.get_stream());
  ASSERT_EQ(int(2), op_problem.get_n_variables());
  ASSERT_EQ(int(2), op_problem.get_n_constraints());
  ASSERT_EQ(int(3), op_problem.get_nnz());
  ASSERT_EQ(op_problem.get_constraint_matrix_values().size(),
            op_problem.get_constraint_matrix_indices().size());
  ASSERT_EQ(std::size_t(3), op_problem.get_constraint_matrix_offsets().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_constraint_bounds().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_objective_coefficients().size());
  ASSERT_EQ(false, op_problem.get_sense());
  auto h_A = op_problem.get_constraint_matrix_values();
  EXPECT_EQ(3., h_A[0]);
  EXPECT_EQ(4., h_A[1]);
  EXPECT_EQ(2.7, h_A[2]);
  auto h_A_indices = op_problem.get_constraint_matrix_indices();
  EXPECT_EQ(int(0), h_A_indices[0]);
  EXPECT_EQ(int(1), h_A_indices[1]);
  EXPECT_EQ(int(0), h_A_indices[2]);
  auto h_A_offsets = op_problem.get_constraint_matrix_offsets();
  EXPECT_EQ(int(0), h_A_offsets[0]);
  EXPECT_EQ(int(2), h_A_offsets[1]);
  EXPECT_EQ(int(3), h_A_offsets[2]);
  auto h_b = op_problem.get_constraint_bounds();
  EXPECT_EQ(5.4, h_b[0]);
  EXPECT_EQ(4.9, h_b[1]);
  auto h_c = op_problem.get_objective_coefficients();
  EXPECT_EQ(0.2, h_c[0]);
  EXPECT_EQ(0.1, h_c[1]);
}

TEST(optimization_problem_t, test_set_get_fields)
{
  raft::handle_t handle;
  auto problem = optimization_problem_t<int, double>(&handle);

  double A_host[]      = {1.0, 2.0, 3.0};
  int indices_host[]   = {0, 1, 2};
  double b_host[]      = {4.0, 5.0, 6.0};
  double c_host[]      = {7.0, 8.0, 9.0};
  double var_lb_host[] = {0.0, 0.1, 0.2};
  double var_ub_host[] = {1.0, 1.1, 1.2};
  double con_lb_host[] = {0.5, 0.6, 0.7};
  double con_ub_host[] = {1.5, 1.6, 1.7};
  std::vector<double> result(3);
  std::vector<int> result_int(3);

  problem.set_csr_constraint_matrix(A_host, 3, indices_host, 3, indices_host, 3);

  // Test set_A_values
  cudaMemcpy(result.data(),
             problem.get_constraint_matrix_values().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(1.0, result[0], 1e-5);
  EXPECT_NEAR(2.0, result[1], 1e-5);
  EXPECT_NEAR(3.0, result[2], 1e-5);

  // Test A_indices
  cudaMemcpy(result_int.data(),
             problem.get_constraint_matrix_indices().data(),
             3 * sizeof(int),
             cudaMemcpyDeviceToHost);
  EXPECT_EQ(0, result_int[0]);
  EXPECT_EQ(1, result_int[1]);
  EXPECT_EQ(2, result_int[2]);

  // Test A_offsets_
  cudaMemcpy(result_int.data(),
             problem.get_constraint_matrix_offsets().data(),
             3 * sizeof(int),
             cudaMemcpyDeviceToHost);
  EXPECT_EQ(0, result_int[0]);
  EXPECT_EQ(1, result_int[1]);
  EXPECT_EQ(2, result_int[2]);

  // Test b_
  problem.set_constraint_bounds(b_host, 3);
  cudaMemcpy(result.data(),
             problem.get_constraint_bounds().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(4.0, result[0], 1e-5);
  EXPECT_NEAR(5.0, result[1], 1e-5);
  EXPECT_NEAR(6.0, result[2], 1e-5);

  // Test c_
  problem.set_objective_coefficients(c_host, 3);
  cudaMemcpy(result.data(),
             problem.get_objective_coefficients().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(7.0, result[0], 1e-5);
  EXPECT_NEAR(8.0, result[1], 1e-5);
  EXPECT_NEAR(9.0, result[2], 1e-5);

  // Test variable_lower_bounds_
  problem.set_variable_lower_bounds(var_lb_host, 3);
  cudaMemcpy(result.data(),
             problem.get_variable_lower_bounds().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(0.0, result[0], 1e-5);
  EXPECT_NEAR(0.1, result[1], 1e-5);
  EXPECT_NEAR(0.2, result[2], 1e-5);

  // Test variable_upper_bounds_
  problem.set_variable_upper_bounds(var_ub_host, 3);
  cudaMemcpy(result.data(),
             problem.get_variable_upper_bounds().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(1.0, result[0], 1e-5);
  EXPECT_NEAR(1.1, result[1], 1e-5);
  EXPECT_NEAR(1.2, result[2], 1e-5);

  // Test constraint_lower_bounds_
  problem.set_constraint_lower_bounds(con_lb_host, 3);
  cudaMemcpy(result.data(),
             problem.get_constraint_lower_bounds().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(0.5, result[0], 1e-5);
  EXPECT_NEAR(0.6, result[1], 1e-5);
  EXPECT_NEAR(0.7, result[2], 1e-5);

  // Test constraint_upper_bounds_
  problem.set_constraint_upper_bounds(con_ub_host, 3);
  cudaMemcpy(result.data(),
             problem.get_constraint_upper_bounds().data(),
             3 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(1.5, result[0], 1e-5);
  EXPECT_NEAR(1.6, result[1], 1e-5);
  EXPECT_NEAR(1.7, result[2], 1e-5);

  // Test objective_scaling_factor_
  double obj_scale = 1.5;
  problem.set_objective_scaling_factor(obj_scale);
  EXPECT_NEAR(obj_scale, problem.get_objective_scaling_factor(), 1e-5);

  // Test objective_offset_
  double obj_offset = 0.5;
  problem.set_objective_offset(obj_offset);
  EXPECT_NEAR(obj_offset, problem.get_objective_offset(), 1e-5);

  // Test objective_name_
  std::string obj_name = "my_objective";
  problem.set_objective_name(obj_name);
  EXPECT_EQ(obj_name, problem.get_objective_name());

  // Test problem_name_
  std::string prob_name = "my_problem";
  problem.set_problem_name(prob_name);
  EXPECT_EQ(prob_name, problem.get_problem_name());

  // Test var_names_
  std::vector<std::string> var_names = {"var1", "var2", "var3"};
  problem.set_variable_names(var_names);
  EXPECT_EQ("var1", problem.get_variable_names()[0]);
  EXPECT_EQ("var2", problem.get_variable_names()[1]);
  EXPECT_EQ("var3", problem.get_variable_names()[2]);

  // Test row_names_
  std::vector<std::string> row_names = {"row1", "row2", "row3"};
  problem.set_row_names(row_names);
  EXPECT_EQ("row1", problem.get_row_names()[0]);
  EXPECT_EQ("row2", problem.get_row_names()[1]);
  EXPECT_EQ("row3", problem.get_row_names()[2]);
}

TEST(optimization_problem_t, test_check_problem_validity)
{
  raft::handle_t handle;
  auto op_problem_ = optimization_problem_t<int, double>(&handle);

  // Test if exception is thrown when A_CSR_matrix are not set
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
               cuopt::logic_error);

  // Set A_CSR_matrix
  double A_host[]    = {1.0};
  int indices_host[] = {0};
  int offset_host[]  = {0, 1};
  op_problem_.set_csr_constraint_matrix(A_host, 1, indices_host, 1, offset_host, 2);

  // Test if exception is thrown when c is not set
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
               cuopt::logic_error);

  // Test that n_vars is not set
  EXPECT_EQ(op_problem_.get_n_variables(), 0);

  // Set c
  double c_host[] = {1.0};
  op_problem_.set_objective_coefficients(c_host, 1);

  // Test if exception is thrown when constraints are not set
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
               cuopt::logic_error);

  // Test that n_vars is now set
  EXPECT_EQ(op_problem_.get_n_variables(), 1);

  // Test that n_constraints is not set
  EXPECT_EQ(op_problem_.get_n_constraints(), 0);

  // Set row type
  char row_type_host[] = {'E'};
  op_problem_.set_row_types(row_type_host, 1);

  // Test if exception is thrown when row_type is set but not b
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
               cuopt::logic_error);

  // Test that n_constraints is now set
  EXPECT_EQ(op_problem_.get_n_constraints(), 1);

  // Set b
  double b_host[] = {1.0};
  op_problem_.set_constraint_bounds(b_host, 1);

  // Test that nothing is thrown when both b and row types are set
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)));

  // Unsetting row types and constraints bounds
  op_problem_.set_row_types(row_type_host, 0);
  op_problem_.set_constraint_bounds(b_host, 0);

  // Test that n_constraints is not set
  EXPECT_EQ(op_problem_.get_n_constraints(), 0);

  // Test again if exception is thrown when constraints bounds are not set
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
               cuopt::logic_error);

  // Seting constraint lower bounds
  double constraint_lower_bounds_host[] = {1.0};
  op_problem_.set_constraint_lower_bounds(constraint_lower_bounds_host, 1);

  // Test that n_constraints is now set
  EXPECT_EQ(op_problem_.get_n_constraints(), 1);

  // Test if exception is thrown when upper constraints bounds are not set
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
               cuopt::logic_error);

  // Seting constraint upper bounds
  double constraint_upper_bounds_host[] = {1.0};
  op_problem_.set_constraint_upper_bounds(constraint_upper_bounds_host, 1);

  // Test if no exception is thrown when constraints bounds are set
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)));
}

TEST(optimization_problem_t, test_csr_validity)
{
  raft::handle_t handle;
  auto op_problem_   = optimization_problem_t<int, double>(&handle);
  double A_host[]    = {1.0, 1.0};
  int indices_host[] = {0, 0};
  int offset_host[]  = {0, 1, 2};
  op_problem_.set_csr_constraint_matrix(A_host, 2, indices_host, 2, offset_host, 3);
  op_problem_.set_constraint_bounds(A_host, 2);
  op_problem_.set_objective_coefficients(A_host, 1);
  char row_type_host[] = {'E', 'E'};
  op_problem_.set_row_types(row_type_host, 2);
  // Valid problem
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)));

  // Test case 0: A_indices and A_values have different size
  {
    int incorrect_indices_size[] = {0};
    op_problem_.set_csr_constraint_matrix(A_host, 2, incorrect_indices_size, 1, offset_host, 3);
    EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
                 cuopt::logic_error);
  }

  // Test case 1: A_offsets first value not 0
  {
    int incorrect_first_offset[] = {1, 1, 2};
    op_problem_.set_csr_constraint_matrix(A_host, 2, indices_host, 2, incorrect_first_offset, 3);
    EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
                 cuopt::logic_error);
  }

  // Test case 2: A_offsets not in increasing order
  {
    int unsorted_offsets[] = {0, 2, 1};
    op_problem_.set_csr_constraint_matrix(A_host, 2, indices_host, 2, unsorted_offsets, 3);
    EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
                 cuopt::logic_error);
  }

  // Test case 3: A_indices value is negative
  {
    int negative_indices_host[] = {0, -1};
    op_problem_.set_csr_constraint_matrix(A_host, 2, negative_indices_host, 2, offset_host, 3);
    EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
                 cuopt::logic_error);
  }

  // Test case 4: A_indices value is greater than number of vars
  {
    int too_big_indices_host[] = {0, 1};
    op_problem_.set_csr_constraint_matrix(A_host, 2, too_big_indices_host, 2, offset_host, 3);
    EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_)),
                 cuopt::logic_error);
  }
}

TEST(optimization_problem_t, test_row_type_invalidity_char)
{
  raft::handle_t handle;

  // Constraints set through row types
  auto op_problem_1  = optimization_problem_t<int, double>(&handle);
  double A_host[]    = {1.0, 1.0, 1.0};
  int indices_host[] = {0, 0, 0};
  int offset_host[]  = {0, 1, 2, 3};
  op_problem_1.set_csr_constraint_matrix(A_host, 3, indices_host, 3, offset_host, 4);
  op_problem_1.set_constraint_bounds(A_host, 3);
  op_problem_1.set_objective_coefficients(A_host, 1);
  char row_type_host[] = {'E', 'L', 'N'};
  op_problem_1.set_row_types(row_type_host, 3);

  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)),
               cuopt::logic_error);
}

TEST(optimization_problem_t, test_row_type_invalidity_size)
{
  raft::handle_t handle;

  // Constraints set through row types
  auto op_problem_1  = optimization_problem_t<int, double>(&handle);
  double A_host[]    = {1.0, 1.0, 1.0};
  int indices_host[] = {0, 0, 0};
  int offset_host[]  = {0, 1, 2, 3};
  op_problem_1.set_csr_constraint_matrix(A_host, 3, indices_host, 3, offset_host, 4);
  op_problem_1.set_constraint_bounds(A_host, 3);
  op_problem_1.set_objective_coefficients(A_host, 1);
  char row_type_host[] = {'E', 'L', 'L'};
  op_problem_1.set_row_types(row_type_host, 2);

  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)),
               cuopt::logic_error);

  op_problem_1.set_row_types(row_type_host, 3);
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)));
}

TEST(optimization_problem_t, test_variable_invalidity_size)
{
  raft::handle_t handle;

  auto op_problem_1  = optimization_problem_t<int, double>(&handle);
  double A_host[]    = {1.0, 1.0, 1.0};
  int indices_host[] = {0, 0, 0};
  int offset_host[]  = {0, 1, 2, 3};
  op_problem_1.set_csr_constraint_matrix(A_host, 3, indices_host, 3, offset_host, 4);
  op_problem_1.set_constraint_lower_bounds(A_host, 3);
  op_problem_1.set_constraint_bounds(A_host, 3);
  op_problem_1.set_constraint_upper_bounds(A_host, 3);
  op_problem_1.set_objective_coefficients(A_host, 1);

  op_problem_1.set_variable_lower_bounds(A_host, 2);
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)),
               cuopt::logic_error);

  op_problem_1.set_variable_lower_bounds(A_host, 1);
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)));

  op_problem_1.set_variable_upper_bounds(A_host, 2);
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)),
               cuopt::logic_error);

  op_problem_1.set_variable_upper_bounds(A_host, 1);
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)));
}

TEST(optimization_problem_t, test_constraints_invalidity_size)
{
  raft::handle_t handle;

  auto op_problem_1  = optimization_problem_t<int, double>(&handle);
  double A_host[]    = {1.0, 1.0, 1.0};
  int indices_host[] = {0, 0, 0};
  int offset_host[]  = {0, 1, 2, 3};
  op_problem_1.set_csr_constraint_matrix(A_host, 3, indices_host, 3, offset_host, 4);
  op_problem_1.set_constraint_lower_bounds(A_host, 2);
  op_problem_1.set_constraint_bounds(A_host, 3);
  op_problem_1.set_constraint_upper_bounds(A_host, 2);
  op_problem_1.set_objective_coefficients(A_host, 1);

  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)),
               cuopt::logic_error);

  op_problem_1.set_constraint_lower_bounds(A_host, 3);
  EXPECT_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)),
               cuopt::logic_error);

  op_problem_1.set_constraint_upper_bounds(A_host, 3);
  EXPECT_NO_THROW((problem_checking_t<int, double>::check_problem_representation(op_problem_1)));
}

TEST(optimization_problem_t, good_mps_mip_file_1)
{
  const raft::handle_t handle_{};
  auto op_problem = read_from_mps("mixed_integer_programming/good-mip-mps-1.mps", false);

  handle_.sync_stream(handle_.get_stream());
  ASSERT_EQ(int(2), op_problem.get_n_variables());
  ASSERT_EQ(int(2), op_problem.get_n_constraints());
  ASSERT_EQ(int(4), op_problem.get_nnz());
  ASSERT_EQ(op_problem.get_constraint_matrix_values().size(),
            op_problem.get_constraint_matrix_indices().size());
  ASSERT_EQ(std::size_t(3), op_problem.get_constraint_matrix_offsets().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_constraint_bounds().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_objective_coefficients().size());
  ASSERT_EQ(true, op_problem.get_sense());
  auto h_A = op_problem.get_constraint_matrix_values();
  EXPECT_EQ(8000., h_A[0]);
  EXPECT_EQ(4000., h_A[1]);
  EXPECT_EQ(15., h_A[2]);
  EXPECT_EQ(30., h_A[3]);
  auto h_A_indices = op_problem.get_constraint_matrix_indices();
  EXPECT_EQ(int(0), h_A_indices[0]);
  EXPECT_EQ(int(1), h_A_indices[1]);
  EXPECT_EQ(int(0), h_A_indices[2]);
  EXPECT_EQ(int(1), h_A_indices[3]);
  auto h_A_offsets = op_problem.get_constraint_matrix_offsets();
  EXPECT_EQ(int(0), h_A_offsets[0]);
  EXPECT_EQ(int(2), h_A_offsets[1]);
  EXPECT_EQ(int(4), h_A_offsets[2]);
  auto h_b = op_problem.get_constraint_bounds();
  EXPECT_EQ(40000., h_b[0]);
  EXPECT_EQ(200., h_b[1]);
  auto h_c = op_problem.get_objective_coefficients();
  EXPECT_EQ(100., h_c[0]);
  EXPECT_EQ(150., h_c[1]);
  auto h_lower_bounds = op_problem.get_variable_lower_bounds();
  EXPECT_EQ(0., h_lower_bounds[0]);
  EXPECT_EQ(0., h_lower_bounds[1]);
  auto h_upper_bounds = op_problem.get_variable_upper_bounds();
  EXPECT_EQ(10., h_upper_bounds[0]);
  EXPECT_EQ(10., h_upper_bounds[1]);
}

TEST(optimization_problem_t, good_mps_mip_file_no_marker)
{
  const raft::handle_t handle_{};
  auto op_problem = read_from_mps("mixed_integer_programming/good-mip-mps-1-no-mark.mps", false);

  handle_.sync_stream(handle_.get_stream());
  ASSERT_EQ(int(2), op_problem.get_n_variables());
  ASSERT_EQ(int(2), op_problem.get_n_constraints());
  ASSERT_EQ(int(4), op_problem.get_nnz());
  ASSERT_EQ(op_problem.get_constraint_matrix_values().size(),
            op_problem.get_constraint_matrix_indices().size());
  ASSERT_EQ(std::size_t(3), op_problem.get_constraint_matrix_offsets().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_constraint_bounds().size());
  ASSERT_EQ(std::size_t(2), op_problem.get_objective_coefficients().size());
  ASSERT_EQ(true, op_problem.get_sense());
  auto h_A = op_problem.get_constraint_matrix_values();
  EXPECT_EQ(8000., h_A[0]);
  EXPECT_EQ(4000., h_A[1]);
  EXPECT_EQ(15., h_A[2]);
  EXPECT_EQ(30., h_A[3]);
  auto h_A_indices = op_problem.get_constraint_matrix_indices();
  EXPECT_EQ(int(0), h_A_indices[0]);
  EXPECT_EQ(int(1), h_A_indices[1]);
  EXPECT_EQ(int(0), h_A_indices[2]);
  EXPECT_EQ(int(1), h_A_indices[3]);
  auto h_A_offsets = op_problem.get_constraint_matrix_offsets();
  EXPECT_EQ(int(0), h_A_offsets[0]);
  EXPECT_EQ(int(2), h_A_offsets[1]);
  EXPECT_EQ(int(4), h_A_offsets[2]);
  auto h_b = op_problem.get_constraint_bounds();
  EXPECT_EQ(40000., h_b[0]);
  EXPECT_EQ(200., h_b[1]);
  auto h_c = op_problem.get_objective_coefficients();
  EXPECT_EQ(100., h_c[0]);
  EXPECT_EQ(150., h_c[1]);
  auto h_lower_bounds = op_problem.get_variable_lower_bounds();
  EXPECT_EQ(0., h_lower_bounds[0]);
  EXPECT_EQ(0., h_lower_bounds[1]);
  auto h_upper_bounds = op_problem.get_variable_upper_bounds();
  EXPECT_EQ(10., h_upper_bounds[0]);
  EXPECT_EQ(10., h_upper_bounds[1]);
}

}  // namespace cuopt::linear_programming
