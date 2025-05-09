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

#include "../utilities/pdlp_test_utilities.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <linear_programming/utils.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/problem/problem.cuh>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

namespace lp  = cuopt::linear_programming;
namespace dtl = cuopt::linear_programming::detail;

template <typename i_t, typename T>
thrust::host_vector<T> rand_vec(i_t size, T dist_beg, T dist_end)
{
  thrust::default_random_engine rng(1337);
  thrust::host_vector<T> vec(size);
  if constexpr (std::is_floating_point_v<T>) {
    thrust::uniform_real_distribution<T> dist(dist_beg, dist_end);
    thrust::generate(vec.begin(), vec.end(), [&] { return dist(rng); });
  } else {
    thrust::uniform_int_distribution<T> dist(dist_beg, dist_end);
    thrust::generate(vec.begin(), vec.end(), [&] { return dist(rng); });
  }
  return vec;
}

template <typename i_t, typename f_t>
lp::optimization_problem_t<i_t, f_t> create_problem(raft::handle_t const* h, i_t n_cnst, i_t n_var)
{
  lp::optimization_problem_t<i_t, f_t> problem(h);
  thrust::default_random_engine rng(1337);
  thrust::uniform_real_distribution<f_t> dist(0, 5);

  thrust::host_vector<var_t> variable_types(n_var, var_t::CONTINUOUS);
  variable_types[0] = var_t::INTEGER;

  // variable_lower_bounds & variable_upper_bounds
  auto variable_lower_bounds = rand_vec<i_t, f_t>(n_var, 0, 5);
  auto variable_upper_bounds = rand_vec<i_t, f_t>(n_var, 0, 5);

  // variable_upper_bounds = variable_upper_bounds + variable_lower_bounds
  thrust::transform(thrust::host,
                    variable_lower_bounds.begin(),
                    variable_lower_bounds.end(),
                    variable_upper_bounds.begin(),
                    variable_upper_bounds.begin(),
                    thrust::plus<f_t>{});

  // constraint_lower_bounds & constraint_upper_bounds
  auto constraint_lower_bounds = rand_vec<i_t, f_t>(n_cnst, 0, 5);
  auto constraint_upper_bounds = rand_vec<i_t, f_t>(n_cnst, 0, 5);

  // constraint_upper_bounds = constraint_upper_bounds + constraint_lower_bounds
  thrust::transform(thrust::host,
                    constraint_lower_bounds.begin(),
                    constraint_lower_bounds.end(),
                    constraint_upper_bounds.begin(),
                    constraint_upper_bounds.begin(),
                    thrust::plus<f_t>{});

  // c - non-zero coefficients
  auto c = rand_vec<i_t, f_t>(n_var, 0.2, 10.0);

  // to ensure every variable is used once in the graph
  i_t average_vars_per_cnst = raft::ceildiv<i_t>(n_var, n_cnst);
  auto offsets = rand_vec<i_t, i_t>(n_cnst + 1, average_vars_per_cnst, average_vars_per_cnst * 5);
  thrust::exclusive_scan(thrust::host, offsets.begin(), offsets.end(), offsets.begin());

  i_t nnz = offsets.back() + 1;

  // a_values - non-zero coefficients
  auto coeff = rand_vec<i_t, f_t>(nnz, 0.2, 10.0);

  thrust::host_vector<i_t> indices(nnz);
  thrust::tabulate(
    thrust::host, indices.begin(), indices.end(), [n_var] __host__(auto i) { return i % n_var; });

  // randomly shuffle variables
  thrust::shuffle(thrust::host, indices.begin(), indices.end(), rng);

  problem.set_variable_types(variable_types.data(), n_var);

  problem.set_csr_constraint_matrix(
    coeff.data(), nnz, indices.data(), nnz, offsets.data(), n_cnst + 1);
  problem.set_objective_coefficients(c.data(), n_var);

  problem.set_variable_lower_bounds(variable_lower_bounds.data(), n_var);

  problem.set_variable_upper_bounds(variable_upper_bounds.data(), n_var);

  problem.set_constraint_lower_bounds(constraint_lower_bounds.data(), n_cnst);

  problem.set_constraint_upper_bounds(constraint_upper_bounds.data(), n_cnst);

  problem.set_constraint_bounds(constraint_upper_bounds.data(), n_cnst);

  return problem;
}

template <typename i_t, typename f_t>
void set_equal_var_bounds(optimization_problem_t<i_t, f_t>& problem,
                          thrust::host_vector<i_t>& selected_vars)
{
  cuopt_assert(selected_vars.size() < problem.get_n_variables(), "invalid number of variables");
  rmm::device_uvector<f_t>& v_lb = problem.get_variable_lower_bounds();
  rmm::device_uvector<f_t>& v_ub = problem.get_variable_upper_bounds();
  rmm::device_uvector<i_t> sel_vars(selected_vars.size(), problem.get_handle_ptr()->get_stream());
  raft::copy(sel_vars.data(),
             selected_vars.data(),
             selected_vars.size(),
             problem.get_handle_ptr()->get_stream());
  auto lb = make_span(v_lb);
  auto ub = make_span(v_ub);
  thrust::for_each(problem.get_handle_ptr()->get_thrust_policy(),
                   sel_vars.begin(),
                   sel_vars.end(),
                   [lb, ub] __device__(auto v) { lb[v] = ub[v]; });
}

template <typename i_t, typename f_t>
thrust::host_vector<i_t> generate_random_vals(i_t total_count, i_t n_count)
{
  thrust::default_random_engine rng(1337);
  thrust::host_vector<i_t> vec(total_count);
  thrust::sequence(thrust::host, vec.begin(), vec.end());
  thrust::shuffle(thrust::host, vec.begin(), vec.end(), rng);
  vec.resize(n_count);
  return vec;
}

template <typename i_t, typename f_t>
void test_equal_val_bounds(i_t n_cnst, i_t n_var)
{
  const raft::handle_t handle_{};

  auto op_problem = create_problem<i_t, f_t>(&handle_, n_cnst, n_var);
  auto selected_vars =
    generate_random_vals<i_t, f_t>(op_problem.get_n_variables(), std::max(n_var * 0.1, 1.));
  set_equal_var_bounds<i_t, f_t>(op_problem, selected_vars);

  dtl::problem_t<i_t, f_t> problem(op_problem);
  problem.preprocess_problem();
  detail::trivial_presolve(problem);

  EXPECT_EQ(selected_vars.size() + problem.n_variables, n_var);
}

TEST(problem, run_small_tests)
{
  std::vector<std::pair<int, int>> cnst_var_vals = {{30, 150}, {40, 200}, {50, 300}};
  for (const auto& val : cnst_var_vals) {
    test_equal_val_bounds<int, double>(val.first, val.second);
  }
}

static void fill_problem(optimization_problem_t<int, double>& op_problem)
{
  // Set A_CSR_matrix
  double A_host[]    = {1.0};
  int indices_host[] = {0};
  int offset_host[]  = {0, 1};
  op_problem.set_csr_constraint_matrix(A_host, 1, indices_host, 1, offset_host, 2);

  // Set c
  double c_host[] = {1.0};
  op_problem.set_objective_coefficients(c_host, 1);

  // Set row type
  char row_type_host[] = {'E'};
  op_problem.set_row_types(row_type_host, 1);

  // Set b
  double b_host[] = {1.0};
  op_problem.set_constraint_bounds(b_host, 1);
}

TEST(problem, setting_both_rhs_and_constraints_bounds)
{
  // Check constraints lower/upper bounds after having filled the row type and rhs
  {
    raft::handle_t handle;
    optimization_problem_t<int, double> op_problem(&handle);
    fill_problem(op_problem);
    cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

    const auto constraints_lower_bounds = host_copy(problem.constraint_lower_bounds);
    const auto constraints_upper_bounds = host_copy(problem.constraint_upper_bounds);

    EXPECT_EQ(constraints_lower_bounds[0], 1.0);
    EXPECT_EQ(constraints_upper_bounds[0], 1.0);
  }

  // Check constraints lower/upper bounds after having set both
  {
    raft::handle_t handle;
    optimization_problem_t<int, double> op_problem(&handle);
    fill_problem(op_problem);
    double lower[] = {2.0};
    double upper[] = {3.0};
    op_problem.set_constraint_lower_bounds(lower, 1);
    op_problem.set_constraint_upper_bounds(upper, 1);
    cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

    const auto constraints_lower_bounds = host_copy(problem.constraint_lower_bounds);
    const auto constraints_upper_bounds = host_copy(problem.constraint_upper_bounds);
    EXPECT_EQ(constraints_lower_bounds[0], 2.0);
    EXPECT_EQ(constraints_upper_bounds[0], 3.0);
  }

  // Check constraints lower/upper bounds after having set both
  // Set upper / lower before
  {
    raft::handle_t handle;
    optimization_problem_t<int, double> op_problem(&handle);
    double lower[] = {2.0};
    double upper[] = {3.0};
    op_problem.set_constraint_lower_bounds(lower, 1);
    op_problem.set_constraint_upper_bounds(upper, 1);
    fill_problem(op_problem);
    cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

    const auto constraints_lower_bounds = host_copy(problem.constraint_lower_bounds);
    const auto constraints_upper_bounds = host_copy(problem.constraint_upper_bounds);
    EXPECT_EQ(constraints_lower_bounds[0], 2.0);
    EXPECT_EQ(constraints_upper_bounds[0], 3.0);
  }
}

// Special setup since this is a "killing" test (because of assert)
TEST(optimization_problem_t_DeathTest, test_check_problem_validity)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  raft::handle_t handle;
  auto op_problem        = optimization_problem_t<int, double>(&handle);
  using custom_problem_t = cuopt::linear_programming::detail::problem_t<int, double>;

  // Check if assert if nothing
  EXPECT_DEATH({ custom_problem_t problem(op_problem); }, "");

  // Set A_CSR_matrix
  /*
   *   1 2 0
   *   3 3 4
   */
  double A_host[]    = {1.0, 2.0, 3.0, 3.0, 4.0};
  int indices_host[] = {0, 1, 0, 1, 2};
  int offset_host[]  = {0, 2, 5};
  op_problem.set_csr_constraint_matrix(A_host, 5, indices_host, 5, offset_host, 3);

  // Test if assert is thrown when c is not set
  EXPECT_DEATH({ custom_problem_t problem(op_problem); }, "");

  // Set c
  double c_host[] = {1.0, 2.0, 3.0};
  op_problem.set_objective_coefficients(c_host, 3);

  // Test if assert is thrown when constraints are not set
  EXPECT_DEATH({ custom_problem_t problem(op_problem); }, "");

  // Set row type
  char row_type_host[] = {'E', 'E'};
  op_problem.set_row_types(row_type_host, 2);

  // Test if assert is thrown when row_type is set but not b
  EXPECT_DEATH({ custom_problem_t problem(op_problem); }, "");

  // Set b
  double b_host[] = {1.0, 2.0};
  op_problem.set_constraint_bounds(b_host, 2);

  // Test that nothing is thrown when both b and row types are set
  custom_problem_t problem(op_problem);

  // Unsetting row types and constraints bounds
  op_problem.set_row_types(row_type_host, 0);
  op_problem.set_constraint_bounds(b_host, 0);

  // Test again if assert is thrown when constraints bounds are not set
  EXPECT_DEATH({ custom_problem_t problem(op_problem); }, "");

  // Seting constraint lower bounds
  double constraint_lower_bounds_host[] = {1.0f, 2.0f};
  op_problem.set_constraint_lower_bounds(constraint_lower_bounds_host, 2);

  // Test if assert is thrown when upper constraints bounds are not set
  EXPECT_DEATH({ custom_problem_t problem(op_problem); }, "");

  // Seting constraint upper bounds
  op_problem.set_constraint_upper_bounds(constraint_lower_bounds_host, 2);

  // Test if no assert is thrown when constraints bounds are set
  custom_problem_t problem2(op_problem);

  // Manually unsetting the tranpose fields in problem2 (automatically created in LP mode)
  problem2.reverse_coefficients = rmm::device_uvector<double>(0, handle.get_stream());
  problem2.reverse_constraints  = rmm::device_uvector<int>(0, handle.get_stream());
  problem2.reverse_offsets      = rmm::device_uvector<int>(0, handle.get_stream());

  // Test if assert is thrown when transpose is not set, yet ask for it to be checked
  EXPECT_DEATH({ problem2.check_problem_representation(true, false); }, "");

  // Set the tranpose matrix
  /*
   *   1 2 0
   *   3 3 4
   *   Tranpose
   *   1 3
   *   2 3
   *   0 4
   */
  std::vector<double> reverse_A_host    = {1.0, 3.0, 2.0, 3.0, 4.0};
  std::vector<int> reverse_indices_host = {0, 1, 0, 1, 1};
  std::vector<int> reverse_offset_host  = {0, 2, 4, 5};
  problem2.reverse_coefficients         = device_copy(reverse_A_host, handle.get_stream());
  problem2.reverse_constraints          = device_copy(reverse_indices_host, handle.get_stream());
  problem2.reverse_offsets              = device_copy(reverse_offset_host, handle.get_stream());

  // Test if assert is not thrown when transpose is set
  problem2.check_problem_representation(true, false);

  // Test if assert is thrown when a basic (not correct size) transpose csr issue occurs
  std::vector<int> incorrect_reverse_offset_host = {0, 2, 4};
  problem2.reverse_offsets = device_copy(incorrect_reverse_offset_host, handle.get_stream());
  EXPECT_DEATH({ problem2.check_problem_representation(true, false); }, "");

  // Test if assert is thrown when a value swap transpose csr issue occurs

  // First make sure putting back offset doesn't trigger anything
  problem2.reverse_offsets = device_copy(reverse_offset_host, handle.get_stream());
  problem2.check_problem_representation(true, false);

  // Now just reverse two value in the transposed which should trigger an assert (since tranpose
  // needs to match and non transposed)
  std::vector<double> incorrect_reverse_A_host = {1.0, 3.0, 3.0, 2.0, 4.0};
  problem2.reverse_coefficients = device_copy(incorrect_reverse_A_host, handle.get_stream());

  // Check that it does assert
  EXPECT_DEATH({ problem2.check_problem_representation(true, false); }, "");
}

}  // namespace cuopt::linear_programming::test
