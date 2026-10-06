/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <pdlp/reduced_matrix.cuh>
#include <raft/core/handle.hpp>
#include <utilities/copy_helpers.hpp>

#include <algorithm>
#include <vector>

namespace cuopt::mathematical_optimization::test {

class ReducedMatrix : public testing::TestWithParam<bool> {};

TEST_P(ReducedMatrix, CompactPrimalRefreshReleaseAndRestart)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();
  optimization_problem_t<int, double> op(&handle);
  std::vector<double> values{1, -2, 3, -4, 5, -6, 7, -8, -2, 1, -4, 3, -6, 5, -8, 7};
  std::vector<int> columns{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> offsets{0, 8, 16};
  std::vector<double> lower{-2, -3, -4, -5, -6, -7, -8, -9};
  std::vector<double> upper{3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> objective{1000, -1000, 1000, -1000, 1000, -1000, 1000, .3};
  std::vector<double> row_lower{-100, -100}, row_upper{100, 100};
  op.set_csr_constraint_matrix(
    values.data(), values.size(), columns.data(), columns.size(), offsets.data(), offsets.size());
  op.set_objective_coefficients(objective.data(), objective.size());
  op.set_variable_lower_bounds(lower.data(), lower.size());
  op.set_variable_upper_bounds(upper.data(), upper.size());
  op.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  op.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  mip::problem_t<int, double> problem(op);
  problem.compute_transpose_of_problem();
  mip::problem_t<int, double> shared_structure(problem, false);
  RAFT_CUSPARSE_TRY(
    cusparseSetPointerMode(handle.get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE));

  std::vector<double> projected(8), current(8), initial(8);
  for (int j = 0; j < 8; ++j) {
    projected[j] = j == 7 ? 0.5 : (j % 2 == 0 ? lower[j] : upper[j]);
    current[j]   = projected[j] + 0.07 * (j + 1);
    initial[j]   = projected[j] - 0.11 * (j + 1);
  }
  auto d_projected = device_copy(projected, stream);
  auto d_current   = device_copy(current, stream);
  auto d_initial   = device_copy(initial, stream);
  std::vector<double> dual{-.2, .3};
  auto d_dual                 = device_copy(dual, stream);
  auto d_step                 = device_copy(std::vector<double>{.05}, stream);
  const double initial_weight = 0.5;
  rmm::device_scalar<double> d_weight(initial_weight, stream);
  rmm::device_uvector<double> activity(2, stream);
  pdlp::reduced_matrix_t<int, double> reduced(&handle, problem, true);
  if (GetParam()) {
    reduced.redirect_csr_structure(shared_structure);
    for (auto* structure : {&problem.offsets,
                            &problem.variables,
                            &problem.reverse_offsets,
                            &problem.reverse_constraints}) {
      structure->resize(0, stream);
      structure->shrink_to_fit(stream);
    }
    // Poison the shared problem's values: only its CSR structure may be used.
    RAFT_CUDA_TRY(cudaMemsetAsync(shared_structure.coefficients.data(),
                                  0,
                                  sizeof(double) * shared_structure.coefficients.size(),
                                  stream.get()));
    RAFT_CUDA_TRY(cudaMemsetAsync(shared_structure.reverse_coefficients.data(),
                                  0,
                                  sizeof(double) * shared_structure.reverse_coefficients.size(),
                                  stream.get()));
  }
  ASSERT_TRUE(reduced.update_mode(1e-4, false, d_projected, d_current, d_initial));
  ASSERT_EQ(reduced.free_count(), 1);

  int iteration = 0;
  for (int epoch = 0; epoch < 3; ++epoch) {
    // Odd/even segment lengths exercise both signs of the fixed-state coefficient.
    const int steps = epoch == 0 ? 1 : (epoch == 1 ? 2 : 197);
    for (int step = 0; step < steps; ++step, ++iteration) {
      const double weight = double(iteration + 1) / (iteration + 2);
      d_weight.set_value_async(weight, stream);
      reduced.compute_At_y(d_dual);
      reduced.update_primal(d_step, d_weight);
      reduced.compute_A_x(activity, d_weight);
      std::vector<double> expected_activity(2, 0.0);
      for (int j = 0; j < 8; ++j) {
        const double aty = values[j] * dual[0] + values[8 + j] * dual[1];
        projected[j]     = std::clamp(current[j] - .05 * (objective[j] - aty), lower[j], upper[j]);
        const double reflected = 2 * projected[j] - current[j];
        current[j]             = weight * reflected + (1 - weight) * initial[j];
        for (int row = 0; row < 2; ++row) {
          expected_activity[row] += values[8 * row + j] * reflected;
        }
      }
      const auto actual_activity = host_copy(activity, stream);
      for (int row = 0; row < 2; ++row) {
        EXPECT_NEAR(actual_activity[row], expected_activity[row], 1e-10);
      }
    }
    ASSERT_TRUE(reduced.request_refresh(true, d_current, d_initial));
    const auto actual_current = host_copy(d_current, stream);
    for (int j = 0; j < 8; ++j) {
      EXPECT_NEAR(actual_current[j], current[j], 1e-11);
    }

    // Supply the result of a full refresh. An interior projection releases column 0;
    // releasing all columns later must fall back to the full operator.
    if (epoch == 0) { projected[0] = current[0] = 0.0; }
    if (epoch == 2) { std::fill(projected.begin(), projected.end(), 0.0); }
    raft::update_device(d_projected.data(), projected.data(), projected.size(), stream);
    raft::update_device(d_current.data(), current.data(), current.size(), stream);
    ASSERT_TRUE(reduced.finish_refresh(d_projected, d_current, d_initial));
    if (epoch == 0) { EXPECT_EQ(reduced.free_count(), 2); }
    if (epoch == 1) {
      // Restart overwrites both full current and anchor vectors, then re-gathers compact state.
      current = initial = projected;
      raft::update_device(d_current.data(), current.data(), current.size(), stream);
      raft::update_device(d_initial.data(), initial.data(), initial.size(), stream);
      reduced.update_mode(1e-4, true, d_projected, d_current, d_initial);
      iteration = 0;
    }
  }
  EXPECT_FALSE(reduced.active());
}

INSTANTIATE_TEST_SUITE_P(SharedStructure, ReducedMatrix, testing::Bool());

}  // namespace cuopt::mathematical_optimization::test
