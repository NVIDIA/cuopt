/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/utils.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(problem, find_implied_integers)
{
  const raft::handle_t handle_{};

  auto path           = make_path_absolute("mip/fiball.mps");
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  auto presolver      = std::make_unique<mip::third_party_presolve_t<int, double>>();
  auto result         = presolver->apply(op_problem,
                                 cuopt::mathematical_optimization::problem_category_t::MIP,
                                 cuopt::mathematical_optimization::presolver_t::Papilo,
                                 false,
                                 1e-6,
                                 1e-12,
                                 20,
                                 1);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::UNBNDORINFEAS);

  auto problem = mip::problem_t<int, double>(result.reduced_problem);
  problem.set_implied_integers(result.implied_integer_indices);
  ASSERT_TRUE(result.implied_integer_indices.size() > 0);
  auto var_types = host_copy(problem.variable_types, handle_.get_stream());
  // Find the index of the one continuous variable
  auto it = std::find_if(var_types.begin(), var_types.end(), [](var_t var_type) {
    return var_type == var_t::CONTINUOUS;
  });
  ASSERT_NE(it, var_types.end());
  ASSERT_EQ(problem.presolve_data.var_flags.size(), var_types.size());
  // Ensure it is an implied integer
  EXPECT_EQ(problem.presolve_data.var_flags.element(it - var_types.begin(), handle_.get_stream()),
            ((int)mip::problem_t<int, double>::var_flags_t::VAR_IMPLIED_INTEGER));
}

TEST(gf2_presolve, uses_compact_constraint_indices)
{
  constexpr int num_packing_vars = 128;
  constexpr int num_gf2_vars     = 6;
  constexpr int num_key_vars     = 6;
  constexpr int num_packing_rows = 128;
  constexpr int num_key_rows     = 2 * num_key_vars;
  constexpr int num_gf2_rows     = 6;
  constexpr int num_vars         = num_packing_vars + num_gf2_vars + num_key_vars;
  constexpr int num_rows         = num_packing_rows + num_key_rows + num_gf2_rows;
  constexpr int x_offset         = num_packing_vars;
  constexpr int y_offset         = x_offset + num_gf2_vars;

  std::vector<double> values;
  std::vector<int> indices;
  std::vector<int> offsets{0};
  std::vector<double> constraint_lb(num_rows, 1.0);
  std::vector<double> constraint_ub(num_rows, 2.0);

  auto add_entry = [&](int column, double value) {
    indices.push_back(column);
    values.push_back(value);
  };
  auto finish_row = [&] { offsets.push_back(static_cast<int>(values.size())); };

  // A normal binary MIP block keeps the GF2 rows at high raw row indices.
  for (int row = 0; row < num_packing_rows; ++row) {
    std::array columns{row, (row + 1) % num_packing_vars, (row + 2) % num_packing_vars};
    std::sort(columns.begin(), columns.end());
    for (int column : columns) {
      add_entry(column, 1.0);
    }
    finish_row();
  }

  // Keep every GF2 key column non-singleton without forcing it.
  for (int key = 0; key < num_key_vars; ++key) {
    add_entry(3 * key, 1.0);
    add_entry(3 * key + 1, 1.0);
    add_entry(y_offset + key, 1.0);
    finish_row();

    add_entry(3 * key + 1, 1.0);
    add_entry(3 * key + 2, 1.0);
    add_entry(y_offset + key, 1.0);
    finish_row();
  }

  // Over GF(2), this is J-I for even dimension 6, hence nonsingular. Three positive and two
  // negative coefficients per row prevent ordinary bound propagation from fixing the key.
  for (int row = 0; row < num_gf2_rows; ++row) {
    int term = 0;
    for (int col = 0; col < num_gf2_vars; ++col) {
      if (col == row) { continue; }
      add_entry(x_offset + col, term < 3 ? 1.0 : -1.0);
      ++term;
    }
    add_entry(y_offset + row, 2.0);
    finish_row();
    constraint_lb[num_packing_rows + num_key_rows + row] = 1.0;
    constraint_ub[num_packing_rows + num_key_rows + row] = 1.0;
  }

  const raft::handle_t handle_{};
  optimization_problem_t<int, double> problem(&handle_);
  std::vector<double> objective(num_vars, 1.0);
  std::vector<double> variable_lb(num_vars, 0.0);
  std::vector<double> variable_ub(num_vars, 1.0);
  std::vector<var_t> variable_types(num_vars, var_t::INTEGER);
  problem.set_csr_constraint_matrix(
    values.data(), values.size(), indices.data(), indices.size(), offsets.data(), offsets.size());
  problem.set_objective_coefficients(objective.data(), objective.size());
  problem.set_variable_lower_bounds(variable_lb.data(), variable_lb.size());
  problem.set_variable_upper_bounds(variable_ub.data(), variable_ub.size());
  problem.set_variable_types(variable_types.data(), variable_types.size());
  problem.set_constraint_lower_bounds(constraint_lb.data(), constraint_lb.size());
  problem.set_constraint_upper_bounds(constraint_ub.data(), constraint_ub.size());

  auto presolver = std::make_unique<mip::third_party_presolve_t<int, double>>();
  auto result    = presolver->apply(
    problem, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20, 1);

  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::REDUCED);
}

}  // namespace cuopt::mathematical_optimization::test
