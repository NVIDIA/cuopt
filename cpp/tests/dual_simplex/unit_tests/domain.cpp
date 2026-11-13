/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/domain.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::mathematical_optimization::simplex::test {

class domain : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // x0 + x1 + s0 = 4 and x0 - x1 + s1 = 2 with 0 <= x0, x1 <= 10 and slacks s0, s1 >= 0.
    lp.A.col_start = {0, 2, 4, 5, 6};
    lp.A.i         = {0, 1, 0, 1, 0, 1};
    lp.A.x         = {1.0, 1.0, 1.0, -1.0, 1.0, 1.0};
    lp.rhs         = {4.0, 2.0};
    lp.lower       = {0.0, 0.0, 0.0, 0.0};
    lp.upper       = {10.0, 10.0, inf, inf};
    var_types.assign(lp.num_cols, variable_type_t::CONTINUOUS);
  }

  lp_problem_t<int, double> lp{nullptr, 2, 4, 6};
  csr_matrix_t<int, double> Arow{1, 1, 1};
  simplex_solver_settings_t<int, double> settings;
  domain_t<int, double> local_domain;
  std::vector<variable_type_t> var_types;
};

TEST_F(domain, continuous_upper_bounds)
{
  // Row 0 gives x0, x1, s0 <= 4, then row 1 gives s1 <= 2 + x1 <= 6.
  lp.A.to_compressed_row(Arow);
  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, lp));
  EXPECT_EQ(lp.lower, (std::vector<double>{0.0, 0.0, 0.0, 0.0}));
  EXPECT_EQ(lp.upper, (std::vector<double>{4.0, 4.0, 4.0, 6.0}));
}

TEST_F(domain, apply_and_propagate)
{
  lp.A.to_compressed_row(Arow);
  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, lp));

  // x0 >= 3 gives x1 <= 1 from row 0, then x0 <= 3 and x1 >= 1 from row 1. The slacks keep their
  // finite bounds from the root.
  const bound_change_t<int, double> branch{
    .var = 0, .new_upper = 4.0, .new_lower = 3.0, .origin = bound_change_origin_t::BRANCH};
  ASSERT_TRUE(local_domain.apply_and_propagate(Arow, var_types, settings, branch, lp));
  EXPECT_EQ(lp.lower, (std::vector<double>{3.0, 1.0, 0.0, 0.0}));
  EXPECT_EQ(lp.upper, (std::vector<double>{3.0, 1.0, 4.0, 6.0}));
  EXPECT_GT(local_domain.last_nnz_processed, 0u);
}

TEST_F(domain, backtrack_to_parent)
{
  lp.A.to_compressed_row(Arow);
  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, lp));
  const size_t root_size = local_domain.size();

  const bound_change_t<int, double> branch{
    .var = 0, .new_upper = 4.0, .new_lower = 3.0, .origin = bound_change_origin_t::BRANCH};
  ASSERT_TRUE(local_domain.apply_and_propagate(Arow, var_types, settings, branch, lp));
  EXPECT_GT(local_domain.size(), root_size + 1);

  // Pops the branching and the changes derived from it, leaving the root changes on the stack.
  local_domain.backtrack_to_parent(lp);
  EXPECT_EQ(local_domain.size(), root_size);
  EXPECT_EQ(lp.lower, (std::vector<double>{0.0, 0.0, 0.0, 0.0}));
  EXPECT_EQ(lp.upper, (std::vector<double>{4.0, 4.0, 4.0, 6.0}));

  // The activities were reverted as well, so the same branching gives the same bounds.
  ASSERT_TRUE(local_domain.apply_and_propagate(Arow, var_types, settings, branch, lp));
  EXPECT_EQ(lp.lower, (std::vector<double>{3.0, 1.0, 0.0, 0.0}));
  EXPECT_EQ(lp.upper, (std::vector<double>{3.0, 1.0, 4.0, 6.0}));
}

TEST_F(domain, propagate_from_stack)
{
  lp.A.to_compressed_row(Arow);
  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, lp));
  local_domain.clear();

  local_domain.apply(
    lp, {.var = 0, .new_upper = 4.0, .new_lower = 3.0, .origin = bound_change_origin_t::BRANCH});
  ASSERT_TRUE(local_domain.propagate_from_stack(Arow, var_types, settings, lp));
  EXPECT_EQ(lp.lower, (std::vector<double>{3.0, 1.0, 0.0, 0.0}));
  EXPECT_EQ(lp.upper, (std::vector<double>{3.0, 1.0, 4.0, 6.0}));
}

TEST_F(domain, propagate_from_variables)
{
  lp.A.to_compressed_row(Arow);
  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, lp));

  lp.lower[0] = 3.0;
  ASSERT_TRUE(local_domain.propagate_from_variables(Arow, var_types, settings, lp, {0}));
  EXPECT_EQ(lp.lower, (std::vector<double>{3.0, 1.0, 0.0, 0.0}));
  EXPECT_EQ(lp.upper, (std::vector<double>{3.0, 1.0, 4.0, 6.0}));
}

TEST_F(domain, integer_rounding_and_gauss_seidel)
{
  // 2 x0 + x1 = 3 and x0 + x1 + s = 10 with x0 integer in [0, 10] and x1 in [0, 2]. Row 0 gives
  // x0 in [0.5, 1.5], rounded to x0 = 1, and x1 sees x0 = 1 in the same pass, giving x1 = 1. Row 1
  // then bounds s <= 8, but its finite lower bound is never tightened.
  lp_problem_t<int, double> integer_lp(nullptr, 2, 3, 5);
  integer_lp.A.col_start = {0, 2, 4, 5};
  integer_lp.A.i         = {0, 1, 0, 1, 1};
  integer_lp.A.x         = {2.0, 1.0, 1.0, 1.0, 1.0};
  integer_lp.rhs         = {3.0, 10.0};
  integer_lp.lower       = {0.0, 0.0, 0.0};
  integer_lp.upper       = {10.0, 2.0, inf};
  var_types = {variable_type_t::INTEGER, variable_type_t::CONTINUOUS, variable_type_t::CONTINUOUS};
  integer_lp.A.to_compressed_row(Arow);

  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, integer_lp));
  EXPECT_EQ(integer_lp.lower, (std::vector<double>{1.0, 1.0, 0.0}));
  EXPECT_EQ(integer_lp.upper, (std::vector<double>{1.0, 1.0, 8.0}));
}

TEST_F(domain, infeasible_row)
{
  // With s0 fixed at 0, the activity of row 0 is at most 20 < 30.
  lp.rhs   = {30.0, 2.0};
  lp.upper = {10.0, 10.0, 0.0, inf};
  lp.A.to_compressed_row(Arow);
  EXPECT_FALSE(local_domain.propagate_full(Arow, var_types, settings, lp));

  // The object is reusable after an infeasible call.
  lp.rhs = {4.0, 2.0};
  ASSERT_TRUE(local_domain.propagate_full(Arow, var_types, settings, lp));
  EXPECT_EQ(lp.upper, (std::vector<double>{4.0, 4.0, 0.0, 6.0}));
}

TEST_F(domain, infeasible_integer_bounds)
{
  // 2 x0 + x1 + s0 = 1 with x1 and s0 fixed at 0 and x0 integer: x0 = 0.5 rounds to [1, 0].
  lp.A.x    = {2.0, 1.0, 1.0, -1.0, 1.0, 1.0};
  lp.rhs    = {1.0, 2.0};
  lp.upper  = {10.0, 0.0, 0.0, inf};
  var_types = {variable_type_t::INTEGER,
               variable_type_t::CONTINUOUS,
               variable_type_t::CONTINUOUS,
               variable_type_t::CONTINUOUS};
  lp.A.to_compressed_row(Arow);
  EXPECT_FALSE(local_domain.propagate_full(Arow, var_types, settings, lp));
}

}  // namespace cuopt::mathematical_optimization::simplex::test
