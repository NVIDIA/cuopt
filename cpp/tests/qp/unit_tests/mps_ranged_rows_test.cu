/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/translate.hpp>
#include <raft/core/handle.hpp>
#include <vector>

namespace cuopt::mathematical_optimization {

class MpsRangedRowsTest : public ::testing::TestWithParam<bool> {};

TEST_P(MpsRangedRowsTest, parsed_and_converted_bounds)
{
  // The six signed-range encodings all describe 0 <= x <= 1.
  auto parsed = io::read_mps_from_string<int, double>(R"MPS(
NAME RANGED_QP
ROWS
 N OBJ
 G GPOS
 G GNEG
 L LPOS
 L LNEG
 E EPOS
 E ENEG
COLUMNS
 X OBJ -4 GPOS 1
 X GNEG 1 LPOS 1
 X LNEG 1 EPOS 1
 X ENEG 1
RHS
 RHS1 LPOS 1 LNEG 1
 RHS1 ENEG 1
RANGES
 RNG1 GPOS 1 GNEG -1
 RNG1 LPOS 1 LNEG -1
 RNG1 EPOS 1 ENEG -1
BOUNDS
 FR BND1 X
QUADOBJ
 X X 2
ENDATA
)MPS");
  EXPECT_EQ(parsed.get_row_types(), (std::vector<char>{'G', 'G', 'L', 'L', 'E', 'E'}));
  EXPECT_EQ(parsed.get_constraint_bounds(), (std::vector<double>{0, 0, 1, 1, 0, 1}));
  EXPECT_EQ(parsed.get_constraint_lower_bounds(), (std::vector<double>{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(parsed.get_constraint_upper_bounds(), (std::vector<double>{1, 1, 1, 1, 1, 1}));

  if (GetParam()) {
    parsed.set_row_types({});
    parsed.set_constraint_bounds({});
  }
  raft::handle_t handle;
  auto model     = mps_data_model_to_optimization_problem(&handle, parsed);
  auto converted = cuopt_optimization_problem_to_user_problem(&handle, model);
  // Check the actual barrier input without solving: each equality has a unit-width range.
  EXPECT_EQ(converted.num_rows, 6);
  EXPECT_EQ(converted.row_sense, (std::vector<char>{'E', 'E', 'E', 'E', 'E', 'E'}));
  EXPECT_EQ(converted.rhs, (std::vector<double>{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(converted.num_range_rows, 6);
  EXPECT_EQ(converted.range_rows, (std::vector<int>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(converted.range_value, (std::vector<double>{1, 1, 1, 1, 1, 1}));
}

INSTANTIATE_TEST_SUITE_P(BoundsRepresentations,
                         MpsRangedRowsTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "BoundsOnly" : "WithRowSenses";
                         });

}  // namespace cuopt::mathematical_optimization
