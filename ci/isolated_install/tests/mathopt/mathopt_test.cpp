// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// libcuopt-mathopt (+ client) only: solve a tiny LP and MILP through the C API.
//   min -x - y   s.t.  x + y <= 3.5,  0 <= x, y <= 3
#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/cuopt_c.h>

#include <gtest/gtest.h>

static double solve(char var_type)
{
  cuopt_int_t row_offsets[] = {0, 2};
  cuopt_int_t col_indices[] = {0, 1};
  cuopt_float_t values[]    = {1.0, 1.0};
  cuopt_float_t objective[] = {-1.0, -1.0};
  cuopt_float_t rhs[]       = {3.5};
  cuopt_float_t lower[]     = {0.0, 0.0};
  cuopt_float_t upper[]     = {3.0, 3.0};
  char constraint_sense[]   = {CUOPT_LESS_THAN};
  char variable_types[]     = {var_type, var_type};

  cuOptOptimizationProblem problem = nullptr;
  cuOptSolverSettings settings     = nullptr;
  cuOptSolution solution           = nullptr;
  EXPECT_EQ(cuOptCreateProblem(1,
                               2,
                               CUOPT_MINIMIZE,
                               0.0,
                               objective,
                               row_offsets,
                               col_indices,
                               values,
                               constraint_sense,
                               rhs,
                               lower,
                               upper,
                               variable_types,
                               &problem),
            CUOPT_SUCCESS);
  EXPECT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  EXPECT_EQ(cuOptSolve(problem, settings, &solution), CUOPT_SUCCESS);

  cuopt_int_t status = -1;
  EXPECT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  cuopt_float_t obj = 0;
  EXPECT_EQ(cuOptGetObjectiveValue(solution, &obj), CUOPT_SUCCESS);

  cuOptDestroySolution(&solution);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
  return obj;
}

TEST(IsolatedMathopt, SolvesLp) { EXPECT_NEAR(solve(CUOPT_CONTINUOUS), -3.5, 1e-2); }

TEST(IsolatedMathopt, SolvesMilp) { EXPECT_NEAR(solve(CUOPT_INTEGER), -3.0, 1e-6); }
