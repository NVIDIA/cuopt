/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class SimpleMip {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple-milp")) {
      Variable x = problem.addVariable(
          0.0, 100.0, 3.0, VariableType.INTEGER, "x");
      Variable y = problem.addVariable(
          0.0, 100.0, 5.0, VariableType.INTEGER, "y");

      problem.addConstraint(
          LinearExpression.of(x).times(2.0).plus(y).le(8.0), "capacity");
      problem.setObjective(
          LinearExpression.of(x).times(3.0).plus(y, 5.0),
          ObjectiveSense.MAXIMIZE);

      try (SolverSettings settings = new SolverSettings()
               .setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
           Solution solution = problem.solve(settings)) {
        System.out.println("Status: " + solution.getTerminationStatus());
        System.out.println("x = " + x.getValue());
        System.out.println("y = " + y.getValue());
        System.out.println("Objective = " + solution.getPrimalObjective());
        System.out.println("MIP gap = " + solution.getMIPGap());
        System.out.println("Bound = " + solution.getSolutionBound());
      }
    }
  }
}
