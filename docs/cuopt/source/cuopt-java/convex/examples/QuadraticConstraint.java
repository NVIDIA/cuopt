/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class QuadraticConstraint {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("quadratic-constraint")) {
      Variable x = problem.addVariable(0.0, 10.0, 1.0, VariableType.CONTINUOUS, "x");
      Variable y = problem.addVariable(0.0, 10.0, 1.0, VariableType.CONTINUOUS, "y");

      // cuOptCreateProblem currently requires at least one linear constraint row.
      problem.addConstraint(LinearExpression.of(x).plus(y).le(15.0), "budget");

      QuadraticExpression radius = QuadraticExpression
          .of(x, x, 1.0)
          .plus(y, y, 1.0);
      problem.addConstraint(radius.le(4.0), "radius");
      problem.setObjective(
          LinearExpression.of(x).plus(y), ObjectiveSense.MAXIMIZE);

      try (Solution solution = problem.solve()) {
        System.out.println(solution.getTerminationStatus());
      }
    }
  }
}
