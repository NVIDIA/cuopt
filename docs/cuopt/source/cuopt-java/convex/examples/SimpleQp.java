/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class SimpleQp {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple-qp")) {
      Variable x = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "x");
      Variable y = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "y");

      QuadraticExpression objective = QuadraticExpression
          .of(x, x, 1.0)
          .plus(y, y, 1.0)
          .plus(LinearExpression.of(x).times(-1.0))
          .plus(LinearExpression.of(y).times(-1.0));

      problem.addConstraint(
          LinearExpression.of(x).plus(y).eq(1.0), "sum");
      problem.setObjective(objective, ObjectiveSense.MINIMIZE);

      try (Solution solution = problem.solve()) {
        System.out.println("x = " + x.getValue());
        System.out.println("y = " + y.getValue());
        System.out.println("Objective = " + solution.getPrimalObjective());
      }
    }
  }
}
