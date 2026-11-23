/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class QpQuickstart {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("quadratic")) {
      Variable x = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "x");
      Variable y = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "y");
      problem.addConstraint(LinearExpression.of(x).plus(y).ge(5.0));
      problem.setObjective(
          QuadraticExpression.of(x, x, 1.0).plus(y, y, 4.0),
          ObjectiveSense.MINIMIZE);
      try (Solution solution = problem.solve()) {
        System.out.println(solution.getPrimalObjective());
      }
    }
  }
}
