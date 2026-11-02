/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class LpRelaxation {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple-milp")) {
      Variable x = problem.addVariable(0.0, 100.0, 3.0, VariableType.INTEGER, "x");
      Variable y = problem.addVariable(0.0, 100.0, 5.0, VariableType.INTEGER, "y");
      problem.addConstraint(LinearExpression.of(x).times(2.0).plus(y).le(8.0), "capacity");
      problem.setObjective(
          LinearExpression.of(x).times(3.0).plus(y, 5.0), ObjectiveSense.MAXIMIZE);

      // start-relax
      for (Variable variable : problem.getVariables()) {
        variable.setVariableType(VariableType.CONTINUOUS);
      }
      try (Solution solution = problem.solve()) {
        System.out.println("LP relaxation objective = " + solution.getPrimalObjective());
      }
      // end-relax
    }
  }
}
