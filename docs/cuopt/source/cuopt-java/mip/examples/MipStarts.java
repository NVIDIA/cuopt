/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class MipStarts {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple-milp")) {
      Variable x = problem.addVariable(0.0, 100.0, 3.0, VariableType.INTEGER, "x");
      Variable y = problem.addVariable(0.0, 100.0, 5.0, VariableType.INTEGER, "y");
      problem.addConstraint(LinearExpression.of(x).times(2.0).plus(y).le(8.0), "capacity");
      problem.setObjective(
          LinearExpression.of(x).times(3.0).plus(y, 5.0), ObjectiveSense.MAXIMIZE);

      // start-per-variable
      x.setMIPStart(3.0);
      y.setMIPStart(2.0);

      try (SolverSettings settings = new SolverSettings();
           Solution solution = problem.solve(settings)) {
        System.out.println(solution.getPrimalObjective());
      }
      // end-per-variable

      // start-array
      try (SolverSettings settings = new SolverSettings()) {
        double[] values = new double[problem.getNumVariables()];
        for (Variable variable : problem.getVariables()) {
          values[variable.getIndex()] = variable.getMIPStart();
        }
        settings.addMIPStart(values);

        try (Solution solution = problem.solve(settings)) {
          System.out.println(solution.getPrimalObjective());
        }
      }
      // end-array
    }
  }
}
