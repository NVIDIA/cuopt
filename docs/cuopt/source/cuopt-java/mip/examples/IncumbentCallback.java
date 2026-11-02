/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class IncumbentCallback {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple-milp")) {
      Variable x = problem.addVariable(0.0, 100.0, 3.0, VariableType.INTEGER, "x");
      Variable y = problem.addVariable(0.0, 100.0, 5.0, VariableType.INTEGER, "y");
      problem.addConstraint(LinearExpression.of(x).times(2.0).plus(y).le(8.0), "capacity");
      problem.setObjective(
          LinearExpression.of(x).times(3.0).plus(y, 5.0), ObjectiveSense.MAXIMIZE);

      // start-basic-callback
      try (SolverSettings settings = new SolverSettings()) {
        settings.setMIPCallback(
            (incumbent, objective, bound, userData) -> {
              System.out.println(
                  "incumbent objective=" + objective + ", bound=" + bound);
            },
            null,
            problem.getNumVariables());

        try (Solution solution = problem.solve(settings)) {
          System.out.println("Final status: " + solution.getTerminationStatus());
        }
      }
      // end-basic-callback

      // start-from-incumbent
      try (SolverSettings settings = new SolverSettings()) {
        settings.setMIPCallback(
            (incumbent, objective, bound, userData) -> {
              double[] picked = Problem.fromIncumbent(incumbent, y, x);
              System.out.println("y=" + picked[0] + " x=" + picked[1]);
            },
            null,
            problem.getNumVariables());

        try (Solution solution = problem.solve(settings)) {
          System.out.println("Final status: " + solution.getTerminationStatus());
        }
      }
      // end-from-incumbent
    }
  }
}
