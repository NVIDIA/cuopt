/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class SimpleLp {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple-lp")) {
      Variable x = problem.addVariable(
          0.0, Double.POSITIVE_INFINITY, 1.0,
          VariableType.CONTINUOUS, "x");
      Variable y = problem.addVariable(
          0.0, Double.POSITIVE_INFINITY, 1.0,
          VariableType.CONTINUOUS, "y");

      problem.addConstraint(
          LinearExpression.of(x).plus(y).ge(10.0), "demand");
      problem.setObjective(
          LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);

      try (SolverSettings settings = new SolverSettings()
               .setMethod(SolverMethod.PDLP);
           Solution solution = problem.solve(settings)) {
        System.out.println("Status: " + solution.getTerminationStatus());
        System.out.println("x = " + x.getValue());
        System.out.println("y = " + y.getValue());
        System.out.println("Objective = " + solution.getPrimalObjective());
      }
    }
  }
}
