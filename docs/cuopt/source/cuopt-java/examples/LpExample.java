/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class LpExample {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("simple")) {
      Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
          VariableType.CONTINUOUS, "x");
      Variable y = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
          VariableType.CONTINUOUS, "y");

      problem.addConstraint(LinearExpression.of(x).plus(y).ge(1.0), "c0");
      problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);

      try (SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
           Solution solution = problem.solve(settings)) {
        System.out.println(solution.getTerminationStatus());
        System.out.println(solution.getPrimalObjective());
      }
    }
  }
}
