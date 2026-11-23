/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class SemiContinuous {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("semi-continuous")) {
      Variable production = problem.addVariable(
          10.0, 100.0, 1.0,
          VariableType.SEMI_CONTINUOUS, "production");

      // cuOptCreateProblem currently requires at least one linear constraint row.
      problem.addConstraint(LinearExpression.of(production).le(1000.0), "capacity");

      problem.setObjective(production, ObjectiveSense.MINIMIZE);

      try (Solution solution = problem.solve()) {
        System.out.println("production = " + production.getValue());
      }
    }
  }
}
