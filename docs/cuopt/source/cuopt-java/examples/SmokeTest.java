/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class SmokeTest {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("smoke-test")) {
      Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
          VariableType.CONTINUOUS, "x");
      Variable y = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
          VariableType.CONTINUOUS, "y");
      problem.addConstraint(LinearExpression.of(x).plus(y).ge(1.0), "c0");
      problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);
      try (Solution solution = problem.solve()) {
        System.out.println(solution.getTerminationStatus());
        System.out.println(solution.getPrimalObjective());
      }
    }
  }
}
