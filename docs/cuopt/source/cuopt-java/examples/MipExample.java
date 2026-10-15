/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class MipExample {
  public static void main(String[] args) throws Exception {
    try (Problem problem = new Problem("integer")) {
      Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
      problem.addConstraint(LinearExpression.of(x).ge(1.0));

      try (SolverSettings settings = new SolverSettings()
               .setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
           Solution solution = problem.solve(settings)) {
        System.out.println(solution.getMIPGap());
        System.out.println(solution.getSolutionBound());
      }
    }
  }
}
