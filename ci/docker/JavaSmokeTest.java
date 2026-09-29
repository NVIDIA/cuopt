// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import com.nvidia.cuopt.mathematicaloptimization.LinearExpression;
import com.nvidia.cuopt.mathematicaloptimization.ObjectiveSense;
import com.nvidia.cuopt.mathematicaloptimization.Problem;
import com.nvidia.cuopt.mathematicaloptimization.Solution;
import com.nvidia.cuopt.mathematicaloptimization.TerminationStatus;
import com.nvidia.cuopt.mathematicaloptimization.Variable;
import com.nvidia.cuopt.mathematicaloptimization.VariableType;

// Solves minimize x0 + x1 s.t. x0 + x1 >= 1, x0,x1 >= 0 (optimal objective 1.0) against the
// cuopt.jar + libcuopt_jni.so shipped in this image, to catch packaging regressions a native
// library loading check alone would miss.
public class JavaSmokeTest {
  public static void main(String[] args) throws Exception {
    Problem problem = new Problem("docker-java-smoke");
    Variable x0 = problem.addVariable(0.0, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x0");
    Variable x1 = problem.addVariable(0.0, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x1");
    problem.addConstraint(LinearExpression.of(x0).plus(x1).ge(1.0), "c0");
    problem.setObjective(LinearExpression.of(x0).plus(x1), ObjectiveSense.MINIMIZE);

    try (Solution solution = problem.solve()) {
      TerminationStatus status = solution.getTerminationStatus();
      double objective = solution.getPrimalObjective();
      if (status != TerminationStatus.OPTIMAL) {
        throw new AssertionError("expected OPTIMAL, got " + status);
      }
      if (Math.abs(objective - 1.0) > 1e-6) {
        throw new AssertionError("expected objective 1.0, got " + objective);
      }
      System.out.println("JavaSmokeTest OK: status=" + status + " objective=" + objective);
    }
  }
}
