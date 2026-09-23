/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.util.Arrays;

public final class MIPCallbackSolution {
  final double[] solution;
  final double objectiveValue;

  public MIPCallbackSolution(double[] solution, double objectiveValue) {
    this.solution = Arrays.copyOf(solution, solution.length);
    this.objectiveValue = objectiveValue;
  }

  public double[] getSolution() {
    return Arrays.copyOf(solution, solution.length);
  }

  public double getObjectiveValue() {
    return objectiveValue;
  }
}
