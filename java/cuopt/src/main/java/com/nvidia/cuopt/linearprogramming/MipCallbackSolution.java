/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.Arrays;

public final class MipCallbackSolution {
  final double[] solution;
  final double objectiveValue;

  public MipCallbackSolution(double[] solution, double objectiveValue) {
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
