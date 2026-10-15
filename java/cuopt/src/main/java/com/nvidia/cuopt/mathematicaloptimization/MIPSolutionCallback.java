/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

@FunctionalInterface
public interface MIPSolutionCallback extends MIPCallback {
  void onSolution(double[] solution, double objectiveValue, double solutionBound, Object userData);
}
