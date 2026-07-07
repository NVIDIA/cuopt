/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

@FunctionalInterface
public interface MipSolutionCallback {
  void onSolution(double[] solution, double objectiveValue, double solutionBound, Object userData);
}
