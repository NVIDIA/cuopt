/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

@FunctionalInterface
public interface MipSetSolutionCallback {
  MipCallbackSolution getSolution(double solutionBound, Object userData);
}
