/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

@FunctionalInterface
public interface MIPSetSolutionCallback extends MIPCallback {
  MIPCallbackSolution getSolution(double solutionBound, Object userData);
}
