/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

/** Common API for linear and quadratic objective expressions. */
public interface ObjectiveExpression {
  /** Return the linear portion of this expression. */
  LinearExpression getLinearExpression();

  double getConstant();

  double getValue();

  boolean isQuadratic();
}
