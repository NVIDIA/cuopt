/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.optimization;

/**
 * Coarse classification of an optimization problem. Returned by
 * {@code Problem.problemCategory()}.
 */
public enum ProblemCategory {
    /** Linear program: linear objective, linear constraints, all continuous variables. */
    LP,
    /** Mixed-integer (linear) program: at least one integer/binary variable. */
    MIP,
    /** Quadratic program: quadratic objective, linear constraints. */
    QP
}
