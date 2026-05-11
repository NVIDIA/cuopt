/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * Coarse classification of an optimization problem. Returned by
 * {@code Problem.problemCategory()}.
 */
public enum ProblemCategory {
    /** Linear program: linear objective, linear constraints, all continuous variables. */
    LP,
    /** Integer program: linear objective and constraints, all variables integer. */
    IP,
    /** Mixed-integer (linear) program: at least one continuous and one integer variable. */
    MIP,
    /** Quadratic program: quadratic objective, linear constraints. */
    QP
}
