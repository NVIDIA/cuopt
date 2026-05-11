/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * Constraint sense for {@code Problem.addConstraint(...)}.
 *
 * <p>Maps to the C API constants {@code CUOPT_LESS_THAN} ({@code 'L'}),
 * {@code CUOPT_GREATER_THAN} ({@code 'G'}), and {@code CUOPT_EQUAL}
 * ({@code 'E'}).
 */
public enum CType {
    /** Less than or equal: {@code lhs <= rhs}. */
    LE('L'),
    /** Greater than or equal: {@code lhs >= rhs}. */
    GE('G'),
    /** Equal: {@code lhs == rhs}. */
    EQ('E');

    private final char code;

    CType(char code) {
        this.code = code;
    }

    /** Returns the C-API char encoding. */
    public char code() {
        return code;
    }
}
