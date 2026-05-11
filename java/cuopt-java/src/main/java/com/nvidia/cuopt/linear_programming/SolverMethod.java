/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * Solver method selection for LP problems.
 *
 * <p>Maps to the C API constants {@code CUOPT_METHOD_*}.
 */
public enum SolverMethod {
    CONCURRENT(0),
    PDLP(1),
    DUAL_SIMPLEX(2),
    BARRIER(3),
    UNSET(4);

    private final int code;

    SolverMethod(int code) {
        this.code = code;
    }

    /** Returns the C-API int encoding. */
    public int code() {
        return code;
    }

    /** Inverse of {@link #code()}. Throws {@link IllegalArgumentException} for unknown codes. */
    public static SolverMethod fromCode(int code) {
        for (SolverMethod m : values()) {
            if (m.code == code) {
                return m;
            }
        }
        throw new IllegalArgumentException("Unknown SolverMethod code: " + code);
    }
}
