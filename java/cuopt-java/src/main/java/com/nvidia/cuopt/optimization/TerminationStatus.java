/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.optimization;

/**
 * Solver termination status. Returned by {@code Problem.status()} after
 * {@code Problem.solve(...)}.
 *
 * <p>Maps to the C API constants {@code CUOPT_TERMINATION_STATUS_*}.
 */
public enum TerminationStatus {
    NO_TERMINATION(0),
    OPTIMAL(1),
    INFEASIBLE(2),
    UNBOUNDED(3),
    ITERATION_LIMIT(4),
    TIME_LIMIT(5),
    NUMERICAL_ERROR(6),
    PRIMAL_FEASIBLE(7),
    FEASIBLE_FOUND(8),
    CONCURRENT_LIMIT(9),
    WORK_LIMIT(10),
    UNBOUNDED_OR_INFEASIBLE(11);

    private final int code;

    TerminationStatus(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static TerminationStatus fromCode(int code) {
        for (TerminationStatus s : values()) {
            if (s.code == code) {
                return s;
            }
        }
        throw new IllegalArgumentException("Unknown TerminationStatus code: " + code);
    }
}
