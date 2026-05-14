/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.optimization;

/**
 * Solver error status, distinct from {@link TerminationStatus}.
 * {@code SUCCESS} indicates the solver ran without errors (regardless of
 * whether it found an optimal solution).
 *
 * <p>Maps to the C API status codes (return values from cuOpt functions).
 */
public enum ErrorStatus {
    SUCCESS(0),
    INVALID_ARGUMENT(1),
    MPS_FILE_ERROR(2),
    MPS_PARSE_ERROR(3),
    VALIDATION_ERROR(4),
    OUT_OF_MEMORY(5),
    RUNTIME_ERROR(6);

    private final int code;

    ErrorStatus(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static ErrorStatus fromCode(int code) {
        for (ErrorStatus s : values()) {
            if (s.code == code) {
                return s;
            }
        }
        // Unknown code — fall back to RUNTIME_ERROR rather than throwing,
        // since this is on the error-reporting path.
        return RUNTIME_ERROR;
    }
}
