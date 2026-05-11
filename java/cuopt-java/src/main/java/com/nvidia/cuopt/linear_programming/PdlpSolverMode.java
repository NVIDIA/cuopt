/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * PDLP-specific solver mode tuning. Set via
 * {@code SolverSettings.setParameter(CuOpt.PDLP_SOLVER_MODE, mode)}.
 *
 * <p>Maps to the C API constants {@code CUOPT_PDLP_SOLVER_MODE_*}.
 */
public enum PdlpSolverMode {
    STABLE1(0),
    STABLE2(1),
    METHODICAL1(2),
    FAST1(3),
    STABLE3(4);

    private final int code;

    PdlpSolverMode(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static PdlpSolverMode fromCode(int code) {
        for (PdlpSolverMode m : values()) {
            if (m.code == code) {
                return m;
            }
        }
        throw new IllegalArgumentException("Unknown PdlpSolverMode code: " + code);
    }
}
