/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.optimization;

/**
 * Objective sense for {@code Problem.setObjective(...)}.
 *
 * <p>Maps to the C API constants {@code CUOPT_MINIMIZE} (1) and
 * {@code CUOPT_MAXIMIZE} (-1).
 */
public enum Sense {
    MINIMIZE(1),
    MAXIMIZE(-1);

    private final int code;

    Sense(int code) {
        this.code = code;
    }

    /** Returns the C-API int encoding (1 for MINIMIZE, -1 for MAXIMIZE). */
    public int code() {
        return code;
    }
}
