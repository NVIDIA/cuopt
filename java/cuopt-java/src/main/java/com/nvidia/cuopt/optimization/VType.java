/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.optimization;

/**
 * Variable type for {@code Problem.addVariable(...)}.
 *
 * <p>The C API only distinguishes {@code CONTINUOUS} (char {@code 'C'})
 * and {@code INTEGER} (char {@code 'I'}). {@code BINARY} is exposed as
 * a convenience — it's equivalent to {@code INTEGER} with bounds
 * {@code [0, 1]}, and is marshalled to the C side as {@code 'I'}.
 */
public enum VType {
    CONTINUOUS('C'),
    INTEGER('I'),
    BINARY('I');

    private final char code;

    VType(char code) {
        this.code = code;
    }

    /** Returns the C-API char encoding ({@code 'C'} or {@code 'I'}). */
    public char code() {
        return code;
    }
}
