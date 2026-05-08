/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt;

import com.nvidia.cuopt.spi.CuOptProvider;

/**
 * Entry point for cuopt-java.
 *
 * <p>This class currently exposes only {@link #getVersion()} as a
 * minimal end-to-end demonstration of the FFM bridge. The full LP /
 * MILP / QP API ({@code Problem}, {@code Variable}, {@code LinearExpr},
 * etc.) lands in subsequent PRs.
 */
public final class Solver {

    private Solver() {}

    /**
     * Returns the cuOpt library version (from {@code libcuopt.so}) as a
     * human-readable string.
     *
     * @throws CuOptException if the native library cannot be loaded or
     *                        the FFM implementation is missing (e.g.,
     *                        running on Java 21).
     */
    public static String getVersion() {
        return CuOptProvider.instance().getVersion();
    }
}
