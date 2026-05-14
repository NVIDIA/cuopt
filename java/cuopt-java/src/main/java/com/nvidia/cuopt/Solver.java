/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt;

import com.nvidia.cuopt.optimization.Problem;
import com.nvidia.cuopt.optimization.SolverSettings;
import com.nvidia.cuopt.spi.cuOptProvider;

/**
 * Static utility entry points for cuopt-java.
 *
 * <p>Most users call {@link Problem#solve()} or
 * {@link Problem#solve(SolverSettings)} on a Problem directly. The
 * static {@code Solver.solve(...)} overloads here are convenience
 * wrappers.
 */
public final class Solver {

    private Solver() {}

    /**
     * Returns the cuOpt library version (from {@code libcuopt.so}) as a
     * human-readable {@code major.minor.patch} string.
     *
     * @throws cuOptException if the native library cannot be loaded.
     */
    public static String getVersion() {
        return cuOptProvider.instance().getVersion();
    }

    // ── solve overloads ──────────────────────────────────────────

    public static void solve(Problem problem) {
        problem.solve();
    }

    public static void solve(Problem problem, SolverSettings settings) {
        problem.solve(settings);
    }
}
