/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt;

import com.nvidia.cuopt.linear_programming.DataModel;
import com.nvidia.cuopt.linear_programming.Problem;
import com.nvidia.cuopt.linear_programming.SolverSettings;
import com.nvidia.cuopt.spi.CuOptProvider;
import java.lang.reflect.Method;

/**
 * Static utility entry points for cuopt-java.
 *
 * <p>Most users call {@link Problem#solve()} or
 * {@link Problem#solve(SolverSettings)} on a Problem directly. The
 * static {@code Solver.solve(...)} overloads here are convenience
 * wrappers that match cuOpt Python's {@code Solver.Solve(dm, settings)}
 * call shape.
 */
public final class Solver {

    private Solver() {}

    /**
     * Returns the cuOpt library version (from {@code libcuopt.so}) as a
     * human-readable {@code major.minor.patch} string.
     *
     * @throws CuOptException if the native library cannot be loaded.
     */
    public static String getVersion() {
        return CuOptProvider.instance().getVersion();
    }

    // ── solve overloads ──────────────────────────────────────────

    public static void solve(Problem problem) {
        problem.solve();
    }

    public static void solve(Problem problem, SolverSettings settings) {
        problem.solve(settings);
    }

    public static void solve(DataModel dm) {
        solve(dm, null);
    }

    public static void solve(DataModel dm, SolverSettings settings) {
        // Package-private call on DataModel — Solver lives in com.nvidia.cuopt,
        // DataModel lives in com.nvidia.cuopt.linear_programming, so we go
        // through reflection to access solveInternal. Replace with a public
        // SPI entry point or move Solver into the same package later.
        try {
            Method m = DataModel.class.getDeclaredMethod("solveInternal", SolverSettings.class);
            m.setAccessible(true);
            m.invoke(dm, settings);
        } catch (ReflectiveOperationException e) {
            throw new CuOptException("Internal: failed to dispatch DataModel.solveInternal", e);
        }
    }
}
