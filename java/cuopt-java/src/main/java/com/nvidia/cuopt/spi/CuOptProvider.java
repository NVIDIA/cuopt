/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.spi;

import com.nvidia.cuopt.CuOptException;
import com.nvidia.cuopt.linear_programming.DataModel;
import com.nvidia.cuopt.linear_programming.Problem;
import com.nvidia.cuopt.linear_programming.SolverSettings;
import java.util.ServiceLoader;

/**
 * Service provider interface bridging the Java 21 public API to the
 * Java 22 FFM implementation.
 *
 * <p>This interface compiles to Java 21 release. Method signatures use
 * only Java 21-compatible types (no {@code MemorySegment}, no
 * {@code Arena}) — native handles cross the boundary as opaque
 * {@code long} values (raw addresses).
 *
 * <p>The implementation lives in
 * {@code com.nvidia.cuopt.internal.CuOptProviderImpl} under
 * {@code src/main/java22/}; the JVM resolves it via
 * {@link java.util.ServiceLoader} on Java 22+ runtimes.
 */
public interface CuOptProvider {

    // ── library-level ────────────────────────────────────────────

    /**
     * Returns the cuOpt library version as a human-readable string in
     * {@code major.minor.patch} format.
     */
    String getVersion();

    // ── SolverSettings lifecycle (native cuOptSolverSettings) ────

    /** Creates a native solver-settings handle. Returns the raw pointer as {@code long}. */
    long createSolverSettings();

    /** Destroys a native solver-settings handle. Idempotent on already-destroyed handles. */
    void destroySolverSettings(long handle);

    void setSolverIntegerParameter(long handle, String name, long value);
    void setSolverFloatParameter(long handle, String name, double value);
    void setSolverStringParameter(long handle, String name, String value);

    // ── Solve entry points ───────────────────────────────────────

    /**
     * Solves the given Problem with the given settings (or default settings
     * if {@code null}). Builds the native problem handle from
     * {@code problem}'s collected variables / constraints / objective,
     * calls {@code cuOptSolve}, extracts all solution data into a
     * {@link SolveResult}, then frees the native solution and problem
     * handles before returning.
     */
    SolveResult solveProblem(Problem problem, SolverSettings settings);

    /**
     * Solves the given DataModel directly (low-level path). Builds the
     * native problem handle from the DataModel's CSR arrays.
     */
    SolveResult solveDataModel(DataModel dm, SolverSettings settings);

    // ── lookup ───────────────────────────────────────────────────

    /**
     * Returns the singleton implementation, resolved via
     * {@link ServiceLoader}.
     *
     * @throws CuOptException if no implementation is registered
     *                        (e.g., running on Java 21 — the FFM
     *                        impl lives in the Java 22 multi-release
     *                        layer of the JAR and is not visible to
     *                        Java 21 JVMs).
     */
    static CuOptProvider instance() {
        return Holder.INSTANCE;
    }

    final class Holder {
        private static final CuOptProvider INSTANCE = ServiceLoader
            .load(CuOptProvider.class)
            .findFirst()
            .orElseThrow(() -> new CuOptException(
                "No CuOptProvider implementation found. cuopt-java requires "
                + "Java 22 or higher at runtime; the FFM implementation lives "
                + "in the Java 22 multi-release layer of the JAR and is not "
                + "visible to Java 21 JVMs."));

        private Holder() {}
    }
}
