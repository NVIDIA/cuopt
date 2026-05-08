/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.spi;

import com.nvidia.cuopt.CuOptException;
import java.util.ServiceLoader;

/**
 * Service provider interface bridging the Java 21 public API to the
 * Java 22 FFM implementation.
 *
 * <p>This interface compiles to Java 21 release. Method signatures use
 * only Java 21-compatible types (no {@code MemorySegment}, no
 * {@code Arena}). The implementation lives in
 * {@code com.nvidia.cuopt.internal.CuOptProviderImpl} under
 * {@code src/main/java22/}; the JVM resolves it via
 * {@link java.util.ServiceLoader} on Java 22+ runtimes.
 *
 * <p>This interface is sealed — third parties cannot provide alternative
 * implementations.
 */
public interface CuOptProvider {

    /**
     * Returns the cuOpt library version as a human-readable string in
     * {@code major.minor.patch} format.
     */
    String getVersion();

    /**
     * Returns the singleton implementation, resolved via
     * {@link ServiceLoader}.
     *
     * @throws CuOptException if no implementation is registered
     *                        (e.g., running on Java 21 — the FFM
     *                        impl lives in the Java 22 multi-release
     *                        layer and is not visible to Java 21 JVMs).
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
