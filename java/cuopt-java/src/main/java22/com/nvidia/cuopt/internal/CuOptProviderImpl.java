/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.internal;

import com.nvidia.cuopt.CuOptException;
import com.nvidia.cuopt.internal.panama.cuopt_c_h;
import com.nvidia.cuopt.spi.CuOptProvider;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * FFM-based implementation of {@link CuOptProvider}.
 *
 * <p>Registered via {@code META-INF/services/com.nvidia.cuopt.spi.CuOptProvider}.
 * Lives in the Java 22 multi-release layer of the JAR; loaded only on
 * JVMs at Java 22 or higher.
 */
public final class CuOptProviderImpl implements CuOptProvider {

    /**
     * No-arg constructor required for {@link java.util.ServiceLoader}.
     * Eagerly loads the native library so a misconfigured runtime fails
     * fast at provider creation rather than on the first call.
     */
    public CuOptProviderImpl() {
        NativeLibraryLoader.ensureLoaded();
    }

    @Override
    public String getVersion() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment major = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment minor = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment patch = arena.allocate(ValueLayout.JAVA_INT);

            int rc = cuopt_c_h.cuOptGetVersion(major, minor, patch);
            if (rc != 0) {
                throw new CuOptException(
                    "cuOptGetVersion returned non-zero status: " + rc);
            }
            return major.get(ValueLayout.JAVA_INT, 0)
                + "." + minor.get(ValueLayout.JAVA_INT, 0)
                + "." + patch.get(ValueLayout.JAVA_INT, 0);
        }
    }
}
