/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.internal;

import com.nvidia.cuopt.CuOptException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Properties;

/**
 * Loads {@code libcuopt.so} (or the platform equivalent) on first use.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>If {@code libcuopt} is on {@code java.library.path}, use it
 *       (typical conda / system install).</li>
 *   <li>Otherwise, look for an embedded copy at
 *       {@code META-INF/native/<os-arch>/libcuopt.so} inside the JAR
 *       (classifier-JAR install path; not yet implemented in this
 *       skeleton).</li>
 * </ol>
 *
 * <p>The platform table at
 * {@code META-INF/cuopt/supported-platforms.properties} controls which
 * OS+arch combinations are accepted — adding arm64 support is a
 * properties-file edit, not a code change.
 */
final class NativeLibraryLoader {

    private static final String PLATFORM_PROPS =
        "/META-INF/cuopt/supported-platforms.properties";

    private static volatile boolean loaded = false;

    private NativeLibraryLoader() {}

    /**
     * Idempotent load of {@code libcuopt}. Safe to call from multiple threads.
     */
    static synchronized void ensureLoaded() {
        if (loaded) {
            return;
        }
        verifyPlatformSupported();
        try {
            // Prefer java.library.path (conda / system install).
            System.loadLibrary("cuopt");
        } catch (UnsatisfiedLinkError e) {
            // TODO: extract from META-INF/native/<os-arch>/libcuopt.so once
            // classifier JARs are wired up. For now, surface a clear error.
            throw new CuOptException(
                "Failed to load libcuopt. Either add the directory containing "
                + "libcuopt.so (or .dll/.dylib) to java.library.path, or rely "
                + "on a future classifier JAR that bundles the native library.",
                e);
        }
        loaded = true;
    }

    private static void verifyPlatformSupported() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String key = osKey(os) + "-" + archKey(arch);

        Properties props = new Properties();
        try (InputStream in = NativeLibraryLoader.class.getResourceAsStream(PLATFORM_PROPS)) {
            if (in == null) {
                throw new CuOptException(
                    "Internal error: " + PLATFORM_PROPS + " not found in JAR.");
            }
            props.load(in);
        } catch (IOException e) {
            throw new CuOptException("Failed to read " + PLATFORM_PROPS, e);
        }

        if (!"true".equalsIgnoreCase(props.getProperty(key))) {
            throw new CuOptException(
                "Unsupported platform: " + key + ". Supported platforms: "
                + props.stringPropertyNames());
        }
    }

    private static String osKey(String os) {
        if (os.contains("linux")) return "linux";
        if (os.contains("mac") || os.contains("darwin")) return "macos";
        if (os.contains("win")) return "windows";
        return os;
    }

    private static String archKey(String arch) {
        if (arch.equals("amd64") || arch.equals("x86_64")) return "amd64";
        if (arch.equals("aarch64") || arch.equals("arm64")) return "aarch64";
        return arch;
    }
}
