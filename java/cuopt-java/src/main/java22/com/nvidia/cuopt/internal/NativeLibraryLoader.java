/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.internal;

import com.nvidia.cuopt.cuOptException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.CodeSource;
import java.util.Locale;
import java.util.Properties;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

/**
 * Loads {@code libcuopt.so} (and its bundled RAPIDS dependencies, when
 * present) on first use.
 *
 * <p>Two modes, chosen automatically by inspecting the JAR manifest:
 *
 * <ol>
 *   <li><b>Embedded mode</b> — the JAR carries the manifest entry
 *       {@code Embedded-Libraries-Cuda-Version}. Native libs are
 *       extracted from {@code <arch>/<os>/} entries in the JAR to a
 *       temporary directory and {@code System.load}-ed in dependency
 *       order. Used by classifier JARs
 *       (e.g. {@code cuopt-java-<version>-x86_64-cuda13.jar}) which bundle
 *       {@code libcuopt.so}, {@code libmps_parser.so}, {@code librmm.so},
 *       and {@code librapids_logger.so}.</li>
 *   <li><b>System (BYO) mode</b> — the JAR has no
 *       {@code Embedded-Libraries-Cuda-Version} entry. Falls back to
 *       {@code System.loadLibrary("cuopt")}, requiring the user to have
 *       {@code libcuopt.so} on {@code java.library.path} (typical conda
 *       install).</li>
 * </ol>
 *
 * <p>The platform table at
 * {@code META-INF/cuopt/supported-platforms.properties} controls which
 * OS+arch combinations are accepted.
 */
public final class NativeLibraryLoader {

    private static final String PLATFORM_PROPS =
        "/META-INF/cuopt/supported-platforms.properties";

    private static final String EMBEDDED_MARKER = "Embedded-Libraries-Cuda-Version";

    /**
     * Libraries bundled in classifier JARs, in load order. {@code libcuopt}
     * dlopen-resolves symbols from {@code librmm} and {@code librapids_logger}
     * at load time, so those must be present in memory before
     * {@code libcuopt} is loaded.
     */
    private static final String[] EMBEDDED_LIBS = {
        "rapids_logger", "rmm", "mps_parser", "cuopt",
    };

    private static volatile boolean loaded = false;

    private NativeLibraryLoader() {}

    /**
     * Idempotent load of {@code libcuopt} and its bundled deps (if any).
     * Safe to call from multiple threads.
     */
    public static synchronized void ensureLoaded() {
        if (loaded) {
            return;
        }
        verifyPlatformSupported();
        if (jarHasEmbeddedLibraries()) {
            loadEmbedded();
        } else {
            loadSystem();
        }
        loaded = true;
    }

    private static void loadSystem() {
        try {
            System.loadLibrary("cuopt");
        } catch (UnsatisfiedLinkError e) {
            throw new cuOptException(
                "Failed to load libcuopt. Either add the directory containing "
                + "libcuopt.so to java.library.path (typical conda install), "
                + "or use a classifier JAR that bundles the native library "
                + "(e.g. cuopt-java-<version>-x86_64-cuda13.jar).",
                e);
        }
    }

    private static void loadEmbedded() {
        String os = System.getProperty("os.name", "");
        String arch = System.getProperty("os.arch", "");
        for (String name : EMBEDDED_LIBS) {
            String libFile = System.mapLibraryName(name);
            String resourcePath = "/" + arch + "/" + os + "/" + libFile;
            try {
                Path extracted = extractResource(resourcePath, libFile);
                System.load(extracted.toAbsolutePath().toString());
            } catch (Throwable t) {
                throw new cuOptException(
                    "Failed to load embedded native dependency: " + libFile
                    + " from resource path " + resourcePath, t);
            }
        }
    }

    private static Path extractResource(String resourcePath, String libFile)
            throws IOException {
        URL url = NativeLibraryLoader.class.getResource(resourcePath);
        if (url == null) {
            throw new IOException(
                "Embedded native library not found at " + resourcePath);
        }
        Path tmp = Files.createTempFile("cuopt-native-", "-" + libFile);
        tmp.toFile().deleteOnExit();
        try (InputStream in = url.openStream()) {
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
        }
        return tmp;
    }

    private static boolean jarHasEmbeddedLibraries() {
        // Locate the JAR that this class was loaded from, not any other
        // META-INF/MANIFEST.MF that happens to be on the classpath.
        CodeSource codeSource = NativeLibraryLoader.class.getProtectionDomain().getCodeSource();
        if (codeSource == null || codeSource.getLocation() == null) {
            return false;
        }
        String path = codeSource.getLocation().getPath();
        if (path == null || !path.endsWith(".jar")) {
            // Running from exploded classes (e.g. surefire on target/classes/);
            // there is no JAR manifest to inspect.
            return false;
        }
        try (JarFile jar = new JarFile(path)) {
            Manifest manifest = jar.getManifest();
            if (manifest == null) {
                return false;
            }
            Attributes attrs = manifest.getMainAttributes();
            return attrs.getValue(EMBEDDED_MARKER) != null;
        } catch (IOException e) {
            return false;
        }
    }

    private static void verifyPlatformSupported() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String key = osKey(os) + "-" + archKey(arch);

        Properties props = new Properties();
        try (InputStream in = NativeLibraryLoader.class.getResourceAsStream(PLATFORM_PROPS)) {
            if (in == null) {
                throw new cuOptException(
                    "Internal error: " + PLATFORM_PROPS + " not found in JAR.");
            }
            props.load(in);
        } catch (IOException e) {
            throw new cuOptException("Failed to read " + PLATFORM_PROPS, e);
        }

        if (!"true".equalsIgnoreCase(props.getProperty(key))) {
            throw new cuOptException(
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
