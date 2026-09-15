/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URL;
import org.junit.jupiter.api.Test;

/**
 * Confirms the suite is exercising a packaged classifier JAR rather than classes built from
 * source.
 *
 * <p>Excluded by default and run only under {@code -Ppackaged-jar-tests}. Without it a stray
 * {@code target/classes} on the classpath would shadow the JAR and the run would pass while
 * testing the wrong thing entirely — which is the failure this whole job exists to rule out.
 */
final class PackagedJarOriginCheck {
  @Test
  void classesComeFromAJarRatherThanADirectory() {
    URL location = Problem.class.getProtectionDomain().getCodeSource().getLocation();
    assertNotNull(location, "no code source for Problem");
    String path = location.getPath();
    assertTrue(
        path.endsWith(".jar"),
        "expected Problem to be loaded from a packaged JAR, but it came from " + path);
  }

  @Test
  void theNativeLibraryIsEmbeddedInThatJar() {
    String resource =
        NativeLibraryLoader.resourcePath(
            System.getProperty("os.arch", ""), System.mapLibraryName("cuopt_jni"));
    URL embedded = PackagedJarOriginCheck.class.getResource(resource);
    assertNotNull(embedded, "no " + resource + " on the classpath");
    assertTrue(
        "jar".equals(embedded.getProtocol()),
        "expected the native library to come from a JAR, but it came from " + embedded);
  }

  @Test
  void noNativeDirectoryOverrideIsInEffect() {
    String nativeDir = System.getProperty("cuopt.native.dir");
    assertTrue(
        nativeDir == null || nativeDir.isBlank(),
        "cuopt.native.dir is set to '" + nativeDir + "', so the JAR's own library was bypassed");
  }
}
