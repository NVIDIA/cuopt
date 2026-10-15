/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * The resource path has to agree with the layout the packaging step writes, and neither side can
 * see the other, so pin it here.
 */
final class NativeLibraryLoaderTest {
  @Test
  void mapsJvmArchitecturesToThePackagedResourcePath() {
    // os.arch reports amd64 on an x86_64 JVM, but the build host reports x86_64; both appear.
    assertEquals(
        "/amd64/Linux/libcuopt_jni.so",
        NativeLibraryLoader.resourcePath("amd64", "libcuopt_jni.so"));
    assertEquals(
        "/amd64/Linux/libcuopt_jni.so",
        NativeLibraryLoader.resourcePath("x86_64", "libcuopt_jni.so"));
    assertEquals(
        "/aarch64/Linux/libcuopt_jni.so",
        NativeLibraryLoader.resourcePath("aarch64", "libcuopt_jni.so"));
    assertEquals(
        "/aarch64/Linux/libcuopt_jni.so",
        NativeLibraryLoader.resourcePath("arm64", "libcuopt_jni.so"));
  }

  @Test
  void rejectsAnArchitectureCuOptDoesNotPublish() {
    IllegalStateException error =
        assertThrows(
            IllegalStateException.class,
            () -> NativeLibraryLoader.resourcePath("ppc64le", "libcuopt_jni.so"));
    assertTrue(error.getMessage().contains("ppc64le"));
  }

  @Test
  void theLibraryThisSuiteRunsAgainstIsLoadable() {
    // Reaching any native method proves whichever strategy applied here resolved the library:
    // cuopt.native.dir for a source build, the embedded copy for a classifier JAR.
    assertEquals(8, NativeCuOpt.getFloatSize());
  }
}
