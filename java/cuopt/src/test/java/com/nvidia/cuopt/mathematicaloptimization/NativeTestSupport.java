/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;

final class NativeTestSupport {
  private NativeTestSupport() {}

  /**
   * Skips when there is no native library to load, by either route the loader accepts: a
   * directory named by {@code cuopt.native.dir} for a source build, or a copy embedded in a
   * classifier JAR on the classpath. Requiring the property alone would silently skip the whole
   * native suite when it runs against a JAR, which is where it matters most.
   */
  static void assumeNativeLibrary() {
    String fileName = System.mapLibraryName("cuopt_jni");
    String nativeDir = System.getProperty("cuopt.native.dir");
    if (nativeDir != null && !nativeDir.isBlank()) {
      Assumptions.assumeTrue(
          Files.exists(Path.of(nativeDir, fileName)), "libcuopt_jni is not built");
      return;
    }
    Assumptions.assumeTrue(
        embeddedLibraryPresent(fileName),
        "no libcuopt_jni: cuopt.native.dir is unset and no copy is embedded on the classpath");
  }

  private static boolean embeddedLibraryPresent(String fileName) {
    try {
      String resource =
          NativeLibraryLoader.resourcePath(System.getProperty("os.arch", ""), fileName);
      return NativeTestSupport.class.getResource(resource) != null;
    } catch (IllegalStateException unsupportedArchitecture) {
      return false;
    }
  }

  private static final long NVIDIA_SMI_TIMEOUT_SECONDS = 30;

  static void assumeCudaDriverAvailable() {
    Process process = null;
    // The assumptions are made after the try block on purpose. Assumptions.assumeTrue signals a
    // skip by throwing TestAbortedException, which a catch here would swallow and re-report under
    // the wrong reason.
    boolean exited = false;
    int exitCode = -1;
    try {
      // Discard the output rather than leaving it in the pipe: nothing reads it, and a full
      // pipe buffer would block nvidia-smi instead of letting it exit.
      process =
          new ProcessBuilder("nvidia-smi")
              .redirectErrorStream(true)
              .redirectOutput(ProcessBuilder.Redirect.DISCARD)
              .start();
      // A wedged driver makes nvidia-smi hang indefinitely, which would hang the whole suite.
      exited = process.waitFor(NVIDIA_SMI_TIMEOUT_SECONDS, TimeUnit.SECONDS);
      if (exited) {
        exitCode = process.exitValue();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      Assumptions.abort("CUDA driver check was interrupted");
    } catch (IOException | SecurityException e) {
      Assumptions.abort("CUDA driver check failed: " + e.getMessage());
    } finally {
      if (process != null && process.isAlive()) {
        process.destroyForcibly();
      }
    }

    Assumptions.assumeTrue(
        exited, "CUDA driver check timed out after " + NVIDIA_SMI_TIMEOUT_SECONDS + "s");
    Assumptions.assumeTrue(exitCode == 0, "CUDA driver is unavailable");
  }
}
