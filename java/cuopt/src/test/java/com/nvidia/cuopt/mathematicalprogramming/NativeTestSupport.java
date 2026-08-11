/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;

final class NativeTestSupport {
  private NativeTestSupport() {}

  static void assumeNativeLibrary() {
    String nativeDir = System.getProperty("cuopt.native.dir");
    Assumptions.assumeTrue(nativeDir != null && !nativeDir.isBlank(), "cuopt.native.dir is unset");
    Assumptions.assumeTrue(
        Files.exists(Path.of(nativeDir, System.mapLibraryName("cuopt_jni"))),
        "libcuopt_jni is not built");
  }

  private static final long NVIDIA_SMI_TIMEOUT_SECONDS = 30;

  static void assumeCudaDriverAvailable() {
    Process process = null;
    try {
      // Discard the output rather than leaving it in the pipe: nothing reads it, and a full
      // pipe buffer would block nvidia-smi instead of letting it exit.
      process =
          new ProcessBuilder("nvidia-smi")
              .redirectErrorStream(true)
              .redirectOutput(ProcessBuilder.Redirect.DISCARD)
              .start();
      // A wedged driver makes nvidia-smi hang indefinitely, which would hang the whole suite.
      Assumptions.assumeTrue(
          process.waitFor(NVIDIA_SMI_TIMEOUT_SECONDS, TimeUnit.SECONDS),
          "CUDA driver check timed out after " + NVIDIA_SMI_TIMEOUT_SECONDS + "s");
      Assumptions.assumeTrue(process.exitValue() == 0, "CUDA driver is unavailable");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      Assumptions.assumeTrue(false, "CUDA driver check was interrupted");
    } catch (Exception e) {
      Assumptions.assumeTrue(false, "CUDA driver check failed: " + e.getMessage());
    } finally {
      if (process != null && process.isAlive()) {
        process.destroyForcibly();
      }
    }
  }
}
