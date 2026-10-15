/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/**
 * Receives cuOpt's console log lines from native code and writes them through {@link
 * System#out}, rather than the native library writing to the process's stdout stream directly.
 *
 * <p>A direct native write bypasses {@code System.out}, so it is invisible to anything that
 * intercepts or redirects it -- {@link System#setOut}, a logging framework bridge, or Maven
 * Surefire, which uses the forked JVM's stdout as its own communication channel and can
 * misinterpret an unexpected raw write on it as the forked process having crashed.
 *
 * <p>Called from {@code cuopt_jni.cpp}; not part of the public API.
 */
final class NativeLogSink {
  private NativeLogSink() {}

  static void onLogLine(String message) {
    System.out.print(message);
  }
}
