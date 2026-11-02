/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Verifies that {@link NativeLogSink#onLogLine} writes through {@link System#out} rather than
 * bypassing it -- the whole point of routing native console log lines through this class instead
 * of a direct native write. See {@link NativeLogSink} for why a direct write is unsafe here.
 */
final class NativeLogSinkTest {

  private final PrintStream originalOut = System.out;

  @AfterEach
  void restoreSystemOut() {
    System.setOut(originalOut);
  }

  @Test
  void onLogLineWritesThroughSystemOut() {
    ByteArrayOutputStream captured = new ByteArrayOutputStream();
    System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));

    NativeLogSink.onLogLine("Solving a problem with 1 constraints, 1 variables\n");

    assertEquals(
        "Solving a problem with 1 constraints, 1 variables\n",
        captured.toString(StandardCharsets.UTF_8));
  }

  @Test
  void onLogLineReflectsSystemSetOutRedirection() {
    ByteArrayOutputStream first = new ByteArrayOutputStream();
    System.setOut(new PrintStream(first, true, StandardCharsets.UTF_8));
    NativeLogSink.onLogLine("first\n");

    ByteArrayOutputStream second = new ByteArrayOutputStream();
    System.setOut(new PrintStream(second, true, StandardCharsets.UTF_8));
    NativeLogSink.onLogLine("second\n");

    assertEquals("first\n", first.toString(StandardCharsets.UTF_8));
    assertEquals("second\n", second.toString(StandardCharsets.UTF_8));
  }
}
