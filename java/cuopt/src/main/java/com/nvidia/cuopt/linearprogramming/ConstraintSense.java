/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public enum ConstraintSense {
  LE((byte) 'L'),
  GE((byte) 'G'),
  EQ((byte) 'E');

  private final byte nativeValue;

  ConstraintSense(byte nativeValue) {
    this.nativeValue = nativeValue;
  }

  byte nativeValue() {
    return nativeValue;
  }

  static ConstraintSense fromNative(byte value) {
    for (ConstraintSense sense : values()) {
      if (sense.nativeValue == value) {
        return sense;
      }
    }
    throw new IllegalArgumentException("Unknown constraint sense: " + (char) value);
  }
}
