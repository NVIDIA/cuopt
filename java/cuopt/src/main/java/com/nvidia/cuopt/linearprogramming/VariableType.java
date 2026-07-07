/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public enum VariableType {
  CONTINUOUS((byte) 'C'),
  INTEGER((byte) 'I'),
  SEMI_CONTINUOUS((byte) 'S');

  private final byte nativeValue;

  VariableType(byte nativeValue) {
    this.nativeValue = nativeValue;
  }

  byte nativeValue() {
    return nativeValue;
  }

  static VariableType fromNative(byte value) {
    for (VariableType type : values()) {
      if (type.nativeValue == value) {
        return type;
      }
    }
    throw new IllegalArgumentException("Unknown variable type: " + (char) value);
  }
}
