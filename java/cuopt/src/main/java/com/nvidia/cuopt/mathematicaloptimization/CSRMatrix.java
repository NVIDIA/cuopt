/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.util.Arrays;

public final class CSRMatrix {
  private final int[] rowOffsets;
  private final int[] columnIndices;
  private final double[] values;

  /** Construct a CSR matrix using the cuOpt values, indices, offsets argument order. */
  public CSRMatrix(double[] values, int[] columnIndices, int[] rowOffsets) {
    validate(values, columnIndices, rowOffsets);
    this.rowOffsets = Arrays.copyOf(rowOffsets, rowOffsets.length);
    this.columnIndices = Arrays.copyOf(columnIndices, columnIndices.length);
    this.values = Arrays.copyOf(values, values.length);
  }

  public int[] getRowOffsets() {
    return Arrays.copyOf(rowOffsets, rowOffsets.length);
  }

  public int[] getColumnIndices() {
    return Arrays.copyOf(columnIndices, columnIndices.length);
  }

  public double[] getValues() {
    return Arrays.copyOf(values, values.length);
  }

  int[] rowOffsetsUnsafe() {
    return rowOffsets;
  }

  int[] columnIndicesUnsafe() {
    return columnIndices;
  }

  double[] valuesUnsafe() {
    return values;
  }

  private static void validate(double[] values, int[] columnIndices, int[] rowOffsets) {
    if (values == null) {
      throw new IllegalArgumentException("CSR values must not be null");
    }
    if (columnIndices == null) {
      throw new IllegalArgumentException("CSR column indices must not be null");
    }
    if (rowOffsets == null) {
      throw new IllegalArgumentException("CSR row offsets must not be null");
    }
    if (values.length != columnIndices.length) {
      throw new IllegalArgumentException("CSR values and column indices must have the same length");
    }
    if (rowOffsets.length == 0) {
      throw new IllegalArgumentException("CSR row offsets must not be empty");
    }
    if (rowOffsets[0] != 0) {
      throw new IllegalArgumentException("CSR row offsets must start at 0");
    }
    for (int i = 1; i < rowOffsets.length; i++) {
      if (rowOffsets[i] < rowOffsets[i - 1]) {
        throw new IllegalArgumentException("CSR row offsets must be monotonic");
      }
    }
    if (rowOffsets[rowOffsets.length - 1] != values.length) {
      throw new IllegalArgumentException("CSR row offsets must end at the number of values");
    }
  }
}
