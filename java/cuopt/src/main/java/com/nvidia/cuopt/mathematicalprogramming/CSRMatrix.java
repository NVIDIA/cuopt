/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

import java.util.Arrays;

public final class CSRMatrix {
  private final int[] rowOffsets;
  private final int[] columnIndices;
  private final double[] values;

  /** Construct a CSR matrix using the cuOpt values, indices, offsets argument order. */
  public CSRMatrix(double[] values, int[] columnIndices, int[] rowOffsets) {
    this.rowOffsets = Arrays.copyOf(rowOffsets, rowOffsets.length);
    this.columnIndices = Arrays.copyOf(columnIndices, columnIndices.length);
    this.values = Arrays.copyOf(values, values.length);
    if (this.values.length != this.columnIndices.length) {
      throw new IllegalArgumentException("CSR values and column indices must have the same length");
    }
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
}
