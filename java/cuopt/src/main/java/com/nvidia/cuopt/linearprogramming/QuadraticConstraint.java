/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.Arrays;

/** Host-side representation of a quadratic constraint in COO form. */
public final class QuadraticConstraint {
  private final int rowIndex;
  private final String rowName;
  private final ConstraintSense sense;
  private final double[] linearValues;
  private final int[] linearIndices;
  private final double rhs;
  private final int[] rows;
  private final int[] columns;
  private final double[] values;

  public QuadraticConstraint(
      int rowIndex,
      String rowName,
      ConstraintSense sense,
      double[] linearValues,
      int[] linearIndices,
      double rhs,
      int[] rows,
      int[] columns,
      double[] values) {
    if (linearValues.length != linearIndices.length) {
      throw new IllegalArgumentException("linearValues and linearIndices must have the same length");
    }
    if (rows.length != columns.length || rows.length != values.length) {
      throw new IllegalArgumentException("quadratic COO arrays must have the same length");
    }
    if (sense == ConstraintSense.EQ) {
      throw new IllegalArgumentException("Equality quadratic constraints are not supported");
    }
    this.rowIndex = rowIndex;
    this.rowName = rowName == null ? "" : rowName;
    this.sense = sense;
    this.linearValues = Arrays.copyOf(linearValues, linearValues.length);
    this.linearIndices = Arrays.copyOf(linearIndices, linearIndices.length);
    this.rhs = rhs;
    this.rows = Arrays.copyOf(rows, rows.length);
    this.columns = Arrays.copyOf(columns, columns.length);
    this.values = Arrays.copyOf(values, values.length);
  }

  public int getRowIndex() {
    return rowIndex;
  }

  public String getRowName() {
    return rowName;
  }

  public ConstraintSense getSense() {
    return sense;
  }

  public double[] getLinearValues() {
    return Arrays.copyOf(linearValues, linearValues.length);
  }

  public int[] getLinearIndices() {
    return Arrays.copyOf(linearIndices, linearIndices.length);
  }

  public double getRHS() {
    return rhs;
  }

  public int[] getRows() {
    return Arrays.copyOf(rows, rows.length);
  }

  public int[] getColumns() {
    return Arrays.copyOf(columns, columns.length);
  }

  public double[] getValues() {
    return Arrays.copyOf(values, values.length);
  }
}
