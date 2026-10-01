/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class NativeProblem implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();
  private final NativeHandle nativeHandle;
  private final Cleaner.Cleanable cleanable;
  private final Map<Integer, String> quadraticConstraintNames = new HashMap<>();

  private NativeProblem(long handle) {
    this.nativeHandle = new NativeHandle(handle);
    this.cleanable = CLEANER.register(this, nativeHandle);
  }

  static NativeProblem createProblem(
      int numConstraints,
      int numVariables,
      ObjectiveSense objectiveSense,
      double objectiveOffset,
      double[] objectiveCoefficients,
      CSRMatrix constraintMatrix,
      byte[] constraintSense,
      double[] rhs,
      double[] variableLowerBounds,
      double[] variableUpperBounds,
      byte[] variableTypes) {
    long handle =
        NativeCuOpt.createProblem(
            numConstraints,
            numVariables,
            objectiveSense.nativeValue(),
            objectiveOffset,
            Arrays.copyOf(objectiveCoefficients, objectiveCoefficients.length),
            constraintMatrix.getRowOffsets(),
            constraintMatrix.getColumnIndices(),
            constraintMatrix.getValues(),
            Arrays.copyOf(constraintSense, constraintSense.length),
            Arrays.copyOf(rhs, rhs.length),
            Arrays.copyOf(variableLowerBounds, variableLowerBounds.length),
            Arrays.copyOf(variableUpperBounds, variableUpperBounds.length),
            Arrays.copyOf(variableTypes, variableTypes.length));
    return new NativeProblem(handle);
  }

  static NativeProblem read(String path, boolean fixedMPSFormat) {
    return new NativeProblem(NativeCuOpt.readProblemWithFormat(path, fixedMPSFormat));
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  int getNumVariables() {
    return NativeCuOpt.getNumVariables(handle());
  }

  int getNumConstraints() {
    return NativeCuOpt.getNumConstraints(handle());
  }

  int getNumNonZeros() {
    return NativeCuOpt.getNumNonZeros(handle());
  }

  ObjectiveSense getObjectiveSense() {
    return NativeCuOpt.getObjectiveSense(handle()) == ObjectiveSense.MAXIMIZE.nativeValue()
        ? ObjectiveSense.MAXIMIZE
        : ObjectiveSense.MINIMIZE;
  }

  double getObjectiveOffset() {
    return NativeCuOpt.getObjectiveOffset(handle());
  }

  double[] getObjectiveCoefficients() {
    return NativeCuOpt.getObjectiveCoefficients(handle());
  }

  CSRMatrix getConstraintMatrix() {
    Object[] matrix = NativeCuOpt.getConstraintMatrix(handle());
    return new CSRMatrix((double[]) matrix[2], (int[]) matrix[1], (int[]) matrix[0]);
  }

  byte[] getConstraintSense() {
    return NativeCuOpt.getConstraintSense(handle());
  }

  double[] getConstraintRHS() {
    return NativeCuOpt.getConstraintRHS(handle());
  }

  double[] getConstraintLowerBounds() {
    return NativeCuOpt.getConstraintLowerBounds(handle());
  }

  double[] getConstraintUpperBounds() {
    return NativeCuOpt.getConstraintUpperBounds(handle());
  }

  double[] getVariableLowerBounds() {
    return NativeCuOpt.getVariableLowerBounds(handle());
  }

  double[] getVariableUpperBounds() {
    return NativeCuOpt.getVariableUpperBounds(handle());
  }

  byte[] getVariableTypes() {
    return NativeCuOpt.getVariableTypes(handle());
  }

  NativeProblem setVariableNames(String[] variableNames) {
    NativeCuOpt.setVariableNames(handle(), variableNames == null ? new String[0] : variableNames.clone());
    return this;
  }

  NativeProblem setRowNames(String[] rowNames) {
    NativeCuOpt.setRowNames(handle(), rowNames == null ? new String[0] : rowNames.clone());
    return this;
  }

  NativeProblem setProblemName(String problemName) {
    NativeCuOpt.setProblemName(handle(), problemName == null ? "" : problemName);
    return this;
  }

  double[] getQuadraticObjectiveValues() {
    return NativeCuOpt.getQuadraticObjectiveValues(handle());
  }

  int[] getQuadraticObjectiveIndices() {
    return NativeCuOpt.getQuadraticObjectiveIndices(handle());
  }

  int[] getQuadraticObjectiveOffsets() {
    return NativeCuOpt.getQuadraticObjectiveOffsets(handle());
  }

  String[] getVariableNames() {
    return NativeCuOpt.getVariableNames(handle());
  }

  String[] getRowNames() {
    return NativeCuOpt.getRowNames(handle());
  }

  String getProblemName() {
    return NativeCuOpt.getProblemName(handle());
  }

  /** The engine reports 0 for LP; every other category has discrete variables. */
  boolean isMIP() {
    return NativeCuOpt.getProblemCategory(handle()) != 0;
  }

  NativeProblem setQuadraticObjective(QuadraticExpression expression) {
    NativeCuOpt.setQuadraticObjective(
        handle(), quadraticRows(expression), quadraticColumns(expression), quadraticValues(expression));
    return this;
  }

  NativeProblem addQuadraticConstraint(Constraint constraint) {
    if (!constraint.isQuadratic()) {
      throw new IllegalArgumentException("Quadratic constraint requires quadratic terms");
    }
    if (constraint.getSense() == ConstraintSense.EQ) {
      throw new IllegalArgumentException("Equality quadratic constraints are not supported");
    }
    QuadraticExpression expression = constraint.getQuadraticExpression();
    LinearExpression linear = constraint.getLinearExpression();
    int[] linearIndices = new int[linear.getTerms().size()];
    double[] linearCoefficients = new double[linear.getTerms().size()];
    int i = 0;
    for (var entry : linear.getTerms().entrySet()) {
      linearIndices[i] = entry.getKey().getIndex();
      linearCoefficients[i] = entry.getValue();
      i++;
    }
    int rowIndex = getNumConstraints();
    NativeCuOpt.addQuadraticConstraint(
        handle(),
        quadraticRows(expression),
        quadraticColumns(expression),
        quadraticValues(expression),
        linearIndices,
        linearCoefficients,
        constraint.getSense().nativeValue(),
        constraint.getRHS());
    quadraticConstraintNames.put(rowIndex, constraint.getConstraintName());
    return this;
  }

  List<QuadraticConstraint> getQuadraticConstraints() {
    Object[] nativeConstraints = NativeCuOpt.getQuadraticConstraints(handle());
    List<QuadraticConstraint> result = new ArrayList<>(nativeConstraints.length);
    for (int i = 0; i < nativeConstraints.length; i++) {
      Object[] entry = (Object[]) nativeConstraints[i];
      int rowIndex = ((int[]) entry[0])[0];
      String rowName = (String) entry[1];
      String addedName = quadraticConstraintNames.get(rowIndex);
      if (addedName != null && !addedName.isEmpty()) {
        rowName = addedName;
      }
      ConstraintSense sense = ConstraintSense.fromNative(((byte[]) entry[2])[0]);
      double rhs = ((double[]) entry[5])[0];
      result.add(
          new QuadraticConstraint(
              rowIndex,
              rowName,
              sense,
              (double[]) entry[3],
              (int[]) entry[4],
              rhs,
              (int[]) entry[6],
              (int[]) entry[7],
              (double[]) entry[8]));
    }
    return List.copyOf(result);
  }

  Solution solve(SolverSettings settings) {
    SolverSettings actualSettings = settings == null ? new SolverSettings() : settings;
    boolean closeSettings = settings == null;
    try {
      long solutionHandle = NativeCuOpt.solve(handle(), actualSettings.handle());
      return new Solution(
          solutionHandle,
          getNumVariables(),
          getNumConstraints(),
          isMIP(),
          getVariableNames());
    } finally {
      if (closeSettings) {
        actualSettings.close();
      }
    }
  }

  void write(String path) {
    NativeCuOpt.writeProblem(handle(), path);
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  private static int[] quadraticRows(QuadraticExpression expression) {
    int[] rows = new int[expression.getQuadraticTerms().size()];
    for (int i = 0; i < rows.length; i++) {
      rows[i] = expression.getQuadraticTerms().get(i).getFirst().getIndex();
    }
    return rows;
  }

  private static int[] quadraticColumns(QuadraticExpression expression) {
    int[] columns = new int[expression.getQuadraticTerms().size()];
    for (int i = 0; i < columns.length; i++) {
      columns[i] = expression.getQuadraticTerms().get(i).getSecond().getIndex();
    }
    return columns;
  }

  private static double[] quadraticValues(QuadraticExpression expression) {
    double[] values = new double[expression.getQuadraticTerms().size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = expression.getQuadraticTerms().get(i).getCoefficient();
    }
    return values;
  }

  private static final class NativeHandle implements Runnable {
    private long handle;

    NativeHandle(long handle) {
      this.handle = handle;
    }

    void requireOpen() {
      if (handle == 0) {
        throw new IllegalStateException("Native problem is closed");
      }
    }

    @Override
    public void run() {
      if (handle != 0) {
        NativeCuOpt.destroyProblem(handle);
        handle = 0;
      }
    }
  }
}
