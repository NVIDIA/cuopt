/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DataModel implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();
  private final NativeHandle nativeHandle;
  private final Cleaner.Cleanable cleanable;
  private double[] initialPrimalSolution = new double[0];
  private double[] initialDualSolution = new double[0];
  // Keep the user-facing CSR representation separate from cuOpt's internal GPU
  // representation. The GPU setter stores Q + Q^T for solving, while Python's
  // DataModel getters expose the matrix supplied by the caller.
  private double[] quadraticObjectiveValues = new double[0];
  private int[] quadraticObjectiveIndices = new int[0];
  private int[] quadraticObjectiveOffsets = new int[0];
  private boolean quadraticObjectiveMatrixSet;
  private final List<String> quadraticConstraintNames = new ArrayList<>();

  /** Create an empty mutable LP/MIP/QP data model. */
  public DataModel() {
    this(NativeCuOpt.createEmptyProblem());
  }

  private DataModel(long handle) {
    this.nativeHandle = new NativeHandle(handle);
    this.cleanable = CLEANER.register(this, nativeHandle);
  }

  public static DataModel createProblem(
      int numConstraints,
      int numVariables,
      ObjectiveSense objectiveSense,
      double objectiveOffset,
      double[] objectiveCoefficients,
      CsrMatrix constraintMatrix,
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
    return new DataModel(handle);
  }

  public static DataModel createRangedProblem(
      int numConstraints,
      int numVariables,
      ObjectiveSense objectiveSense,
      double objectiveOffset,
      double[] objectiveCoefficients,
      CsrMatrix constraintMatrix,
      double[] constraintLowerBounds,
      double[] constraintUpperBounds,
      double[] variableLowerBounds,
      double[] variableUpperBounds,
      byte[] variableTypes) {
    long handle =
        NativeCuOpt.createRangedProblem(
            numConstraints,
            numVariables,
            objectiveSense.nativeValue(),
            objectiveOffset,
            Arrays.copyOf(objectiveCoefficients, objectiveCoefficients.length),
            constraintMatrix.getRowOffsets(),
            constraintMatrix.getColumnIndices(),
            constraintMatrix.getValues(),
            Arrays.copyOf(constraintLowerBounds, constraintLowerBounds.length),
            Arrays.copyOf(constraintUpperBounds, constraintUpperBounds.length),
            Arrays.copyOf(variableLowerBounds, variableLowerBounds.length),
            Arrays.copyOf(variableUpperBounds, variableUpperBounds.length),
            Arrays.copyOf(variableTypes, variableTypes.length));
    return new DataModel(handle);
  }

  public static DataModel read(String path) {
    return new DataModel(NativeCuOpt.readProblem(path));
  }

  public static DataModel read(String path, boolean fixedMpsFormat) {
    return new DataModel(NativeCuOpt.readProblemWithFormat(path, fixedMpsFormat));
  }

  /** Parse an MPS/QPS file directly, optionally using fixed-format parsing. */
  public static DataModel parseMps(String path) {
    return parseMps(path, false);
  }

  public static DataModel parseMps(String path, boolean fixedMpsFormat) {
    return new DataModel(NativeCuOpt.parseMpsProblem(path, fixedMpsFormat));
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  public int getNumVariables() {
    return NativeCuOpt.getNumVariables(handle());
  }

  public int getNumConstraints() {
    return NativeCuOpt.getNumConstraints(handle());
  }

  public int getNumNonZeros() {
    return NativeCuOpt.getNumNonZeros(handle());
  }

  public ObjectiveSense getObjectiveSense() {
    return NativeCuOpt.getObjectiveSense(handle()) == ObjectiveSense.MAXIMIZE.nativeValue()
        ? ObjectiveSense.MAXIMIZE
        : ObjectiveSense.MINIMIZE;
  }

  /** Return true for maximize and false for minimize, matching Python get_sense(). */
  public boolean getSense() {
    return getObjectiveSense() == ObjectiveSense.MAXIMIZE;
  }

  public double getObjectiveOffset() {
    return NativeCuOpt.getObjectiveOffset(handle());
  }

  public double getObjectiveScalingFactor() {
    return NativeCuOpt.getObjectiveScalingFactor(handle());
  }

  public double[] getObjectiveCoefficients() {
    return NativeCuOpt.getObjectiveCoefficients(handle());
  }

  public CsrMatrix getConstraintMatrix() {
    Object[] matrix = NativeCuOpt.getConstraintMatrix(handle());
    return new CsrMatrix((int[]) matrix[0], (int[]) matrix[1], (double[]) matrix[2]);
  }

  public double[] getConstraintMatrixValues() {
    return getConstraintMatrix().getValues();
  }

  public int[] getConstraintMatrixIndices() {
    return getConstraintMatrix().getColumnIndices();
  }

  public int[] getConstraintMatrixOffsets() {
    return getConstraintMatrix().getRowOffsets();
  }

  public byte[] getConstraintSense() {
    return NativeCuOpt.getConstraintSense(handle());
  }

  public byte[] getRowTypes() {
    return getConstraintSense();
  }

  public double[] getConstraintRhs() {
    return NativeCuOpt.getConstraintRhs(handle());
  }

  public double[] getConstraintBounds() {
    return getConstraintRhs();
  }

  public double[] getConstraintLowerBounds() {
    return NativeCuOpt.getConstraintLowerBounds(handle());
  }

  public double[] getConstraintUpperBounds() {
    return NativeCuOpt.getConstraintUpperBounds(handle());
  }

  public double[] getVariableLowerBounds() {
    return NativeCuOpt.getVariableLowerBounds(handle());
  }

  public double[] getVariableUpperBounds() {
    return NativeCuOpt.getVariableUpperBounds(handle());
  }

  public byte[] getVariableTypes() {
    return NativeCuOpt.getVariableTypes(handle());
  }

  public boolean isMip() {
    return NativeCuOpt.isMip(handle());
  }

  public ProblemCategory getProblemCategory() {
    return ProblemCategory.fromNative(NativeCuOpt.getProblemCategory(handle()));
  }

  public DataModel setMaximize(boolean maximize) {
    NativeCuOpt.setMaximize(handle(), maximize);
    return this;
  }

  public DataModel setCsrConstraintMatrix(double[] values, int[] indices, int[] offsets) {
    NativeCuOpt.setConstraintMatrix(handle(), copy(values), copy(indices), copy(offsets));
    return this;
  }

  public DataModel setConstraintBounds(double[] bounds) {
    NativeCuOpt.setConstraintBounds(handle(), copy(bounds));
    return this;
  }

  public DataModel setObjectiveCoefficients(double[] coefficients) {
    NativeCuOpt.setObjectiveCoefficients(handle(), copy(coefficients));
    return this;
  }

  public DataModel setObjectiveScalingFactor(double scalingFactor) {
    NativeCuOpt.setObjectiveScalingFactor(handle(), scalingFactor);
    return this;
  }

  public DataModel setObjectiveOffset(double offset) {
    NativeCuOpt.setObjectiveOffset(handle(), offset);
    return this;
  }

  public DataModel setQuadraticObjectiveMatrix(double[] values, int[] indices, int[] offsets) {
    double[] copiedValues = copy(values);
    int[] copiedIndices = copy(indices);
    int[] copiedOffsets = copy(offsets);
    NativeCuOpt.setQuadraticObjectiveMatrix(handle(), copiedValues, copiedIndices, copiedOffsets);
    quadraticObjectiveValues = copiedValues;
    quadraticObjectiveIndices = copiedIndices;
    quadraticObjectiveOffsets = copiedOffsets;
    quadraticObjectiveMatrixSet = true;
    return this;
  }

  public DataModel setVariableLowerBounds(double[] bounds) {
    NativeCuOpt.setVariableLowerBounds(handle(), copy(bounds));
    return this;
  }

  public DataModel setVariableUpperBounds(double[] bounds) {
    NativeCuOpt.setVariableUpperBounds(handle(), copy(bounds));
    return this;
  }

  public DataModel setConstraintLowerBounds(double[] bounds) {
    NativeCuOpt.setConstraintLowerBounds(handle(), copy(bounds));
    return this;
  }

  public DataModel setConstraintUpperBounds(double[] bounds) {
    NativeCuOpt.setConstraintUpperBounds(handle(), copy(bounds));
    return this;
  }

  public DataModel setRowTypes(byte[] rowTypes) {
    NativeCuOpt.setRowTypes(handle(), copy(rowTypes));
    return this;
  }

  public DataModel setVariableTypes(byte[] variableTypes) {
    NativeCuOpt.setVariableTypes(handle(), copy(variableTypes));
    return this;
  }

  public DataModel setVariableNames(String[] variableNames) {
    NativeCuOpt.setVariableNames(handle(), variableNames == null ? new String[0] : variableNames.clone());
    return this;
  }

  public DataModel setRowNames(String[] rowNames) {
    NativeCuOpt.setRowNames(handle(), rowNames == null ? new String[0] : rowNames.clone());
    return this;
  }

  public DataModel setObjectiveName(String objectiveName) {
    NativeCuOpt.setObjectiveName(handle(), objectiveName == null ? "" : objectiveName);
    return this;
  }

  public DataModel setProblemName(String problemName) {
    NativeCuOpt.setProblemName(handle(), problemName == null ? "" : problemName);
    return this;
  }

  public DataModel setInitialPrimalSolution(double[] values) {
    initialPrimalSolution = copy(values);
    NativeCuOpt.setInitialPrimalSolutionOnProblem(handle(), initialPrimalSolution);
    return this;
  }

  public DataModel setInitialDualSolution(double[] values) {
    initialDualSolution = copy(values);
    NativeCuOpt.setInitialDualSolutionOnProblem(handle(), initialDualSolution);
    return this;
  }

  public double[] getInitialPrimalSolution() {
    return copy(initialPrimalSolution);
  }

  public double[] getInitialDualSolution() {
    return copy(initialDualSolution);
  }

  public double[] getQuadraticObjectiveValues() {
    return quadraticObjectiveMatrixSet
        ? copy(quadraticObjectiveValues)
        : NativeCuOpt.getQuadraticObjectiveValues(handle());
  }

  public int[] getQuadraticObjectiveIndices() {
    return quadraticObjectiveMatrixSet
        ? copy(quadraticObjectiveIndices)
        : NativeCuOpt.getQuadraticObjectiveIndices(handle());
  }

  public int[] getQuadraticObjectiveOffsets() {
    return quadraticObjectiveMatrixSet
        ? copy(quadraticObjectiveOffsets)
        : NativeCuOpt.getQuadraticObjectiveOffsets(handle());
  }

  public String[] getVariableNames() {
    return NativeCuOpt.getVariableNames(handle());
  }

  public String[] getRowNames() {
    return NativeCuOpt.getRowNames(handle());
  }

  public String getObjectiveName() {
    return NativeCuOpt.getObjectiveName(handle());
  }

  public String getProblemName() {
    return NativeCuOpt.getProblemName(handle());
  }

  public byte[] getAsciiRowTypes() {
    return getConstraintSense();
  }

  /** Return a Java map with the same logical fields as Python's parser.toDict. */
  public Map<String, Object> toDict() {
    CsrMatrix matrix = getConstraintMatrix();
    Map<String, Object> csr = new LinkedHashMap<>();
    csr.put("offsets", matrix.getRowOffsets());
    csr.put("indices", matrix.getColumnIndices());
    csr.put("values", matrix.getValues());

    Map<String, Object> bounds = new LinkedHashMap<>();
    bounds.put("bounds", getConstraintRhs());
    bounds.put("upper_bounds", getConstraintUpperBounds());
    bounds.put("lower_bounds", getConstraintLowerBounds());
    bounds.put("types", getConstraintSense());

    Map<String, Object> objective = new LinkedHashMap<>();
    objective.put("coefficients", getObjectiveCoefficients());
    objective.put("scalability_factor", getObjectiveScalingFactor());
    objective.put("offset", getObjectiveOffset());

    Map<String, Object> variableBounds = new LinkedHashMap<>();
    variableBounds.put("upper_bounds", getVariableUpperBounds());
    variableBounds.put("lower_bounds", getVariableLowerBounds());

    Map<String, Object> result = new LinkedHashMap<>();
    result.put("csr_constraint_matrix", csr);
    result.put("constraint_bounds", bounds);
    result.put("objective_data", objective);
    result.put("variable_bounds", variableBounds);
    result.put("maximize", getObjectiveSense() == ObjectiveSense.MAXIMIZE);
    result.put("variable_types", getVariableTypes());
    result.put("variable_names", getVariableNames());
    return result;
  }

  public DataModel setQuadraticObjective(QuadraticExpression expression) {
    NativeCuOpt.setQuadraticObjective(
        handle(), quadraticRows(expression), quadraticColumns(expression), quadraticValues(expression));
    return this;
  }

  public DataModel addQuadraticConstraint(Constraint constraint) {
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
    NativeCuOpt.addQuadraticConstraint(
        handle(),
        quadraticRows(expression),
        quadraticColumns(expression),
        quadraticValues(expression),
        linearIndices,
        linearCoefficients,
        constraint.getSense().nativeValue(),
        constraint.getRHS());
    quadraticConstraintNames.add(constraint.getConstraintName());
    return this;
  }

  public DataModel addQuadraticConstraint(
      String rowName,
      double[] linearValues,
      int[] linearIndices,
      double rhs,
      double[] values,
      int[] rows,
      int[] columns,
      ConstraintSense sense) {
    if (sense == ConstraintSense.EQ) {
      throw new IllegalArgumentException("Equality quadratic constraints are not supported");
    }
    if (linearValues == null || linearIndices == null || linearValues.length != linearIndices.length) {
      throw new IllegalArgumentException("linearValues and linearIndices must have the same length");
    }
    if (values == null || rows == null || columns == null
        || values.length != rows.length || values.length != columns.length) {
      throw new IllegalArgumentException("quadratic COO arrays must have the same length");
    }
    NativeCuOpt.addQuadraticConstraint(
        handle(),
        copy(rows),
        copy(columns),
        copy(values),
        copy(linearIndices),
        copy(linearValues),
        sense.nativeValue(),
        rhs);
    quadraticConstraintNames.add(rowName == null ? "" : rowName);
    return this;
  }

  public List<QuadraticConstraint> getQuadraticConstraints() {
    Object[] nativeConstraints = NativeCuOpt.getQuadraticConstraints(handle());
    List<QuadraticConstraint> result = new ArrayList<>(nativeConstraints.length);
    for (int i = 0; i < nativeConstraints.length; i++) {
      Object[] entry = (Object[]) nativeConstraints[i];
      int rowIndex = ((int[]) entry[0])[0];
      String rowName = (String) entry[1];
      if (i < quadraticConstraintNames.size() && !quadraticConstraintNames.get(i).isEmpty()) {
        rowName = quadraticConstraintNames.get(i);
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

  public DataModel clearQuadraticConstraints() {
    NativeCuOpt.clearQuadraticConstraints(handle());
    quadraticConstraintNames.clear();
    return this;
  }

  public Solution solve(SolverSettings settings) {
    SolverSettings actualSettings = settings == null ? new SolverSettings() : settings;
    boolean closeSettings = settings == null;
    try {
      long solutionHandle = NativeCuOpt.solve(handle(), actualSettings.handle());
      return new Solution(
          solutionHandle,
          getNumVariables(),
          getNumConstraints(),
          getProblemCategory(),
          getVariableNames());
    } finally {
      if (closeSettings) {
        actualSettings.close();
      }
    }
  }

  public void writeMPS(String path) {
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

  private static double[] copy(double[] values) {
    return values == null ? new double[0] : Arrays.copyOf(values, values.length);
  }

  private static int[] copy(int[] values) {
    return values == null ? new int[0] : Arrays.copyOf(values, values.length);
  }

  private static byte[] copy(byte[] values) {
    return values == null ? new byte[0] : Arrays.copyOf(values, values.length);
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
        throw new IllegalStateException("DataModel is closed");
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
