/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

public final class Problem implements AutoCloseable {
  private final String name;
  private final List<Variable> variables = new ArrayList<>();
  private final List<Constraint> constraints = new ArrayList<>();
  private LinearExpression linearObjective = new LinearExpression();
  private QuadraticExpression quadraticObjective = null;
  private ObjectiveSense objectiveSense = ObjectiveSense.MINIMIZE;
  private boolean objectiveSet = false;
  private TerminationStatus status = TerminationStatus.NO_TERMINATION;
  private double objectiveValue = Double.NaN;
  private double solveTime = Double.NaN;

  public Problem() {
    this("");
  }

  public Problem(String name) {
    this.name = name == null ? "" : name;
  }

  public String getName() {
    return name;
  }

  public Variable addVariable() {
    return addVariable(0.0, Double.POSITIVE_INFINITY, 0.0, VariableType.CONTINUOUS, "");
  }

  public Variable addVariable(
      double lowerBound,
      double upperBound,
      double objectiveCoefficient,
      VariableType variableType,
      String name) {
    Variable variable =
        new Variable(
            variables.size(), lowerBound, upperBound, objectiveCoefficient, variableType, name);
    variables.add(variable);
    resetSolvedValues();
    return variable;
  }

  public Constraint addConstraint(Constraint constraint) {
    return addConstraint(constraint, "");
  }

  public Constraint addConstraint(Constraint constraint, String name) {
    requireOwnedVariables(constraint);
    constraint.setConstraintName(name);
    constraint.setIndex(constraints.size());
    constraints.add(constraint);
    resetSolvedValues();
    return constraint;
  }

  public Problem setObjective(LinearExpression expression, ObjectiveSense sense) {
    this.linearObjective = expression;
    this.quadraticObjective = null;
    this.objectiveSense = sense;
    this.objectiveSet = true;
    syncVariableObjectiveCoefficients(expression);
    resetSolvedValues();
    return this;
  }

  public Problem setObjective(Variable variable, ObjectiveSense sense) {
    return setObjective(LinearExpression.of(variable), sense);
  }

  public Problem setObjective(double constant, ObjectiveSense sense) {
    return setObjective(LinearExpression.ofConstant(constant), sense);
  }

  public Problem setObjective(QuadraticExpression expression, ObjectiveSense sense) {
    this.linearObjective = expression.getLinearExpression();
    this.quadraticObjective = expression;
    this.objectiveSense = sense;
    this.objectiveSet = true;
    syncVariableObjectiveCoefficients(expression.getLinearExpression());
    resetSolvedValues();
    return this;
  }

  public List<Variable> getVariables() {
    return List.copyOf(variables);
  }

  /**
   * Reads the values of {@code variables} out of an array laid out in variable-index order, such
   * as the incumbent a MIP callback receives, and returns them in the order asked for.
   *
   * <pre>{@code
   * double[] picked = Problem.fromIncumbent(incumbent, z, y, x);
   * }</pre>
   *
   * @throws IllegalArgumentException if a variable's index falls outside {@code indexOrdered}
   */
  public static double[] fromIncumbent(double[] indexOrdered, Variable... variables) {
    double[] values = new double[variables.length];
    for (int i = 0; i < variables.length; i++) {
      int index = variables[i].getIndex();
      if (index < 0 || index >= indexOrdered.length) {
        throw new IllegalArgumentException(
            "Variable '"
                + variables[i].getVariableName()
                + "' has index "
                + index
                + ", outside an array of length "
                + indexOrdered.length);
      }
      values[i] = indexOrdered[index];
    }
    return values;
  }

  public Variable getVariable(int index) {
    return variables.get(index);
  }

  public Variable getVariable(String variableName) {
    for (Variable variable : variables) {
      if (variable.getVariableName().equals(variableName)) {
        return variable;
      }
    }
    return null;
  }

  public List<Constraint> getConstraints() {
    return List.copyOf(constraints);
  }

  public Constraint getConstraint(int index) {
    return constraints.get(index);
  }

  public Constraint getConstraint(String constraintName) {
    for (Constraint constraint : constraints) {
      if (constraint.getConstraintName().equals(constraintName)) {
        return constraint;
      }
    }
    return null;
  }

  public int getNumVariables() {
    return variables.size();
  }

  public int getNumConstraints() {
    return constraints.size();
  }

  public boolean isMIP() {
    return variables.stream().anyMatch(v -> v.getVariableType() != VariableType.CONTINUOUS);
  }

  public TerminationStatus getStatus() {
    return status;
  }

  public double getObjectiveValue() {
    return objectiveValue;
  }

  public double getSolveTime() {
    return solveTime;
  }

  /** The linear constraint matrix in CSR form. Quadratic constraints are not included. */
  public CSRMatrix getConstraintMatrix() {
    return buildLinearConstraintMatrix().matrix;
  }

  private NativeProblem toNativeProblem() {
    MatrixBuild matrixBuild = buildLinearConstraintMatrix();
    double[] objectiveCoefficients = objectiveCoefficients();
    double[] lowerBounds = new double[variables.size()];
    double[] upperBounds = new double[variables.size()];
    byte[] variableTypes = new byte[variables.size()];
    for (Variable variable : variables) {
      int index = variable.getIndex();
      lowerBounds[index] = variable.getLowerBound();
      upperBounds[index] = variable.getUpperBound();
      variableTypes[index] = variable.getVariableType().nativeValue();
    }

    NativeProblem nativeProblem =
        NativeProblem.createProblem(
            matrixBuild.linearConstraints.size(),
            variables.size(),
            objectiveSense,
            objectiveSet ? linearObjective.getConstant() : 0.0,
            objectiveCoefficients,
            matrixBuild.matrix,
            matrixBuild.constraintSense,
            matrixBuild.rhs,
            lowerBounds,
            upperBounds,
            variableTypes);

    if (quadraticObjective != null && !quadraticObjective.getQuadraticTerms().isEmpty()) {
      nativeProblem.setQuadraticObjective(quadraticObjective);
    }
    for (Constraint constraint : constraints) {
      if (constraint.isQuadratic()) {
        nativeProblem.addQuadraticConstraint(constraint);
      }
    }
    String[] variableNames = new String[variables.size()];
    for (Variable variable : variables) {
      variableNames[variable.getIndex()] = variable.getVariableName();
    }
    String[] rowNames = new String[matrixBuild.linearConstraints.size()];
    for (int i = 0; i < matrixBuild.linearConstraints.size(); i++) {
      rowNames[i] = matrixBuild.linearConstraints.get(i).getConstraintName();
    }
    nativeProblem.setVariableNames(variableNames).setRowNames(rowNames).setProblemName(name);
    return nativeProblem;
  }

  public Solution solve() {
    return solve(null);
  }

  public Solution solve(SolverSettings settings) {
    SolverSettings actualSettings = settings == null ? new SolverSettings() : settings;
    boolean closeSettings = settings == null;
    addMIPStarts(actualSettings);
    try (NativeProblem nativeProblem = toNativeProblem()) {
      Solution solution = nativeProblem.solve(actualSettings);
      try {
        populateSolution(solution);
      } catch (RuntimeException | Error e) {
        solution.close();
        throw e;
      }
      return solution;
    } finally {
      if (closeSettings) {
        actualSettings.close();
      }
    }
  }

  /**
   * Writes the problem to {@code path} in MPS format, which is the only format the engine can
   * write. {@code path} must end in {@code .mps} or {@code .qps}.
   *
   * @throws IllegalArgumentException if {@code path} has any other extension
   */
  public void write(String path) {
    String lower = path.toLowerCase(Locale.ROOT);
    if (!lower.endsWith(".mps") && !lower.endsWith(".qps")) {
      throw new IllegalArgumentException(
          "Problem.write only writes MPS; expected a .mps or .qps path but got '" + path + "'");
    }
    try (NativeProblem nativeProblem = toNativeProblem()) {
      nativeProblem.write(path);
    }
  }

  /** Reads a problem from {@code path}. The parser is chosen from the file extension. */
  public static Problem read(String path) {
    return read(path, false);
  }

  public static Problem read(String path, boolean fixedMPSFormat) {
    try (NativeProblem nativeProblem = NativeProblem.read(path, fixedMPSFormat)) {
      return fromNativeProblem(nativeProblem);
    }
  }

  private static Problem fromNativeProblem(NativeProblem nativeProblem) {
    Problem problem = new Problem(nativeProblem.getProblemName());
    double[] lowerBounds = nativeProblem.getVariableLowerBounds();
    double[] upperBounds = nativeProblem.getVariableUpperBounds();
    byte[] variableTypes = nativeProblem.getVariableTypes();
    double[] objectiveCoefficients = nativeProblem.getObjectiveCoefficients();
    String[] variableNames = nativeProblem.getVariableNames();
    for (int i = 0; i < nativeProblem.getNumVariables(); i++) {
      problem.addVariable(
          lowerBounds[i],
          upperBounds[i],
          objectiveCoefficients[i],
          VariableType.fromNative(variableTypes[i]),
          variableNames.length > i && !variableNames[i].isEmpty() ? variableNames[i] : "x" + i);
    }

    CSRMatrix matrix = nativeProblem.getConstraintMatrix();
    int[] rowOffsets = matrix.getRowOffsets();
    int[] columnIndices = matrix.getColumnIndices();
    double[] values = matrix.getValues();
    byte[] senses = nativeProblem.getConstraintSense();
    double[] rhs = nativeProblem.getConstraintRHS();
    double[] constraintLowerBounds = nativeProblem.getConstraintLowerBounds();
    double[] constraintUpperBounds = nativeProblem.getConstraintUpperBounds();
    String[] rowNames = nativeProblem.getRowNames();
    for (int row = 0; row < nativeProblem.getNumConstraints(); row++) {
      LinearExpression expression = new LinearExpression();
      for (int p = rowOffsets[row]; p < rowOffsets[row + 1]; p++) {
        expression = expression.plus(problem.getVariable(columnIndices[p]), values[p]);
      }
      Constraint constraint =
          constraintFromNativeBounds(
              expression,
              senses.length > row ? ConstraintSense.fromNative(senses[row]) : null,
              rhs[row],
              constraintLowerBounds,
              constraintUpperBounds,
              row);
      problem.addConstraint(
          constraint,
          rowNames.length > row && !rowNames[row].isEmpty() ? rowNames[row] : "c" + row);
    }

    int[] qOffsets = nativeProblem.getQuadraticObjectiveOffsets();
    int[] qIndices = nativeProblem.getQuadraticObjectiveIndices();
    double[] qValues = nativeProblem.getQuadraticObjectiveValues();
    if (qValues.length == 0) {
      LinearExpression objective = LinearExpression.ofConstant(nativeProblem.getObjectiveOffset());
      for (int i = 0; i < objectiveCoefficients.length; i++) {
        if (objectiveCoefficients[i] != 0.0) {
          objective = objective.plus(problem.getVariable(i), objectiveCoefficients[i]);
        }
      }
      problem.setObjective(objective, nativeProblem.getObjectiveSense());
    } else {
      QuadraticExpression objective =
          new QuadraticExpression().constant(nativeProblem.getObjectiveOffset());
      for (int i = 0; i < objectiveCoefficients.length; i++) {
        if (objectiveCoefficients[i] != 0.0) {
          objective = objective.plus(problem.getVariable(i), objectiveCoefficients[i]);
        }
      }
      for (int row = 0; row + 1 < qOffsets.length; row++) {
        for (int p = qOffsets[row]; p < qOffsets[row + 1]; p++) {
          objective =
              objective.plus(
                  problem.getVariable(row), problem.getVariable(qIndices[p]), qValues[p]);
        }
      }
      problem.setObjective(objective, nativeProblem.getObjectiveSense());
    }

    for (QuadraticConstraint quadraticConstraint : nativeProblem.getQuadraticConstraints()) {
      QuadraticExpression expression = new QuadraticExpression();
      double[] linearValues = quadraticConstraint.getLinearValues();
      int[] linearIndices = quadraticConstraint.getLinearIndices();
      for (int i = 0; i < linearValues.length; i++) {
        expression = expression.plus(problem.getVariable(linearIndices[i]), linearValues[i]);
      }
      int[] rows = quadraticConstraint.getRows();
      int[] columns = quadraticConstraint.getColumns();
      double[] quadraticValues = quadraticConstraint.getValues();
      for (int i = 0; i < quadraticValues.length; i++) {
        expression =
            expression.plus(
                problem.getVariable(rows[i]), problem.getVariable(columns[i]), quadraticValues[i]);
      }
      Constraint constraint =
          quadraticConstraint.getSense() == ConstraintSense.LE
              ? expression.le(quadraticConstraint.getRHS())
              : expression.ge(quadraticConstraint.getRHS());
      problem.addConstraint(constraint, quadraticConstraint.getRowName());
    }
    return problem;
  }



  @Override
  public void close() {
    // Problem is a Java-side model; native handles are scoped to solve/read/write calls.
  }

  void resetSolvedValues() {
    variables.forEach(Variable::resetSolvedValues);
    constraints.forEach(Constraint::resetSolvedValues);
    status = TerminationStatus.NO_TERMINATION;
    objectiveValue = Double.NaN;
    solveTime = Double.NaN;
  }

  /**
   * The linear part of the objective. The quadratic part, when there is one, is available as a
   * matrix from {@link #getQuadraticObjectiveMatrix()}.
   */
  public LinearExpression getObjective() {
    return linearObjective;
  }

  public ObjectiveSense getObjectiveSense() {
    return objectiveSense;
  }

  public double getObjectiveConstant() {
    return objectiveSet ? linearObjective.getConstant() : 0.0;
  }

  public int getNumNonZeros() {
    return buildLinearConstraintMatrix().matrix.getValues().length;
  }

  public List<Constraint> getQuadraticConstraints() {
    List<Constraint> result = new ArrayList<>();
    for (Constraint constraint : constraints) {
      if (constraint.isQuadratic()) {
        result.add(constraint);
      }
    }
    return List.copyOf(result);
  }

  /** The quadratic objective matrix Q in CSR form, or null when the objective is linear. */
  public CSRMatrix getQuadraticObjectiveMatrix() {
    if (quadraticObjective == null) {
      return null;
    }
    int n = variables.size();
    int[] offsets = new int[n + 1];
    Map<Integer, Map<Integer, Double>> byRow = new TreeMap<>();
    for (int i = 0; i < n; i++) {
      byRow.put(i, new TreeMap<>());
    }
    for (QuadraticExpression.QuadraticTerm term : quadraticObjective.getQuadraticTerms()) {
      byRow
          .get(term.getFirst().getIndex())
          .merge(term.getSecond().getIndex(), term.getCoefficient(), Double::sum);
    }
    int nnz = 0;
    for (int row = 0; row < n; row++) {
      offsets[row] = nnz;
      nnz += byRow.get(row).size();
    }
    offsets[n] = nnz;
    int[] columns = new int[nnz];
    double[] coefficients = new double[nnz];
    int position = 0;
    for (int row = 0; row < n; row++) {
      for (Map.Entry<Integer, Double> entry : byRow.get(row).entrySet()) {
        columns[position] = entry.getKey();
        coefficients[position++] = entry.getValue();
      }
    }
    return new CSRMatrix(coefficients, columns, offsets);
  }

  private void populateSolution(Solution solution) {
    resetSolvedValues();
    status = solution.getTerminationStatus();
    solveTime = solution.getSolveTime();

    double[] primal = solution.getPrimalSolution();
    if (primal.length > 0) {
      for (int i = 0; i < variables.size(); i++) {
        variables.get(i).setValue(primal[i]);
      }
    }
    if (!solution.isMIP()) {
      double[] reducedCosts = solution.getReducedCost();
      if (reducedCosts.length > 0) {
        for (int i = 0; i < variables.size(); i++) {
          variables.get(i).setReducedCost(reducedCosts[i]);
        }
      }
      double[] dual = solution.getDualSolution();
      int linearRow = 0;
      for (Constraint constraint : constraints) {
        if (!constraint.isQuadratic()) {
          if (dual.length > linearRow) {
            constraint.setDualValue(dual[linearRow]);
          }
          linearRow++;
        }
      }
    }
    for (Constraint constraint : constraints) {
      constraint.setSlack(constraint.computeSlack());
    }
    objectiveValue = solution.getPrimalObjective();
  }

  private void addMIPStarts(SolverSettings settings) {
    if (!isMIP()) {
      return;
    }
    double[] starts = new double[variables.size()];
    boolean any = false;
    for (Variable variable : variables) {
      starts[variable.getIndex()] = variable.getMIPStart();
      any |= !Double.isNaN(variable.getMIPStart());
    }
    if (any) {
      settings.addMIPStart(starts);
    }
  }

  private double[] objectiveCoefficients() {
    double[] coefficients = new double[variables.size()];
    if (!objectiveSet) {
      for (Variable variable : variables) {
        coefficients[variable.getIndex()] = variable.getObjectiveCoefficient();
      }
    } else {
      for (Map.Entry<Variable, Double> entry : linearObjective.getTerms().entrySet()) {
        coefficients[entry.getKey().getIndex()] += entry.getValue();
      }
    }
    return coefficients;
  }

  /**
   * A variable carries only its index, and the matrix is built from those indices, so a variable
   * from another problem would be read as whichever variable holds that index here and would
   * silently solve a different model.
   */
  private void requireOwnedVariables(Constraint constraint) {
    for (Variable variable : constraint.getLinearExpression().getTerms().keySet()) {
      requireOwnedVariable(variable);
    }
    if (constraint.isQuadratic()) {
      for (QuadraticExpression.QuadraticTerm term :
          constraint.getQuadraticExpression().getQuadraticTerms()) {
        requireOwnedVariable(term.getFirst());
        requireOwnedVariable(term.getSecond());
      }
    }
  }

  private void requireOwnedVariable(Variable variable) {
    int index = variable.getIndex();
    if (index < 0 || index >= variables.size() || variables.get(index) != variable) {
      throw new IllegalArgumentException(
          "Constraint variable '"
              + variable.getVariableName()
              + "' does not belong to this problem");
    }
  }

  private void syncVariableObjectiveCoefficients(LinearExpression expression) {
    for (Variable variable : variables) {
      variable.setObjectiveCoefficient(0.0);
    }
    for (Map.Entry<Variable, Double> entry : expression.getTerms().entrySet()) {
      if (!variables.contains(entry.getKey())) {
        throw new IllegalArgumentException("Objective variable does not belong to this problem");
      }
      entry.getKey().setObjectiveCoefficient(entry.getValue());
    }
  }

  private LinearExpression objectiveFromVariableCoefficients(double constant) {
    LinearExpression expression = LinearExpression.ofConstant(constant);
    for (Variable variable : variables) {
      if (variable.getObjectiveCoefficient() != 0.0) {
        expression = expression.plus(variable, variable.getObjectiveCoefficient());
      }
    }
    return expression;
  }

  private static Constraint constraintFromNativeBounds(
      LinearExpression expression,
      ConstraintSense sense,
      double rhs,
      double[] constraintLowerBounds,
      double[] constraintUpperBounds,
      int row) {
    if (constraintLowerBounds.length > row && constraintUpperBounds.length > row) {
      double lowerBound = constraintLowerBounds[row];
      double upperBound = constraintUpperBounds[row];
      boolean hasLowerBound = !Double.isInfinite(lowerBound);
      boolean hasUpperBound = !Double.isInfinite(upperBound);
      if (hasLowerBound && hasUpperBound) {
        if (Double.compare(lowerBound, upperBound) == 0) {
          return expression.eq(lowerBound);
        }
        throw new IllegalArgumentException(
            "Ranged constraints are not supported by Problem.read: row "
                + row
                + " has lower bound "
                + lowerBound
                + " and upper bound "
                + upperBound);
      }
      if (hasLowerBound) {
        return expression.ge(lowerBound);
      }
      if (hasUpperBound) {
        return expression.le(upperBound);
      }
    }

    if (sense == null) {
      throw new IllegalStateException(
          "Native constraint row " + row + " does not provide bounds or a constraint sense");
    }
    switch (sense) {
      case LE:
        return expression.le(rhs);
      case GE:
        return expression.ge(rhs);
      case EQ:
        return expression.eq(rhs);
      default:
        throw new IllegalStateException("Unsupported sense " + sense);
    }
  }

  private MatrixBuild buildLinearConstraintMatrix() {
    List<Constraint> linearConstraints = new ArrayList<>();
    for (Constraint constraint : constraints) {
      if (!constraint.isQuadratic()) {
        linearConstraints.add(constraint);
      }
    }

    int nnz = 0;
    for (Constraint constraint : linearConstraints) {
      nnz += constraint.getLinearExpression().getTerms().size();
    }

    int[] rowOffsets = new int[linearConstraints.size() + 1];
    int[] columnIndices = new int[nnz];
    double[] values = new double[nnz];
    byte[] senses = new byte[linearConstraints.size()];
    double[] rhs = new double[linearConstraints.size()];

    int position = 0;
    for (int row = 0; row < linearConstraints.size(); row++) {
      Constraint constraint = linearConstraints.get(row);
      rowOffsets[row] = position;
      for (Map.Entry<Variable, Double> entry : constraint.getLinearExpression().getTerms().entrySet()) {
        columnIndices[position] = entry.getKey().getIndex();
        values[position] = entry.getValue();
        position++;
      }
      senses[row] = constraint.getSense().nativeValue();
      rhs[row] = constraint.getRHS();
    }
    rowOffsets[linearConstraints.size()] = position;
    return new MatrixBuild(
        new CSRMatrix(values, columnIndices, rowOffsets), linearConstraints, senses, rhs);
  }

  private static final class MatrixBuild {
    private final CSRMatrix matrix;
    private final List<Constraint> linearConstraints;
    private final byte[] constraintSense;
    private final double[] rhs;

    private MatrixBuild(
        CSRMatrix matrix, List<Constraint> linearConstraints, byte[] constraintSense, double[] rhs) {
      this.matrix = matrix;
      this.linearConstraints = linearConstraints;
      this.constraintSense = Arrays.copyOf(constraintSense, constraintSense.length);
      this.rhs = Arrays.copyOf(rhs, rhs.length);
    }
  }
}
