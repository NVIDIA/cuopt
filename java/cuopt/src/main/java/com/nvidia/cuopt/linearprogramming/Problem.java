/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
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
  private boolean solved = false;
  private TerminationStatus status = TerminationStatus.NO_TERMINATION;
  private double objectiveValue = Double.NaN;
  private double solveTime = Double.NaN;
  private PDLPWarmStartData warmStartData;

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
    solved = false;
    return variable;
  }

  public Constraint addConstraint(Constraint constraint) {
    return addConstraint(constraint, "");
  }

  public Constraint addConstraint(Constraint constraint, String name) {
    constraint.setConstraintName(name);
    constraint.setIndex(constraints.size());
    constraints.add(constraint);
    solved = false;
    return constraint;
  }

  public Problem setObjective(LinearExpression expression, ObjectiveSense sense) {
    this.linearObjective = expression;
    this.quadraticObjective = null;
    this.objectiveSense = sense;
    this.objectiveSet = true;
    syncVariableObjectiveCoefficients(expression);
    solved = false;
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
    solved = false;
    resetSolvedValues();
    return this;
  }

  public List<Variable> getVariables() {
    return List.copyOf(variables);
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

  public boolean isMip() {
    return variables.stream().anyMatch(v -> v.getVariableType() != VariableType.CONTINUOUS);
  }

  public boolean isSolved() {
    return solved;
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

  public CsrMatrix getCSR() {
    return buildLinearConstraintMatrix().matrix;
  }

  public DataModel toDataModel() {
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

    DataModel dataModel =
        DataModel.createProblem(
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
      dataModel.setQuadraticObjective(quadraticObjective);
    }
    for (Constraint constraint : constraints) {
      if (constraint.isQuadratic()) {
        dataModel.addQuadraticConstraint(constraint);
      }
    }
    String[] variableNames = new String[variables.size()];
    for (Variable variable : variables) {
      variableNames[variable.getIndex()] = variable.getVariableName();
    }
    String[] rowNames = new String[constraints.size()];
    for (int i = 0; i < constraints.size(); i++) {
      rowNames[i] = constraints.get(i).getConstraintName();
    }
    dataModel.setVariableNames(variableNames).setRowNames(rowNames).setProblemName(name);
    return dataModel;
  }

  public Solution solve() {
    return solve(null);
  }

  public Solution solve(SolverSettings settings) {
    SolverSettings actualSettings = settings == null ? new SolverSettings() : settings;
    boolean closeSettings = settings == null;
    addMipStarts(actualSettings);
    try (DataModel dataModel = toDataModel()) {
      Solution solution = dataModel.solve(actualSettings);
      populateSolution(solution);
      return solution;
    } finally {
      if (closeSettings) {
        actualSettings.close();
      }
    }
  }

  public void writeMPS(String path) {
    try (DataModel dataModel = toDataModel()) {
      dataModel.writeMPS(path);
    }
  }

  public static Problem read(String path) {
    return read(path, false);
  }

  public static Problem read(String path, boolean fixedMpsFormat) {
    try (DataModel dataModel = DataModel.read(path, fixedMpsFormat)) {
      return fromDataModel(dataModel);
    }
  }

  private static Problem fromDataModel(DataModel dataModel) {
    Problem problem = new Problem(dataModel.getProblemName());
    double[] lowerBounds = dataModel.getVariableLowerBounds();
    double[] upperBounds = dataModel.getVariableUpperBounds();
    byte[] variableTypes = dataModel.getVariableTypes();
    double[] objectiveCoefficients = dataModel.getObjectiveCoefficients();
    String[] variableNames = dataModel.getVariableNames();
    for (int i = 0; i < dataModel.getNumVariables(); i++) {
      problem.addVariable(
          lowerBounds[i],
          upperBounds[i],
          objectiveCoefficients[i],
          VariableType.fromNative(variableTypes[i]),
          variableNames.length > i && !variableNames[i].isEmpty() ? variableNames[i] : "x" + i);
    }

      CsrMatrix matrix = dataModel.getConstraintMatrix();
      int[] rowOffsets = matrix.getRowOffsets();
      int[] columnIndices = matrix.getColumnIndices();
      double[] values = matrix.getValues();
      byte[] senses = dataModel.getConstraintSense();
      double[] rhs = dataModel.getConstraintRhs();
      double[] constraintLowerBounds = dataModel.getConstraintLowerBounds();
      double[] constraintUpperBounds = dataModel.getConstraintUpperBounds();
      String[] rowNames = dataModel.getRowNames();
      for (int row = 0; row < dataModel.getNumConstraints(); row++) {
        LinearExpression expression = new LinearExpression();
        for (int p = rowOffsets[row]; p < rowOffsets[row + 1]; p++) {
          expression = expression.plus(problem.getVariable(columnIndices[p]), values[p]);
        }
        ConstraintSense sense = ConstraintSense.fromNative(senses[row]);
        if (constraintLowerBounds.length > row && constraintUpperBounds.length > row) {
          if (Double.compare(constraintLowerBounds[row], constraintUpperBounds[row]) == 0) {
            sense = ConstraintSense.EQ;
          } else if (Double.compare(constraintLowerBounds[row], rhs[row]) == 0) {
            sense = ConstraintSense.GE;
          } else if (Double.compare(constraintUpperBounds[row], rhs[row]) == 0) {
            sense = ConstraintSense.LE;
          }
        }
        Constraint constraint;
        switch (sense) {
          case LE:
            constraint = expression.le(rhs[row]);
            break;
          case GE:
            constraint = expression.ge(rhs[row]);
            break;
          case EQ:
            constraint = expression.eq(rhs[row]);
            break;
          default:
            throw new IllegalStateException("Unsupported sense " + sense);
        }
        problem.addConstraint(
            constraint,
            rowNames.length > row && !rowNames[row].isEmpty() ? rowNames[row] : "c" + row);
      }

      int[] qOffsets = dataModel.getQuadraticObjectiveOffsets();
      int[] qIndices = dataModel.getQuadraticObjectiveIndices();
      double[] qValues = dataModel.getQuadraticObjectiveValues();
      if (qValues.length == 0) {
        LinearExpression objective = LinearExpression.ofConstant(dataModel.getObjectiveOffset());
        for (int i = 0; i < objectiveCoefficients.length; i++) {
          if (objectiveCoefficients[i] != 0.0) {
            objective = objective.plus(problem.getVariable(i), objectiveCoefficients[i]);
          }
        }
        problem.setObjective(objective, dataModel.getObjectiveSense());
      } else {
        QuadraticExpression objective =
            new QuadraticExpression().constant(dataModel.getObjectiveOffset());
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
        problem.setObjective(objective, dataModel.getObjectiveSense());
      }

    for (QuadraticConstraint quadraticConstraint : dataModel.getQuadraticConstraints()) {
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

  public static Problem readMPS(String path) {
    return readMPS(path, false);
  }

  public static Problem readMPS(String path, boolean fixedMpsFormat) {
    try (DataModel dataModel = DataModel.parseMps(path, fixedMpsFormat)) {
      return fromDataModel(dataModel);
    }
  }

  @Override
  public void close() {
    // Problem owns no native handle; DataModel and Solution carry native lifetimes.
  }

  public void update() {
    resetSolvedValues();
  }

  public void resetSolvedValues() {
    variables.forEach(Variable::resetSolvedValues);
    constraints.forEach(Constraint::resetSolvedValues);
    solved = false;
    status = TerminationStatus.NO_TERMINATION;
    objectiveValue = Double.NaN;
    solveTime = Double.NaN;
    warmStartData = null;
  }

  public void updateConstraint(Constraint constraint, Map<Variable, Double> coefficients, Double rhs) {
    if (!constraints.contains(constraint)) {
      throw new IllegalArgumentException("Constraint does not belong to this problem");
    }
    if (constraint.isQuadratic()) {
      throw new IllegalArgumentException("updateConstraint applies to linear constraints only");
    }
    LinearExpression expression = new LinearExpression();
    for (Map.Entry<Variable, Double> entry : constraint.getLinearExpression().getTerms().entrySet()) {
      expression = expression.plus(entry.getKey(), entry.getValue());
    }
    if (coefficients != null) {
      for (Map.Entry<Variable, Double> entry : coefficients.entrySet()) {
        expression = expression.plus(entry.getKey(), entry.getValue() - constraint.getCoefficient(entry.getKey()));
      }
    }
    constraint.updateLinearExpression(expression);
    if (rhs != null) {
      constraint.updateRHS(rhs);
    }
    resetSolvedValues();
  }

  public void updateObjective(Map<Variable, Double> coefficients, Double constant, ObjectiveSense sense) {
    if (coefficients != null) {
      for (Map.Entry<Variable, Double> entry : coefficients.entrySet()) {
        if (!variables.contains(entry.getKey())) {
          throw new IllegalArgumentException("Objective variable does not belong to this problem");
        }
        entry.getKey().setObjectiveCoefficient(entry.getValue());
      }
      if (objectiveSet) {
        LinearExpression updated =
            LinearExpression.ofConstant(constant == null ? linearObjective.getConstant() : constant);
        for (Map.Entry<Variable, Double> entry : linearObjective.getTerms().entrySet()) {
          updated = updated.plus(entry.getKey(), coefficients.getOrDefault(entry.getKey(), entry.getValue()));
        }
        for (Map.Entry<Variable, Double> entry : coefficients.entrySet()) {
          if (!linearObjective.getTerms().containsKey(entry.getKey())) {
            updated = updated.plus(entry.getKey(), entry.getValue());
          }
        }
        linearObjective = updated;
        if (quadraticObjective != null) {
          QuadraticExpression updatedQuadratic =
              new QuadraticExpression().constant(linearObjective.getConstant());
          for (Map.Entry<Variable, Double> entry : linearObjective.getTerms().entrySet()) {
            updatedQuadratic = updatedQuadratic.plus(entry.getKey(), entry.getValue());
          }
          for (QuadraticExpression.QuadraticTerm term : quadraticObjective.getQuadraticTerms()) {
            updatedQuadratic =
                updatedQuadratic.plus(term.getFirst(), term.getSecond(), term.getCoefficient());
          }
          quadraticObjective = updatedQuadratic;
        }
      } else if (constant != null) {
        linearObjective = LinearExpression.ofConstant(constant);
        for (Variable variable : variables) {
          if (variable.getObjectiveCoefficient() != 0.0) {
            linearObjective = linearObjective.plus(variable, variable.getObjectiveCoefficient());
          }
        }
        objectiveSet = true;
      }
    } else if (constant != null) {
      linearObjective = linearObjective.constant(constant - linearObjective.getConstant());
      if (quadraticObjective != null) {
        quadraticObjective = quadraticObjective.plus(LinearExpression.ofConstant(constant - quadraticObjective.getLinearExpression().getConstant()));
      }
      objectiveSet = true;
    }
    if (sense != null) {
      objectiveSense = sense;
    }
    resetSolvedValues();
  }

  public Object getObjective() {
    return quadraticObjective == null ? linearObjective : quadraticObjective;
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

  public CsrMatrix getQCSR() {
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
    return new CsrMatrix(offsets, columns, coefficients);
  }

  public CsrMatrix getQcsr() {
    return getQCSR();
  }

  public List<Double> getIncumbentValues(double[] solution, List<Variable> requestedVariables) {
    List<Double> values = new ArrayList<>();
    for (Variable variable : requestedVariables) {
      values.add(solution[variable.getIndex()]);
    }
    return List.copyOf(values);
  }

  public List<Double> get_incumbent_values(double[] solution, List<Variable> requestedVariables) {
    return getIncumbentValues(solution, requestedVariables);
  }

  public PDLPWarmStartData getWarmstartData() {
    return warmStartData;
  }

  public PDLPWarmStartData getPdlpWarmStartData() {
    return getWarmstartData();
  }

  public PDLPWarmStartData get_pdlp_warm_start_data() {
    return getWarmstartData();
  }

  public Problem relax() {
    Map<Variable, Variable> mapping = new LinkedHashMap<>();
    Problem relaxed = new Problem(name);
    for (Variable variable : variables) {
      mapping.put(variable, relaxed.addVariable(variable.getLowerBound(), variable.getUpperBound(),
          variable.getObjectiveCoefficient(), VariableType.CONTINUOUS, variable.getVariableName()));
    }
    for (Constraint constraint : constraints) {
      if (constraint.isQuadratic()) {
        QuadraticExpression expression = new QuadraticExpression();
        for (Map.Entry<Variable, Double> entry : constraint.getLinearExpression().getTerms().entrySet()) {
          expression = expression.plus(mapping.get(entry.getKey()), entry.getValue());
        }
        for (QuadraticExpression.QuadraticTerm term : constraint.getQuadraticExpression().getQuadraticTerms()) {
          expression = expression.plus(mapping.get(term.getFirst()), mapping.get(term.getSecond()), term.getCoefficient());
        }
        relaxed.addConstraint(constraint.getSense() == ConstraintSense.LE
            ? expression.le(constraint.getRHS()) : expression.ge(constraint.getRHS()), constraint.getConstraintName());
      } else {
        LinearExpression expression = new LinearExpression();
        for (Map.Entry<Variable, Double> entry : constraint.getLinearExpression().getTerms().entrySet()) {
          expression = expression.plus(mapping.get(entry.getKey()), entry.getValue());
        }
        Constraint copy;
        switch (constraint.getSense()) {
          case LE:
            copy = expression.le(constraint.getRHS());
            break;
          case GE:
            copy = expression.ge(constraint.getRHS());
            break;
          case EQ:
            copy = expression.eq(constraint.getRHS());
            break;
          default:
            throw new IllegalStateException("Unsupported constraint sense");
        }
        relaxed.addConstraint(copy, constraint.getConstraintName());
      }
    }
    if (quadraticObjective != null) {
      QuadraticExpression expression = new QuadraticExpression().constant(quadraticObjective.getLinearExpression().getConstant());
      for (Map.Entry<Variable, Double> entry : quadraticObjective.getLinearExpression().getTerms().entrySet()) {
        expression = expression.plus(mapping.get(entry.getKey()), entry.getValue());
      }
      for (QuadraticExpression.QuadraticTerm term : quadraticObjective.getQuadraticTerms()) {
        expression = expression.plus(mapping.get(term.getFirst()), mapping.get(term.getSecond()), term.getCoefficient());
      }
      relaxed.setObjective(expression, objectiveSense);
    } else {
      LinearExpression expression = new LinearExpression().constant(linearObjective.getConstant());
      for (Map.Entry<Variable, Double> entry : linearObjective.getTerms().entrySet()) {
        expression = expression.plus(mapping.get(entry.getKey()), entry.getValue());
      }
      relaxed.setObjective(expression, objectiveSense);
    }
    return relaxed;
  }

  private void populateSolution(Solution solution) {
    double[] primal = solution.getPrimalSolution();
    for (int i = 0; i < variables.size(); i++) {
      variables.get(i).setValue(primal[i]);
    }
    if (!solution.isMip()) {
      double[] reducedCosts = solution.getReducedCost();
      for (int i = 0; i < variables.size(); i++) {
        variables.get(i).setReducedCost(reducedCosts[i]);
      }
      double[] dual = solution.getDualSolution();
      int linearRow = 0;
      for (Constraint constraint : constraints) {
        if (!constraint.isQuadratic()) {
          constraint.setDualValue(dual[linearRow++]);
        }
      }
    }
    for (Constraint constraint : constraints) {
      constraint.setSlack(constraint.computeSlack());
    }
    status = solution.getTerminationStatus();
    objectiveValue = solution.getPrimalObjective();
    solveTime = solution.getSolveTime();
    warmStartData = solution.isMip() ? null : solution.getPdlpWarmStartData();
    solved = true;
  }

  private void addMipStarts(SolverSettings settings) {
    if (!isMip()) {
      return;
    }
    double[] starts = new double[variables.size()];
    boolean any = false;
    for (Variable variable : variables) {
      starts[variable.getIndex()] = variable.getMipStart();
      any |= !Double.isNaN(variable.getMipStart());
    }
    if (any) {
      settings.addMipStart(starts);
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
        new CsrMatrix(rowOffsets, columnIndices, values), linearConstraints, senses, rhs);
  }

  private static final class MatrixBuild {
    private final CsrMatrix matrix;
    private final List<Constraint> linearConstraints;
    private final byte[] constraintSense;
    private final double[] rhs;

    private MatrixBuild(
        CsrMatrix matrix, List<Constraint> linearConstraints, byte[] constraintSense, double[] rhs) {
      this.matrix = matrix;
      this.linearConstraints = linearConstraints;
      this.constraintSense = Arrays.copyOf(constraintSense, constraintSense.length);
      this.rhs = Arrays.copyOf(rhs, rhs.length);
    }
  }
}
