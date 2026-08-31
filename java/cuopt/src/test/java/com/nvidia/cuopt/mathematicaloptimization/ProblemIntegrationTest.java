/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

final class ProblemIntegrationTest {
  private static final double SOLVE_TOLERANCE = 1.0e-3;

  @TestFactory
  Stream<DynamicTest> problemsBuildAndSolve() {
    return cases().stream()
        .map(testCase -> DynamicTest.dynamicTest(testCase.name, () -> verify(testCase)));
  }

  private static void verify(CaseSpec testCase) {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();

    try (Problem problem = testCase.createProblem()) {
      assertProblemConstruction(testCase, problem);
      if (!testCase.shouldSolve()) {
        // QP callability is covered by NativeIntegrationTest. This case owns the independent
        // Problem construction contract for quadratic objectives and constraints.
        return;
      }
      try (SolverSettings settings = createSettings(testCase);
          Solution solution = problem.solve(settings)) {
        assertSolution(testCase, problem, solution);
      }
    }
  }

  private static void assertProblemConstruction(CaseSpec testCase, Problem problem) {
    assertEquals(testCase.numVariables, problem.getNumVariables());
    assertEquals(testCase.problemConstraintCount(), problem.getNumConstraints());
    assertEquals(testCase.linearProblemConstraintCount(), problem.getConstraintMatrix().getRowOffsets().length - 1);
    assertEquals(testCase.objectiveSense, problem.getObjectiveSense());
    assertEquals(testCase.objectiveOffset, problem.getObjectiveConstant(), 0.0);
    assertEquals(testCase.problemName, problem.getName());

    for (int i = 0; i < testCase.numVariables; i++) {
      Variable variable = problem.getVariable(i);
      assertEquals(testCase.variableLowerBounds[i], variable.getLowerBound(), 0.0);
      assertEquals(testCase.variableUpperBounds[i], variable.getUpperBound(), 0.0);
      assertEquals(VariableType.fromNative(testCase.variableTypes[i]), variable.getVariableType());
      assertEquals(testCase.objectiveCoefficients[i], variable.getObjectiveCoefficient(), 0.0);
      assertEquals(testCase.variableName(i), variable.getVariableName());
    }

    if (!testCase.isRanged()) {
      CSRMatrix matrix = problem.getConstraintMatrix();
      assertArrayEquals(testCase.rowOffsets, matrix.getRowOffsets());
      assertArrayEquals(testCase.columnIndices, matrix.getColumnIndices());
      assertDoubleArrayEquals(testCase.values, matrix.getValues(), 0.0);
      for (int row = 0; row < testCase.numConstraints; row++) {
        Constraint constraint = problem.getConstraint(row);
        assertEquals(ConstraintSense.fromNative(testCase.constraintSense[row]), constraint.getSense());
        assertEquals(testCase.rhs[row], constraint.getRHS(), 0.0);
        assertEquals(testCase.rowName(row), constraint.getConstraintName());
      }
    }

    if (testCase.hasQuadraticObjective()) {
      assertNotNull(problem.getQuadraticObjectiveMatrix());
      CSRMatrix matrix = problem.getQuadraticObjectiveMatrix();
      assertArrayEquals(testCase.quadraticObjectiveRowOffsets, matrix.getRowOffsets());
      assertArrayEquals(testCase.quadraticObjectiveColumnIndices, matrix.getColumnIndices());
      assertDoubleArrayEquals(testCase.quadraticObjectiveValues, matrix.getValues(), 0.0);
    }

    List<Constraint> quadraticConstraints = problem.getQuadraticConstraints();
    assertEquals(testCase.hasQuadraticConstraint() ? 1 : 0, quadraticConstraints.size());
    if (testCase.hasQuadraticConstraint()) {
      Constraint constraint = quadraticConstraints.get(0);
      assertEquals(testCase.quadraticConstraintName, constraint.getConstraintName());
      assertEquals(ConstraintSense.fromNative(testCase.quadraticConstraintSense), constraint.getSense());
      assertEquals(testCase.quadraticConstraintRHS, constraint.getRHS(), 0.0);
    }
  }

  private static void assertSolution(CaseSpec testCase, Problem problem, Solution solution) {
    assertEquals(testCase.hasIntegerVariables(), solution.isMIP());
    // getStatus is the way to tell a solved problem from an unsolved one: it stays
    // NO_TERMINATION until a solve populates it.
    assertNotEquals(TerminationStatus.NO_TERMINATION, problem.getStatus());
    assertEquals(solution.getTerminationStatus(), problem.getStatus());
    assertTrue(
        Double.isNaN(solution.getSolveTime()) || solution.getSolveTime() >= 0.0,
        "solve time must be non-negative when available");

    if (!testCase.expectSolutionValues) {
      assertTrue(
          solution.getTerminationStatus() == TerminationStatus.INFEASIBLE
              || solution.getTerminationStatus() == TerminationStatus.UNBOUNDED_OR_INFEASIBLE,
          "expected an infeasible status, got " + solution.getTerminationStatus());
      for (Variable variable : problem.getVariables()) {
        assertTrue(Double.isNaN(variable.getReducedCost()));
      }
      for (Constraint constraint : problem.getConstraints()) {
        assertTrue(Double.isNaN(constraint.getDualValue()));
      }
      return;
    }

    assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
    double[] primal = solution.getPrimalSolution();
    assertEquals(testCase.numVariables, primal.length);
    testCase.assertFeasible(primal);

    if (!Double.isNaN(testCase.expectedObjective)) {
      assertEquals(
          testCase.expectedObjective,
          solution.getPrimalObjective(),
          testCase.solutionTolerance,
          "objective value");
    }

    // Dual values and reduced costs belong to an LP solve and are rejected for a MIP.
    if (testCase.hasIntegerVariables()) {
      assertThrows(IllegalStateException.class, solution::getDualSolution);
      assertThrows(IllegalStateException.class, solution::getReducedCost);
    } else {
      assertDoesNotThrow(solution::getDualSolution);
      assertDoesNotThrow(solution::getReducedCost);
    }
  }

  private static SolverSettings createSettings(CaseSpec testCase) {
    SolverSettings settings = new SolverSettings();
    settings.setSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false);
    settings.setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 30.0);
    settings.setSetting(CuOptConstants.CUOPT_RANDOM_SEED, 1);
    if (testCase.hasIntegerVariables()) {
      settings.setSetting(
          CuOptConstants.CUOPT_MIP_DETERMINISM_MODE, CuOptConstants.CUOPT_MODE_DETERMINISTIC);
      settings.setSetting(CuOptConstants.CUOPT_MIP_ABSOLUTE_GAP, 1.0e-8);
      settings.setSetting(CuOptConstants.CUOPT_MIP_RELATIVE_GAP, 1.0e-8);
    } else if (testCase.hasQuadraticObjective()) {
      settings.setSetting(CuOptConstants.CUOPT_ITERATION_LIMIT, 50);
    } else {
      settings.setMethod(SolverMethod.PDLP);
      settings.setPDLPSolverMode(PDLPSolverMode.STABLE1);
      settings.setSetting(CuOptConstants.CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1.0e-7);
      settings.setSetting(CuOptConstants.CUOPT_RELATIVE_PRIMAL_TOLERANCE, 1.0e-7);
      settings.setSetting(CuOptConstants.CUOPT_ABSOLUTE_DUAL_TOLERANCE, 1.0e-7);
      settings.setSetting(CuOptConstants.CUOPT_RELATIVE_DUAL_TOLERANCE, 1.0e-7);
      settings.setSetting(CuOptConstants.CUOPT_ABSOLUTE_GAP_TOLERANCE, 1.0e-7);
      settings.setSetting(CuOptConstants.CUOPT_RELATIVE_GAP_TOLERANCE, 1.0e-7);
    }
    return settings;
  }

  private static void assertDoubleArrayEquals(
      double[] expected, double[] actual, double tolerance) {
    assertEquals(expected.length, actual.length, "array length");
    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i], actual[i], tolerance, "array value at index " + i);
    }
  }

  private static List<CaseSpec> cases() {
    return List.of(
        new CaseSpec(
            "lp_min_ge_unique_solution",
            1,
            2,
            ObjectiveSense.MINIMIZE,
            0.25,
            new double[] {1.0, 2.0},
            new int[] {0, 2},
            new int[] {0, 1},
            new double[] {1.0, 1.0},
            new byte[] {'G'},
            new double[] {3.0},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {'C', 'C'},
            true,
            3.25),
        new CaseSpec(
            "lp_max_le_unique_solution",
            3,
            2,
            ObjectiveSense.MAXIMIZE,
            -1.0,
            new double[] {3.0, 2.0},
            new int[] {0, 2, 3, 4},
            new int[] {0, 1, 0, 1},
            new double[] {1.0, 1.0, 1.0, 1.0},
            new byte[] {'L', 'L', 'L'},
            new double[] {4.0, 2.0, 3.0},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {'C', 'C'},
            true,
            9.0),
        new CaseSpec(
            "lp_equal_with_offset",
            1,
            2,
            ObjectiveSense.MINIMIZE,
            7.0,
            new double[] {0.0, 1.0},
            new int[] {0, 2},
            new int[] {0, 1},
            new double[] {1.0, 1.0},
            new byte[] {'E'},
            new double[] {5.0},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {5.0, 5.0},
            new byte[] {'C', 'C'},
            true,
            7.0),
        new CaseSpec(
            "lp_ranged_bounds",
            2,
            2,
            ObjectiveSense.MINIMIZE,
            0.0,
            new double[] {0.2, 1.0},
            new int[] {0, 2, 4},
            new int[] {0, 1, 0, 1},
            new double[] {1.0, 1.0, 2.0, 1.0},
            null,
            null,
            new double[] {1.0, 2.0},
            new double[] {3.0, 4.0},
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {'C', 'C'},
            true,
            0.2),
        new CaseSpec(
            "lp_mixed_bounds_negative_coefficients",
            1,
            2,
            ObjectiveSense.MINIMIZE,
            -2.0,
            new double[] {-1.0, 2.0},
            new int[] {0, 2},
            new int[] {0, 1},
            new double[] {1.0, 1.0},
            new byte[] {'E'},
            new double[] {1.0},
            null,
            null,
            new double[] {-2.0, -1.0},
            new double[] {2.0, 3.0},
            new byte[] {'C', 'C'},
            true,
            -6.0),
        new CaseSpec(
            "lp_max_ranged_bounds",
            1,
            2,
            ObjectiveSense.MAXIMIZE,
            0.0,
            new double[] {2.0, 1.0},
            new int[] {0, 2},
            new int[] {0, 1},
            new double[] {1.0, 1.0},
            null,
            null,
            new double[] {0.0},
            new double[] {3.0},
            new double[] {0.0, 0.0},
            new double[] {2.0, 2.0},
            new byte[] {'C', 'C'},
            true,
            5.0),
        new CaseSpec(
            "milp_integer_unique_solution",
            1,
            2,
            ObjectiveSense.MINIMIZE,
            0.0,
            new double[] {1.0, 2.0},
            new int[] {0, 2},
            new int[] {0, 1},
            new double[] {1.0, 1.0},
            new byte[] {'G'},
            new double[] {2.5},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {'I', 'I'},
            true,
            3.0),
        new CaseSpec(
            "milp_mixed_integer_continuous_max",
            1,
            2,
            ObjectiveSense.MAXIMIZE,
            0.0,
            new double[] {5.0, 1.0},
            new int[] {0, 2},
            new int[] {0, 1},
            new double[] {1.0, 1.0},
            new byte[] {'L'},
            new double[] {2.5},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {3.0, 10.0},
            new byte[] {'I', 'C'},
            true,
            10.5),
        new CaseSpec(
                "qp_diagonal_objective",
                1,
                2,
                ObjectiveSense.MINIMIZE,
                0.0,
                new double[] {-8.0, -16.0},
                new int[] {0, 2},
                new int[] {0, 1},
                new double[] {1.0, 1.0},
                null,
                null,
                new double[] {5.0},
                new double[] {1.0e20},
                new double[] {0.0, 0.0},
                new double[] {10.0, 10.0},
                new byte[] {'C', 'C'},
                true,
                Double.NaN)
            .withQuadraticObjective(
                new int[] {0, 1, 2}, new int[] {0, 1}, new double[] {1.0, 4.0})
            .withMetadata(
                new String[] {"x0", "long_variable_1"},
                new String[] {"constraint_0"},
                "qp_model")
            .withQuadraticConstraint(
                "qc0",
                (byte) 'L',
                100.0,
                new double[] {1.0},
                new int[] {0},
                new double[] {1.0},
                new int[] {0},
                new int[] {0})
            .withoutSolve(),
        new CaseSpec(
            "lp_infeasible_status",
            2,
            1,
            ObjectiveSense.MINIMIZE,
            0.0,
            new double[] {1.0},
            new int[] {0, 1, 2},
            new int[] {0, 0},
            new double[] {1.0, 1.0},
            new byte[] {'G', 'L'},
            new double[] {1.0, 0.0},
            null,
            null,
            new double[] {0.0},
            new double[] {10.0},
            new byte[] {'C'},
            false,
            Double.NaN));
  }

  private static final class CaseSpec {
    private final String name;
    private final int numConstraints;
    private final int numVariables;
    private final ObjectiveSense objectiveSense;
    private final double objectiveOffset;
    private final double[] objectiveCoefficients;
    private final int[] rowOffsets;
    private final int[] columnIndices;
    private final double[] values;
    private final byte[] constraintSense;
    private final double[] rhs;
    private final double[] constraintLowerBounds;
    private final double[] constraintUpperBounds;
    private final double[] variableLowerBounds;
    private final double[] variableUpperBounds;
    private final byte[] variableTypes;
    private final boolean expectSolutionValues;
    private final double expectedObjective;
    private final double solutionTolerance = SOLVE_TOLERANCE;
    private String[] variableNames = new String[0];
    private String[] rowNames = new String[0];
    private String problemName = "";
    private String quadraticConstraintName;
    private byte quadraticConstraintSense;
    private double quadraticConstraintRHS;
    private double[] quadraticConstraintLinearValues;
    private int[] quadraticConstraintLinearIndices;
    private double[] quadraticConstraintValues;
    private int[] quadraticConstraintRows;
    private int[] quadraticConstraintColumns;
    private int[] quadraticObjectiveRowOffsets;
    private int[] quadraticObjectiveColumnIndices;
    private double[] quadraticObjectiveValues;
    private boolean solveCase = true;

    private CaseSpec(
        String name,
        int numConstraints,
        int numVariables,
        ObjectiveSense objectiveSense,
        double objectiveOffset,
        double[] objectiveCoefficients,
        int[] rowOffsets,
        int[] columnIndices,
        double[] values,
        byte[] constraintSense,
        double[] rhs,
        double[] constraintLowerBounds,
        double[] constraintUpperBounds,
        double[] variableLowerBounds,
        double[] variableUpperBounds,
        byte[] variableTypes,
        boolean expectSolutionValues,
        double expectedObjective) {
      this.name = name;
      this.numConstraints = numConstraints;
      this.numVariables = numVariables;
      this.objectiveSense = objectiveSense;
      this.objectiveOffset = objectiveOffset;
      this.objectiveCoefficients = Arrays.copyOf(objectiveCoefficients, objectiveCoefficients.length);
      this.rowOffsets = Arrays.copyOf(rowOffsets, rowOffsets.length);
      this.columnIndices = Arrays.copyOf(columnIndices, columnIndices.length);
      this.values = Arrays.copyOf(values, values.length);
      this.constraintSense =
          constraintSense == null ? null : Arrays.copyOf(constraintSense, constraintSense.length);
      this.rhs = rhs == null ? null : Arrays.copyOf(rhs, rhs.length);
      this.constraintLowerBounds =
          constraintLowerBounds == null
              ? null
              : Arrays.copyOf(constraintLowerBounds, constraintLowerBounds.length);
      this.constraintUpperBounds =
          constraintUpperBounds == null
              ? null
              : Arrays.copyOf(constraintUpperBounds, constraintUpperBounds.length);
      this.variableLowerBounds = Arrays.copyOf(variableLowerBounds, variableLowerBounds.length);
      this.variableUpperBounds = Arrays.copyOf(variableUpperBounds, variableUpperBounds.length);
      this.variableTypes = Arrays.copyOf(variableTypes, variableTypes.length);
      this.expectSolutionValues = expectSolutionValues;
      this.expectedObjective = expectedObjective;
    }

    private Problem createProblem() {
      Problem problem = new Problem(problemName);
      for (int i = 0; i < numVariables; i++) {
        problem.addVariable(
            variableLowerBounds[i],
            variableUpperBounds[i],
            objectiveCoefficients[i],
            VariableType.fromNative(variableTypes[i]),
            variableName(i));
      }

      if (hasQuadraticObjective()) {
        problem.setObjective(buildQuadraticObjective(problem), objectiveSense);
      } else {
        problem.setObjective(buildLinearObjective(problem), objectiveSense);
      }

      for (int row = 0; row < numConstraints; row++) {
        LinearExpression expression = buildRowExpression(problem, row);
        if (isRanged()) {
          if (!Double.isInfinite(constraintLowerBounds[row])) {
            problem.addConstraint(expression.ge(constraintLowerBounds[row]), rangedRowName(row, "lower"));
          }
          if (!Double.isInfinite(constraintUpperBounds[row])) {
            problem.addConstraint(expression.le(constraintUpperBounds[row]), rangedRowName(row, "upper"));
          }
        } else {
          problem.addConstraint(
              toConstraint(expression, ConstraintSense.fromNative(constraintSense[row]), rhs[row]),
              rowName(row));
        }
      }

      if (hasQuadraticConstraint()) {
        QuadraticExpression expression = new QuadraticExpression();
        for (int i = 0; i < quadraticConstraintLinearValues.length; i++) {
          expression =
              expression.plus(
                  problem.getVariable(quadraticConstraintLinearIndices[i]),
                  quadraticConstraintLinearValues[i]);
        }
        for (int i = 0; i < quadraticConstraintValues.length; i++) {
          expression =
              expression.plus(
                  problem.getVariable(quadraticConstraintRows[i]),
                  problem.getVariable(quadraticConstraintColumns[i]),
                  quadraticConstraintValues[i]);
        }
        Constraint constraint =
            ConstraintSense.fromNative(quadraticConstraintSense) == ConstraintSense.LE
                ? expression.le(quadraticConstraintRHS)
                : expression.ge(quadraticConstraintRHS);
        problem.addConstraint(constraint, quadraticConstraintName);
      }
      return problem;
    }

    private LinearExpression buildLinearObjective(Problem problem) {
      LinearExpression objective = LinearExpression.ofConstant(objectiveOffset);
      for (int i = 0; i < objectiveCoefficients.length; i++) {
        if (objectiveCoefficients[i] != 0.0) {
          objective = objective.plus(problem.getVariable(i), objectiveCoefficients[i]);
        }
      }
      return objective;
    }

    private QuadraticExpression buildQuadraticObjective(Problem problem) {
      QuadraticExpression objective = new QuadraticExpression().constant(objectiveOffset);
      for (int i = 0; i < objectiveCoefficients.length; i++) {
        if (objectiveCoefficients[i] != 0.0) {
          objective = objective.plus(problem.getVariable(i), objectiveCoefficients[i]);
        }
      }
      for (int row = 0; row + 1 < quadraticObjectiveRowOffsets.length; row++) {
        for (int p = quadraticObjectiveRowOffsets[row]; p < quadraticObjectiveRowOffsets[row + 1]; p++) {
          objective =
              objective.plus(
                  problem.getVariable(row),
                  problem.getVariable(quadraticObjectiveColumnIndices[p]),
                  quadraticObjectiveValues[p]);
        }
      }
      return objective;
    }

    private LinearExpression buildRowExpression(Problem problem, int row) {
      LinearExpression expression = new LinearExpression();
      for (int p = rowOffsets[row]; p < rowOffsets[row + 1]; p++) {
        expression = expression.plus(problem.getVariable(columnIndices[p]), values[p]);
      }
      return expression;
    }

    private Constraint toConstraint(LinearExpression expression, ConstraintSense sense, double rhs) {
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

    private boolean isRanged() {
      return constraintLowerBounds != null;
    }

    private boolean hasQuadraticObjective() {
      return quadraticObjectiveValues != null;
    }

    private boolean shouldSolve() {
      return solveCase;
    }

    private boolean hasQuadraticConstraint() {
      return quadraticConstraintValues != null;
    }

    private boolean hasIntegerVariables() {
      for (byte type : variableTypes) {
        if (type == 'I' || type == 'S') {
          return true;
        }
      }
      return false;
    }

    private int linearProblemConstraintCount() {
      return isRanged() ? numConstraints * 2 : numConstraints;
    }

    private int problemConstraintCount() {
      return linearProblemConstraintCount() + (hasQuadraticConstraint() ? 1 : 0);
    }

    private String variableName(int index) {
      return variableNames.length > index ? variableNames[index] : "";
    }

    private String rowName(int index) {
      return rowNames.length > index ? rowNames[index] : "";
    }

    private String rangedRowName(int index, String boundName) {
      String base = rowName(index);
      return base.isEmpty() ? "" : base + "_" + boundName;
    }

    private CaseSpec withQuadraticObjective(
        int[] rowOffsets, int[] columnIndices, double[] values) {
      this.quadraticObjectiveRowOffsets = Arrays.copyOf(rowOffsets, rowOffsets.length);
      this.quadraticObjectiveColumnIndices = Arrays.copyOf(columnIndices, columnIndices.length);
      this.quadraticObjectiveValues = Arrays.copyOf(values, values.length);
      return this;
    }

    private CaseSpec withMetadata(
        String[] variableNames, String[] rowNames, String problemName) {
      this.variableNames = Arrays.copyOf(variableNames, variableNames.length);
      this.rowNames = Arrays.copyOf(rowNames, rowNames.length);
      this.problemName = problemName;
      return this;
    }

    private CaseSpec withQuadraticConstraint(
        String name,
        byte sense,
        double rhs,
        double[] linearValues,
        int[] linearIndices,
        double[] values,
        int[] rows,
        int[] columns) {
      this.quadraticConstraintName = name;
      this.quadraticConstraintSense = sense;
      this.quadraticConstraintRHS = rhs;
      this.quadraticConstraintLinearValues = Arrays.copyOf(linearValues, linearValues.length);
      this.quadraticConstraintLinearIndices = Arrays.copyOf(linearIndices, linearIndices.length);
      this.quadraticConstraintValues = Arrays.copyOf(values, values.length);
      this.quadraticConstraintRows = Arrays.copyOf(rows, rows.length);
      this.quadraticConstraintColumns = Arrays.copyOf(columns, columns.length);
      return this;
    }

    private CaseSpec withoutSolve() {
      this.solveCase = false;
      return this;
    }

    private void assertFeasible(double[] primal) {
      for (int variable = 0; variable < numVariables; variable++) {
        assertTrue(
            primal[variable] >= variableLowerBounds[variable] - solutionTolerance,
            "variable " + variable + " violates its lower bound");
        assertTrue(
            primal[variable] <= variableUpperBounds[variable] + solutionTolerance,
            "variable " + variable + " violates its upper bound");
        if (variableTypes[variable] == 'I') {
          assertEquals(
              Math.rint(primal[variable]),
              primal[variable],
              solutionTolerance,
              "variable " + variable + " must be integral");
        }
      }

      for (int row = 0; row < numConstraints; row++) {
        double activity = 0.0;
        for (int index = rowOffsets[row]; index < rowOffsets[row + 1]; index++) {
          activity += values[index] * primal[columnIndices[index]];
        }
        if (isRanged()) {
          assertTrue(
              activity >= constraintLowerBounds[row] - solutionTolerance,
              "row " + row + " violates its lower bound");
          assertTrue(
              activity <= constraintUpperBounds[row] + solutionTolerance,
              "row " + row + " violates its upper bound");
        } else if (constraintSense[row] == 'L') {
          assertTrue(activity <= rhs[row] + solutionTolerance, "row " + row + " violates <=");
        } else if (constraintSense[row] == 'G') {
          assertTrue(activity >= rhs[row] - solutionTolerance, "row " + row + " violates >=");
        } else {
          assertEquals(rhs[row], activity, solutionTolerance, "row " + row + " violates =");
        }
      }

      if (hasQuadraticConstraint()) {
        double activity = 0.0;
        for (int i = 0; i < quadraticConstraintLinearValues.length; i++) {
          activity +=
              quadraticConstraintLinearValues[i]
                  * primal[quadraticConstraintLinearIndices[i]];
        }
        for (int i = 0; i < quadraticConstraintValues.length; i++) {
          activity +=
              quadraticConstraintValues[i]
                  * primal[quadraticConstraintRows[i]]
                  * primal[quadraticConstraintColumns[i]];
        }
        if (quadraticConstraintSense == 'L') {
          assertTrue(
              activity <= quadraticConstraintRHS + solutionTolerance,
              "quadratic constraint violates <=");
        } else {
          assertTrue(
              activity >= quadraticConstraintRHS - solutionTolerance,
              "quadratic constraint violates >=");
        }
      }
    }
  }
}
