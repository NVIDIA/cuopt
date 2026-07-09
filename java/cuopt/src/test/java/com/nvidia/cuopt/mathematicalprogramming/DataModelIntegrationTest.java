/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

final class DataModelIntegrationTest {
  private static final double SOLVE_TOLERANCE = 1.0e-3;

  @TestFactory
  Stream<DynamicTest> dataModelsRoundTripAndSolve() {
    return cases().stream()
        .map(testCase -> DynamicTest.dynamicTest(testCase.name, () -> verify(testCase)));
  }

  private static void verify(CaseSpec testCase) {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();

    try (DataModel model = testCase.createDataModel()) {
      assertModelRoundTrip(testCase, model);
      if (testCase.hasQuadraticObjective()) {
        // QP callability is covered by NativeIntegrationTest. This case owns the independent
        // Java-to-JNI marshalling contract for quadratic objectives and constraints.
        return;
      }
      try (SolverSettings settings = createSettings(testCase);
          Solution solution = model.solve(settings)) {
        assertSolution(testCase, solution);
      }
    }
  }

  private static void assertModelRoundTrip(CaseSpec testCase, DataModel model) {
    assertEquals(testCase.numVariables, model.getNumVariables());
    assertEquals(testCase.numConstraints, model.getNumConstraints());
    assertEquals(testCase.values.length, model.getNumNonZeros());
    assertEquals(testCase.objectiveSense, model.getObjectiveSense());
    assertEquals(testCase.objectiveOffset, model.getObjectiveOffset(), 0.0);
    assertEquals(testCase.objectiveScalingFactor, model.getObjectiveScalingFactor(), 0.0);
    assertDoubleArrayEquals(testCase.objectiveCoefficients, model.getObjectiveCoefficients(), 0.0);

    CSRMatrix matrix = model.getConstraintMatrix();
    assertArrayEquals(testCase.rowOffsets, matrix.getRowOffsets());
    assertArrayEquals(testCase.columnIndices, matrix.getColumnIndices());
    assertDoubleArrayEquals(testCase.values, matrix.getValues(), 0.0);
    assertDoubleArrayEquals(testCase.variableLowerBounds, model.getVariableLowerBounds(), 0.0);
    assertDoubleArrayEquals(testCase.variableUpperBounds, model.getVariableUpperBounds(), 0.0);
    assertArrayEquals(testCase.variableTypes, model.getVariableTypes());
    assertArrayEquals(testCase.variableNames, model.getVariableNames());
    assertArrayEquals(testCase.rowNames, model.getRowNames());
    assertEquals(testCase.objectiveName, model.getObjectiveName());
    assertEquals(testCase.problemName, model.getProblemName());

    if (testCase.isRanged()) {
      assertDoubleArrayEquals(
          testCase.constraintLowerBounds, model.getConstraintLowerBounds(), 0.0);
      assertDoubleArrayEquals(
          testCase.constraintUpperBounds, model.getConstraintUpperBounds(), 0.0);
    } else {
      assertArrayEquals(testCase.constraintSense, model.getConstraintSense());
      assertDoubleArrayEquals(testCase.rhs, model.getConstraintRHS(), 0.0);
    }

    if (testCase.hasQuadraticObjective()) {
      assertArrayEquals(
          testCase.quadraticObjectiveRowOffsets, model.getQuadraticObjectiveOffsets());
      assertArrayEquals(
          testCase.quadraticObjectiveColumnIndices, model.getQuadraticObjectiveIndices());
      assertDoubleArrayEquals(
          testCase.quadraticObjectiveValues, model.getQuadraticObjectiveValues(), 0.0);
    }

    List<QuadraticConstraint> constraints = model.getQuadraticConstraints();
    assertEquals(testCase.hasQuadraticConstraint() ? 1 : 0, constraints.size());
    if (testCase.hasQuadraticConstraint()) {
      QuadraticConstraint constraint = constraints.get(0);
      assertEquals(testCase.quadraticConstraintName, constraint.getRowName());
      assertEquals(
          ConstraintSense.fromNative(testCase.quadraticConstraintSense), constraint.getSense());
      assertEquals(testCase.quadraticConstraintRHS, constraint.getRHS(), 0.0);
      assertArrayEquals(testCase.quadraticConstraintLinearIndices, constraint.getLinearIndices());
      assertDoubleArrayEquals(
          testCase.quadraticConstraintLinearValues, constraint.getLinearValues(), 0.0);
      assertArrayEquals(testCase.quadraticConstraintRows, constraint.getRows());
      assertArrayEquals(testCase.quadraticConstraintColumns, constraint.getColumns());
      assertDoubleArrayEquals(
          testCase.quadraticConstraintValues, constraint.getValues(), 0.0);
    }
  }

  private static void assertSolution(CaseSpec testCase, Solution solution) {
    assertEquals(testCase.hasIntegerVariables(), solution.isMIP());
    assertEquals(testCase.expectedCategory(), solution.getProblemCategory());
    assertTrue(
        Double.isNaN(solution.getSolveTime()) || solution.getSolveTime() >= 0.0,
        "solve time must be non-negative when available");

    if (!testCase.expectSolutionValues) {
      assertTrue(
          solution.getTerminationStatus() == TerminationStatus.INFEASIBLE
              || solution.getTerminationStatus() == TerminationStatus.UNBOUNDED_OR_INFEASIBLE,
          "expected an infeasible status, got " + solution.getTerminationStatus());
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

    if (testCase.hasIntegerVariables()) {
      assertDoesNotThrow(solution::getMIPStats);
      assertThrows(IllegalStateException.class, solution::getDualSolution);
      assertThrows(IllegalStateException.class, solution::getLPStats);
    } else {
      assertDoesNotThrow(solution::getDualSolution);
      assertDoesNotThrow(solution::getReducedCost);
      assertDoesNotThrow(solution::getLPStats);
      assertThrows(IllegalStateException.class, solution::getMIPStats);
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

  private static void assumeNativeLibrary() {
    String nativeDir = System.getProperty("cuopt.native.dir");
    Assumptions.assumeTrue(nativeDir != null && !nativeDir.isBlank(), "cuopt.native.dir is unset");
    Assumptions.assumeTrue(
        Files.exists(Path.of(nativeDir, System.mapLibraryName("cuopt_jni"))),
        "libcuopt_jni is not built");
  }

  private static void assumeCudaDriverAvailable() {
    try {
      Process process = new ProcessBuilder("nvidia-smi").redirectErrorStream(true).start();
      Assumptions.assumeTrue(process.waitFor() == 0, "CUDA driver is unavailable");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      Assumptions.assumeTrue(false, "CUDA driver check was interrupted");
    } catch (Exception e) {
      Assumptions.assumeTrue(false, "CUDA driver check failed: " + e.getMessage());
    }
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
                2.0,
                new String[] {"x0", "long_variable_1"},
                new String[] {"constraint_0"},
                "qp_objective",
                "qp_model")
            .withQuadraticConstraint(
                "qc0",
                (byte) 'L',
                100.0,
                new double[] {1.0},
                new int[] {0},
                new double[] {1.0},
                new int[] {0},
                new int[] {0}),
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
    private double objectiveScalingFactor = 1.0;
    private String[] variableNames = new String[0];
    private String[] rowNames = new String[0];
    private String objectiveName = "";
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

    private DataModel createDataModel() {
      CSRMatrix matrix = new CSRMatrix(values, columnIndices, rowOffsets);
      DataModel model =
          isRanged()
              ? DataModel.createRangedProblem(
                  numConstraints,
                  numVariables,
                  objectiveSense,
                  objectiveOffset,
                  objectiveCoefficients,
                  matrix,
                  constraintLowerBounds,
                  constraintUpperBounds,
                  variableLowerBounds,
                  variableUpperBounds,
                  variableTypes)
              : DataModel.createProblem(
                  numConstraints,
                  numVariables,
                  objectiveSense,
                  objectiveOffset,
                  objectiveCoefficients,
                  matrix,
                  constraintSense,
                  rhs,
                  variableLowerBounds,
                  variableUpperBounds,
                  variableTypes);
      if (hasQuadraticObjective()) {
        model.setQuadraticObjectiveMatrix(
            quadraticObjectiveValues,
            quadraticObjectiveColumnIndices,
            quadraticObjectiveRowOffsets);
      }
      model
          .setObjectiveScalingFactor(objectiveScalingFactor)
          .setVariableNames(variableNames)
          .setRowNames(rowNames)
          .setObjectiveName(objectiveName)
          .setProblemName(problemName);
      if (hasQuadraticConstraint()) {
        model.addQuadraticConstraint(
            quadraticConstraintName,
            quadraticConstraintLinearValues,
            quadraticConstraintLinearIndices,
            quadraticConstraintRHS,
            quadraticConstraintValues,
            quadraticConstraintRows,
            quadraticConstraintColumns,
            ConstraintSense.fromNative(quadraticConstraintSense));
      }
      return model;
    }

    private boolean isRanged() {
      return constraintLowerBounds != null;
    }

    private boolean hasQuadraticObjective() {
      return quadraticObjectiveValues != null;
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

    private ProblemCategory expectedCategory() {
      return hasIntegerVariables() ? ProblemCategory.MIP : ProblemCategory.LP;
    }

    private CaseSpec withQuadraticObjective(
        int[] rowOffsets, int[] columnIndices, double[] values) {
      this.quadraticObjectiveRowOffsets = Arrays.copyOf(rowOffsets, rowOffsets.length);
      this.quadraticObjectiveColumnIndices = Arrays.copyOf(columnIndices, columnIndices.length);
      this.quadraticObjectiveValues = Arrays.copyOf(values, values.length);
      return this;
    }

    private CaseSpec withMetadata(
        double scalingFactor,
        String[] variableNames,
        String[] rowNames,
        String objectiveName,
        String problemName) {
      this.objectiveScalingFactor = scalingFactor;
      this.variableNames = Arrays.copyOf(variableNames, variableNames.length);
      this.rowNames = Arrays.copyOf(rowNames, rowNames.length);
      this.objectiveName = objectiveName;
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
