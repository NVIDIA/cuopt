/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

final class PythonParityTest {
  private static final double MODEL_TOLERANCE = 0.0;
  private static final double SOLVE_TOLERANCE = 1.0e-4;
  private static ProcessResult pythonProbe;

  @TestFactory
  Stream<DynamicTest> pythonAndJavaBindingsMatch() {
    return cases().stream()
        .map(
            testCase ->
                DynamicTest.dynamicTest(testCase.name, () -> assertMatchesPython(testCase)));
  }

  private static void assertMatchesPython(CaseSpec testCase)
      throws IOException, InterruptedException, URISyntaxException {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    assumePythonCuOptAvailable();

    PythonResult pythonResult = runPython(testCase);
    try (DataModel model = testCase.createDataModel()) {
      assertModelMatchesPython(testCase, model, pythonResult);

      try (SolverSettings settings = createSettings(testCase);
          Solution solution = model.solve(settings)) {
        JavaResult javaResult = JavaResult.from(solution);
        assertSolutionMatchesPython(testCase, javaResult, pythonResult);
      }
    }
  }

  private static void assertModelMatchesPython(
      CaseSpec testCase, DataModel javaModel, PythonResult pythonResult) {
    assertEquals(
        javaModel.getNumVariables(),
        pythonResult.intValue("model.num_variables"),
        testCase.name + " num variables");
    assertEquals(
        javaModel.getNumConstraints(),
        pythonResult.intValue("model.num_constraints"),
        testCase.name + " num constraints");
    assertEquals(
        javaModel.getNumNonZeros(),
        pythonResult.intValue("model.num_nonzeros"),
        testCase.name + " num nonzeros");
    assertEquals(
        javaModel.getObjectiveSense().nativeValue(),
        pythonResult.intValue("model.objective_sense"),
        testCase.name + " objective sense");
    assertEquals(
        javaModel.getObjectiveOffset(),
        pythonResult.doubleValue("model.objective_offset"),
        MODEL_TOLERANCE,
        testCase.name + " objective offset");
    assertEquals(
        javaModel.getObjectiveScalingFactor(),
        pythonResult.doubleValue("model.objective_scaling_factor"),
        MODEL_TOLERANCE,
        testCase.name + " objective scaling factor");
    assertDoubleArrayEquals(
        testCase.name + " objective coefficients",
        javaModel.getObjectiveCoefficients(),
        pythonResult.doubleArray("model.objective_coefficients"),
        MODEL_TOLERANCE);

    CsrMatrix matrix = javaModel.getConstraintMatrix();
    assertDoubleArrayEquals(
        testCase.name + " CSR values",
        matrix.getValues(),
        pythonResult.doubleArray("model.csr_values"),
        MODEL_TOLERANCE);
    assertArrayEquals(
        matrix.getColumnIndices(),
        pythonResult.intArray("model.csr_column_indices"),
        testCase.name + " CSR column indices");
    assertArrayEquals(
        matrix.getRowOffsets(),
        pythonResult.intArray("model.csr_row_offsets"),
        testCase.name + " CSR row offsets");

    assertDoubleArrayEquals(
        testCase.name + " variable lower bounds",
        javaModel.getVariableLowerBounds(),
        pythonResult.doubleArray("model.variable_lower_bounds"),
        MODEL_TOLERANCE);
    assertDoubleArrayEquals(
        testCase.name + " variable upper bounds",
        javaModel.getVariableUpperBounds(),
        pythonResult.doubleArray("model.variable_upper_bounds"),
        MODEL_TOLERANCE);
    assertEquals(
        byteArrayAsCsv(javaModel.getVariableTypes()),
        pythonResult.stringValue("model.variable_types"),
        testCase.name + " variable types");
    assertEquals(
        stringArrayAsCsv(javaModel.getVariableNames()),
        pythonResult.stringValue("model.variable_names"),
        testCase.name + " variable names");
    assertEquals(
        stringArrayAsCsv(javaModel.getRowNames()),
        pythonResult.stringValue("model.row_names"),
        testCase.name + " row names");
    assertEquals(
        javaModel.getObjectiveName(),
        pythonResult.stringValue("model.objective_name"),
        testCase.name + " objective name");
    assertEquals(
        javaModel.getProblemName(),
        pythonResult.stringValue("model.problem_name"),
        testCase.name + " problem name");

    if (testCase.hasQuadraticObjective()) {
      assertDoubleArrayEquals(
          testCase.name + " quadratic objective values",
          javaModel.getQuadraticObjectiveValues(),
          pythonResult.doubleArray("model.quadratic_objective_values"),
          MODEL_TOLERANCE);
      assertArrayEquals(
          javaModel.getQuadraticObjectiveIndices(),
          pythonResult.intArray("model.quadratic_objective_column_indices"),
          testCase.name + " quadratic objective column indices");
      assertArrayEquals(
          javaModel.getQuadraticObjectiveOffsets(),
          pythonResult.intArray("model.quadratic_objective_row_offsets"),
          testCase.name + " quadratic objective row offsets");
    }

    List<QuadraticConstraint> quadraticConstraints = javaModel.getQuadraticConstraints();
    assertEquals(
        quadraticConstraints.size(),
        pythonResult.intValue("model.quadratic_constraint_count"),
        testCase.name + " quadratic constraint count");
    for (int i = 0; i < quadraticConstraints.size(); i++) {
      QuadraticConstraint constraint = quadraticConstraints.get(i);
      String prefix = "model.quadratic_constraint." + i;
      assertEquals(
          constraint.getRowIndex(),
          pythonResult.intValue(prefix + ".row_index"),
          testCase.name + " quadratic constraint row index");
      assertEquals(
          constraint.getRowName(),
          pythonResult.stringValue(prefix + ".row_name"),
          testCase.name + " quadratic constraint row name");
      assertEquals(
          (char) constraint.getSense().nativeValue(),
          pythonResult.stringValue(prefix + ".sense").charAt(0),
          testCase.name + " quadratic constraint sense");
      assertDoubleArrayEquals(
          testCase.name + " quadratic constraint linear values",
          constraint.getLinearValues(),
          pythonResult.doubleArray(prefix + ".linear_values"),
          MODEL_TOLERANCE);
      assertArrayEquals(
          constraint.getLinearIndices(),
          pythonResult.intArray(prefix + ".linear_indices"),
          testCase.name + " quadratic constraint linear indices");
      assertDoubleEquals(
          testCase.name + " quadratic constraint rhs",
          constraint.getRHS(),
          pythonResult.doubleValue(prefix + ".rhs"),
          MODEL_TOLERANCE);
      assertArrayEquals(
          constraint.getRows(),
          pythonResult.intArray(prefix + ".rows"),
          testCase.name + " quadratic constraint rows");
      assertArrayEquals(
          constraint.getColumns(),
          pythonResult.intArray(prefix + ".columns"),
          testCase.name + " quadratic constraint columns");
      assertDoubleArrayEquals(
          testCase.name + " quadratic constraint values",
          constraint.getValues(),
          pythonResult.doubleArray(prefix + ".values"),
          MODEL_TOLERANCE);
    }

    if (testCase.isRanged()) {
      assertDoubleArrayEquals(
          testCase.name + " constraint lower bounds",
          javaModel.getConstraintLowerBounds(),
          pythonResult.doubleArray("model.constraint_lower_bounds"),
          MODEL_TOLERANCE);
      assertDoubleArrayEquals(
          testCase.name + " constraint upper bounds",
          javaModel.getConstraintUpperBounds(),
          pythonResult.doubleArray("model.constraint_upper_bounds"),
          MODEL_TOLERANCE);
    } else {
      assertDoubleArrayEquals(
          testCase.name + " rhs",
          javaModel.getConstraintRhs(),
          pythonResult.doubleArray("model.rhs"),
          MODEL_TOLERANCE);
      assertEquals(
          byteArrayAsCsv(javaModel.getConstraintSense()),
          pythonResult.stringValue("model.constraint_sense"),
          testCase.name + " constraint sense");
    }
  }

  private static void assertSolutionMatchesPython(
      CaseSpec testCase, JavaResult javaResult, PythonResult pythonResult) {
    assertEquals(
        javaResult.isMip,
        pythonResult.booleanValue("solution.is_mip"),
        testCase.name + " solution category");
    assertEquals(
        javaResult.status.nativeValue(),
        pythonResult.intValue("solution.status"),
        testCase.name + " termination status");
    assertEquals(
        javaResult.errorStatus,
        pythonResult.intValue("solution.error_status"),
        testCase.name + " error status");
    assertEquals(
        javaResult.errorMessage,
        pythonResult.stringValue("solution.error_message"),
        testCase.name + " error message");
    assertNonNegativeIfAvailable(testCase.name + " Java solve time", javaResult.solveTime);
    assertNonNegativeIfAvailable(
        testCase.name + " Python solve time", pythonResult.doubleValue("solution.solve_time"));
    assertEquals(
        javaResult.dualUnavailable,
        pythonResult.booleanValue("solution.dual_unavailable"),
        testCase.name + " dual availability");
    assertEquals(
        javaResult.dualObjectiveUnavailable,
        pythonResult.booleanValue("solution.dual_objective_unavailable"),
        testCase.name + " dual objective availability");
    assertEquals(
        javaResult.reducedCostUnavailable,
        pythonResult.booleanValue("solution.reduced_cost_unavailable"),
        testCase.name + " reduced-cost availability");
    assertEquals(
        javaResult.lpStatsUnavailable,
        pythonResult.booleanValue("solution.lp_stats_unavailable"),
        testCase.name + " LP stats availability");
    assertEquals(
        javaResult.mipStatsUnavailable,
        pythonResult.booleanValue("solution.mip_stats_unavailable"),
        testCase.name + " MIP stats availability");

    if (!testCase.expectSolutionValues) {
      return;
    }

    assertDoubleArrayEquals(
        testCase.name + " primal solution",
        javaResult.primal,
        pythonResult.doubleArray("solution.primal"),
        testCase.solutionTolerance);
    assertDoubleEquals(
        testCase.name + " objective",
        javaResult.objective,
        pythonResult.doubleValue("solution.objective"),
        testCase.solutionTolerance);

    if (!javaResult.dualUnavailable) {
      assertDoubleArrayEquals(
          testCase.name + " dual solution",
          javaResult.dual,
          pythonResult.doubleArray("solution.dual"),
          testCase.solutionTolerance);
    }
    if (!javaResult.dualObjectiveUnavailable) {
      assertDoubleEquals(
          testCase.name + " dual objective",
          javaResult.dualObjective,
          pythonResult.doubleValue("solution.dual_objective"),
          testCase.solutionTolerance);
    }
    if (!javaResult.reducedCostUnavailable) {
      assertDoubleArrayEquals(
          testCase.name + " reduced cost",
          javaResult.reducedCost,
          pythonResult.doubleArray("solution.reduced_cost"),
          testCase.solutionTolerance);
    }
    if (!javaResult.lpStatsUnavailable) {
      assertDoubleEquals(
          testCase.name + " LP primal residual",
          javaResult.lpPrimalResidual,
          pythonResult.doubleValue("solution.lp_primal_residual"),
          testCase.solutionTolerance);
      assertDoubleEquals(
          testCase.name + " LP dual residual",
          javaResult.lpDualResidual,
          pythonResult.doubleValue("solution.lp_dual_residual"),
          testCase.solutionTolerance);
      assertDoubleEquals(
          testCase.name + " LP gap",
          javaResult.lpGap,
          pythonResult.doubleValue("solution.lp_gap"),
          testCase.solutionTolerance);
      assertEquals(
          javaResult.lpIterations,
          pythonResult.intValue("solution.lp_iterations"),
          testCase.name + " LP iterations");
      assertEquals(
          javaResult.lpSolvedBy,
          pythonResult.intValue("solution.solved_by"),
          testCase.name + " LP solved-by method");
    }
    if (!javaResult.mipStatsUnavailable) {
      assertDoubleEquals(
          testCase.name + " MIP gap",
          javaResult.mipGap,
          pythonResult.doubleValue("solution.mip_gap"),
          testCase.solutionTolerance);
      assertDoubleEquals(
          testCase.name + " solution bound",
          javaResult.solutionBound,
          pythonResult.doubleValue("solution.solution_bound"),
          testCase.solutionTolerance);
      assertNonNegativeIfAvailable(
          testCase.name + " Java MIP presolve time", javaResult.mipPresolveTime);
      assertNonNegativeIfAvailable(
          testCase.name + " Python MIP presolve time",
          pythonResult.doubleValue("solution.mip_presolve_time"));
      assertDoubleEquals(
          testCase.name + " max constraint violation",
          javaResult.maxConstraintViolation,
          pythonResult.doubleValue("solution.max_constraint_violation"),
          testCase.solutionTolerance);
      assertDoubleEquals(
          testCase.name + " max integer violation",
          javaResult.maxIntViolation,
          pythonResult.doubleValue("solution.max_int_violation"),
          testCase.solutionTolerance);
      assertDoubleEquals(
          testCase.name + " max variable bound violation",
          javaResult.maxVariableBoundViolation,
          pythonResult.doubleValue("solution.max_variable_bound_violation"),
          testCase.solutionTolerance);
      assertEquals(
          javaResult.numNodes,
          pythonResult.intValue("solution.num_nodes"),
          testCase.name + " MIP node count");
      assertEquals(
          javaResult.numSimplexIterations,
          pythonResult.intValue("solution.num_simplex_iterations"),
          testCase.name + " MIP simplex iterations");
    }
  }

  private static SolverSettings createSettings(CaseSpec testCase) {
    SolverSettings settings = new SolverSettings();
    settings.setParameter(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false);
    settings.setParameter(CuOptConstants.CUOPT_TIME_LIMIT, 30.0);
    settings.setParameter(CuOptConstants.CUOPT_RANDOM_SEED, 1);
    if (testCase.hasIntegerVariables()) {
      settings.setParameter(
          CuOptConstants.CUOPT_MIP_DETERMINISM_MODE, CuOptConstants.CUOPT_MODE_DETERMINISTIC);
      settings.setParameter(CuOptConstants.CUOPT_MIP_ABSOLUTE_GAP, 1.0e-8);
      settings.setParameter(CuOptConstants.CUOPT_MIP_RELATIVE_GAP, 1.0e-8);
    } else if (testCase.hasQuadraticObjective()) {
      settings.setParameter(CuOptConstants.CUOPT_ITERATION_LIMIT, 50);
    } else {
      settings.setMethod(SolverMethod.PDLP);
      settings.setPdlpSolverMode(PDLPSolverMode.STABLE1);
      settings.setParameter(CuOptConstants.CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1.0e-7);
      settings.setParameter(CuOptConstants.CUOPT_RELATIVE_PRIMAL_TOLERANCE, 1.0e-7);
      settings.setParameter(CuOptConstants.CUOPT_ABSOLUTE_DUAL_TOLERANCE, 1.0e-7);
      settings.setParameter(CuOptConstants.CUOPT_RELATIVE_DUAL_TOLERANCE, 1.0e-7);
      settings.setParameter(CuOptConstants.CUOPT_ABSOLUTE_GAP_TOLERANCE, 1.0e-7);
      settings.setParameter(CuOptConstants.CUOPT_RELATIVE_GAP_TOLERANCE, 1.0e-7);
    }
    return settings;
  }

  private static PythonResult runPython(CaseSpec testCase)
      throws IOException, InterruptedException, URISyntaxException {
    Path caseFile = Files.createTempFile("cuopt-java-python-parity-", ".json");
    try {
      Files.writeString(caseFile, testCase.toJson(), StandardCharsets.UTF_8);
      ProcessResult result =
          runPythonProcess(List.of(pythonHelperPath().toString(), caseFile.toString()));
      assertEquals(0, result.exitCode, "Python oracle failed:\n" + result.output);
      return new PythonResult(result.output);
    } finally {
      Files.deleteIfExists(caseFile);
    }
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
      boolean exited = process.waitFor() == 0;
      Assumptions.assumeTrue(exited, "CUDA driver is unavailable");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      Assumptions.assumeTrue(false, "CUDA driver check was interrupted");
    } catch (Exception e) {
      Assumptions.assumeTrue(false, "CUDA driver check failed: " + e.getMessage());
    }
  }

  private static void assumePythonCuOptAvailable()
      throws IOException, InterruptedException, URISyntaxException {
    if (pythonProbe == null) {
      pythonProbe = runPythonProcess(List.of(pythonHelperPath().toString(), "--probe"));
    }
    Assumptions.assumeTrue(
        pythonProbe.exitCode == 0, "Python cuOpt import failed:\n" + pythonProbe.output);
  }

  private static ProcessResult runPythonProcess(List<String> arguments)
      throws IOException, InterruptedException {
    List<String> command = new ArrayList<>();
    command.add(pythonExecutable());
    command.addAll(arguments);
    ProcessBuilder builder = new ProcessBuilder(command);
    builder.redirectErrorStream(true);
    builder.environment().put("CUOPT_EXTRA_TIMESTAMPS", "false");
    Process process = builder.start();
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    int exitCode = process.waitFor();
    return new ProcessResult(exitCode, output);
  }

  private static String pythonExecutable() {
    String property = System.getProperty("cuopt.python");
    if (property != null && !property.isBlank()) {
      return property;
    }
    String environment = System.getenv("PYTHON");
    if (environment != null && !environment.isBlank()) {
      return environment;
    }
    return "python3";
  }

  private static Path pythonHelperPath() throws URISyntaxException {
    URL resource = PythonParityTest.class.getResource("/python_binding_parity.py");
    assertNotNull(resource, "python_binding_parity.py test resource is missing");
    return Path.of(resource.toURI());
  }

  private static void assertDoubleArrayEquals(
      String message, double[] actual, double[] expected, double tolerance) {
    assertEquals(expected.length, actual.length, message + " length");
    for (int i = 0; i < expected.length; i++) {
      assertDoubleEquals(message + "[" + i + "]", actual[i], expected[i], tolerance);
    }
  }

  private static void assertDoubleEquals(
      String message, double actual, double expected, double tolerance) {
    if (Double.isNaN(actual) || Double.isNaN(expected)) {
      assertTrue(
          Double.isNaN(actual) && Double.isNaN(expected),
          message + " expected both values to be NaN, actual=" + actual + ", expected=" + expected);
      return;
    }
    if (Double.isInfinite(actual) || Double.isInfinite(expected)) {
      assertTrue(
          Double.compare(actual, expected) == 0,
          message + " expected matching infinities, actual=" + actual + ", expected=" + expected);
      return;
    }
    assertEquals(expected, actual, tolerance, message);
  }

  private static void assertNonNegativeIfAvailable(String message, double value) {
    assertTrue(
        Double.isNaN(value) || value >= 0.0,
        message + " should be NaN when unavailable or non-negative when available");
  }

  private static String byteArrayAsCsv(byte[] values) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append((char) values[i]);
    }
    return builder.toString();
  }

  private static String stringArrayAsCsv(String[] values) {
    return String.join(",", values);
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
            new byte[] {(byte) 'G'},
            new double[] {3.0},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {(byte) 'C', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'L', (byte) 'L', (byte) 'L'},
            new double[] {4.0, 2.0, 3.0},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {(byte) 'C', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'E'},
            new double[] {5.0},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {5.0, 5.0},
            new byte[] {(byte) 'C', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'C', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'E'},
            new double[] {1.0},
            null,
            null,
            new double[] {-2.0, -1.0},
            new double[] {2.0, 3.0},
            new byte[] {(byte) 'C', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'C', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'G'},
            new double[] {2.5},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {10.0, 10.0},
            new byte[] {(byte) 'I', (byte) 'I'},
            true,
            SOLVE_TOLERANCE),
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
            new byte[] {(byte) 'L'},
            new double[] {2.5},
            null,
            null,
            new double[] {0.0, 0.0},
            new double[] {3.0, 10.0},
            new byte[] {(byte) 'I', (byte) 'C'},
            true,
            SOLVE_TOLERANCE),
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
                new byte[] {(byte) 'C', (byte) 'C'},
                true,
                1.0e-3)
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
            new byte[] {(byte) 'G', (byte) 'L'},
            new double[] {1.0, 0.0},
            null,
            null,
            new double[] {0.0},
            new double[] {10.0},
            new byte[] {(byte) 'C'},
            false,
            SOLVE_TOLERANCE));
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
    private final double solutionTolerance;
    private double objectiveScalingFactor = 1.0;
    private String[] variableNames = new String[0];
    private String[] rowNames = new String[0];
    private String objectiveName = "";
    private String problemName = "";
    private String quadraticConstraintName;
    private byte quadraticConstraintSense;
    private double quadraticConstraintRhs;
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
        double solutionTolerance) {
      this.name = name;
      this.numConstraints = numConstraints;
      this.numVariables = numVariables;
      this.objectiveSense = objectiveSense;
      this.objectiveOffset = objectiveOffset;
      this.objectiveCoefficients =
          Arrays.copyOf(objectiveCoefficients, objectiveCoefficients.length);
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
      this.solutionTolerance = solutionTolerance;
    }

    private DataModel createDataModel() {
      CsrMatrix matrix = new CsrMatrix(rowOffsets, columnIndices, values);
      DataModel model;
      if (isRanged()) {
        model =
            DataModel.createRangedProblem(
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
                variableTypes);
      } else {
        model =
            DataModel.createProblem(
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
      }
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
            quadraticConstraintRhs,
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
        if (type == (byte) 'I' || type == (byte) 'S') {
          return true;
        }
      }
      return false;
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
      this.quadraticConstraintRhs = rhs;
      this.quadraticConstraintLinearValues = Arrays.copyOf(linearValues, linearValues.length);
      this.quadraticConstraintLinearIndices = Arrays.copyOf(linearIndices, linearIndices.length);
      this.quadraticConstraintValues = Arrays.copyOf(values, values.length);
      this.quadraticConstraintRows = Arrays.copyOf(rows, rows.length);
      this.quadraticConstraintColumns = Arrays.copyOf(columns, columns.length);
      return this;
    }

    private QuadraticExpression createQuadraticObjective() {
      Problem shell = new Problem();
      Variable[] variables = new Variable[numVariables];
      for (int i = 0; i < numVariables; i++) {
        variables[i] = shell.addVariable();
      }

      QuadraticExpression expression = new QuadraticExpression();
      for (int row = 0; row < quadraticObjectiveRowOffsets.length - 1; row++) {
        for (int offset = quadraticObjectiveRowOffsets[row];
            offset < quadraticObjectiveRowOffsets[row + 1];
            offset++) {
          expression =
              expression.plus(
                  variables[row],
                  variables[quadraticObjectiveColumnIndices[offset]],
                  quadraticObjectiveValues[offset]);
        }
      }
      return expression;
    }

    private String toJson() {
      StringBuilder builder = new StringBuilder();
      builder.append('{');
      appendField(builder, "name", name);
      appendField(builder, "num_constraints", numConstraints);
      appendField(builder, "num_variables", numVariables);
      appendField(builder, "objective_sense", objectiveSense.nativeValue());
      appendField(builder, "objective_offset", objectiveOffset);
      appendField(builder, "objective_scaling_factor", objectiveScalingFactor);
      appendField(builder, "objective_coefficients", objectiveCoefficients);
      appendField(builder, "csr_row_offsets", rowOffsets);
      appendField(builder, "csr_column_indices", columnIndices);
      appendField(builder, "csr_values", values);
      if (hasQuadraticObjective()) {
        appendField(builder, "quadratic_objective_row_offsets", quadraticObjectiveRowOffsets);
        appendField(builder, "quadratic_objective_column_indices", quadraticObjectiveColumnIndices);
        appendField(builder, "quadratic_objective_values", quadraticObjectiveValues);
      }
      if (hasQuadraticConstraint()) {
        appendField(builder, "quadratic_constraint_name", quadraticConstraintName);
        appendField(builder, "quadratic_constraint_sense", String.valueOf((char) quadraticConstraintSense));
        appendField(builder, "quadratic_constraint_rhs", quadraticConstraintRhs);
        appendField(builder, "quadratic_constraint_linear_values", quadraticConstraintLinearValues);
        appendField(builder, "quadratic_constraint_linear_indices", quadraticConstraintLinearIndices);
        appendField(builder, "quadratic_constraint_values", quadraticConstraintValues);
        appendField(builder, "quadratic_constraint_rows", quadraticConstraintRows);
        appendField(builder, "quadratic_constraint_columns", quadraticConstraintColumns);
      }
      if (isRanged()) {
        appendField(builder, "constraint_lower_bounds", constraintLowerBounds);
        appendField(builder, "constraint_upper_bounds", constraintUpperBounds);
      } else {
        appendField(builder, "constraint_sense", constraintSense);
        appendField(builder, "rhs", rhs);
      }
      appendField(builder, "variable_lower_bounds", variableLowerBounds);
      appendField(builder, "variable_upper_bounds", variableUpperBounds);
      appendField(builder, "variable_types", variableTypes);
      if (variableNames.length > 0) {
        appendField(builder, "variable_names", variableNames);
      }
      if (rowNames.length > 0) {
        appendField(builder, "row_names", rowNames);
      }
      if (!objectiveName.isEmpty()) {
        appendField(builder, "objective_name", objectiveName);
      }
      if (!problemName.isEmpty()) {
        appendField(builder, "problem_name", problemName);
      }
      builder.append('}');
      return builder.toString();
    }
  }

  private static final class JavaResult {
    private final boolean isMip;
    private final TerminationStatus status;
    private final int errorStatus;
    private final String errorMessage;
    private final double solveTime;
    private final double objective;
    private final double dualObjective;
    private final double[] primal;
    private final double[] dual;
    private final double[] reducedCost;
    private final boolean dualUnavailable;
    private final boolean dualObjectiveUnavailable;
    private final boolean reducedCostUnavailable;
    private final boolean lpStatsUnavailable;
    private final boolean mipStatsUnavailable;
    private final double lpPrimalResidual;
    private final double lpDualResidual;
    private final double lpGap;
    private final int lpIterations;
    private final int lpSolvedBy;
    private final double mipGap;
    private final double solutionBound;
    private final double mipPresolveTime;
    private final double maxConstraintViolation;
    private final double maxIntViolation;
    private final double maxVariableBoundViolation;
    private final int numNodes;
    private final int numSimplexIterations;

    private JavaResult(
        boolean isMip,
        TerminationStatus status,
        int errorStatus,
        String errorMessage,
        double solveTime,
        double objective,
        double dualObjective,
        double[] primal,
        double[] dual,
        double[] reducedCost,
        boolean dualUnavailable,
        boolean dualObjectiveUnavailable,
        boolean reducedCostUnavailable,
        boolean lpStatsUnavailable,
        boolean mipStatsUnavailable,
        double lpPrimalResidual,
        double lpDualResidual,
        double lpGap,
        int lpIterations,
        int lpSolvedBy,
        double mipGap,
        double solutionBound,
        double mipPresolveTime,
        double maxConstraintViolation,
        double maxIntViolation,
        double maxVariableBoundViolation,
        int numNodes,
        int numSimplexIterations) {
      this.isMip = isMip;
      this.status = status;
      this.errorStatus = errorStatus;
      this.errorMessage = errorMessage;
      this.solveTime = solveTime;
      this.objective = objective;
      this.dualObjective = dualObjective;
      this.primal = Arrays.copyOf(primal, primal.length);
      this.dual = Arrays.copyOf(dual, dual.length);
      this.reducedCost = Arrays.copyOf(reducedCost, reducedCost.length);
      this.dualUnavailable = dualUnavailable;
      this.dualObjectiveUnavailable = dualObjectiveUnavailable;
      this.reducedCostUnavailable = reducedCostUnavailable;
      this.lpStatsUnavailable = lpStatsUnavailable;
      this.mipStatsUnavailable = mipStatsUnavailable;
      this.lpPrimalResidual = lpPrimalResidual;
      this.lpDualResidual = lpDualResidual;
      this.lpGap = lpGap;
      this.lpIterations = lpIterations;
      this.lpSolvedBy = lpSolvedBy;
      this.mipGap = mipGap;
      this.solutionBound = solutionBound;
      this.mipPresolveTime = mipPresolveTime;
      this.maxConstraintViolation = maxConstraintViolation;
      this.maxIntViolation = maxIntViolation;
      this.maxVariableBoundViolation = maxVariableBoundViolation;
      this.numNodes = numNodes;
      this.numSimplexIterations = numSimplexIterations;
    }

    private static JavaResult from(Solution solution) {
      boolean dualUnavailable = false;
      boolean dualObjectiveUnavailable = false;
      boolean reducedCostUnavailable = false;
      boolean lpStatsUnavailable = false;
      boolean mipStatsUnavailable = false;
      double[] dual = new double[0];
      double[] reducedCost = new double[0];
      double dualObjective = 0.0;
      double lpPrimalResidual = 0.0;
      double lpDualResidual = 0.0;
      double lpGap = 0.0;
      int lpIterations = 0;
      int lpSolvedBy = SolverMethod.UNSET.nativeValue();
      double mipGap = 0.0;
      double solutionBound = 0.0;
      double mipPresolveTime = 0.0;
      double maxConstraintViolation = 0.0;
      double maxIntViolation = 0.0;
      double maxVariableBoundViolation = 0.0;
      int numNodes = 0;
      int numSimplexIterations = 0;

      try {
        dual = solution.getDualSolution();
      } catch (IllegalStateException e) {
        dualUnavailable = true;
      }
      try {
        dualObjective = solution.getDualObjective();
      } catch (IllegalStateException e) {
        dualObjectiveUnavailable = true;
      }
      try {
        reducedCost = solution.getReducedCost();
      } catch (IllegalStateException e) {
        reducedCostUnavailable = true;
      }
      try {
        LPStats lpStats = solution.getLpStats();
        lpPrimalResidual = lpStats.getPrimalResidual();
        lpDualResidual = lpStats.getDualResidual();
        lpGap = lpStats.getGap();
        lpIterations = lpStats.getNumIterations();
        lpSolvedBy = lpStats.getSolvedBy().nativeValue();
      } catch (IllegalStateException e) {
        lpStatsUnavailable = true;
      }
      try {
        MIPStats mipStats = solution.getMipStats();
        mipGap = solution.getMipGap();
        solutionBound = solution.getSolutionBound();
        mipPresolveTime = mipStats.getPresolveTime();
        maxConstraintViolation = mipStats.getMaxConstraintViolation();
        maxIntViolation = mipStats.getMaxIntViolation();
        maxVariableBoundViolation = mipStats.getMaxVariableBoundViolation();
        numNodes = mipStats.getNumNodes();
        numSimplexIterations = mipStats.getNumSimplexIterations();
      } catch (IllegalStateException e) {
        mipStatsUnavailable = true;
      }

      if (solution.isMip()) {
        assertThrows(IllegalStateException.class, solution::getDualSolution);
        assertThrows(IllegalStateException.class, solution::getDualObjective);
        assertThrows(IllegalStateException.class, solution::getReducedCost);
        assertFalse(mipStatsUnavailable);
        assertTrue(lpStatsUnavailable);
      } else {
        assertThrows(IllegalStateException.class, solution::getMipStats);
        assertFalse(dualUnavailable);
        assertFalse(reducedCostUnavailable);
        assertFalse(lpStatsUnavailable);
        assertTrue(mipStatsUnavailable);
      }

      return new JavaResult(
          solution.isMip(),
          solution.getTerminationStatus(),
          solution.getErrorStatus(),
          solution.getErrorMessage(),
          solution.getSolveTime(),
          solution.getPrimalObjective(),
          dualObjective,
          solution.getPrimalSolution(),
          dual,
          reducedCost,
          dualUnavailable,
          dualObjectiveUnavailable,
          reducedCostUnavailable,
          lpStatsUnavailable,
          mipStatsUnavailable,
          lpPrimalResidual,
          lpDualResidual,
          lpGap,
          lpIterations,
          lpSolvedBy,
          mipGap,
          solutionBound,
          mipPresolveTime,
          maxConstraintViolation,
          maxIntViolation,
          maxVariableBoundViolation,
          numNodes,
          numSimplexIterations);
    }
  }

  private static final class PythonResult {
    private final Map<String, String> values = new LinkedHashMap<>();

    private PythonResult(String output) {
      for (String line : output.split("\\R")) {
        if (!line.startsWith("CUOPT_COMPARE ")) {
          continue;
        }
        String result = line.substring("CUOPT_COMPARE ".length());
        int equals = result.indexOf('=');
        if (equals > 0) {
          values.put(result.substring(0, equals), result.substring(equals + 1));
        }
      }
    }

    private String stringValue(String key) {
      String value = values.get(key);
      assertNotNull(value, "Python result is missing " + key);
      return value;
    }

    private int intValue(String key) {
      return Integer.parseInt(stringValue(key));
    }

    private boolean booleanValue(String key) {
      return Boolean.parseBoolean(stringValue(key));
    }

    private double doubleValue(String key) {
      return parseDouble(stringValue(key));
    }

    private int[] intArray(String key) {
      String value = stringValue(key);
      if (value.isEmpty()) {
        return new int[0];
      }
      String[] parts = value.split(",");
      int[] result = new int[parts.length];
      for (int i = 0; i < parts.length; i++) {
        result[i] = Integer.parseInt(parts[i]);
      }
      return result;
    }

    private double[] doubleArray(String key) {
      String value = stringValue(key);
      if (value.isEmpty()) {
        return new double[0];
      }
      String[] parts = value.split(",");
      double[] result = new double[parts.length];
      for (int i = 0; i < parts.length; i++) {
        result[i] = parseDouble(parts[i]);
      }
      return result;
    }

    private static double parseDouble(String value) {
      if ("nan".equalsIgnoreCase(value)) {
        return Double.NaN;
      }
      if ("inf".equalsIgnoreCase(value) || "+inf".equalsIgnoreCase(value)) {
        return Double.POSITIVE_INFINITY;
      }
      if ("-inf".equalsIgnoreCase(value)) {
        return Double.NEGATIVE_INFINITY;
      }
      return Double.parseDouble(value);
    }
  }

  private static final class ProcessResult {
    private final int exitCode;
    private final String output;

    private ProcessResult(int exitCode, String output) {
      this.exitCode = exitCode;
      this.output = output;
    }
  }

  private static void appendField(StringBuilder builder, String name, String value) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":\"");
    builder.append(value.replace("\\", "\\\\").replace("\"", "\\\""));
    builder.append('"');
  }

  private static void appendField(StringBuilder builder, String name, int value) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":").append(value);
  }

  private static void appendField(StringBuilder builder, String name, double value) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":").append(Double.toString(value));
  }

  private static void appendField(StringBuilder builder, String name, int[] values) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":[");
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append(values[i]);
    }
    builder.append(']');
  }

  private static void appendField(StringBuilder builder, String name, double[] values) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":[");
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append(Double.toString(values[i]));
    }
    builder.append(']');
  }

  private static void appendField(StringBuilder builder, String name, byte[] values) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":[");
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append('"').append((char) values[i]).append('"');
    }
    builder.append(']');
  }

  private static void appendField(StringBuilder builder, String name, String[] values) {
    appendSeparator(builder);
    builder.append('"').append(name).append("\":[");
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append('"')
          .append(values[i].replace("\\", "\\\\").replace("\"", "\\\""))
          .append('"');
    }
    builder.append(']');
  }

  private static void appendSeparator(StringBuilder builder) {
    if (builder.charAt(builder.length() - 1) != '{') {
      builder.append(',');
    }
  }
}
