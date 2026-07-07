/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

final class NativeIntegrationTest {
  private static void assumeNativeLibrary() {
    String nativeDir = System.getProperty("cuopt.native.dir");
    Assumptions.assumeTrue(nativeDir != null && !nativeDir.isBlank(), "cuopt.native.dir is unset");
    Assumptions.assumeTrue(
        Files.exists(Path.of(nativeDir, System.mapLibraryName("cuopt_jni"))),
        "libcuopt_jni is not built");
  }

  @Test
  void solverParameterNamesAreAvailable() {
    assumeNativeLibrary();
    assertTrue(SolverSettings.getSolverParameterNames().contains(CuOptConstants.CUOPT_TIME_LIMIT));
  }

  @Test
  void settingsExposeTypedValuesAndParameterFileRoundTrip() throws Exception {
    assumeNativeLibrary();
    Path file = Files.createTempFile("cuopt-java-settings-", ".cfg");
    try (SolverSettings settings = new SolverSettings()) {
      settings.setParameter(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false);
      settings.setParameter(CuOptConstants.CUOPT_TIME_LIMIT, 12.5);
      settings.setOptimalityTolerance(1.0e-6);
      assertEquals(Boolean.FALSE, settings.getTypedParameter(CuOptConstants.CUOPT_LOG_TO_CONSOLE));
      assertEquals(12.5, (Double) settings.getTypedParameter(CuOptConstants.CUOPT_TIME_LIMIT), 1e-12);
      assertEquals(
          1.0e-6,
          (Double) settings.getTypedParameter(CuOptConstants.CUOPT_ABSOLUTE_PRIMAL_TOLERANCE),
          1e-12);
      assertEquals(
          12.5,
          Double.parseDouble(settings.getParameterAsString(CuOptConstants.CUOPT_TIME_LIMIT)),
          1e-12);
      assertTrue(settings.toDict().containsKey("tolerances"));
      MipSolutionCallback callback = (solution, objective, bound, userData) -> {};
      settings.setMipCallback(callback, "test-user-data", 2);
      assertTrue(settings.getMipCallbacks().contains(callback));
      assertDoesNotThrow(() -> settings.dumpParametersToFile(file.toString(), true));
      settings.loadParametersFromFile(file.toString());
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Test
  void emptyDataModelCanBeClosed() {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    try (DataModel ignored = new DataModel()) {
      // Lifecycle regression test for JNI-owned empty problems.
    }
  }

  @Test
  void mutableDataModelExposesPythonMetadataAndQuadraticFields() {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    try (DataModel model = new DataModel()) {
      model.setCsrConstraintMatrix(
              new double[] {1.0, 2.0}, new int[] {0, 1}, new int[] {0, 2})
          .setObjectiveCoefficients(new double[] {3.0, 4.0})
          .setObjectiveScalingFactor(2.0)
          .setObjectiveOffset(5.0)
          .setVariableLowerBounds(new double[] {0.0, 0.0})
          .setVariableUpperBounds(new double[] {10.0, 10.0})
          .setVariableTypes(new byte[] {'C', 'C'})
          .setConstraintBounds(new double[] {7.0})
          .setRowTypes(new byte[] {'L'})
          .setVariableNames(new String[] {"x", "y"})
          .setRowNames(new String[] {"c"})
          .setObjectiveName("obj")
          .setProblemName("model")
          .setQuadraticObjectiveMatrix(
              new double[] {1.0, 2.0}, new int[] {0, 1}, new int[] {0, 1, 2});
      model
          .setInitialPrimalSolution(new double[] {1.0, 2.0})
          .setInitialDualSolution(new double[] {3.0});

      assertEquals(2.0, model.getObjectiveScalingFactor());
      assertEquals("model", model.getProblemName());
      assertEquals("obj", model.getObjectiveName());
      assertArrayEquals(new String[] {"x", "y"}, model.getVariableNames());
      assertArrayEquals(new String[] {"c"}, model.getRowNames());
      assertArrayEquals(new byte[] {'L'}, model.getAsciiRowTypes());
      assertArrayEquals(new double[0], model.getConstraintLowerBounds());
      assertArrayEquals(new double[0], model.getConstraintUpperBounds());
      assertArrayEquals(new double[] {1.0, 2.0}, model.getQuadraticObjectiveValues());
      assertArrayEquals(new int[] {0, 1}, model.getQuadraticObjectiveIndices());
      assertArrayEquals(new int[] {0, 1, 2}, model.getQuadraticObjectiveOffsets());
      assertEquals(ProblemCategory.LP, model.getProblemCategory());
      assertArrayEquals(new double[] {1.0, 2.0}, model.getInitialPrimalSolution());
      assertArrayEquals(new double[] {3.0}, model.getInitialDualSolution());
      assertTrue(model.toDict().containsKey("objective_data"));

      model.addQuadraticConstraint(
          "qc", new double[] {1.0}, new int[] {0}, 4.0,
          new double[] {2.0}, new int[] {0}, new int[] {0}, ConstraintSense.LE);
      assertEquals(1, model.getQuadraticConstraints().size());
      assertEquals("qc", model.getQuadraticConstraints().get(0).getRowName());
      model.clearQuadraticConstraints();
      assertTrue(model.getQuadraticConstraints().isEmpty());
    }
  }

  @Test
  void solvesSmallLpAndReportsStats() {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    try (DataModel model = tinyLp();
        SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        Solution solution = model.solve(settings)) {
      assertFalse(solution.isMip());
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, solution.getPrimalObjective(), 1e-3);
      double[] primal = solution.getPrimalSolution();
      assertEquals(1.0, primal[0] + primal[1], 1e-3);
      assertDoesNotThrow(solution::getLpStats);
      assertDoesNotThrow(solution::getPdlpWarmStartData);
      assertThrows(IllegalStateException.class, solution::getMipStats);
    }
  }

  @Test
  void solvesProblemApiMilpAndLifecycleCloseIsIdempotent() {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    Problem problem = new Problem("integer");
    Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
    problem.addConstraint(LinearExpression.of(x).ge(1.0));

    try (SolverSettings settings = new SolverSettings().setParameter(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
        Solution solution = problem.solve(settings)) {
      assertTrue(solution.isMip());
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, x.getValue(), 1e-6);
      assertDoesNotThrow(solution::getMipStats);
      assertThrows(IllegalStateException.class, solution::getDualSolution);
      solution.close();
      solution.close();
    }
  }

  @Test
  void solvesSmallQp() {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    try (DataModel model = tinyLp()) {
      Problem shell = new Problem();
      Variable x0 = shell.addVariable();
      Variable x1 = shell.addVariable();
      model.setQuadraticObjective(
          QuadraticExpression.of(x0, x0, 1.0).plus(x1, x1, 4.0));
      try (SolverSettings settings = new SolverSettings().setParameter(CuOptConstants.CUOPT_ITERATION_LIMIT, 50);
          Solution solution = model.solve(settings)) {
        assertFalse(solution.isMip());
        assertDoesNotThrow(solution::getPrimalSolution);
      }
    }
  }

  @Test
  void rejectsMissingFileThroughCuOptException() {
    assumeNativeLibrary();
    CuOptException exception =
        assertThrows(CuOptException.class, () -> DataModel.read("missing-file-does-not-exist.mps"));
    assertEquals(CuOptConstants.CUOPT_MPS_FILE_ERROR, exception.getStatusCode());
  }

  @Test
  void writesAndReadsMpsThroughReadAndParseMps() throws Exception {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    Path file = Files.createTempFile("cuopt-java-roundtrip-", ".mps");
    try {
      try (DataModel source = tinyLp()) {
        source.writeMPS(file.toString());
      }
      try (DataModel read = DataModel.read(file.toString());
          DataModel parsed = DataModel.parseMps(file.toString());
          Problem problem = Problem.read(file.toString(), false)) {
        assertEquals(2, read.getNumVariables());
        assertEquals(1, read.getNumConstraints());
        assertEquals(read.getNumVariables(), parsed.getNumVariables());
        assertEquals(read.getNumConstraints(), parsed.getNumConstraints());
        assertEquals(read.getNumVariables(), problem.getNumVariables());
        assertEquals(read.getNumConstraints(), problem.getNumConstraints());
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Test
  void batchSolveCompatibilityReturnsAllSolutions() {
    assumeNativeLibrary();
    assumeCudaDriverAvailable();
    try (DataModel first = tinyLp();
        DataModel second = tinyLp();
        SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        BatchSolveResult result = BatchSolve.solve(List.of(first, second), settings)) {
      assertEquals(2, result.getSolutions().size());
      assertTrue(result.getSolveTime() >= 0.0);
      assertEquals(TerminationStatus.OPTIMAL, result.getSolutions().get(0).getTerminationStatus());
      assertEquals(TerminationStatus.OPTIMAL, result.getSolutions().get(1).getTerminationStatus());
    }
  }

  private static DataModel tinyLp() {
    CsrMatrix matrix = new CsrMatrix(new int[] {0, 2}, new int[] {0, 1}, new double[] {1.0, 1.0});
    return DataModel.createProblem(
        1,
        2,
        ObjectiveSense.MINIMIZE,
        0.0,
        new double[] {1.0, 1.0},
        matrix,
        new byte[] {(byte) 'G'},
        new double[] {1.0},
        new double[] {0.0, 0.0},
        new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
        new byte[] {(byte) 'C', (byte) 'C'});
  }

  private static void assumeCudaDriverAvailable() {
    try {
      Process process = new ProcessBuilder("nvidia-smi").start();
      boolean exited = process.waitFor() == 0;
      Assumptions.assumeTrue(exited, "CUDA driver is unavailable");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      Assumptions.assumeTrue(false, "CUDA driver check was interrupted");
    } catch (Exception e) {
      Assumptions.assumeTrue(false, "CUDA driver check failed: " + e.getMessage());
    }
  }
}
