/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

final class NativeIntegrationTest {


  @Test
  void settingsExposeTypedValues() {
    NativeTestSupport.assumeNativeLibrary();
    try (SolverSettings settings = new SolverSettings()) {
      settings.setSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false);
      settings.setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 12.5);
      settings.setOptimalityTolerance(1.0e-6);
      assertEquals(
          Boolean.FALSE,
          settings.getSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, Boolean.class));
      assertEquals(
          12.5,
          settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT, Double.class),
          1e-12);
      assertEquals(
          SolverSettings.getSolverSetting(CuOptConstants.CUOPT_TIME_LIMIT),
          SolverSettings.getSolverSetting(CuOptConstants.CUOPT_TIME_LIMIT, String.class));
      assertEquals(
          1.0e-6,
          settings.getSetting(CuOptConstants.CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, Double.class),
          1e-12);
      assertEquals(
          12.5,
          Double.parseDouble(settings.getSettingAsString(CuOptConstants.CUOPT_TIME_LIMIT)),
          1e-12);
      MIPSolutionCallback callback = (solution, objective, bound, userData) -> {};
      settings.setMIPCallback(callback, "test-user-data", 2);
      assertTrue(settings.getMIPCallbacks().contains(callback));
    }
  }

  @Test
  void solvesSmallLPAndReportsStats() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    try (Problem problem = tinyLP();
        SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        Solution solution = problem.solve(settings)) {
      assertFalse(solution.isMIP());
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, solution.getPrimalObjective(), 1e-3);
      double[] primal = solution.getPrimalSolution();
      assertEquals(1.0, primal[0] + primal[1], 1e-3);
      assertDoesNotThrow(solution::getLPStats);
      assertThrows(IllegalStateException.class, solution::getMIPStats);
    }
  }

  @Test
  void warmStartsPDLPFromInitialPrimalAndDualSolutions() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    // tinyLP has two variables and one constraint; the optimum lies on x0 + x1 == 1.
    try (Problem problem = tinyLP();
        SolverSettings settings =
            new SolverSettings()
                .setMethod(SolverMethod.PDLP)
                .setInitialPrimalSolution(new double[] {0.5, 0.5})
                .setInitialDualSolution(new double[] {1.0});
        Solution solution = problem.solve(settings)) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, solution.getPrimalObjective(), 1e-3);
    }
  }

  @Test
  void solvesProblemApiMIPAndLifecycleCloseIsIdempotent() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    Problem problem = new Problem("integer");
    Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
    problem.addConstraint(LinearExpression.of(x).ge(1.0));

    try (SolverSettings settings = new SolverSettings().setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
        Solution solution = problem.solve(settings)) {
      assertTrue(solution.isMIP());
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, x.getValue(), 1e-6);
      assertDoesNotThrow(solution::getMIPStats);
      assertThrows(IllegalStateException.class, solution::getDualSolution);
      solution.close();
      solution.close();
    }
  }

  @Test
  void solvesSmallQP() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    try (Problem problem = tinyLP()) {
      Variable x0 = problem.getVariable(0);
      Variable x1 = problem.getVariable(1);
      problem.setObjective(
          QuadraticExpression.of(x0, x0, 1.0).plus(x1, x1, 4.0),
          ObjectiveSense.MINIMIZE);
      try (SolverSettings settings = new SolverSettings().setSetting(CuOptConstants.CUOPT_ITERATION_LIMIT, 50);
          Solution solution = problem.solve(settings)) {
        assertFalse(solution.isMIP());
        assertDoesNotThrow(solution::getPrimalSolution);
      }
    }
  }

  @Test
  void rejectsMissingFileThroughCuOptException() {
    NativeTestSupport.assumeNativeLibrary();
    CuOptException exception =
        assertThrows(CuOptException.class, () -> Problem.read("missing-file-does-not-exist.mps"));
    assertEquals(CuOptConstants.CUOPT_MPS_FILE_ERROR, exception.getStatusCode());
  }

  @Test
  void writesAndReadsMPSThroughReadAndParseMPS() throws Exception {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    Path file = Files.createTempFile("cuopt-java-roundtrip-", ".mps");
    try {
      try (Problem source = tinyLP()) {
        source.writeMPS(file.toString());
      }
      try (Problem read = Problem.read(file.toString(), false);
          Problem parsed = Problem.readMPS(file.toString(), false)) {
        assertEquals(2, read.getNumVariables());
        assertEquals(1, read.getNumConstraints());
        assertEquals(read.getNumVariables(), parsed.getNumVariables());
        assertEquals(read.getNumConstraints(), parsed.getNumConstraints());
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }

  private static Problem tinyLP() {
    Problem problem = new Problem("tiny");
    Variable x0 = problem.addVariable(0.0, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x0");
    Variable x1 = problem.addVariable(0.0, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x1");
    problem.addConstraint(LinearExpression.of(x0).plus(x1).ge(1.0), "c0");
    problem.setObjective(LinearExpression.of(x0).plus(x1), ObjectiveSense.MINIMIZE);
    return problem;
  }

}
