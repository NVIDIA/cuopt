/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
  void writesAndReadsProblemFiles() throws Exception {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    Path file = Files.createTempFile("cuopt-java-roundtrip-", ".mps");
    try {
      try (Problem source = tinyLP()) {
        source.write(file.toString());
      }
      // The extension drives the parser; the boolean overload forces fixed-format MPS.
      try (Problem read = Problem.read(file.toString());
          Problem fixedFormat = Problem.read(file.toString(), false)) {
        assertEquals(2, read.getNumVariables());
        assertEquals(1, read.getNumConstraints());
        assertEquals(read.getNumVariables(), fixedFormat.getNumVariables());
        assertEquals(read.getNumConstraints(), fixedFormat.getNumConstraints());
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Test
  void readsSolverStatisticsAsSolutionAttributes() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    try (Problem problem = tinyLP();
        SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        Solution solution = problem.solve(settings)) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());

      // Requesting a method does not mean that method is credited with the solve; a problem
      // resolved without one reports CUOPT_METHOD_UNSET. What must hold is that the value is a
      // method the API defines.
      int solvedBy = solution.getIntAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_SOLVED_BY);
      assertTrue(
          solvedBy == CuOptConstants.CUOPT_METHOD_CONCURRENT
              || solvedBy == CuOptConstants.CUOPT_METHOD_PDLP
              || solvedBy == CuOptConstants.CUOPT_METHOD_DUAL_SIMPLEX
              || solvedBy == CuOptConstants.CUOPT_METHOD_BARRIER
              || solvedBy == CuOptConstants.CUOPT_METHOD_UNSET,
          "solved-by was " + solvedBy);
      assertTrue(
          solution.getIntAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_NUM_ITERATIONS) >= 0);

      // An optimal solve has converged, so the residuals and gap are at most the tolerance.
      assertEquals(
          0.0,
          solution.getFloatAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_PRIMAL_RESIDUAL),
          1e-3);
      assertEquals(
          0.0,
          solution.getFloatAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_DUAL_RESIDUAL),
          1e-3);
      assertEquals(
          0.0, solution.getFloatAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_GAP), 1e-3);

      // A float selector through the integer accessor, and a MIP selector on an LP solution, are
      // both rejected rather than silently returning something.
      assertThrows(
          CuOptException.class,
          () -> solution.getIntAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_GAP));
      assertThrows(
          CuOptException.class,
          () -> solution.getIntAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_MIP_NUM_NODES));
    }
  }

  @Test
  void readsMIPStatisticsAsSolutionAttributes() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    Problem problem = new Problem("integer");
    Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
    problem.addConstraint(LinearExpression.of(x).ge(1.0));

    try (SolverSettings settings = new SolverSettings().setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
        Solution solution = problem.solve(settings)) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertTrue(solution.getIntAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_MIP_NUM_NODES) >= 0);
      // Violations are magnitudes on a solved problem, so they are non-negative and small.
      double violation =
          solution.getFloatAttribute(
              CuOptConstants.CUOPT_SOLUTION_ATTR_MIP_MAX_CONSTRAINT_VIOLATION);
      assertTrue(violation >= 0.0 && violation < 1e-3, "constraint violation was " + violation);

      // LP selectors do not apply to a MIP solution.
      assertThrows(
          CuOptException.class,
          () -> solution.getFloatAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_GAP));
    }
  }

  @Test
  void mutatingTheModelChangesTheSolve() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    // maximize x subject to x <= 10, so the optimum sits on whichever bound binds. Each setter
    // below has to reach the solver, which only a change in the answer can demonstrate.
    Problem problem = new Problem("mutable");
    Variable x = problem.addVariable(0.0, 10.0, 1.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).le(10.0), "cap");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MAXIMIZE);

    assertEquals(1, problem.getNumNonZeros());
    try (Solution solution = problem.solve()) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(10.0, problem.getObjectiveValue(), 1e-6);
      assertEquals(0, solution.getErrorStatus());
      assertEquals("", solution.getErrorMessage());
      assertEquals(10.0, solution.getDualObjective(), 1e-6);
    }

    // Tightening the upper bound must move the optimum.
    x.setUpperBound(4.0);
    try (Solution solution = problem.solve()) {
      assertEquals(4.0, problem.getObjectiveValue(), 1e-6);
      assertEquals(4.0, x.getValue(), 1e-6);
    }

    // Raising the lower bound above the objective's preference pins the variable.
    x.setLowerBound(4.0).setUpperBound(4.0);
    try (Solution solution = problem.solve()) {
      assertEquals(4.0, x.getValue(), 1e-6);
    }

    // Doubling the objective coefficient doubles the objective at a fixed solution.
    x.setLowerBound(0.0).setUpperBound(4.0).setObjectiveCoefficient(2.0);
    problem.setObjective(LinearExpression.of(x, 2.0), ObjectiveSense.MAXIMIZE);
    try (Solution solution = problem.solve()) {
      assertEquals(8.0, problem.getObjectiveValue(), 1e-6);
    }

    // Making the variable integral makes the solve a MIP, which changes which accessors apply.
    x.setVariableType(VariableType.INTEGER).setVariableName("x_int");
    assertEquals("x_int", x.getVariableName());
    assertTrue(problem.isMIP());
    try (Solution solution = problem.solve()) {
      assertTrue(solution.isMIP());
      assertEquals(4.0, x.getValue(), 1e-6);
      assertThrows(IllegalStateException.class, solution::getDualObjective);
    }

    Constraint constraint = problem.getConstraint(0);
    assertFalse(constraint.isQuadratic());
    assertEquals(1.0, constraint.getLinearExpression().getCoefficient(x), 0.0);
  }

  @Test
  void mipStartsAndMIPOnlySolutionFields() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
    // A small knapsack: maximise value subject to a weight cap.
    Problem problem = new Problem("knapsack");
    Variable a = problem.addVariable(0.0, 1.0, 5.0, VariableType.INTEGER, "a");
    Variable b = problem.addVariable(0.0, 1.0, 4.0, VariableType.INTEGER, "b");
    problem.addConstraint(LinearExpression.of(a, 3.0).plus(b, 2.0).le(3.0), "weight");
    problem.setObjective(LinearExpression.of(a, 5.0).plus(b, 4.0), ObjectiveSense.MAXIMIZE);

    // Seed a feasible starting point. Problem.addMIPStarts collects these by variable index and
    // hands them to the settings, so a wrong index would seed the wrong variable.
    a.setMIPStart(1.0);
    b.setMIPStart(0.0);
    assertEquals(1.0, a.getMIPStart(), 0.0);
    assertEquals(0.0, b.getMIPStart(), 0.0);

    try (SolverSettings settings =
            new SolverSettings().setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
        Solution solution = problem.solve(settings)) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      // Taking a alone scores 5 and weighs 3; taking b alone scores 4. a is optimal.
      assertEquals(5.0, solution.getPrimalObjective(), 1e-6);
      assertEquals(1.0, a.getValue(), 1e-6);
      assertEquals(0.0, b.getValue(), 1e-6);

      // MIP-only reporting: an optimal solve has closed the gap and bounds its own objective.
      assertEquals(0.0, solution.getMIPGap(), 1e-4);
      assertEquals(5.0, solution.getSolutionBound(), 1e-4);
    }

    // addMIPStart is also callable directly, taking one value per variable in index order.
    try (SolverSettings settings = new SolverSettings()) {
      assertDoesNotThrow(() -> settings.addMIPStart(new double[] {1.0, 0.0}));
    }

    // The callback payload is a value type, so it can be checked without waiting for the solver
    // to produce an incumbent.
    MIPCallbackSolution payload = new MIPCallbackSolution(new double[] {1.0, 0.0}, 5.0);
    assertArrayEquals(new double[] {1.0, 0.0}, payload.getSolution(), 0.0);
    assertEquals(5.0, payload.getObjectiveValue(), 0.0);
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
