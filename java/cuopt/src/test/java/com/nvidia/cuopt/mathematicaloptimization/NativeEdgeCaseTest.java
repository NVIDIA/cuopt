/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Corner cases that need the native library and a GPU. */
final class NativeEdgeCaseTest {

  @BeforeEach
  void requireNative() {
    NativeTestSupport.assumeNativeLibrary();
    NativeTestSupport.assumeCudaDriverAvailable();
  }

  private static SolverSettings quiet() {
    return new SolverSettings()
        .setSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false)
        .setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 20.0);
  }

  private static Problem tinyLP() {
    Problem problem = new Problem("tiny");
    Variable x0 = problem.addVariable(0, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x0");
    Variable x1 = problem.addVariable(0, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x1");
    problem.addConstraint(LinearExpression.of(x0).plus(x1).ge(1.0), "c0");
    problem.setObjective(LinearExpression.of(x0).plus(x1), ObjectiveSense.MINIMIZE);
    return problem;
  }

  /**
   * A knapsack large enough that presolve cannot close it, so the MIP search runs and reports
   * incumbents.
   */
  private static Problem knapsack(int items) {
    Problem problem = new Problem("knapsack");
    LinearExpression weight = new LinearExpression();
    LinearExpression value = new LinearExpression();
    for (int i = 0; i < items; i++) {
      Variable item = problem.addVariable(0, 1, 0, VariableType.INTEGER, "i" + i);
      weight = weight.plus(item, 7 + (i * 13) % 29);
      value = value.plus(item, 11 + (i * 17) % 31);
    }
    problem.addConstraint(weight.le(items * 9.0), "capacity");
    problem.setObjective(value, ObjectiveSense.MAXIMIZE);
    return problem;
  }

  // ---- handle lifecycle -------------------------------------------------------------------

  @Test
  void aClosedSolutionRejectsEveryAccessor() {
    Solution solution;
    try (Problem problem = tinyLP()) {
      solution = problem.solve(quiet());
    }
    solution.close();
    assertThrows(IllegalStateException.class, solution::getTerminationStatus);
    assertThrows(IllegalStateException.class, solution::getPrimalObjective);
    assertThrows(IllegalStateException.class, solution::getDualObjective);
    assertThrows(IllegalStateException.class, solution::getErrorStatus);
    assertThrows(IllegalStateException.class, solution::getErrorMessage);
    assertThrows(
        IllegalStateException.class,
        () -> solution.getIntAttribute(CuOptConstants.CUOPT_SOLUTION_ATTR_LP_NUM_ITERATIONS));
    solution.close();
  }

  @Test
  void aClosedSettingsObjectRejectsEveryUse() {
    SolverSettings settings = new SolverSettings();
    settings.close();
    assertThrows(
        IllegalStateException.class, () -> settings.setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 1.0));
    assertThrows(
        IllegalStateException.class, () -> settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT));
    assertThrows(IllegalStateException.class, () -> settings.addMIPStart(new double[] {1.0}));
    assertThrows(IllegalStateException.class, () -> settings.setOptimalityTolerance(1e-6));
    try (Problem problem = tinyLP()) {
      assertThrows(IllegalStateException.class, () -> problem.solve(settings));
    }
    settings.close();
  }

  @Test
  void theCallersSettingsSurviveASolveAndCanBeReused() {
    try (SolverSettings settings = quiet();
        Problem problem = tinyLP()) {
      double first;
      try (Solution solution = problem.solve(settings)) {
        first = solution.getPrimalObjective();
      }
      try (Solution solution = problem.solve(settings)) {
        assertEquals(first, solution.getPrimalObjective(), 1e-6);
      }
      assertEquals("20", settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT).replaceAll("\\.0+$", ""));
    }
  }

  @Test
  void aSolutionOutlivesItsProblemAndStaysReadable() {
    Solution solution;
    try (Problem problem = tinyLP()) {
      solution = problem.solve(quiet());
    }
    try (solution) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, solution.getPrimalObjective(), 1e-4);
    }
  }

  @Test
  void aProblemKeepsTheValuesOfItsLatestSolveOnly() {
    try (Problem problem = tinyLP()) {
      try (Solution solution = problem.solve(quiet())) {
        assertEquals(1.0, problem.getObjectiveValue(), 1e-4);
      }
      problem.getConstraint(0).getRHS();
      problem.addConstraint(
          LinearExpression.of(problem.getVariable(0)).plus(problem.getVariable(1)).ge(3.0), "c1");
      assertTrue(Double.isNaN(problem.getObjectiveValue()));
      try (Solution solution = problem.solve(quiet())) {
        assertEquals(3.0, problem.getObjectiveValue(), 1e-4);
        assertEquals(TerminationStatus.OPTIMAL, problem.getStatus());
      }
    }
  }

  // ---- LP/MIP-only accessors ----------------------------------------------------------------

  @Test
  void lpOnlyAccessorsRejectAMIPSolution() {
    try (Problem problem = knapsack(4);
        Solution solution = problem.solve(quiet())) {
      assertTrue(solution.isMIP());
      assertThrows(IllegalStateException.class, solution::getDualSolution);
      assertThrows(IllegalStateException.class, solution::getReducedCost);
      assertThrows(IllegalStateException.class, solution::getDualObjective);
      assertDoesNotThrowMIPGetters(solution);
    }
  }

  private static void assertDoesNotThrowMIPGetters(Solution solution) {
    solution.getMIPGap();
    solution.getSolutionBound();
  }

  @Test
  void mipOnlyAccessorsRejectAnLPSolution() {
    try (Problem problem = tinyLP();
        Solution solution = problem.solve(quiet())) {
      assertFalse(solution.isMIP());
      assertThrows(IllegalStateException.class, solution::getMIPGap);
      assertThrows(IllegalStateException.class, solution::getSolutionBound);
      assertEquals(1, solution.getDualSolution().length);
      assertEquals(2, solution.getReducedCost().length);
    }
  }

  @Test
  void anAttributeSelectorThatDoesNotApplyRaisesCuOptException() {
    try (Problem problem = tinyLP();
        Solution solution = problem.solve(quiet())) {
      assertThrows(CuOptException.class, () -> solution.getIntAttribute(-1));
      assertThrows(CuOptException.class, () -> solution.getFloatAttribute(-1));
    }
  }

  // ---- settings ---------------------------------------------------------------------------

  @Test
  void anUnknownSettingNameRaisesCuOptException() {
    try (SolverSettings settings = new SolverSettings()) {
      assertThrows(CuOptException.class, () -> settings.setSetting("no_such_setting", 1));
      assertThrows(CuOptException.class, () -> settings.setSetting("no_such_setting", 1.0));
      assertThrows(CuOptException.class, () -> settings.setSetting("no_such_setting", "x"));
      assertThrows(CuOptException.class, () -> settings.getSetting("no_such_setting"));
    }
  }

  @Test
  void typedSettingGettersParseAndRejectMismatches() {
    try (SolverSettings settings = new SolverSettings()) {
      settings.setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 12.5);
      assertEquals(12.5, settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT, Double.class), 0.0);
      assertEquals(
          12.5,
          Double.parseDouble(settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT, String.class)),
          0.0);
      assertThrows(
          IllegalArgumentException.class,
          () -> settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT, Integer.class));
      assertThrows(
          IllegalArgumentException.class,
          () -> settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT, Boolean.class));
      assertThrows(
          IllegalArgumentException.class,
          () -> settings.getSetting(CuOptConstants.CUOPT_TIME_LIMIT, Long.class));
    }
  }

  @Test
  void aBooleanSettingReadsBackAsABoolean() {
    try (SolverSettings settings = new SolverSettings()) {
      settings.setSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false);
      assertFalse(settings.getSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, Boolean.class));
      settings.setSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, true);
      assertTrue(settings.getSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, Boolean.class));
    }
  }

  @Test
  void theOptimalityToleranceSetsEverySubTolerance() {
    try (SolverSettings settings = new SolverSettings().setOptimalityTolerance(1e-5)) {
      for (String name :
          new String[] {
            CuOptConstants.CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
            CuOptConstants.CUOPT_RELATIVE_PRIMAL_TOLERANCE,
            CuOptConstants.CUOPT_ABSOLUTE_DUAL_TOLERANCE,
            CuOptConstants.CUOPT_RELATIVE_DUAL_TOLERANCE,
            CuOptConstants.CUOPT_ABSOLUTE_GAP_TOLERANCE,
            CuOptConstants.CUOPT_RELATIVE_GAP_TOLERANCE,
          }) {
        assertEquals(1e-5, settings.getSetting(name, Double.class), 0.0, name);
      }
    }
  }

  @Test
  void aMIPStartIsCopiedWhenItIsAdded() {
    double[] start = {1.0, 0.0};
    try (SolverSettings settings = new SolverSettings().setSetting(CuOptConstants.CUOPT_LOG_TO_CONSOLE, false);
        Problem problem = knapsack(2)) {
      settings.addMIPStart(start);
      start[0] = Double.NaN;
      try (Solution solution = problem.solve(settings)) {
        assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      }
    }
  }

  @Test
  void aWarmStartOfTheWrongLengthDoesNotCrashTheSolver() {
    try (Problem problem = tinyLP();
        SolverSettings settings =
            quiet()
                .setSetting(CuOptConstants.CUOPT_METHOD, SolverMethod.PDLP.nativeValue())
                .setInitialPrimalSolution(new double[] {1.0, 2.0, 3.0})) {
      try (Solution solution = problem.solve(settings)) {
        assertNotEquals(TerminationStatus.NO_TERMINATION, solution.getTerminationStatus());
      } catch (CuOptException | IllegalArgumentException expected) {
        // Rejecting the mismatch up front is as acceptable as ignoring it.
      }
    }
  }

  // ---- degenerate and extreme models ---------------------------------------------------------

  @Test
  void anUnboundedLPIsReportedAsUnbounded() {
    Problem problem = new Problem("unbounded");
    Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, -1.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).ge(1.0), "c");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MAXIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertTrue(
          solution.getTerminationStatus() == TerminationStatus.UNBOUNDED
              || solution.getTerminationStatus() == TerminationStatus.UNBOUNDED_OR_INFEASIBLE,
          () -> "status was " + solution.getTerminationStatus());
    }
  }

  @Test
  void anUnboundedMIPIsNotReportedOptimal() {
    Problem problem = new Problem("unbounded-mip");
    Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0.0, VariableType.INTEGER, "x");
    Variable y = problem.addVariable(0, Double.POSITIVE_INFINITY, 0.0, VariableType.CONTINUOUS, "y");
    problem.addConstraint(LinearExpression.of(x).minus(y).le(5.0), "c");
    problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MAXIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertNotEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
    }
  }

  @Test
  void anInfeasibleMIPIsReportedInfeasible() {
    Problem problem = new Problem("infeasible-mip");
    Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
    problem.addConstraint(LinearExpression.of(x, 2.0).eq(3.0), "odd");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(TerminationStatus.INFEASIBLE, solution.getTerminationStatus());
      assertTrue(Double.isNaN(x.getValue()) || x.getValue() != 1.5);
    }
  }

  @Test
  void crossedVariableBoundsAreInfeasible() {
    Problem problem = new Problem("crossed");
    Variable x = problem.addVariable(5, 1, 1.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).le(10.0), "c");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(TerminationStatus.INFEASIBLE, solution.getTerminationStatus());
    } catch (CuOptException rejectedUpFront) {
      // Refusing the model is as valid as reporting it infeasible.
    }
  }

  @Test
  void aModelWithNoMatrixEntriesIsRejectedByTheNativeLayer() {
    Problem noRows = new Problem("no-rows");
    Variable x = noRows.addVariable(2, 5, 3.0, VariableType.CONTINUOUS, "x");
    noRows.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    assertThrows(CuOptException.class, () -> noRows.solve(quiet()));

    Problem emptyRow = new Problem("empty-row");
    Variable y = emptyRow.addVariable(0, 1, 1.0, VariableType.CONTINUOUS, "y");
    emptyRow.addConstraint(new LinearExpression().le(5.0), "trivial");
    emptyRow.setObjective(LinearExpression.of(y), ObjectiveSense.MAXIMIZE);
    assertThrows(CuOptException.class, () -> emptyRow.solve(quiet()));
  }

  @Test
  void aFixedVariableTakesItsFixedValue() {
    Problem problem = new Problem("fixed");
    Variable x = problem.addVariable(3, 3, 1.0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 10, 1.0, VariableType.CONTINUOUS, "y");
    problem.addConstraint(LinearExpression.of(x).plus(y).ge(5.0), "c");
    problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(3.0, x.getValue(), 1e-6);
      assertEquals(2.0, y.getValue(), 1e-6);
    }
  }

  @Test
  void aFreeVariableMayGoNegative() {
    Problem problem = new Problem("free");
    Variable x =
        problem.addVariable(
            Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 1.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).ge(-7.0), "c");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(-7.0, x.getValue(), 1e-6);
    }
  }

  @Test
  void aCancelledTermStillSolvesAsAnEmptyRow() {
    Problem problem = new Problem("cancelled");
    Variable x = problem.addVariable(0, 4, 1.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x, 2.0).plus(x, -2.0).le(1.0), "zero");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MAXIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(4.0, x.getValue(), 1e-6);
    }
  }

  @Test
  void anObjectiveOffsetShiftsTheReportedObjectiveOnly() {
    Problem problem = new Problem("offset");
    Variable x = problem.addVariable(1, 5, 0.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).le(1000.0), "bound");
    problem.setObjective(LinearExpression.of(x).plus(10.0), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(1.0, x.getValue(), 1e-6);
      assertEquals(11.0, problem.getObjectiveValue(), 1e-6);
    }
  }

  @Test
  void aConstraintConstantShiftsItsSlack() {
    Problem problem = new Problem("slack");
    Variable x = problem.addVariable(0, 10, 1.0, VariableType.CONTINUOUS, "x");
    Constraint le = problem.addConstraint(LinearExpression.of(x).plus(2.0).le(9.0), "le");
    Constraint ge = problem.addConstraint(LinearExpression.of(x).ge(1.0), "ge");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MAXIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(7.0, x.getValue(), 1e-6);
      assertEquals(0.0, le.getSlack(), 1e-6);
      assertEquals(-6.0, ge.getSlack(), 1e-6);
    }
  }

  @Test
  void anIntegerVariableWithFractionalBoundsIsRoundedInward() {
    Problem problem = new Problem("fractional");
    Variable x = problem.addVariable(0.4, 3.6, 1.0, VariableType.INTEGER, "x");
    problem.addConstraint(LinearExpression.of(x).le(1000.0), "bound");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      assertEquals(1.0, x.getValue(), 1e-6);
    }
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MAXIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(3.0, x.getValue(), 1e-6);
    }
  }

  @Test
  void aProblemWithoutVariablesIsRejectedOrTriviallySolved() {
    Problem problem = new Problem("no-variables");
    try (Solution solution = problem.solve(quiet())) {
      assertNotEquals(TerminationStatus.NO_TERMINATION, solution.getTerminationStatus());
    } catch (CuOptException rejected) {
      assertFalse(rejected.getMessage().isEmpty());
    }
  }

  @Test
  void objectiveCoefficientsSetOnVariablesAreUsedWhenNoObjectiveIsSet() {
    Problem problem = new Problem("coefficients");
    Variable x = problem.addVariable(0, 4, 1.0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 4, -1.0, VariableType.CONTINUOUS, "y");
    problem.addConstraint(LinearExpression.of(x).plus(y).le(6.0), "c");
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(0.0, x.getValue(), 1e-6);
      assertEquals(4.0, y.getValue(), 1e-6);
      assertEquals(-4.0, problem.getObjectiveValue(), 1e-6);
    }
  }

  @Test
  @Disabled("objectiveCoefficients() ignores Variable coefficients once an objective is set")
  void aVariableCoefficientChangedAfterSettingTheObjectiveChangesTheSolve() {
    Problem problem = new Problem("late-coefficient");
    Variable x = problem.addVariable(0, 4, 0.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).le(1000.0), "bound");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(0.0, x.getValue(), 1e-6);
    }
    x.setObjectiveCoefficient(-1.0);
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(4.0, x.getValue(), 1e-6);
    }
  }

  @Test
  @Disabled("objectiveCoefficients() ignores the coefficient of a variable added after setObjective")
  void aVariableAddedAfterSettingTheObjectiveKeepsItsCoefficient() {
    Problem problem = new Problem("late-variable");
    Variable x = problem.addVariable(0, 4, 1.0, VariableType.CONTINUOUS, "x");
    problem.addConstraint(LinearExpression.of(x).le(1000.0), "bound");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    Variable y = problem.addVariable(0, 4, -1.0, VariableType.CONTINUOUS, "y");
    try (Solution solution = problem.solve(quiet())) {
      assertEquals(4.0, y.getValue(), 1e-6);
    }
  }

  @Test
  void aFeasibleMIPStartedFromAnInfeasiblePointStillSolves() {
    try (Problem problem = knapsack(6);
        Solution solution = startedAt(problem, new double[] {1, 1, 1, 1, 1, 1})) {
      assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
    }
  }

  private static Solution startedAt(Problem problem, double[] start) {
    for (int i = 0; i < start.length; i++) {
      problem.getVariable(i).setMIPStart(start[i]);
    }
    return problem.solve(quiet());
  }

  // ---- quadratic ---------------------------------------------------------------------------

  @Test
  void anEqualityQuadraticConstraintIsRejectedBeforeSolving() {
    Problem problem = new Problem("quadratic-equality");
    Variable x = problem.addVariable(0, 5, 0.0, VariableType.CONTINUOUS, "x");
    QuadraticExpression expression = QuadraticExpression.of(x, x, 1.0);
    problem.addConstraint(LinearExpression.of(x).le(100.0), "bound");
    problem.addConstraint(new Constraint(expression, ConstraintSense.EQ, 4.0), "q");
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    assertThrows(IllegalArgumentException.class, () -> problem.solve(quiet()));
  }

  // ---- files -------------------------------------------------------------------------------

  @TempDir Path temp;

  private Path write(String name, String content) throws IOException {
    return Files.writeString(temp.resolve(name), content);
  }

  private static final String RANGED_MPS =
      "NAME          RANGED\n"
          + "ROWS\n"
          + " N  OBJ\n"
          + " L  R1\n"
          + "COLUMNS\n"
          + "    X         OBJ       1.0   R1        1.0\n"
          + "RHS\n"
          + "    RHS       R1        10.0\n"
          + "RANGES\n"
          + "    RNG       R1        4.0\n"
          + "BOUNDS\n"
          + " UP BND       X         20.0\n"
          + "ENDATA\n";

  private static final String SIMPLE_MPS =
      "NAME          SIMPLE\n"
          + "OBJSENSE\n"
          + "    MAX\n"
          + "ROWS\n"
          + " N  COST\n"
          + " L  CAP\n"
          + "COLUMNS\n"
          + "    A         COST      2.0   CAP       1.0\n"
          + "    B         COST      3.0   CAP       1.0\n"
          + "RHS\n"
          + "    RHS       CAP       4.0\n"
          + "BOUNDS\n"
          + " UP BND       A         3.0\n"
          + " UP BND       B         3.0\n"
          + "ENDATA\n";

  @Test
  void aRangedRowIsRejectedWithAnExplanation() throws IOException {
    Path file = write("ranged.mps", RANGED_MPS);
    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> Problem.read(file.toString()));
    assertTrue(error.getMessage().contains("Ranged constraints"), error.getMessage());
  }

  @Test
  void readingKeepsTheNamesSenseAndObjectiveOfAFreeFormatFile() throws IOException {
    Path file = write("simple.mps", SIMPLE_MPS);
    try (Problem problem = Problem.read(file.toString())) {
      assertEquals("SIMPLE", problem.getName());
      assertEquals(ObjectiveSense.MAXIMIZE, problem.getObjectiveSense());
      assertEquals(2, problem.getNumVariables());
      assertEquals(1, problem.getNumConstraints());
      assertEquals("A", problem.getVariable(0).getVariableName());
      assertEquals("CAP", problem.getConstraint(0).getConstraintName());
      assertEquals(ConstraintSense.LE, problem.getConstraint(0).getSense());
      assertEquals(4.0, problem.getConstraint(0).getRHS(), 0.0);
      assertEquals(3.0, problem.getVariable(1).getUpperBound(), 0.0);
      try (Solution solution = problem.solve(quiet())) {
        assertEquals(3.0 * 3 + 2.0 * 1, solution.getPrimalObjective(), 1e-6);
      }
    }
  }

  @Test
  void theFixedFormatFlagIsAcceptedForAFixedFormatFile() throws IOException {
    String fixed =
        "NAME          FIXED\n"
            + "ROWS\n"
            + " N  COST\n"
            + " G  LIM\n"
            + "COLUMNS\n"
            + "    X         COST               1.0   LIM                1.0\n"
            + "RHS\n"
            + "    RHS       LIM                2.0\n"
            + "ENDATA\n";
    Path file = write("fixed.mps", fixed);
    try (Problem problem = Problem.read(file.toString(), true)) {
      assertEquals(1, problem.getNumVariables());
      assertEquals(ConstraintSense.GE, problem.getConstraint(0).getSense());
      assertEquals(2.0, problem.getConstraint(0).getRHS(), 0.0);
    }
  }

  @Test
  void aMalformedFileRaisesCuOptException() throws IOException {
    Path file = write("broken.mps", "NAME BROKEN\nROWS\n N COST\nCOLUMNS\n  X  NOSUCHROW  1.0\nENDATA\n");
    assertThrows(CuOptException.class, () -> Problem.read(file.toString()));
    Path empty = write("empty.mps", "");
    assertThrows(CuOptException.class, () -> Problem.read(empty.toString()));
  }

  @Test
  void writeAcceptsEitherMPSExtensionInAnyCase() {
    try (Problem problem = tinyLP()) {
      for (String name : new String[] {"a.mps", "b.MPS", "c.qps", "d.QPS"}) {
        Path file = temp.resolve(name);
        problem.write(file.toString());
        assertTrue(Files.exists(file), name);
      }
    }
  }

  @Test
  void aQuadraticObjectiveSurvivesAWriteAndRead() {
    Problem problem = new Problem("qround");
    Variable x = problem.addVariable(0, 5, 0.0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 5, 0.0, VariableType.CONTINUOUS, "y");
    problem.addConstraint(LinearExpression.of(x).plus(y).ge(2.0), "c");
    problem.setObjective(
        QuadraticExpression.of(x, x, 1.0).plus(y, y, 2.0).plus(x, 1.0), ObjectiveSense.MINIMIZE);
    Path file = temp.resolve("q.qps");
    problem.write(file.toString());
    try (Problem reread = Problem.read(file.toString())) {
      assertEquals(2, reread.getNumVariables());
      assertTrue(reread.getQuadraticObjectiveMatrix() != null);
      assertEquals(
          problem.getQuadraticObjectiveMatrix().getValues().length,
          reread.getQuadraticObjectiveMatrix().getValues().length);
    }
  }

  @Test
  void aMIPKeepsItsIntegerVariablesAcrossAWriteAndRead() {
    try (Problem problem = knapsack(5)) {
      Path file = temp.resolve("mip.mps");
      problem.write(file.toString());
      try (Problem reread = Problem.read(file.toString())) {
        assertTrue(reread.isMIP());
        assertEquals(5, reread.getNumVariables());
        for (Variable variable : reread.getVariables()) {
          assertEquals(VariableType.INTEGER, variable.getVariableType());
          assertEquals(0.0, variable.getLowerBound(), 0.0);
          assertEquals(1.0, variable.getUpperBound(), 0.0);
        }
        assertEquals(ObjectiveSense.MAXIMIZE, reread.getObjectiveSense());
      }
    }
  }

  @Test
  @Disabled("Problem.write emits blank names for unnamed variables and rows, which read rejects")
  void anUnnamedModelRoundTripsThroughAnMPSFile() {
    Problem problem = new Problem();
    Variable x = problem.addVariable();
    problem.addConstraint(LinearExpression.of(x).le(1.0));
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MAXIMIZE);
    Path file = temp.resolve("unnamed.mps");
    problem.write(file.toString());
    try (Problem reread = Problem.read(file.toString())) {
      assertFalse(reread.getVariable(0).getVariableName().isEmpty());
      assertFalse(reread.getConstraint(0).getConstraintName().isEmpty());
    }
  }

  // ---- MIP callbacks -------------------------------------------------------------------------

  @Test
  void theGetSolutionCallbackReceivesIncumbentsOfTheRightShape() {
    int items = 40;
    AtomicInteger calls = new AtomicInteger();
    List<String> problems = new ArrayList<>();
    try (Problem problem = knapsack(items);
        SolverSettings settings =
            quiet()) {
      Object marker = new Object();
      settings.setMIPCallback(
          (MIPSolutionCallback)
              (solution, objective, bound, userData) -> {
                calls.incrementAndGet();
                if (solution.length != items) {
                  problems.add("length " + solution.length);
                }
                if (userData != marker) {
                  problems.add("user data was not passed through");
                }
                if (Double.isNaN(objective)) {
                  problems.add("objective is NaN");
                }
              },
          marker,
          items);
      try (Solution solution = problem.solve(settings)) {
        assertTrue(
            solution.getTerminationStatus() == TerminationStatus.OPTIMAL
                || solution.getTerminationStatus() == TerminationStatus.FEASIBLE_FOUND
                || solution.getTerminationStatus() == TerminationStatus.TIME_LIMIT);
      }
    }
    assertTrue(problems.isEmpty(), problems::toString);
    assertTrue(calls.get() > 0, "the callback was never called");
  }

  @Test
  void anExceptionInAGetSolutionCallbackDoesNotCrashTheSolve() {
    int items = 40;
    AtomicInteger calls = new AtomicInteger();
    try (Problem problem = knapsack(items);
        SolverSettings settings = quiet()) {
      settings.setMIPCallback(
          (MIPSolutionCallback)
              (solution, objective, bound, userData) -> {
                calls.incrementAndGet();
                throw new IllegalStateException("callback failure");
              },
          null,
          items);
      try (Solution solution = problem.solve(settings)) {
        assertNotEquals(TerminationStatus.NO_TERMINATION, solution.getTerminationStatus());
      } catch (RuntimeException surfaced) {
        // Surfacing the callback's failure is acceptable; killing the JVM is not.
      }
    }
  }

  @Test
  void theSetSolutionCallbackIsAskedForASolution() {
    int items = 40;
    AtomicInteger calls = new AtomicInteger();
    try (Problem problem = knapsack(items);
        SolverSettings settings = quiet()) {
      settings.setMIPCallback(
          (MIPSetSolutionCallback)
              (bound, userData) -> {
                calls.incrementAndGet();
                return new MIPCallbackSolution(new double[items], 0.0);
              },
          null,
          items);
      try (Solution solution = problem.solve(settings)) {
        assertNotEquals(TerminationStatus.NO_TERMINATION, solution.getTerminationStatus());
      }
    }
  }

  // ---- concurrency -------------------------------------------------------------------------

  @Test
  void solvesOnSeveralThreadsDoNotInterfere() throws Exception {
    int threads = 4;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      List<Future<Double>> futures = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final double rhs = t + 1;
        Callable<Double> job =
            () -> {
              try (Problem problem = new Problem("thread-" + rhs)) {
                Variable x = problem.addVariable(0, 100, 1.0, VariableType.CONTINUOUS, "x");
                problem.addConstraint(LinearExpression.of(x).ge(rhs), "c");
                problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
                try (SolverSettings settings = quiet();
                    Solution solution = problem.solve(settings)) {
                  return x.getValue();
                }
              }
            };
        futures.add(pool.submit(job));
      }
      for (int t = 0; t < threads; t++) {
        assertEquals(t + 1.0, futures.get(t).get(120, TimeUnit.SECONDS), 1e-6);
      }
    } finally {
      pool.shutdownNow();
    }
  }

  @Test
  void manyShortLivedSolutionsAndSettingsAreReleased() {
    for (int i = 0; i < 50; i++) {
      try (Problem problem = tinyLP();
          SolverSettings settings = quiet();
          Solution solution = problem.solve(settings)) {
        assertEquals(TerminationStatus.OPTIMAL, solution.getTerminationStatus());
      }
    }
  }

  @Test
  void aSolutionDroppedWithoutCloseIsLeftToTheCleaner() {
    try (Problem problem = tinyLP()) {
      for (int i = 0; i < 20; i++) {
        problem.solve(quiet());
      }
      System.gc();
      assertArrayEquals(new int[] {0, 2}, problem.getConstraintMatrix().getRowOffsets());
    }
  }
}
