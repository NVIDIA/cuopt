/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

final class ProblemModelingTest {
  @Test
  void generatedSolverEnumsMatchCuOptConstants() {
    assertEquals(CuOptConstants.CUOPT_METHOD_PDLP, SolverMethod.PDLP.nativeValue());
    assertEquals(
        CuOptConstants.CUOPT_PDLP_SOLVER_MODE_STABLE1,
        PDLPSolverMode.STABLE1.nativeValue());
    assertEquals(
        CuOptConstants.CUOPT_TERMINATION_STATUS_OPTIMAL,
        TerminationStatus.OPTIMAL.nativeValue());
  }

  @Test
  void mapsLegacyIPCategoryToMIP() {
    assertEquals(ProblemCategory.MIP, ProblemCategory.fromNative(2));
  }

  @Test
  void buildsLinearProblemAndCSR() {
    Problem problem = new Problem("Simple MIP");
    Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0, VariableType.INTEGER, "x");
    Variable y = problem.addVariable(10, 50, 0, VariableType.INTEGER, "y");

    assertEquals(0, x.getIndex());
    assertEquals(1, y.getIndex());
    assertTrue(problem.isMIP());

    problem.addConstraint(LinearExpression.of(x, 2).plus(y, 4).ge(230), "c1");
    problem.addConstraint(LinearExpression.of(x, 3).plus(y, 2).constant(10).le(200), "c2");
    problem.setObjective(LinearExpression.of(x, 5).plus(y, 3).constant(50), ObjectiveSense.MAXIMIZE);

    LinearExpression objective = problem.getObjective();
    assertEquals(50.0, objective.getConstant());

    CSRMatrix csr = problem.getConstraintMatrix();
    assertArrayEquals(new int[] {0, 2, 4}, csr.getRowOffsets());
    assertArrayEquals(new int[] {0, 1, 0, 1}, csr.getColumnIndices());
    assertArrayEquals(new double[] {2.0, 4.0, 3.0, 2.0}, csr.getValues());

    assertEquals(2, problem.getNumVariables());
    assertEquals(2, problem.getNumConstraints());
    assertEquals(230, problem.getConstraint(0).getRHS());
    assertEquals(190, problem.getConstraint(1).getRHS());
  }

  @Test
  void duplicateLinearTermsAreMergedForSlack() {
    Problem problem = new Problem();
    Variable x = problem.addVariable();
    Constraint constraint = problem.addConstraint(LinearExpression.of(x, 5).plus(x, 7).le(18));

    x.setValue(1.0);

    assertEquals(12.0, constraint.getCoefficient(x));
    assertEquals(6.0, constraint.computeSlack());
    assertFalse(problem.isMIP());
  }

  @Test
  void csrMatrixRejectsMalformedInputs() {
    assertThrows(IllegalArgumentException.class, () -> new CSRMatrix(null, new int[0], new int[] {0}));
    assertThrows(IllegalArgumentException.class, () -> new CSRMatrix(new double[0], null, new int[] {0}));
    assertThrows(IllegalArgumentException.class, () -> new CSRMatrix(new double[0], new int[0], null));
    assertThrows(IllegalArgumentException.class, () -> new CSRMatrix(new double[0], new int[0], new int[0]));
    assertThrows(
        IllegalArgumentException.class,
        () -> new CSRMatrix(new double[0], new int[0], new int[] {1}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new CSRMatrix(new double[] {1.0}, new int[] {0}, new int[] {0, 2}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new CSRMatrix(new double[] {1.0}, new int[] {0}, new int[] {0, 1, 0}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new CSRMatrix(new double[] {1.0}, new int[0], new int[] {0, 1}));
  }

  @Test
  void expressionDivisionRejectsZero() {
    Problem problem = new Problem();
    Variable x = problem.addVariable();

    assertThrows(IllegalArgumentException.class, () -> LinearExpression.of(x).dividedBy(0.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> QuadraticExpression.of(x, x, 1.0).dividedBy(-0.0));
  }

  @Test
  void structuralChangesClearSolvedValues() {
    Problem problem = new Problem();
    Variable x = problem.addVariable();
    Constraint constraint = problem.addConstraint(LinearExpression.of(x).le(1.0));

    x.setValue(1.0);
    constraint.setSlack(0.0);
    problem.addVariable();
    assertTrue(Double.isNaN(x.getValue()));
    assertTrue(Double.isNaN(constraint.getSlack()));

    x.setValue(1.0);
    constraint.setSlack(0.0);
    problem.addConstraint(LinearExpression.of(x).ge(0.0));
    assertTrue(Double.isNaN(x.getValue()));
    assertTrue(Double.isNaN(constraint.getSlack()));
  }
}
