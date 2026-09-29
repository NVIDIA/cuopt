/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

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

  @Test
  void rejectsLinearConstraintOverAForeignVariable() {
    Problem owner = new Problem("owner");
    owner.addVariable(0, 10, 1, VariableType.CONTINUOUS, "mine");

    Problem other = new Problem("other");
    Variable foreign = other.addVariable(0, 10, 1, VariableType.CONTINUOUS, "theirs");

    // foreign.getIndex() is 0, the same index 'mine' holds, so without the check the term would
    // be applied to 'mine' and the wrong model would solve without an error.
    assertEquals(0, foreign.getIndex());
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> owner.addConstraint(LinearExpression.of(foreign, 2.0).le(5.0)));
    assertTrue(error.getMessage().contains("theirs"));
    assertEquals(0, owner.getConstraints().size());
  }

  @Test
  void rejectsQuadraticConstraintOverAForeignVariable() {
    Problem owner = new Problem("owner");
    Variable mine = owner.addVariable(0, 10, 1, VariableType.CONTINUOUS, "mine");

    Problem other = new Problem("other");
    Variable foreign = other.addVariable(0, 10, 1, VariableType.CONTINUOUS, "theirs");

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> owner.addConstraint(QuadraticExpression.of(mine, foreign, 1.0).le(5.0)));
    assertTrue(error.getMessage().contains("theirs"));
    assertEquals(0, owner.getConstraints().size());
  }

  @Test
  void fromIncumbentReordersByVariableIndex() {
    Problem problem = new Problem("reorder");
    Variable x = problem.addVariable(0, 10, 1, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 10, 1, VariableType.CONTINUOUS, "y");
    Variable z = problem.addVariable(0, 10, 1, VariableType.CONTINUOUS, "z");

    // An incumbent is always in variable-index order: x=1, y=2, z=3.
    double[] incumbent = {1.0, 2.0, 3.0};

    assertArrayEquals(new double[] {3.0, 2.0, 1.0}, Problem.fromIncumbent(incumbent, z, y, x));
    assertArrayEquals(new double[] {2.0}, Problem.fromIncumbent(incumbent, y));
    assertArrayEquals(new double[] {}, Problem.fromIncumbent(incumbent));
  }

  @Test
  void fromIncumbentRejectsAnIndexOutsideTheArray() {
    Problem problem = new Problem("short");
    Variable a = problem.addVariable(0, 1, 1, VariableType.CONTINUOUS, "a");
    Variable b = problem.addVariable(0, 1, 1, VariableType.CONTINUOUS, "b");

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> Problem.fromIncumbent(new double[] {1.0}, b));
    assertTrue(error.getMessage().contains("'b'"));
  }

  @Test
  void writeRejectsANonMPSExtension() {
    Problem problem = new Problem("write");
    problem.addVariable(0, 1, 1, VariableType.CONTINUOUS, "x");

    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> problem.write("/tmp/model.lp"));
    assertTrue(error.getMessage().contains(".mps"));
  }
}
