/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

final class ProblemModelingTest {
  @Test
  void buildsLinearModelAndCsr() {
    Problem problem = new Problem("Simple MIP");
    Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0, VariableType.INTEGER, "x");
    Variable y = problem.addVariable(10, 50, 0, VariableType.INTEGER, "y");

    assertEquals(0, x.getIndex());
    assertEquals(1, y.getIndex());
    assertTrue(problem.isMip());

    problem.addConstraint(LinearExpression.of(x, 2).plus(y, 4).ge(230), "c1");
    problem.addConstraint(LinearExpression.of(x, 3).plus(y, 2).constant(10).le(200), "c2");
    problem.setObjective(LinearExpression.of(x, 5).plus(y, 3).constant(50), ObjectiveSense.MAXIMIZE);

    CsrMatrix csr = problem.getCSR();
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
    assertFalse(problem.isMip());
  }

  @Test
  void updateRelaxAndQuadraticInspectionMatchPythonModelingContracts() {
    Problem problem = new Problem("model");
    Variable x = problem.addVariable(0.0, 5.0, 1.0, VariableType.INTEGER, "x");
    Variable y = problem.addVariable(0.0, 5.0, 2.0, VariableType.CONTINUOUS, "y");
    Constraint constraint =
        problem.addConstraint(LinearExpression.of(x, 2.0).plus(y).le(7.0), "c");

    problem.updateConstraint(constraint, Map.of(x, 1.0), 8.0);
    assertEquals(1.0, constraint.getCoefficient(x));
    assertEquals(8.0, constraint.getRHS());

    problem.updateObjective(Map.of(x, 3.0, y, 4.0), 5.0, ObjectiveSense.MAXIMIZE);
    assertEquals(ObjectiveSense.MAXIMIZE, problem.getObjectiveSense());
    assertEquals(5.0, problem.getObjectiveConstant());
    assertEquals(3.0, x.getObjectiveCoefficient());
    assertEquals(4.0, y.getObjectiveCoefficient());

    QuadraticExpression quadratic = QuadraticExpression.of(x, x, 2.0).plus(y, y, 3.0);
    problem.setObjective(quadratic, ObjectiveSense.MINIMIZE);
    CsrMatrix qcsr = problem.getQCSR();
    assertArrayEquals(new int[] {0, 1, 2}, qcsr.getRowOffsets());
    assertArrayEquals(new int[] {0, 1}, qcsr.getColumnIndices());
    assertArrayEquals(new double[] {2.0, 3.0}, qcsr.getValues());
    assertEquals(2, quadratic.getCoefficients().size());
    assertEquals(2.0, quadratic.getCoefficient(0));

    Constraint quadraticConstraint =
        problem.addConstraint(QuadraticExpression.of(x, x, 1.0).plus(y, y, 1.0).le(10.0), "qc");
    x.setValue(1.0);
    y.setValue(2.0);
    assertEquals(1, quadraticConstraint.getIndex());
    assertEquals(5.0, quadraticConstraint.computeSlack());
    assertEquals(1, problem.getQuadraticConstraints().size());

    Problem relaxed = problem.relax();
    assertFalse(relaxed.isMip());
    assertEquals(VariableType.CONTINUOUS, relaxed.getVariable(0).getVariableType());
    assertEquals("x", relaxed.getVariable(0).getVariableName());
    assertEquals("y", relaxed.getVariable(1).getVariableName());
  }
}
