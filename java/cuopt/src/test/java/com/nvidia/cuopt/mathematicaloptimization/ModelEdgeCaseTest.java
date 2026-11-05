/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Model-building corner cases that need no native library. */
final class ModelEdgeCaseTest {
  private Problem problem;

  @BeforeEach
  void newProblem() {
    problem = new Problem();
  }

  @Test
  void anEmptyProblemHasNoStructure() {
    assertEquals("", problem.getName());
    assertEquals(0, problem.getNumVariables());
    assertEquals(0, problem.getNumConstraints());
    assertEquals(0, problem.getNumNonZeros());
    assertFalse(problem.isMIP());
    assertEquals(TerminationStatus.NO_TERMINATION, problem.getStatus());
    assertTrue(Double.isNaN(problem.getObjectiveValue()));
    assertTrue(Double.isNaN(problem.getSolveTime()));
    assertEquals(0.0, problem.getObjectiveConstant(), 0.0);
    assertEquals(ObjectiveSense.MINIMIZE, problem.getObjectiveSense());
    assertEquals(0, problem.getConstraintMatrix().getValues().length);
    assertArrayEquals(new int[] {0}, problem.getConstraintMatrix().getRowOffsets());
    assertNull(problem.getQuadraticObjectiveMatrix());
  }

  @Test
  void aNullNameBecomesTheEmptyString() {
    assertEquals("", new Problem(null).getName());
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, null);
    assertEquals("", x.getVariableName());
    assertEquals("", x.setVariableName(null).getVariableName());
    Constraint constraint = problem.addConstraint(LinearExpression.of(x).le(1), null);
    assertEquals("", constraint.getConstraintName());
  }

  @Test
  void lookupByNameReturnsNullWhenMissingAndByIndexThrows() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Constraint c = problem.addConstraint(LinearExpression.of(x).le(1), "c");
    assertSame(x, problem.getVariable("x"));
    assertSame(c, problem.getConstraint("c"));
    assertNull(problem.getVariable("missing"));
    assertNull(problem.getConstraint("missing"));
    assertThrows(IndexOutOfBoundsException.class, () -> problem.getVariable(1));
    assertThrows(IndexOutOfBoundsException.class, () -> problem.getVariable(-1));
    assertThrows(IndexOutOfBoundsException.class, () -> problem.getConstraint(1));
  }

  @Test
  void theVariableAndConstraintListsAreSnapshots() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    List<Variable> before = problem.getVariables();
    problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "y");
    assertEquals(1, before.size());
    assertEquals(2, problem.getVariables().size());
    assertThrows(UnsupportedOperationException.class, () -> before.add(x));
    assertThrows(UnsupportedOperationException.class, () -> problem.getConstraints().add(null));
  }

  @Test
  void variableIndicesFollowCreationOrder() {
    for (int i = 0; i < 5; i++) {
      assertEquals(i, problem.addVariable().getIndex());
    }
    Variable defaulted = problem.getVariable(0);
    assertEquals(0.0, defaulted.getLowerBound(), 0.0);
    assertEquals(Double.POSITIVE_INFINITY, defaulted.getUpperBound(), 0.0);
    assertEquals(VariableType.CONTINUOUS, defaulted.getVariableType());
    assertEquals(0.0, defaulted.getObjectiveCoefficient(), 0.0);
    assertTrue(Double.isNaN(defaulted.getMIPStart()));
  }

  @Test
  void aMIPIsAnyProblemWithADiscreteVariable() {
    problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "c");
    assertFalse(problem.isMIP());
    Variable integer = problem.addVariable(0, 1, 0, VariableType.INTEGER, "i");
    assertTrue(problem.isMIP());
    integer.setVariableType(VariableType.CONTINUOUS);
    assertFalse(problem.isMIP());
    problem.addVariable(0, 1, 0, VariableType.SEMI_CONTINUOUS, "s");
    assertTrue(problem.isMIP());
  }

  @Test
  void aVariableOfAnotherProblemIsRejectedEvenWhenItsIndexIsInRange() {
    Problem first = new Problem();
    Problem second = new Problem();
    Variable a = first.addVariable(0, 1, 0, VariableType.CONTINUOUS, "a");
    Variable b = second.addVariable(0, 1, 0, VariableType.CONTINUOUS, "b");
    // Both variables have index 0, so only an identity check can tell them apart.
    assertEquals(a.getIndex(), b.getIndex());
    assertThrows(IllegalArgumentException.class, () -> first.addConstraint(LinearExpression.of(b).le(1)));
    assertThrows(
        IllegalArgumentException.class,
        () -> first.setObjective(LinearExpression.of(b), ObjectiveSense.MINIMIZE));
    assertEquals(0, first.getNumConstraints());
  }

  @Test
  void aRejectedConstraintLeavesTheProblemUnchanged() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable foreign = new Problem().addVariable(0, 1, 0, VariableType.CONTINUOUS, "f");
    problem.addConstraint(LinearExpression.of(x).le(1));
    assertThrows(
        IllegalArgumentException.class,
        () -> problem.addConstraint(LinearExpression.of(x).plus(foreign).le(1)));
    assertEquals(1, problem.getNumConstraints());
    assertEquals(1, problem.getNumNonZeros());
  }

  @Test
  void aVariableWithAnInRangeIndexButNoOwnerIsRejectedInAQuadraticObjective() {
    Variable owned = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable foreign = new Problem().addVariable(0, 1, 0, VariableType.CONTINUOUS, "f");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            problem.setObjective(
                QuadraticExpression.of(foreign, foreign, 1.0), ObjectiveSense.MINIMIZE));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            problem.setObjective(
                QuadraticExpression.of(foreign, owned, 1.0), ObjectiveSense.MINIMIZE));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            problem.setObjective(
                QuadraticExpression.of(owned, foreign, 1.0), ObjectiveSense.MINIMIZE));
  }

  @Test
  void aRejectedLinearObjectiveLeavesTheCurrentObjectiveUntouched() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable foreign = new Problem().addVariable(0, 1, 0, VariableType.CONTINUOUS, "f");
    problem.setObjective(LinearExpression.of(x, 3.0), ObjectiveSense.MAXIMIZE);
    assertThrows(
        IllegalArgumentException.class,
        () ->
            problem.setObjective(
                LinearExpression.of(x, 9.0).plus(foreign, 5.0), ObjectiveSense.MINIMIZE));
    assertEquals(3.0, x.getObjectiveCoefficient(), 0.0);
    assertEquals(ObjectiveSense.MAXIMIZE, problem.getObjectiveSense());
  }

  @Test
  void aRejectedQuadraticObjectiveLeavesTheCurrentObjectiveUntouched() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable foreign = new Problem().addVariable(0, 1, 0, VariableType.CONTINUOUS, "f");
    problem.setObjective(LinearExpression.of(x, 3.0), ObjectiveSense.MAXIMIZE);
    assertThrows(
        IllegalArgumentException.class,
        () ->
            problem.setObjective(
                QuadraticExpression.of(foreign, foreign, 1.0).plus(x, 2.0),
                ObjectiveSense.MINIMIZE));
    assertEquals(3.0, x.getObjectiveCoefficient(), 0.0);
    assertEquals(ObjectiveSense.MAXIMIZE, problem.getObjectiveSense());
    assertNull(problem.getQuadraticObjectiveMatrix());
  }

  @Test
  void settingTheObjectiveReplacesVariableCoefficients() {
    Variable x = problem.addVariable(0, 1, 7.0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 1, 9.0, VariableType.CONTINUOUS, "y");
    problem.setObjective(LinearExpression.of(y, 2.0), ObjectiveSense.MAXIMIZE);
    assertEquals(0.0, x.getObjectiveCoefficient(), 0.0);
    assertEquals(2.0, y.getObjectiveCoefficient(), 0.0);
    assertEquals(ObjectiveSense.MAXIMIZE, problem.getObjectiveSense());
  }

  @Test
  void aConstantObjectiveHasNoCoefficients() {
    problem.addVariable(0, 1, 5.0, VariableType.CONTINUOUS, "x");
    problem.setObjective(3.5, ObjectiveSense.MINIMIZE);
    assertEquals(3.5, problem.getObjectiveConstant(), 0.0);
    assertEquals(0.0, problem.getVariable(0).getObjectiveCoefficient(), 0.0);
  }

  @Test
  void aQuadraticObjectiveReplacesALinearOneAndBack() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    problem.setObjective(QuadraticExpression.of(x, x, 2.0), ObjectiveSense.MINIMIZE);
    assertEquals(1, problem.getQuadraticObjectiveMatrix().getValues().length);
    problem.setObjective(LinearExpression.of(x), ObjectiveSense.MINIMIZE);
    assertNull(problem.getQuadraticObjectiveMatrix());
  }

  @Test
  void duplicateQuadraticObjectiveTermsAreSummedIntoOneCSREntry() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "y");
    QuadraticExpression objective =
        QuadraticExpression.of(x, y, 1.0).plus(x, y, 2.0).plus(y, x, 4.0).plus(x, x, 8.0);
    problem.setObjective(objective, ObjectiveSense.MINIMIZE);
    CSRMatrix q = problem.getQuadraticObjectiveMatrix();
    // Row x holds (x,x)=8 and (x,y)=3; row y holds (y,x)=4. The two orderings stay separate.
    assertArrayEquals(new int[] {0, 2, 3}, q.getRowOffsets());
    assertArrayEquals(new int[] {0, 1, 0}, q.getColumnIndices());
    assertArrayEquals(new double[] {8.0, 3.0, 4.0}, q.getValues(), 0.0);
  }

  @Test
  void aConstraintMovesItsConstantToTheRightHandSide() {
    Variable x = problem.addVariable(0, 10, 0, VariableType.CONTINUOUS, "x");
    Constraint le = LinearExpression.of(x).plus(3.0).le(10.0);
    Constraint ge = LinearExpression.of(x).minus(2.0).ge(1.0);
    Constraint eq = LinearExpression.of(x, 2.0).constant(5.0).eq(9.0);
    assertEquals(7.0, le.getRHS(), 0.0);
    assertEquals(3.0, ge.getRHS(), 0.0);
    assertEquals(4.0, eq.getRHS(), 0.0);
    assertEquals(ConstraintSense.LE, le.getSense());
    assertEquals(ConstraintSense.GE, ge.getSense());
    assertEquals(ConstraintSense.EQ, eq.getSense());
  }

  @Test
  void comparingTwoExpressionsMovesEverythingToTheLeft() {
    Variable x = problem.addVariable(0, 10, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 10, 0, VariableType.CONTINUOUS, "y");
    Constraint c = LinearExpression.of(x, 2.0).plus(1.0).le(LinearExpression.of(y, 3.0).plus(4.0));
    assertEquals(2.0, c.getCoefficient(x), 0.0);
    assertEquals(-3.0, c.getCoefficient(y), 0.0);
    assertEquals(3.0, c.getRHS(), 0.0);
    Constraint v = LinearExpression.of(x).ge(y);
    assertEquals(1.0, v.getCoefficient(x), 0.0);
    assertEquals(-1.0, v.getCoefficient(y), 0.0);
    assertEquals(0.0, v.getRHS(), 0.0);
  }

  @Test
  void aVariableAbsentFromAConstraintHasCoefficientZero() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "y");
    Constraint c = LinearExpression.of(x).le(1);
    assertEquals(0.0, c.getCoefficient(y), 0.0);
    assertEquals(0.0, LinearExpression.of(x).getCoefficient(y), 0.0);
  }

  @Test
  void expressionsAreImmutable() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    LinearExpression base = LinearExpression.of(x, 2.0);
    LinearExpression grown = base.plus(x, 3.0).plus(1.0).times(2.0);
    assertEquals(2.0, base.getCoefficient(x), 0.0);
    assertEquals(0.0, base.getConstant(), 0.0);
    assertEquals(10.0, grown.getCoefficient(x), 0.0);
    assertEquals(2.0, grown.getConstant(), 0.0);
    assertNotSame(base, grown);
    assertThrows(
        UnsupportedOperationException.class, () -> base.getTerms().put(x, 1.0));

    QuadraticExpression q = QuadraticExpression.of(x, x, 1.0);
    q.plus(x, x, 5.0).times(3.0);
    assertEquals(1, q.getQuadraticTerms().size());
    assertEquals(1.0, q.getCoefficient(0), 0.0);
    assertThrows(
        UnsupportedOperationException.class, () -> q.getQuadraticTerms().add(null));
  }

  @Test
  void mergingTermsForOneVariableSumsTheirCoefficientsAndKeepsZeros() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    LinearExpression cancelled = LinearExpression.of(x, 2.0).plus(x, -2.0);
    assertEquals(0.0, cancelled.getCoefficient(x), 0.0);
    // A cancelled term stays in the expression, so it still occupies a CSR slot.
    assertEquals(1, cancelled.getVariables().size());
    problem.addConstraint(cancelled.le(1.0));
    assertEquals(1, problem.getNumNonZeros());

    LinearExpression zeroed = LinearExpression.of(x, 4.0).times(0.0);
    assertEquals(1, zeroed.getVariables().size());
    assertEquals(0.0, zeroed.getCoefficient(x), 0.0);
  }

  @Test
  void expressionAccessorsByPositionFollowInsertionOrder() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "y");
    LinearExpression e = LinearExpression.of(y, 5.0).plus(x, 7.0);
    assertSame(y, e.getVariable(0));
    assertSame(x, e.getVariable(1));
    assertEquals(5.0, e.getCoefficient(0), 0.0);
    assertEquals(7.0, e.getCoefficient(1), 0.0);
    assertEquals(List.of(5.0, 7.0), e.getCoefficients());
    assertThrows(IndexOutOfBoundsException.class, () -> e.getVariable(2));
    assertThrows(IndexOutOfBoundsException.class, () -> e.getCoefficient(2));
  }

  @Test
  void anExpressionEvaluatesToNaNUntilTheVariablesHaveValues() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    assertTrue(Double.isNaN(LinearExpression.of(x).getValue()));
    assertEquals(4.0, LinearExpression.ofConstant(4.0).getValue(), 0.0);
    assertEquals(4.0, new LinearExpression().plus(4.0).getValue(), 0.0);
  }

  @Test
  void dividingByZeroIsRejectedAndNegativeDivisorsFlipTheSign() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    assertThrows(IllegalArgumentException.class, () -> LinearExpression.of(x).dividedBy(0.0));
    assertThrows(IllegalArgumentException.class, () -> LinearExpression.of(x).dividedBy(-0.0));
    assertThrows(
        IllegalArgumentException.class, () -> QuadraticExpression.of(x, x, 1.0).dividedBy(0.0));
    assertEquals(-2.0, LinearExpression.of(x, 4.0).dividedBy(-2.0).getCoefficient(x), 0.0);
  }

  @Test
  void nonFiniteCoefficientsAreCarriedIntoTheConstraintMatrix() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "y");
    LinearExpression nan = LinearExpression.of(x, Double.NaN);
    LinearExpression inf = LinearExpression.of(y, Double.POSITIVE_INFINITY);
    assertTrue(Double.isNaN(nan.getCoefficient(x)));
    assertEquals(Double.POSITIVE_INFINITY, inf.getCoefficient(y), 0.0);
    assertTrue(Double.isNaN(inf.plus(y, Double.NEGATIVE_INFINITY).getCoefficient(y)));
    problem.addConstraint(nan.plus(inf).le(1.0));
    double[] values = problem.getConstraintMatrix().getValues();
    assertEquals(2, values.length);
    assertTrue(Double.isNaN(values[0]));
    assertEquals(Double.POSITIVE_INFINITY, values[1], 0.0);
  }

  @Test
  void aRejectedConstraintKeepsItsIndexAndName() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Constraint shared = LinearExpression.of(x).le(1);
    problem.addConstraint(shared, "first");
    Problem other = new Problem();
    assertThrows(IllegalArgumentException.class, () -> other.addConstraint(shared, "second"));
    assertEquals(0, shared.getIndex());
    assertEquals("first", shared.getConstraintName());
  }

  @Test
  @Disabled("addConstraint accepts a constraint that is already in another problem if it has no variables")
  void aConstraintWithoutVariablesCannotBeAddedToTwoProblems() {
    Constraint shared = LinearExpression.ofConstant(0.0).le(1);
    problem.addConstraint(shared, "first");
    Problem other = new Problem();
    assertThrows(IllegalArgumentException.class, () -> other.addConstraint(shared, "second"));
    assertEquals(0, shared.getIndex());
  }

  @Test
  void addingTheSameConstraintTwiceToOneProblemIsRejected() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Constraint shared = LinearExpression.of(x).le(1);
    problem.addConstraint(shared);
    assertThrows(IllegalArgumentException.class, () -> problem.addConstraint(shared));
    assertEquals(1, problem.getNumConstraints());
        assertEquals(0, shared.getIndex());
  }

  @Test
  void aQuadraticConstraintIsKeptOutOfTheLinearMatrix() {
    Variable x = problem.addVariable(0, 5, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 5, 0, VariableType.CONTINUOUS, "y");
    problem.addConstraint(LinearExpression.of(x).plus(y).le(4.0), "linear");
    Constraint quadratic =
        problem.addConstraint(QuadraticExpression.of(x, x, 1.0).plus(y, 1.0).le(9.0), "quad");
    assertTrue(quadratic.isQuadratic());
    assertEquals(2, problem.getNumConstraints());
    assertEquals(2, problem.getNumNonZeros());
    assertEquals(List.of(quadratic), problem.getQuadraticConstraints());
    assertEquals(1, problem.getConstraintMatrix().getRowOffsets().length - 1);
  }

  @Test
  void aQuadraticExpressionWithoutQuadraticTermsBuildsALinearConstraint() {
    Variable x = problem.addVariable(0, 5, 0, VariableType.CONTINUOUS, "x");
    Constraint c = problem.addConstraint(new QuadraticExpression().plus(x, 2.0).le(4.0));
    assertFalse(c.isQuadratic());
    assertEquals(1, problem.getNumNonZeros());
    assertTrue(problem.getQuadraticConstraints().isEmpty());
  }

  @Test
  void quadraticComparisonsMoveBothSidesToTheLeft() {
    Variable x = problem.addVariable(0, 5, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 5, 0, VariableType.CONTINUOUS, "y");
    QuadraticExpression left = QuadraticExpression.of(x, x, 1.0).plus(1.0);
    Constraint c = left.le(QuadraticExpression.of(y, y, 2.0).plus(4.0));
    assertEquals(2, c.getQuadraticExpression().getQuadraticTerms().size());
    assertEquals(3.0, c.getRHS(), 0.0);
    Constraint viaLinear = left.ge(LinearExpression.of(y, 3.0));
    assertEquals(-3.0, viaLinear.getCoefficient(y), 0.0);
    Constraint viaVariable = left.le(y);
    assertEquals(-1.0, viaVariable.getCoefficient(y), 0.0);
  }

  @Test
  void aLinearExpressionComparedWithAQuadraticOneBecomesAQuadraticConstraint() {
    Variable x = problem.addVariable(0, 5, 0, VariableType.CONTINUOUS, "x");
    Constraint c = LinearExpression.of(x).le(QuadraticExpression.of(x, x, 1.0));
    assertTrue(c.isQuadratic());
    assertEquals(-1.0, c.getQuadraticExpression().getCoefficient(0), 0.0);
    assertEquals(1.0, c.getCoefficient(x), 0.0);
  }

  @Test
  void quadraticConstraintRecordValidatesItsArrays() {
    IllegalArgumentException linear =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new QuadraticConstraint(
                    0, "q", ConstraintSense.LE, new double[] {1.0}, new int[] {}, 1.0, new int[0],
                    new int[0], new double[0]));
    assertTrue(linear.getMessage().contains("linearValues"), linear.getMessage());
    IllegalArgumentException quadratic =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new QuadraticConstraint(
                    0, "q", ConstraintSense.LE, new double[0], new int[0], 1.0, new int[] {0},
                    new int[] {0, 1}, new double[] {1.0}));
    assertTrue(quadratic.getMessage().contains("quadratic COO"), quadratic.getMessage());
    IllegalArgumentException equality =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new QuadraticConstraint(
                    0, "q", ConstraintSense.EQ, new double[0], new int[0], 1.0, new int[0],
                    new int[0], new double[0]));
    assertTrue(equality.getMessage().contains("Equality"), equality.getMessage());
  }

  @Test
  void quadraticConstraintRecordCopiesItsInputsAndDefaultsTheName() {
    double[] values = {2.0};
    QuadraticConstraint record =
        new QuadraticConstraint(
            3, null, ConstraintSense.GE, new double[0], new int[0], 1.0, new int[] {0},
            new int[] {0}, values);
    values[0] = 99.0;
    assertEquals(2.0, record.getValues()[0], 0.0);
    assertEquals("", record.getRowName());
    record.getValues()[0] = 7.0;
    assertEquals(2.0, record.getValues()[0], 0.0);
    assertEquals(3, record.getRowIndex());
    assertEquals(ConstraintSense.GE, record.getSense());
  }

  @Test
  void csrMatrixRejectsNullArraysAndCopiesItsInputs() {
    assertThrows(
        IllegalArgumentException.class, () -> new CSRMatrix(null, new int[0], new int[] {0}));
    assertThrows(
        IllegalArgumentException.class, () -> new CSRMatrix(new double[0], null, new int[] {0}));
    assertThrows(
        IllegalArgumentException.class, () -> new CSRMatrix(new double[0], new int[0], null));
    assertThrows(
        IllegalArgumentException.class, () -> new CSRMatrix(new double[0], new int[0], new int[0]));

    double[] values = {1.0, 2.0};
    int[] columns = {0, 1};
    int[] offsets = {0, 2};
    CSRMatrix matrix = new CSRMatrix(values, columns, offsets);
    values[0] = 9.0;
    columns[0] = 9;
    offsets[1] = 9;
    assertArrayEquals(new double[] {1.0, 2.0}, matrix.getValues(), 0.0);
    assertArrayEquals(new int[] {0, 1}, matrix.getColumnIndices());
    assertArrayEquals(new int[] {0, 2}, matrix.getRowOffsets());
    matrix.getValues()[0] = 9.0;
    assertEquals(1.0, matrix.getValues()[0], 0.0);
  }

  @Test
  void csrMatrixAcceptsEmptyRows() {
    CSRMatrix matrix = new CSRMatrix(new double[] {5.0}, new int[] {2}, new int[] {0, 0, 1, 1});
    assertArrayEquals(new double[] {5.0}, matrix.getValues(), 0.0);
    assertArrayEquals(new int[] {2}, matrix.getColumnIndices());
    assertArrayEquals(new int[] {0, 0, 1, 1}, matrix.getRowOffsets());
    assertThrows(
        IllegalArgumentException.class,
        () -> new CSRMatrix(new double[] {5.0}, new int[] {2}, new int[] {0, 1, 0, 1}));
  }

  @Test
  void mipCallbackSolutionCopiesItsArray() {
    double[] values = {1.0, 2.0};
    MIPCallbackSolution solution = new MIPCallbackSolution(values, 3.0);
    values[0] = 9.0;
    assertArrayEquals(new double[] {1.0, 2.0}, solution.getSolution(), 0.0);
    solution.getSolution()[0] = 9.0;
    assertEquals(1.0, solution.getSolution()[0], 0.0);
  }

  @Test
  void fromIncumbentHandlesAnEmptyRequestAndRepeatedVariables() {
    Variable x = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "x");
    Variable y = problem.addVariable(0, 1, 0, VariableType.CONTINUOUS, "y");
    double[] incumbent = {10.0, 20.0};
    assertEquals(0, Problem.fromIncumbent(incumbent).length);
    assertArrayEquals(new double[] {20.0, 20.0, 10.0}, Problem.fromIncumbent(incumbent, y, y, x), 0.0);
    assertThrows(IllegalArgumentException.class, () -> Problem.fromIncumbent(new double[0], x));
  }

  @Test
  void unknownNativeCodesAreRejectedOrMappedToUnknown() {
    assertThrows(IllegalArgumentException.class, () -> VariableType.fromNative((byte) 'Z'));
    assertThrows(IllegalArgumentException.class, () -> ConstraintSense.fromNative((byte) 'Z'));
    assertEquals(TerminationStatus.UNKNOWN, TerminationStatus.fromNative(Integer.MAX_VALUE - 1));
    assertEquals(TerminationStatus.UNKNOWN, TerminationStatus.fromNative(-12345));
  }

  @Test
  void everyNativeCodeRoundTripsThroughItsEnum() {
    for (VariableType type : VariableType.values()) {
      assertSame(type, VariableType.fromNative(type.nativeValue()));
    }
    for (ConstraintSense sense : ConstraintSense.values()) {
      assertSame(sense, ConstraintSense.fromNative(sense.nativeValue()));
    }
    for (TerminationStatus status : TerminationStatus.values()) {
      if (status != TerminationStatus.UNKNOWN) {
        assertSame(status, TerminationStatus.fromNative(status.nativeValue()));
      }
    }
  }
}
