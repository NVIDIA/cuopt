/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class QuadraticExpression {
  public static final class QuadraticTerm {
    private final Variable first;
    private final Variable second;
    private final double coefficient;

    QuadraticTerm(Variable first, Variable second, double coefficient) {
      this.first = first;
      this.second = second;
      this.coefficient = coefficient;
    }

    public Variable getFirst() {
      return first;
    }

    public Variable getSecond() {
      return second;
    }

    public double getCoefficient() {
      return coefficient;
    }
  }

  private final LinearExpression linearExpression;
  private final List<QuadraticTerm> quadraticTerms;

  public QuadraticExpression() {
    this(new LinearExpression(), new ArrayList<>());
  }

  private QuadraticExpression(LinearExpression linearExpression, List<QuadraticTerm> quadraticTerms) {
    this.linearExpression = linearExpression;
    this.quadraticTerms = quadraticTerms;
  }

  public static QuadraticExpression of(Variable first, Variable second, double coefficient) {
    return new QuadraticExpression().plus(first, second, coefficient);
  }

  public QuadraticExpression plus(Variable first, Variable second, double coefficient) {
    List<QuadraticTerm> copy = new ArrayList<>(quadraticTerms);
    copy.add(new QuadraticTerm(first, second, coefficient));
    return new QuadraticExpression(linearExpression, copy);
  }

  public QuadraticExpression plus(Variable variable, double coefficient) {
    return new QuadraticExpression(linearExpression.plus(variable, coefficient), quadraticTerms);
  }

  public QuadraticExpression plus(Variable variable) {
    return plus(variable, 1.0);
  }

  public QuadraticExpression plus(double value) {
    return constant(value);
  }

  public QuadraticExpression plus(LinearExpression expression) {
    return new QuadraticExpression(linearExpression.plus(expression), quadraticTerms);
  }

  public QuadraticExpression plus(QuadraticExpression expression) {
    List<QuadraticTerm> copy = new ArrayList<>(quadraticTerms);
    copy.addAll(expression.quadraticTerms);
    return new QuadraticExpression(linearExpression.plus(expression.linearExpression), copy);
  }

  public QuadraticExpression constant(double constant) {
    return new QuadraticExpression(linearExpression.constant(constant), quadraticTerms);
  }

  public QuadraticExpression minus(QuadraticExpression expression) {
    return plus(expression.times(-1.0));
  }

  public QuadraticExpression minus(LinearExpression expression) {
    return plus(expression.times(-1.0));
  }

  public QuadraticExpression minus(Variable variable) {
    return plus(variable, -1.0);
  }

  public QuadraticExpression minus(double value) {
    return plus(-value);
  }

  public QuadraticExpression times(double scalar) {
    List<QuadraticTerm> terms = new ArrayList<>();
    for (QuadraticTerm term : quadraticTerms) {
      terms.add(new QuadraticTerm(term.first, term.second, term.coefficient * scalar));
    }
    return new QuadraticExpression(linearExpression.times(scalar), terms);
  }

  public QuadraticExpression dividedBy(double scalar) {
    return times(1.0 / scalar);
  }

  public Constraint le(double rhs) {
    return new Constraint(this, ConstraintSense.LE, rhs);
  }

  public Constraint le(Variable variable) {
    return minus(variable).le(0.0);
  }

  public Constraint le(LinearExpression expression) {
    return minus(expression).le(0.0);
  }

  public Constraint le(QuadraticExpression expression) {
    return minus(expression).le(0.0);
  }

  public Constraint ge(double rhs) {
    return new Constraint(this, ConstraintSense.GE, rhs);
  }

  public Constraint ge(Variable variable) {
    return minus(variable).ge(0.0);
  }

  public Constraint ge(LinearExpression expression) {
    return minus(expression).ge(0.0);
  }

  public Constraint ge(QuadraticExpression expression) {
    return minus(expression).ge(0.0);
  }

  public Constraint eq(double rhs) {
    throw new IllegalArgumentException("Equality quadratic constraints are not supported");
  }

  public LinearExpression getLinearExpression() {
    return linearExpression;
  }

  public double getConstant() {
    return linearExpression.getConstant();
  }

  public List<QuadraticTerm> getQuadraticTerms() {
    return Collections.unmodifiableList(quadraticTerms);
  }

  public List<Variable[]> getVariables() {
    List<Variable[]> result = new ArrayList<>();
    for (QuadraticTerm term : quadraticTerms) {
      result.add(new Variable[] {term.first, term.second});
    }
    return Collections.unmodifiableList(result);
  }

  public Variable getVariable1(int index) {
    return quadraticTerms.get(index).first;
  }

  public Variable getVariable2(int index) {
    return quadraticTerms.get(index).second;
  }

  public List<Double> getCoefficients() {
    List<Double> result = new ArrayList<>();
    for (QuadraticTerm term : quadraticTerms) {
      result.add(term.coefficient);
    }
    return Collections.unmodifiableList(result);
  }

  public double getCoefficient(int index) {
    return quadraticTerms.get(index).coefficient;
  }

  public double getValue() {
    double value = linearExpression.getValue();
    for (QuadraticTerm term : quadraticTerms) {
      value += term.coefficient * term.first.getValue() * term.second.getValue();
    }
    return value;
  }
}
