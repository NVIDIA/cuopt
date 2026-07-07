/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class LinearExpression {
  private final LinkedHashMap<Variable, Double> terms;
  private final double constant;

  public LinearExpression() {
    this(new LinkedHashMap<>(), 0.0);
  }

  private LinearExpression(LinkedHashMap<Variable, Double> terms, double constant) {
    this.terms = terms;
    this.constant = constant;
  }

  public static LinearExpression of(Variable variable) {
    return of(variable, 1.0);
  }

  public static LinearExpression of(Variable variable, double coefficient) {
    return new LinearExpression().plus(variable, coefficient);
  }

  public static LinearExpression ofConstant(double constant) {
    return new LinearExpression(new LinkedHashMap<>(), constant);
  }

  public LinearExpression plus(Variable variable) {
    return plus(variable, 1.0);
  }

  public LinearExpression plus(Variable variable, double coefficient) {
    LinkedHashMap<Variable, Double> copy = new LinkedHashMap<>(terms);
    copy.merge(variable, coefficient, Double::sum);
    return new LinearExpression(copy, constant);
  }

  public LinearExpression plus(LinearExpression other) {
    LinkedHashMap<Variable, Double> copy = new LinkedHashMap<>(terms);
    for (Map.Entry<Variable, Double> entry : other.terms.entrySet()) {
      copy.merge(entry.getKey(), entry.getValue(), Double::sum);
    }
    return new LinearExpression(copy, constant + other.constant);
  }

  public QuadraticExpression plus(QuadraticExpression other) {
    return other.plus(this);
  }

  public LinearExpression constant(double additionalConstant) {
    return new LinearExpression(new LinkedHashMap<>(terms), constant + additionalConstant);
  }

  public LinearExpression plus(double value) {
    return constant(value);
  }

  public LinearExpression minus(double value) {
    return constant(-value);
  }

  public LinearExpression minus(Variable variable) {
    return plus(variable, -1.0);
  }

  public LinearExpression minus(Variable variable, double coefficient) {
    return plus(variable, -coefficient);
  }

  public LinearExpression minus(LinearExpression other) {
    return plus(other.times(-1.0));
  }

  public QuadraticExpression minus(QuadraticExpression other) {
    return other.times(-1.0).plus(this);
  }

  public LinearExpression times(double scalar) {
    LinkedHashMap<Variable, Double> copy = new LinkedHashMap<>();
    for (Map.Entry<Variable, Double> entry : terms.entrySet()) {
      copy.put(entry.getKey(), entry.getValue() * scalar);
    }
    return new LinearExpression(copy, constant * scalar);
  }

  public LinearExpression dividedBy(double scalar) {
    return times(1.0 / scalar);
  }

  public Map<Variable, Double> getVariablesAndCoefficients() {
    return getTerms();
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
    return expression.times(-1.0).plus(this).le(0.0);
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
    return expression.times(-1.0).plus(this).ge(0.0);
  }

  public Constraint eq(double rhs) {
    return new Constraint(this, ConstraintSense.EQ, rhs);
  }

  public Constraint eq(Variable variable) {
    return minus(variable).eq(0.0);
  }

  public Constraint eq(LinearExpression expression) {
    return minus(expression).eq(0.0);
  }

  public Map<Variable, Double> getTerms() {
    return Collections.unmodifiableMap(terms);
  }

  public List<Variable> getVariables() {
    return List.copyOf(terms.keySet());
  }

  public Variable getVariable(int index) {
    return new ArrayList<>(terms.keySet()).get(index);
  }

  public List<Double> getCoefficients() {
    return List.copyOf(terms.values());
  }

  public double getCoefficient(int index) {
    return new ArrayList<>(terms.values()).get(index);
  }

  public double getCoefficient(Variable variable) {
    return terms.getOrDefault(variable, 0.0);
  }

  public double getConstant() {
    return constant;
  }

  public double getValue() {
    double value = constant;
    for (Map.Entry<Variable, Double> entry : terms.entrySet()) {
      value += entry.getValue() * entry.getKey().getValue();
    }
    return value;
  }
}
