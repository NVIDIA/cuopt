/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.Map;

public final class Constraint {
  private int index = -1;
  private LinearExpression linearExpression;
  private final QuadraticExpression quadraticExpression;
  private final ConstraintSense sense;
  private double rhs;
  private String name = "";
  private double slack = Double.NaN;
  private double dualValue = Double.NaN;

  Constraint(LinearExpression expression, ConstraintSense sense, double rhs) {
    this.linearExpression = expression;
    this.quadraticExpression = null;
    this.sense = sense;
    this.rhs = rhs - expression.getConstant();
  }

  Constraint(QuadraticExpression expression, ConstraintSense sense, double rhs) {
    this.linearExpression = expression.getLinearExpression();
    this.quadraticExpression = expression;
    this.sense = sense;
    this.rhs = rhs - expression.getLinearExpression().getConstant();
  }

  public String getConstraintName() {
    return name;
  }

  public int getIndex() {
    return index;
  }

  void setIndex(int index) {
    this.index = index;
  }

  Constraint setConstraintName(String name) {
    this.name = name == null ? "" : name;
    return this;
  }

  public ConstraintSense getSense() {
    return sense;
  }

  public double getRHS() {
    return rhs;
  }

  Constraint updateLinearExpression(LinearExpression expression) {
    this.linearExpression = expression;
    return this;
  }

  Constraint updateRHS(double rhs) {
    this.rhs = rhs;
    return this;
  }

  public double getCoefficient(Variable variable) {
    return linearExpression.getTerms().getOrDefault(variable, 0.0);
  }

  public LinearExpression getLinearExpression() {
    return linearExpression;
  }

  public boolean isQuadratic() {
    return quadraticExpression != null && !quadraticExpression.getQuadraticTerms().isEmpty();
  }

  public QuadraticExpression getQuadraticExpression() {
    return quadraticExpression;
  }

  public double computeSlack() {
    double lhs = 0.0;
    for (Map.Entry<Variable, Double> entry : linearExpression.getTerms().entrySet()) {
      lhs += entry.getValue() * entry.getKey().getValue();
    }
    if (isQuadratic()) {
      for (QuadraticExpression.QuadraticTerm term : quadraticExpression.getQuadraticTerms()) {
        lhs +=
            term.getCoefficient()
                * term.getFirst().getValue()
                * term.getSecond().getValue();
      }
    }
    // Match Python's Constraint.compute_slack contract: RHS minus the evaluated LHS for
    // every row sense. This intentionally keeps the sign convention stable for GE rows too.
    return rhs - lhs;
  }

  public double getSlack() {
    return slack;
  }

  void setSlack(double slack) {
    this.slack = slack;
  }

  public double getDualValue() {
    return dualValue;
  }

  void setDualValue(double dualValue) {
    this.dualValue = dualValue;
  }

  void resetSolvedValues() {
    slack = Double.NaN;
    dualValue = Double.NaN;
  }
}
