/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public final class Variable {
  private final int index;
  private double lowerBound;
  private double upperBound;
  private double objectiveCoefficient;
  private VariableType variableType;
  private String name;
  private double value = Double.NaN;
  private double reducedCost = Double.NaN;
  private double mipStart = Double.NaN;

  Variable(
      int index,
      double lowerBound,
      double upperBound,
      double objectiveCoefficient,
      VariableType variableType,
      String name) {
    this.index = index;
    this.lowerBound = lowerBound;
    this.upperBound = upperBound;
    this.objectiveCoefficient = objectiveCoefficient;
    this.variableType = variableType;
    this.name = name == null ? "" : name;
  }

  public int getIndex() {
    return index;
  }

  public double getLowerBound() {
    return lowerBound;
  }

  public Variable setLowerBound(double lowerBound) {
    this.lowerBound = lowerBound;
    return this;
  }

  public double getUpperBound() {
    return upperBound;
  }

  public Variable setUpperBound(double upperBound) {
    this.upperBound = upperBound;
    return this;
  }

  public double getObjectiveCoefficient() {
    return objectiveCoefficient;
  }

  public Variable setObjectiveCoefficient(double objectiveCoefficient) {
    this.objectiveCoefficient = objectiveCoefficient;
    return this;
  }

  public VariableType getVariableType() {
    return variableType;
  }

  public Variable setVariableType(VariableType variableType) {
    this.variableType = variableType;
    return this;
  }

  public String getVariableName() {
    return name;
  }

  public Variable setVariableName(String name) {
    this.name = name == null ? "" : name;
    return this;
  }

  public double getValue() {
    return value;
  }

  void setValue(double value) {
    this.value = value;
  }

  public double getReducedCost() {
    return reducedCost;
  }

  void setReducedCost(double reducedCost) {
    this.reducedCost = reducedCost;
  }

  public double getMipStart() {
    return mipStart;
  }

  public Variable setMipStart(double mipStart) {
    this.mipStart = mipStart;
    return this;
  }

  void resetSolvedValues() {
    value = Double.NaN;
    reducedCost = Double.NaN;
  }
}
