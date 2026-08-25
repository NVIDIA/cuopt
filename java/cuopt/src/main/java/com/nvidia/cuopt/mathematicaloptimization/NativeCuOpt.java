/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.nio.file.Path;

final class NativeCuOpt {
  static {
    String nativeDir = System.getProperty("cuopt.native.dir");
    if (nativeDir == null || nativeDir.isBlank()) {
      System.loadLibrary("cuopt_jni");
    } else {
      System.load(Path.of(nativeDir, System.mapLibraryName("cuopt_jni")).toAbsolutePath().toString());
    }
  }

  private NativeCuOpt() {}

  static native int getFloatSize();
  static native long readProblemWithFormat(String path, boolean fixedMPSFormat);

  static native long createSolverSettings();
  static native void destroySolverSettings(long handle);
  static native void setSetting(long handle, String name, String value);
  static native void setIntegerSetting(long handle, String name, int value);
  static native void setFloatSetting(long handle, String name, double value);
  static native String getSetting(long handle, String name);
  static native void addMIPStart(long handle, double[] values);
  static native void setInitialPrimalSolution(long handle, double[] values);
  static native void setInitialDualSolution(long handle, double[] values);
  static native void registerMIPGetSolutionCallback(
      long handle, MIPSolutionCallback callback, Object userData, int numVariables);
  static native void registerMIPSetSolutionCallback(
      long handle, MIPSetSolutionCallback callback, Object userData, int numVariables);
  static native long createProblem(
      int numConstraints,
      int numVariables,
      int objectiveSense,
      double objectiveOffset,
      double[] objectiveCoefficients,
      int[] rowOffsets,
      int[] columnIndices,
      double[] values,
      byte[] constraintSense,
      double[] rhs,
      double[] lowerBounds,
      double[] upperBounds,
      byte[] variableTypes);

  static native void writeProblem(long handle, String path);
  static native void destroyProblem(long handle);
  static native void setQuadraticObjective(long handle, int[] rows, int[] columns, double[] values);
  static native void addQuadraticConstraint(
      long handle,
      int[] rows,
      int[] columns,
      double[] values,
      int[] linearIndices,
      double[] linearCoefficients,
      byte sense,
      double rhs);

  static native int getNumVariables(long handle);
  static native int getNumConstraints(long handle);
  static native int getNumNonZeros(long handle);
  static native int getObjectiveSense(long handle);
  static native double getObjectiveOffset(long handle);
  static native double[] getObjectiveCoefficients(long handle);
  static native Object[] getConstraintMatrix(long handle);
  static native byte[] getConstraintSense(long handle);
  static native double[] getConstraintRHS(long handle);
  static native double[] getConstraintLowerBounds(long handle);
  static native double[] getConstraintUpperBounds(long handle);
  static native double[] getVariableLowerBounds(long handle);
  static native double[] getVariableUpperBounds(long handle);
  static native byte[] getVariableTypes(long handle);
  static native void setVariableNames(long handle, String[] values);
  static native void setRowNames(long handle, String[] values);
  static native void setProblemName(long handle, String value);
  static native double[] getQuadraticObjectiveValues(long handle);
  static native int[] getQuadraticObjectiveIndices(long handle);
  static native int[] getQuadraticObjectiveOffsets(long handle);
  static native String[] getVariableNames(long handle);
  static native String[] getRowNames(long handle);
  static native String getProblemName(long handle);
  static native int getProblemCategory(long handle);
  static native Object[] getQuadraticConstraints(long handle);
  static native long solve(long problemHandle, long settingsHandle);

  static native void destroySolution(long handle);
  static native int getTerminationStatus(long handle);
  static native int getErrorStatus(long handle);
  static native String getErrorString(long handle);
  static native double[] getPrimalSolution(long handle, int size);
  static native int getDualSolutionSize(long handle);
  static native double[] getDualSolution(long handle, int size);
  static native double[] getReducedCosts(long handle, int size);
  static native double getObjectiveValue(long handle);
  static native double getDualObjectiveValue(long handle);
  static native double getSolveTime(long handle);
  static native int getSolutionIntAttribute(long handle, int attribute);
  static native double getSolutionFloatAttribute(long handle, int attribute);
  static native double getMIPGap(long handle);
  static native double getSolutionBound(long handle);
}
