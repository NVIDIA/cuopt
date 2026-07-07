/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.nio.file.Path;

final class NativeCuOpt {
  static final int PDLP_WARM_START_CURRENT_PRIMAL_SOLUTION = 0;
  static final int PDLP_WARM_START_CURRENT_DUAL_SOLUTION = 1;
  static final int PDLP_WARM_START_INITIAL_PRIMAL_AVERAGE = 2;
  static final int PDLP_WARM_START_INITIAL_DUAL_AVERAGE = 3;
  static final int PDLP_WARM_START_CURRENT_ATY = 4;
  static final int PDLP_WARM_START_SUM_PRIMAL_SOLUTIONS = 5;
  static final int PDLP_WARM_START_SUM_DUAL_SOLUTIONS = 6;
  static final int PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_PRIMAL_SOLUTION = 7;
  static final int PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_DUAL_SOLUTION = 8;
  static final int PDLP_WARM_START_INITIAL_PRIMAL_WEIGHT = 9;
  static final int PDLP_WARM_START_INITIAL_STEP_SIZE = 10;
  static final int PDLP_WARM_START_TOTAL_PDLP_ITERATIONS = 11;
  static final int PDLP_WARM_START_TOTAL_PDHG_ITERATIONS = 12;
  static final int PDLP_WARM_START_LAST_CANDIDATE_KKT_SCORE = 13;
  static final int PDLP_WARM_START_LAST_RESTART_KKT_SCORE = 14;
  static final int PDLP_WARM_START_SUM_SOLUTION_WEIGHT = 15;
  static final int PDLP_WARM_START_ITERATIONS_SINCE_LAST_RESTART = 16;

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
  static native String[] getSolverParameterNames();
  static native long createEmptyProblem();
  static native long parseMpsProblem(String path, boolean fixedMpsFormat);
  static native long readProblemWithFormat(String path, boolean fixedMpsFormat);

  static native long createSolverSettings();
  static native void destroySolverSettings(long handle);
  static native void setParameter(long handle, String name, String value);
  static native void setIntegerParameter(long handle, String name, int value);
  static native void setFloatParameter(long handle, String name, double value);
  static native String getParameter(long handle, String name);
  static native void loadParametersFromFile(long handle, String path);
  static native boolean dumpParametersToFile(long handle, String path, boolean hyperparametersOnly);
  static native void setInitialPrimalSolution(long handle, double[] values);
  static native void setInitialDualSolution(long handle, double[] values);
  static native void addMipStart(long handle, double[] values);
  static native void registerMipGetSolutionCallback(
      long handle, MipSolutionCallback callback, Object userData, int numVariables);
  static native void registerMipSetSolutionCallback(
      long handle, MipSetSolutionCallback callback, Object userData, int numVariables);
  static native void setPdlpWarmStartData(
      long handle,
      double[] currentPrimalSolution,
      double[] currentDualSolution,
      double[] initialPrimalAverage,
      double[] initialDualAverage,
      double[] currentAty,
      double[] sumPrimalSolutions,
      double[] sumDualSolutions,
      double[] lastRestartDualityGapPrimalSolution,
      double[] lastRestartDualityGapDualSolution,
      double initialPrimalWeight,
      double initialStepSize,
      int totalPdlpIterations,
      int totalPdhgIterations,
      double lastCandidateKktScore,
      double lastRestartKktScore,
      double sumSolutionWeight,
      int iterationsSinceLastRestart);

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

  static native long createRangedProblem(
      int numConstraints,
      int numVariables,
      int objectiveSense,
      double objectiveOffset,
      double[] objectiveCoefficients,
      int[] rowOffsets,
      int[] columnIndices,
      double[] values,
      double[] constraintLowerBounds,
      double[] constraintUpperBounds,
      double[] variableLowerBounds,
      double[] variableUpperBounds,
      byte[] variableTypes);

  static native long readProblem(String path);
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
  static native double[] getConstraintRhs(long handle);
  static native double[] getConstraintLowerBounds(long handle);
  static native double[] getConstraintUpperBounds(long handle);
  static native double[] getVariableLowerBounds(long handle);
  static native double[] getVariableUpperBounds(long handle);
  static native byte[] getVariableTypes(long handle);
  static native void setMaximize(long handle, boolean maximize);
  static native void setConstraintMatrix(long handle, double[] values, int[] indices, int[] offsets);
  static native void setConstraintBounds(long handle, double[] values);
  static native void setObjectiveCoefficients(long handle, double[] values);
  static native void setObjectiveScalingFactor(long handle, double value);
  static native double getObjectiveScalingFactor(long handle);
  static native void setObjectiveOffset(long handle, double value);
  static native void setQuadraticObjectiveMatrix(long handle, double[] values, int[] indices, int[] offsets);
  static native void setVariableLowerBounds(long handle, double[] values);
  static native void setVariableUpperBounds(long handle, double[] values);
  static native void setConstraintLowerBounds(long handle, double[] values);
  static native void setConstraintUpperBounds(long handle, double[] values);
  static native void setRowTypes(long handle, byte[] values);
  static native void setVariableTypes(long handle, byte[] values);
  static native void setVariableNames(long handle, String[] values);
  static native void setRowNames(long handle, String[] values);
  static native void setObjectiveName(long handle, String value);
  static native void setProblemName(long handle, String value);
  static native void setInitialPrimalSolutionOnProblem(long handle, double[] values);
  static native void setInitialDualSolutionOnProblem(long handle, double[] values);
  static native double[] getQuadraticObjectiveValues(long handle);
  static native int[] getQuadraticObjectiveIndices(long handle);
  static native int[] getQuadraticObjectiveOffsets(long handle);
  static native String[] getVariableNames(long handle);
  static native String[] getRowNames(long handle);
  static native String getObjectiveName(long handle);
  static native String getProblemName(long handle);
  static native int getProblemCategory(long handle);
  static native Object[] getQuadraticConstraints(long handle);
  static native void clearQuadraticConstraints(long handle);
  static native boolean isMip(long handle);
  static native long solve(long problemHandle, long settingsHandle);

  static native void destroySolution(long handle);
  static native boolean solutionIsMip(long handle);
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
  static native double getMipGap(long handle);
  static native double getSolutionBound(long handle);
  static native double[] getLpStats(long handle);
  static native double[] getMipStats(long handle);
  static native boolean hasPdlpWarmStartData(long handle);
  static native double[] getPdlpWarmStartVector(long handle, int fieldId);
  static native double getPdlpWarmStartScalar(long handle, int fieldId);
  static native int getPdlpWarmStartInteger(long handle, int fieldId);
}
