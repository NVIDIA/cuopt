/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.Arrays;

public final class PDLPWarmStartData {
  private final double[] currentPrimalSolution;
  private final double[] currentDualSolution;
  private final double[] initialPrimalAverage;
  private final double[] initialDualAverage;
  private final double[] currentAty;
  private final double[] sumPrimalSolutions;
  private final double[] sumDualSolutions;
  private final double[] lastRestartDualityGapPrimalSolution;
  private final double[] lastRestartDualityGapDualSolution;
  private final double initialPrimalWeight;
  private final double initialStepSize;
  private final int totalPdlpIterations;
  private final int totalPdhgIterations;
  private final double lastCandidateKktScore;
  private final double lastRestartKktScore;
  private final double sumSolutionWeight;
  private final int iterationsSinceLastRestart;

  public PDLPWarmStartData(
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
      int iterationsSinceLastRestart) {
    this.currentPrimalSolution = copy(currentPrimalSolution);
    this.currentDualSolution = copy(currentDualSolution);
    this.initialPrimalAverage = copy(initialPrimalAverage);
    this.initialDualAverage = copy(initialDualAverage);
    this.currentAty = copy(currentAty);
    this.sumPrimalSolutions = copy(sumPrimalSolutions);
    this.sumDualSolutions = copy(sumDualSolutions);
    this.lastRestartDualityGapPrimalSolution = copy(lastRestartDualityGapPrimalSolution);
    this.lastRestartDualityGapDualSolution = copy(lastRestartDualityGapDualSolution);
    this.initialPrimalWeight = initialPrimalWeight;
    this.initialStepSize = initialStepSize;
    this.totalPdlpIterations = totalPdlpIterations;
    this.totalPdhgIterations = totalPdhgIterations;
    this.lastCandidateKktScore = lastCandidateKktScore;
    this.lastRestartKktScore = lastRestartKktScore;
    this.sumSolutionWeight = sumSolutionWeight;
    this.iterationsSinceLastRestart = iterationsSinceLastRestart;
  }

  static PDLPWarmStartData fromSolution(long solutionHandle) {
    return new PDLPWarmStartData(
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_CURRENT_PRIMAL_SOLUTION),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_CURRENT_DUAL_SOLUTION),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_INITIAL_PRIMAL_AVERAGE),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_INITIAL_DUAL_AVERAGE),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_CURRENT_ATY),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_SUM_PRIMAL_SOLUTIONS),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_SUM_DUAL_SOLUTIONS),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle,
            NativeCuOpt.PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_PRIMAL_SOLUTION),
        NativeCuOpt.getPdlpWarmStartVector(
            solutionHandle,
            NativeCuOpt.PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_DUAL_SOLUTION),
        NativeCuOpt.getPdlpWarmStartScalar(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_INITIAL_PRIMAL_WEIGHT),
        NativeCuOpt.getPdlpWarmStartScalar(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_INITIAL_STEP_SIZE),
        NativeCuOpt.getPdlpWarmStartInteger(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_TOTAL_PDLP_ITERATIONS),
        NativeCuOpt.getPdlpWarmStartInteger(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_TOTAL_PDHG_ITERATIONS),
        NativeCuOpt.getPdlpWarmStartScalar(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_LAST_CANDIDATE_KKT_SCORE),
        NativeCuOpt.getPdlpWarmStartScalar(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_LAST_RESTART_KKT_SCORE),
        NativeCuOpt.getPdlpWarmStartScalar(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_SUM_SOLUTION_WEIGHT),
        NativeCuOpt.getPdlpWarmStartInteger(
            solutionHandle, NativeCuOpt.PDLP_WARM_START_ITERATIONS_SINCE_LAST_RESTART));
  }

  void applyTo(long settingsHandle) {
    NativeCuOpt.setPdlpWarmStartData(
        settingsHandle,
        currentPrimalSolution,
        currentDualSolution,
        initialPrimalAverage,
        initialDualAverage,
        currentAty,
        sumPrimalSolutions,
        sumDualSolutions,
        lastRestartDualityGapPrimalSolution,
        lastRestartDualityGapDualSolution,
        initialPrimalWeight,
        initialStepSize,
        totalPdlpIterations,
        totalPdhgIterations,
        lastCandidateKktScore,
        lastRestartKktScore,
        sumSolutionWeight,
        iterationsSinceLastRestart);
  }

  private static double[] copy(double[] values) {
    return Arrays.copyOf(values, values.length);
  }

  public double[] getCurrentPrimalSolution() {
    return copy(currentPrimalSolution);
  }

  public double[] getCurrentDualSolution() {
    return copy(currentDualSolution);
  }

  public double[] getInitialPrimalAverage() {
    return copy(initialPrimalAverage);
  }

  public double[] getInitialDualAverage() {
    return copy(initialDualAverage);
  }

  public double[] getCurrentAty() {
    return copy(currentAty);
  }

  public double[] getSumPrimalSolutions() {
    return copy(sumPrimalSolutions);
  }

  public double[] getSumDualSolutions() {
    return copy(sumDualSolutions);
  }

  public double[] getLastRestartDualityGapPrimalSolution() {
    return copy(lastRestartDualityGapPrimalSolution);
  }

  public double[] getLastRestartDualityGapDualSolution() {
    return copy(lastRestartDualityGapDualSolution);
  }

  public double getInitialPrimalWeight() {
    return initialPrimalWeight;
  }

  public double getInitialStepSize() {
    return initialStepSize;
  }

  public int getTotalPdlpIterations() {
    return totalPdlpIterations;
  }

  public int getTotalPdhgIterations() {
    return totalPdhgIterations;
  }

  public double getLastCandidateKktScore() {
    return lastCandidateKktScore;
  }

  public double getLastRestartKktScore() {
    return lastRestartKktScore;
  }

  public double getSumSolutionWeight() {
    return sumSolutionWeight;
  }

  public int getIterationsSinceLastRestart() {
    return iterationsSinceLastRestart;
  }
}
