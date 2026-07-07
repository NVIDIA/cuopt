/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class Solution implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();
  private final NativeHandle nativeHandle;
  private final Cleaner.Cleanable cleanable;
  private final int numVariables;
  private final int numConstraints;
  private final boolean mip;
  private final ProblemCategory problemCategory;
  private final String[] variableNames;

  Solution(long handle, int numVariables, int numConstraints) {
    this(handle, numVariables, numConstraints, ProblemCategory.LP, new String[0]);
  }

  Solution(
      long handle,
      int numVariables,
      int numConstraints,
      ProblemCategory problemCategory,
      String[] variableNames) {
    this.nativeHandle = new NativeHandle(handle);
    this.cleanable = CLEANER.register(this, nativeHandle);
    this.numVariables = numVariables;
    this.numConstraints = numConstraints;
    this.mip = NativeCuOpt.solutionIsMip(handle);
    this.problemCategory = problemCategory;
    this.variableNames = variableNames == null ? new String[0] : Arrays.copyOf(variableNames, variableNames.length);
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  public boolean isMip() {
    return mip;
  }

  public ProblemCategory getProblemCategory() {
    return problemCategory;
  }

  public double[] getPrimalSolution() {
    return NativeCuOpt.getPrimalSolution(handle(), numVariables);
  }

  public double[] getDualSolution() {
    requireLp("getDualSolution");
    return NativeCuOpt.getDualSolution(handle(), NativeCuOpt.getDualSolutionSize(handle()));
  }

  public double[] getReducedCost() {
    requireLp("getReducedCost");
    return NativeCuOpt.getReducedCosts(handle(), numVariables);
  }

  public double getPrimalObjective() {
    return NativeCuOpt.getObjectiveValue(handle());
  }

  public double getDualObjective() {
    requireLp("getDualObjective");
    return NativeCuOpt.getDualObjectiveValue(handle());
  }

  public TerminationStatus getTerminationStatus() {
    return TerminationStatus.fromNative(NativeCuOpt.getTerminationStatus(handle()));
  }

  public String getTerminationReason() {
    return getTerminationStatus().name();
  }

  public int getErrorStatus() {
    return NativeCuOpt.getErrorStatus(handle());
  }

  public String getErrorMessage() {
    return NativeCuOpt.getErrorString(handle());
  }

  public double getSolveTime() {
    return NativeCuOpt.getSolveTime(handle());
  }

  public SolverMethod getSolvedBy() {
    return mip ? SolverMethod.UNSET : getLpStats().getSolvedBy();
  }

  public boolean getSolvedByPdlp() {
    return !mip && getSolvedBy() == SolverMethod.PDLP;
  }

  public Map<String, Double> getVars() {
    double[] values = getPrimalSolution();
    Map<String, Double> result = new LinkedHashMap<>();
    int count = Math.min(variableNames.length, values.length);
    for (int i = 0; i < count; ++i) {
      result.put(variableNames[i], values[i]);
    }
    return Collections.unmodifiableMap(result);
  }

  public double getMipGap() {
    requireMip("getMipGap");
    return NativeCuOpt.getMipGap(handle());
  }

  public double getSolutionBound() {
    requireMip("getSolutionBound");
    return NativeCuOpt.getSolutionBound(handle());
  }

  public LPStats getLpStats() {
    requireLp("getLpStats");
    return new LPStats(NativeCuOpt.getLpStats(handle()));
  }

  public MIPStats getMipStats() {
    requireMip("getMipStats");
    return new MIPStats(NativeCuOpt.getMipStats(handle()));
  }

  public boolean hasPdlpWarmStartData() {
    requireLp("hasPdlpWarmStartData");
    return NativeCuOpt.hasPdlpWarmStartData(handle());
  }

  public PDLPWarmStartData getPdlpWarmStartData() {
    requireLp("getPdlpWarmStartData");
    return PDLPWarmStartData.fromSolution(handle());
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  private void requireLp(String method) {
    if (mip) {
      throw new IllegalStateException(method + " is not available for MIP solutions");
    }
  }

  private void requireMip(String method) {
    if (!mip) {
      throw new IllegalStateException(method + " is not available for LP solutions");
    }
  }

  private static final class NativeHandle implements Runnable {
    private long handle;

    NativeHandle(long handle) {
      this.handle = handle;
    }

    void requireOpen() {
      if (handle == 0) {
        throw new IllegalStateException("Solution is closed");
      }
    }

    @Override
    public void run() {
      if (handle != 0) {
        NativeCuOpt.destroySolution(handle);
        handle = 0;
      }
    }
  }
}
