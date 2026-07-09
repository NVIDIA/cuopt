/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

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
    this.mip = NativeCuOpt.solutionIsMIP(handle);
    this.problemCategory = problemCategory;
    this.variableNames = variableNames == null ? new String[0] : Arrays.copyOf(variableNames, variableNames.length);
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  public boolean isMIP() {
    return mip;
  }

  public ProblemCategory getProblemCategory() {
    return problemCategory;
  }

  public double[] getPrimalSolution() {
    return NativeCuOpt.getPrimalSolution(handle(), numVariables);
  }

  public double[] getDualSolution() {
    requireLP("getDualSolution");
    return NativeCuOpt.getDualSolution(handle(), NativeCuOpt.getDualSolutionSize(handle()));
  }

  public double[] getReducedCost() {
    requireLP("getReducedCost");
    return NativeCuOpt.getReducedCosts(handle(), numVariables);
  }

  public double getPrimalObjective() {
    return NativeCuOpt.getObjectiveValue(handle());
  }

  public double getDualObjective() {
    requireLP("getDualObjective");
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
    return mip ? SolverMethod.UNSET : getLPStats().getSolvedBy();
  }

  public boolean getSolvedByPDLP() {
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

  public double getMIPGap() {
    requireMIP("getMIPGap");
    return NativeCuOpt.getMIPGap(handle());
  }

  public double getSolutionBound() {
    requireMIP("getSolutionBound");
    return NativeCuOpt.getSolutionBound(handle());
  }

  public LPStats getLPStats() {
    requireLP("getLPStats");
    return new LPStats(NativeCuOpt.getLPStats(handle()));
  }

  public MIPStats getMIPStats() {
    requireMIP("getMIPStats");
    return new MIPStats(NativeCuOpt.getMIPStats(handle()));
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  private void requireLP(String method) {
    if (mip) {
      throw new IllegalStateException(method + " is not available for MIP solutions");
    }
  }

  private void requireMIP(String method) {
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
