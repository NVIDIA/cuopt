/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

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
  private final String[] variableNames;

  Solution(long handle, int numVariables, int numConstraints) {
    this(handle, numVariables, numConstraints, false, new String[0]);
  }

  Solution(
      long handle,
      int numVariables,
      int numConstraints,
      boolean mip,
      String[] variableNames) {
    this.nativeHandle = new NativeHandle(handle);
    this.cleanable = CLEANER.register(this, nativeHandle);
    this.numVariables = numVariables;
    this.numConstraints = numConstraints;
    // Taken from the problem this solution was produced for, so the native layer is not
    // asked a second time.
    this.mip = mip;
    this.variableNames = variableNames == null ? new String[0] : Arrays.copyOf(variableNames, variableNames.length);
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  public boolean isMIP() {
    return mip;
  }

  double[] getPrimalSolution() {
    return NativeCuOpt.getPrimalSolution(handle(), numVariables);
  }

  double[] getDualSolution() {
    requireLP("getDualSolution");
    return NativeCuOpt.getDualSolution(handle(), NativeCuOpt.getDualSolutionSize(handle()));
  }

  double[] getReducedCost() {
    requireLP("getReducedCost");
    return NativeCuOpt.getReducedCosts(handle(), numVariables);
  }

  /**
   * Package-private: Problem copies this onto itself after a solve, and Problem.getSolveTime is
   * the public way to read it.
   */
  double getSolveTime() {
    return NativeCuOpt.getSolveTime(handle());
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

  public int getErrorStatus() {
    return NativeCuOpt.getErrorStatus(handle());
  }

  public String getErrorMessage() {
    return NativeCuOpt.getErrorString(handle());
  }

  public double getMIPGap() {
    requireMIP("getMIPGap");
    return NativeCuOpt.getMIPGap(handle());
  }

  public double getSolutionBound() {
    requireMIP("getSolutionBound");
    return NativeCuOpt.getSolutionBound(handle());
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  /**
   * Read a scalar integer solution attribute, selected by a
   * {@code CuOptConstants.CUOPT_SOLUTION_ATTR_*} value.
   *
   * <p>Which attributes a solution carries depends on the class of problem that produced it. A
   * selector that does not apply, or that does not have an integer value, raises
   * {@link CuOptException}.
   */
  public int getIntAttribute(int attribute) {
    return NativeCuOpt.getSolutionIntAttribute(handle(), attribute);
  }

  /**
   * Read a scalar floating-point solution attribute, selected by a
   * {@code CuOptConstants.CUOPT_SOLUTION_ATTR_*} value.
   *
   * <p>Which attributes a solution carries depends on the class of problem that produced it. A
   * selector that does not apply, or that does not have a floating-point value, raises
   * {@link CuOptException}.
   */
  public double getFloatAttribute(int attribute) {
    return NativeCuOpt.getSolutionFloatAttribute(handle(), attribute);
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
