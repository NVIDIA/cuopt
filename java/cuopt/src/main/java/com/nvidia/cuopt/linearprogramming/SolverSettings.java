/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class SolverSettings implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();
  private final NativeHandle nativeHandle;
  private final Cleaner.Cleanable cleanable;
  private PDLPWarmStartData pdlpWarmStartData;
  private final List<Object> mipCallbacks = new ArrayList<>();

  public SolverSettings() {
    this.nativeHandle = new NativeHandle(NativeCuOpt.createSolverSettings());
    this.cleanable = CLEANER.register(this, nativeHandle);
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  public static Set<String> getSolverParameterNames() {
    return new LinkedHashSet<>(Arrays.asList(NativeCuOpt.getSolverParameterNames()));
  }

  public static String getSolverSetting(String name) {
    try (SolverSettings settings = new SolverSettings()) {
      return settings.getParameterAsString(name);
    }
  }

  public static Object getTypedSolverSetting(String name) {
    return typedValue(getSolverSetting(name));
  }

  public static String getSolverSettingAsString(String name) {
    try (SolverSettings settings = new SolverSettings()) {
      return settings.getParameterAsString(name);
    }
  }

  public SolverSettings setParameter(String name, String value) {
    NativeCuOpt.setParameter(handle(), name, value);
    return this;
  }

  public SolverSettings setParameter(String name, int value) {
    NativeCuOpt.setIntegerParameter(handle(), name, value);
    return this;
  }

  public SolverSettings setParameter(String name, double value) {
    NativeCuOpt.setFloatParameter(handle(), name, value);
    return this;
  }

  public SolverSettings setParameter(String name, boolean value) {
    NativeCuOpt.setIntegerParameter(handle(), name, value ? 1 : 0);
    return this;
  }

  public String getParameter(String name) {
    return getParameterAsString(name);
  }

  public Object getTypedParameter(String name) {
    return typedValue(getParameterAsString(name));
  }

  public String getParameterAsString(String name) {
    return NativeCuOpt.getParameter(handle(), name);
  }

  public SolverSettings setMethod(SolverMethod method) {
    return setParameter(CuOptConstants.CUOPT_METHOD, method.nativeValue());
  }

  public SolverSettings setPdlpSolverMode(PDLPSolverMode mode) {
    return setParameter(CuOptConstants.CUOPT_PDLP_SOLVER_MODE, mode.nativeValue());
  }

  public SolverSettings setOptimalityTolerance(double tolerance) {
    for (String parameter : getSolverParameterNames()) {
      if (parameter.endsWith("tolerance")
          && !parameter.startsWith("mip")
          && !parameter.contains("infeasible")) {
        setParameter(parameter, tolerance);
      }
    }
    return this;
  }

  public SolverSettings setInitialPrimalSolution(double[] values) {
    NativeCuOpt.setInitialPrimalSolution(handle(), Arrays.copyOf(values, values.length));
    return this;
  }

  public SolverSettings setInitialDualSolution(double[] values) {
    NativeCuOpt.setInitialDualSolution(handle(), Arrays.copyOf(values, values.length));
    return this;
  }

  public SolverSettings addMipStart(double[] values) {
    NativeCuOpt.addMipStart(handle(), Arrays.copyOf(values, values.length));
    return this;
  }

  public SolverSettings setPdlpWarmStartData(PDLPWarmStartData warmStartData) {
    if (warmStartData == null) {
      throw new IllegalArgumentException("warmStartData must not be null");
    }
    warmStartData.applyTo(handle());
    this.pdlpWarmStartData = warmStartData;
    return this;
  }

  public SolverSettings setMipCallback(
      MipSolutionCallback callback, Object userData, int numVariables) {
    NativeCuOpt.registerMipGetSolutionCallback(handle(), callback, userData, numVariables);
    mipCallbacks.add(callback);
    return this;
  }

  public SolverSettings setMipCallback(
      MipSetSolutionCallback callback, Object userData, int numVariables) {
    NativeCuOpt.registerMipSetSolutionCallback(handle(), callback, userData, numVariables);
    mipCallbacks.add(callback);
    return this;
  }

  public List<Object> getMipCallbacks() {
    return Collections.unmodifiableList(mipCallbacks);
  }

  public PDLPWarmStartData getPdlpWarmStartData() {
    return pdlpWarmStartData;
  }

  public boolean dumpParametersToFile(String path, boolean hyperparametersOnly) {
    return NativeCuOpt.dumpParametersToFile(handle(), path, hyperparametersOnly);
  }

  public boolean dumpParametersToFile(String path) {
    return dumpParametersToFile(path, true);
  }

  public SolverSettings loadParametersFromFile(String path) {
    NativeCuOpt.loadParametersFromFile(handle(), path);
    return this;
  }

  public Map<String, Object> toDict() {
    Map<String, Object> result = new LinkedHashMap<>();
    Map<String, Object> tolerances = new LinkedHashMap<>();
    for (String parameter : getSolverParameterNames()) {
      Object value = getTypedParameter(parameter);
      if (parameter.endsWith("tolerance")) {
        tolerances.put(parameter, value);
      } else {
        result.put(parameter, value instanceof Double && ((Double) value).isInfinite() ? null : value);
      }
    }
    result.put("tolerances", tolerances);
    return Collections.unmodifiableMap(result);
  }

  private static Object typedValue(String value) {
    if ("true".equalsIgnoreCase(value)) {
      return true;
    }
    if ("false".equalsIgnoreCase(value)) {
      return false;
    }
    try {
      return Integer.valueOf(value);
    } catch (NumberFormatException ignored) {
      try {
        return Double.valueOf(value);
      } catch (NumberFormatException ignoredAgain) {
        return value;
      }
    }
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  private static final class NativeHandle implements Runnable {
    private long handle;

    NativeHandle(long handle) {
      this.handle = handle;
    }

    void requireOpen() {
      if (handle == 0) {
        throw new IllegalStateException("SolverSettings is closed");
      }
    }

    @Override
    public void run() {
      if (handle != 0) {
        NativeCuOpt.destroySolverSettings(handle);
        handle = 0;
      }
    }
  }
}
